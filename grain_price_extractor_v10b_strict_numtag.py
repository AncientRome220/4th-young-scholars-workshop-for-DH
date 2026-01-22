#!/usr/bin/env python3
r"""
grain_price_extractor_v10b_strict_numtag.py

Stage B (CLEAN / strict mention extraction):
- Read Stage A strict candidate CSV.
- Re-open each DDB XML and extract ALL mentions (multiple rows per papyrus)
  BUT ONLY from blocks that contain <num value="...">.

Mention rule (strict):
  grain AND (<num value exists) AND (money OR priceword)
  (unit optional; can be required with --require-unit)

Structured parsing improvements:
- Quantity pairs in BOTH orders:
    <num> + <unit>   and   <unit> + <num>
- Price pairs in BOTH orders:
    <num> + <currency> and <currency> + <num>
- Half fractions:
    <num value="1/2">𐅵</num> or literal 𐅵 are normalized to 1/2 and added (+0.5).

Output: mention-level CSV (UTF-8 with BOM by default).
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from lxml import etree

NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def strip_accents(s: str) -> str:
    if not s:
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def normalize_for_search(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = " ".join(s.split())
    return strip_accents(s).casefold()


def safe_string(node) -> str:
    if node is None:
        return ""
    try:
        return node.xpath("string()").strip()
    except Exception:
        return ""


def iter_blocks(div_node: etree._Element):
    blocks = div_node.xpath(".//tei:ab | .//tei:p | .//tei:seg | .//tei:l", namespaces=NS)
    for b in blocks:
        yield etree.QName(b).localname, b


def has_num_tag(block: etree._Element) -> Tuple[bool, List[str], List[str]]:
    vals: List[str] = []
    txts: List[str] = []
    nodes = block.xpath(".//tei:num[@value]", namespaces=NS)
    if not nodes:
        return False, vals, txts

    for n in nodes:
        v = (n.get("value") or "").strip()
        if v:
            vals.append(v)
        t = safe_string(n)
        if t:
            txts.append(t)

    return (len(vals) > 0), vals, txts


def block_text_with_num_values(elem: etree._Element) -> str:
    """
    Convert <num value="25">κε</num> -> "25" in the output string,
    so regex extraction is much easier/cleaner.
    """
    pieces: List[str] = []

    def walk(node: etree._Element):
        if node.text:
            pieces.append(node.text)

        for ch in node:
            if etree.QName(ch).localname == "num":
                v = (ch.get("value") or "").strip()
                if v:
                    pieces.append(v)
                else:
                    pieces.append(safe_string(ch))
                if ch.tail:
                    pieces.append(ch.tail)
            else:
                walk(ch)
                if ch.tail:
                    pieces.append(ch.tail)

    walk(elem)
    return " ".join(" ".join(pieces).split())


def parse_num_maybe_half(num_str: str, half_str: str | None) -> str:
    try:
        n = float(num_str)
    except Exception:
        return num_str
    if half_str and half_str.strip() == "1/2":
        n += 0.5
    if abs(n - int(n)) < 1e-9:
        return str(int(n))
    return str(n)


def compile_patterns() -> Dict[str, re.Pattern]:
    grain = re.compile(r"(σιτ|πυρ|κριθ|ζεα|ολυρ|σταχυ|αλευρ)", re.UNICODE)
    unit = re.compile(r"(αρταβ|μεδιμν|χοινι|μετρ|κοτυλ|χοινιξ|γομφ|καλαθ)", re.UNICODE)
    money = re.compile(r"(δραχμ|οβολ|δηναρ|μνα|ταλαντ|σεστ|ἀσσ|𐅵|χαλκ)", re.UNICODE)
    priceword = re.compile(r"(τιμη|τιμ(?=[\.\)\(]))", re.UNICODE)

    unit_core = r"(?:αρταβ\w*|μεδιμν\w*|χοιν\w*|μετρ\w*|κοτυλ\w*)"
    cur_core = r"(?:δραχμ\w*|οβολ\w*|δηναρ\w*|μνα\w*|ταλαντ\w*|σεστ\w*|ἀσσ\w*|χαλκ\w*|𐅵)"

    num_maybe_half = r"(?P<num>[0-9]+)(?:\W+(?P<half>1/2))?"

    qty_num_unit = re.compile(rf"{num_maybe_half}\W+(?P<unit>{unit_core})", re.UNICODE)
    qty_unit_num = re.compile(rf"(?P<unit>{unit_core})\W+{num_maybe_half}", re.UNICODE)

    price_num_cur = re.compile(rf"{num_maybe_half}\W+(?P<cur>{cur_core})", re.UNICODE)
    price_cur_num = re.compile(rf"(?P<cur>{cur_core})\W+{num_maybe_half}", re.UNICODE)

    return {
        "grain": grain,
        "unit": unit,
        "money": money,
        "priceword": priceword,
        "qty_num_unit": qty_num_unit,
        "qty_unit_num": qty_unit_num,
        "price_num_cur": price_num_cur,
        "price_cur_num": price_cur_num,
    }


@dataclass
class MentionRow:
    DDB_ID: str
    Mention_ID: int
    Block_Tag: str

    Title: str
    Place: str
    Date_Text: str
    Date_When: str
    Date_NotBefore: str
    Date_NotAfter: str

    Grain_Hits: str
    Unit_Hits: str
    Money_Hits: str
    Priceword_Hit: str

    Number_Hit: str
    Number_Type: str
    Num_Tag_Values: str
    Num_Tag_Texts: str

    Quantities: str
    Prices: str

    Raw_Block: str
    Value_Block: str


def extract_mentions_for_doc(
    ddb_xml_path: Path,
    ddb_id: str,
    meta: Dict[str, str],
    scan_commentary: bool,
    require_unit: bool,
    logger: logging.Logger,
) -> List[MentionRow]:
    pats = compile_patterns()
    mentions: List[MentionRow] = []

    try:
        root = etree.parse(str(ddb_xml_path)).getroot()
    except Exception as e:
        logger.debug("Failed to parse %s (%s)", ddb_xml_path, e)
        return mentions

    div_xpath = "//tei:div[@type='edition']"
    if scan_commentary:
        div_xpath += " | //tei:div[@type='commentary']"
    divs = root.xpath(div_xpath, namespaces=NS)

    mention_id = 0

    for div in divs:
        for tag, block in iter_blocks(div):
            raw = safe_string(block)
            if not raw:
                continue
            raw = " ".join(raw.split())
            norm = normalize_for_search(raw)

            if not pats["grain"].search(norm):
                continue

            ok_num, num_vals, num_txts = has_num_tag(block)
            if not ok_num:
                continue

            has_unit_sig = bool(pats["unit"].search(norm))
            if require_unit and not has_unit_sig:
                continue

            has_money_sig = bool(pats["money"].search(norm))
            has_priceword_sig = bool(pats["priceword"].search(norm))
            if not (has_money_sig or has_priceword_sig):
                continue

            grain_hits = sorted(set(pats["grain"].findall(norm)))
            unit_hits = sorted(set(pats["unit"].findall(norm)))
            money_hits = sorted(set(pats["money"].findall(norm)))
            priceword_hit = "yes" if has_priceword_sig else "no"

            value_text = block_text_with_num_values(block)
            value_text = value_text.replace("𐅵", " 1/2 ")
            value_norm = normalize_for_search(value_text)

            quantities = []
            seenq = set()
            for m in pats["qty_num_unit"].finditer(value_norm):
                n = parse_num_maybe_half(m.group("num"), m.groupdict().get("half"))
                u = m.group("unit")
                item = f"{n} {u}"
                if item not in seenq:
                    seenq.add(item)
                    quantities.append(item)
            for m in pats["qty_unit_num"].finditer(value_norm):
                n = parse_num_maybe_half(m.group("num"), m.groupdict().get("half"))
                u = m.group("unit")
                item = f"{n} {u}"
                if item not in seenq:
                    seenq.add(item)
                    quantities.append(item)

            prices = []
            seenp = set()
            for m in pats["price_num_cur"].finditer(value_norm):
                n = parse_num_maybe_half(m.group("num"), m.groupdict().get("half"))
                c = m.group("cur")
                item = f"{n} {c}"
                if item not in seenp:
                    seenp.add(item)
                    prices.append(item)
            for m in pats["price_cur_num"].finditer(value_norm):
                n = parse_num_maybe_half(m.group("num"), m.groupdict().get("half"))
                c = m.group("cur")
                item = f"{n} {c}"
                if item not in seenp:
                    seenp.add(item)
                    prices.append(item)

            mention_id += 1
            mentions.append(
                MentionRow(
                    DDB_ID=ddb_id,
                    Mention_ID=mention_id,
                    Block_Tag=tag,
                    Title=meta.get("Title", ""),
                    Place=meta.get("Place", ""),
                    Date_Text=meta.get("Date_Text", ""),
                    Date_When=meta.get("Date_When", ""),
                    Date_NotBefore=meta.get("Date_NotBefore", ""),
                    Date_NotAfter=meta.get("Date_NotAfter", ""),
                    Grain_Hits=";".join(grain_hits),
                    Unit_Hits=";".join(unit_hits),
                    Money_Hits=";".join(money_hits),
                    Priceword_Hit=priceword_hit,
                    Number_Hit=f"num:{num_vals[0]}",
                    Number_Type="num_tag",
                    Num_Tag_Values=";".join([v for v in num_vals if v]),
                    Num_Tag_Texts=";".join([t for t in num_txts if t]),
                    Quantities=";".join(quantities),
                    Prices=";".join(prices),
                    Raw_Block=raw[:2000],
                    Value_Block=value_text[:2000],
                )
            )

    return mentions


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage B mention extractor (STRICT: requires <num value>).")
    parser.add_argument("--candidates", required=True, help="Stage A strict candidates CSV")
    parser.add_argument("--ddb-dir", required=True, help="Path to DDB EpiDoc XML directory")
    parser.add_argument("--out", default="grain_mentions_v10b_strict.csv", help="Output mention-level CSV")

    parser.add_argument("--scan-commentary", action="store_true", help="Also scan <div type='commentary'>")
    parser.add_argument("--require-unit", action="store_true", help="Require unit term (may miss some)")
    parser.add_argument("--max-docs", type=int, default=0, help="Process only N docs (0 = all)")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding (default utf-8-sig for Excel)")

    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("grain_price_extractor_v10b_strict")

    cand_path = Path(args.candidates)
    ddb_dir = Path(args.ddb_dir)
    out_csv = Path(args.out)

    if not cand_path.exists():
        raise SystemExit(f"Candidates CSV not found: {cand_path}")
    if not ddb_dir.exists():
        raise SystemExit(f"DDB dir not found: {ddb_dir}")

    df = pd.read_csv(cand_path).fillna("")
    if not {"DDB_ID", "XML_RelPath"}.issubset(df.columns):
        raise SystemExit("Candidates CSV must contain columns: DDB_ID, XML_RelPath")

    mentions: List[MentionRow] = []
    processed = 0

    for _, r in df.iterrows():
        ddb_id = str(r.get("DDB_ID", "")).strip()
        rel = str(r.get("XML_RelPath", "")).strip().replace("\\", "/")
        if not ddb_id or not rel:
            continue

        xml_path = ddb_dir / rel
        if not xml_path.exists():
            logger.debug("Missing XML for %s: %s", ddb_id, xml_path)
            continue

        meta = {
            "Title": str(r.get("Title", "")),
            "Place": str(r.get("Place", "")),
            "Date_Text": str(r.get("Date_Text", "")),
            "Date_When": str(r.get("Date_When", "")),
            "Date_NotBefore": str(r.get("Date_NotBefore", "")),
            "Date_NotAfter": str(r.get("Date_NotAfter", "")),
        }

        mentions.extend(
            extract_mentions_for_doc(
                ddb_xml_path=xml_path,
                ddb_id=ddb_id,
                meta=meta,
                scan_commentary=args.scan_commentary,
                require_unit=args.require_unit,
                logger=logger,
            )
        )

        processed += 1
        if args.max_docs and processed >= args.max_docs:
            break

    if not mentions:
        logger.warning("No mentions extracted. Nothing written.")
        return

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding=args.encoding) as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(mentions[0]).keys()))
        writer.writeheader()
        for m in mentions:
            writer.writerow(asdict(m))

    logger.info("Stage B (strict num_tag) done. Docs processed: %d. Mentions: %d", processed, len(mentions))
    logger.info("Wrote: %s", out_csv.resolve())


if __name__ == "__main__":
    main()
