#!/usr/bin/env python3
r"""
grain_candidate_harvester_v10a_strict_numtag.py

Stage A (CLEAN / strict):
- Scan DDB/EpiDoc TEI XML for grain-related contexts.
- Candidate rule (strict, per block):
    grain AND (<num value="..."> exists) AND (money OR priceword)
  (unit optional; you can require it with --require-unit)

Why strict?
- If your corpus is consistently encoded with <num value="...">, this yields a cleaner dataset
  with fewer false positives and fewer weird number interpretations.

Output: ONE ROW per papyrus (DDB_ID), aggregating hits across matching blocks, plus sample snippets.
CSV encoding: UTF-8 with BOM (utf-8-sig) by default for Excel compatibility.

Usage:
  python grain_candidate_harvester_v10a_strict_numtag.py ^
    --ddb-dir "C:\research\DDB_EpiDoc_XML" ^
    --hgv-dir "C:\research\HGV_meta_EpiDoc" ^
    --out wheat_candidates_v10a_strict.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

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


def iter_xml_files(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".xml"):
                yield Path(dirpath) / fn


@dataclass
class HgvMeta:
    title: str = ""
    place: str = ""
    date_text: str = ""
    date_when: str = ""
    date_notbefore: str = ""
    date_notafter: str = ""


def get_ddb_id(root: etree._Element) -> str:
    idno = root.xpath('string(//tei:idno[@type="ddb-hybrid"][1])', namespaces=NS).strip()
    if idno:
        return idno
    return root.xpath("string(//tei:idno[1])", namespaces=NS).strip()


def parse_hgv_meta(hgv_root: etree._Element) -> Tuple[str, HgvMeta]:
    ddb_id = hgv_root.xpath('string(//tei:idno[@type="ddb-hybrid"][1])', namespaces=NS).strip()
    meta = HgvMeta()

    title_node = hgv_root.xpath("//tei:titleStmt/tei:title[1]", namespaces=NS)
    if title_node:
        meta.title = safe_string(title_node[0])

    place_node = hgv_root.xpath("//tei:origPlace[1]", namespaces=NS)
    if place_node:
        meta.place = safe_string(place_node[0])
    else:
        place_node2 = hgv_root.xpath("//tei:placeName[1]", namespaces=NS)
        if place_node2:
            meta.place = safe_string(place_node2[0])

    date_node = hgv_root.xpath("//tei:origDate[1]", namespaces=NS)
    if date_node:
        dn = date_node[0]
        meta.date_text = safe_string(dn)
        meta.date_when = (dn.get("when") or "").strip()
        meta.date_notbefore = (dn.get("notBefore") or "").strip()
        meta.date_notafter = (dn.get("notAfter") or "").strip()

    return ddb_id, meta


def build_hgv_index(hgv_dir: Path, logger: logging.Logger) -> Dict[str, HgvMeta]:
    idx: Dict[str, HgvMeta] = {}
    total = 0
    kept = 0

    for fp in iter_xml_files(hgv_dir):
        total += 1
        try:
            root = etree.parse(str(fp)).getroot()
            ddb_id, meta = parse_hgv_meta(root)
            if ddb_id:
                idx[ddb_id] = meta
                kept += 1
        except Exception as e:
            logger.debug("HGV parse failed: %s (%s)", fp, e)

    logger.info("HGV index built: %d/%d files with ddb-hybrid id.", kept, total)
    return idx


@dataclass
class CandidateRow:
    DDB_ID: str
    XML_RelPath: str
    Title: str
    Place: str
    Date_Text: str
    Date_When: str
    Date_NotBefore: str
    Date_NotAfter: str

    Score: int
    Block_Match_Count: int

    Grain_Hits: str
    Unit_Hits: str
    Money_Hits: str
    Priceword_Hit: str

    Num_Tag_Values: str
    Num_Tag_Texts: str

    Snippet_1: str
    Snippet_2: str
    Snippet_3: str


def compile_patterns() -> Dict[str, re.Pattern]:
    # Run on NORMALIZED strings
    grain = re.compile(r"(σιτ|πυρ|κριθ|ζεα|ολυρ|σταχυ|αλευρ)", re.UNICODE)
    unit = re.compile(r"(αρταβ|μεδιμν|χοινι|μετρ|κοτυλ|χοινιξ|γομφ|καλαθ)", re.UNICODE)
    money = re.compile(r"(δραχμ|οβολ|δηναρ|μνα|ταλαντ|σεστ|ἀσσ|𐅵|χαλκ)", re.UNICODE)

    # Keep τιμή vocabulary as a separate signal (good recall for texts like P.Mich. II 127)
    priceword = re.compile(r"(τιμη|τιμ(?=[\.\)\(]))", re.UNICODE)

    return {
        "grain": grain,
        "unit": unit,
        "money": money,
        "priceword": priceword,
    }


def iter_blocks(div_node: etree._Element):
    blocks = div_node.xpath(".//tei:ab | .//tei:p | .//tei:seg | .//tei:l", namespaces=NS)
    for b in blocks:
        yield etree.QName(b).localname, b


def has_num_tag(block: etree._Element) -> Tuple[bool, List[str], List[str]]:
    """
    Strict number evidence: require <num value="...">.
    Returns (has_num, values, texts).
    """
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


def harvest_candidates(
    ddb_dir: Path,
    hgv_index: Dict[str, HgvMeta],
    out_csv: Path,
    scan_commentary: bool,
    require_unit: bool,
    max_snippets: int,
    logger: logging.Logger,
    csv_encoding: str,
) -> None:
    pats = compile_patterns()
    rows: List[CandidateRow] = []

    total_files = 0
    parsed_ok = 0

    for fp in iter_xml_files(ddb_dir):
        total_files += 1
        rel_path = str(fp.relative_to(ddb_dir)).replace("\\", "/")

        try:
            root = etree.parse(str(fp)).getroot()
        except Exception as e:
            logger.debug("DDB parse failed: %s (%s)", fp, e)
            continue

        parsed_ok += 1
        ddb_id = get_ddb_id(root).strip() or fp.stem
        meta = hgv_index.get(ddb_id, HgvMeta())

        div_xpath = "//tei:div[@type='edition']"
        if scan_commentary:
            div_xpath += " | //tei:div[@type='commentary']"
        divs = root.xpath(div_xpath, namespaces=NS)

        grain_hits: Set[str] = set()
        unit_hits: Set[str] = set()
        money_hits: Set[str] = set()
        priceword_hit = False
        num_tag_values_all: Set[str] = set()
        num_tag_texts_all: Set[str] = set()

        block_match_count = 0
        snippet_samples: List[str] = []

        for div in divs:
            for _, block in iter_blocks(div):
                raw = safe_string(block)
                if not raw:
                    continue
                raw = " ".join(raw.split())
                norm = normalize_for_search(raw)

                if not pats["grain"].search(norm):
                    continue

                # Strict number evidence
                ok_num, num_vals, num_txts = has_num_tag(block)
                if not ok_num:
                    continue

                has_unit = bool(pats["unit"].search(norm))
                if require_unit and not has_unit:
                    continue

                has_money = bool(pats["money"].search(norm))
                has_priceword = bool(pats["priceword"].search(norm))
                if not (has_money or has_priceword):
                    continue

                block_match_count += 1

                grain_hits.update(set(pats["grain"].findall(norm)))
                if has_unit:
                    unit_hits.update(set(pats["unit"].findall(norm)))
                if has_money:
                    money_hits.update(set(pats["money"].findall(norm)))
                if has_priceword:
                    priceword_hit = True

                num_tag_values_all.update([v for v in num_vals if v])
                num_tag_texts_all.update([t for t in num_txts if t])

                if len(snippet_samples) < max_snippets:
                    snippet_samples.append(raw[:800])

        if block_match_count == 0:
            continue

        # Simple score: unit + money + priceword
        score = 0
        if unit_hits:
            score += 1
        if money_hits:
            score += 1
        if priceword_hit:
            score += 1

        while len(snippet_samples) < 3:
            snippet_samples.append("")

        rows.append(
            CandidateRow(
                DDB_ID=ddb_id,
                XML_RelPath=rel_path,
                Title=meta.title,
                Place=meta.place,
                Date_Text=meta.date_text,
                Date_When=meta.date_when,
                Date_NotBefore=meta.date_notbefore,
                Date_NotAfter=meta.date_notafter,
                Score=score,
                Block_Match_Count=block_match_count,
                Grain_Hits=";".join(sorted(grain_hits)),
                Unit_Hits=";".join(sorted(unit_hits)),
                Money_Hits=";".join(sorted(money_hits)),
                Priceword_Hit="yes" if priceword_hit else "no",
                Num_Tag_Values=";".join(sorted(num_tag_values_all)),
                Num_Tag_Texts=";".join(sorted(num_tag_texts_all)),
                Snippet_1=snippet_samples[0],
                Snippet_2=snippet_samples[1],
                Snippet_3=snippet_samples[2],
            )
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        logger.warning("No candidates found. Nothing written.")
        return

    with out_csv.open("w", newline="", encoding=csv_encoding) as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    logger.info("Stage A (strict num_tag) done. Parsed %d/%d DDB XML files. Candidates: %d", parsed_ok, total_files, len(rows))
    logger.info("Wrote: %s", out_csv.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage A candidate harvester (STRICT: requires <num value>).")
    parser.add_argument("--ddb-dir", required=True, help="Path to DDB EpiDoc XML directory")
    parser.add_argument("--hgv-dir", required=True, help="Path to HGV meta EpiDoc directory")
    parser.add_argument("--out", default="grain_candidates_v10a_strict.csv", help="Output CSV file")

    parser.add_argument("--scan-commentary", action="store_true", help="Also scan <div type='commentary'>")
    parser.add_argument("--require-unit", action="store_true", help="Require a unit term (higher precision, may miss some)")
    parser.add_argument("--max-snippets", type=int, default=3, help="How many sample snippets to store per papyrus")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding (default utf-8-sig for Excel)")

    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("grain_candidate_harvester_v10a_strict")

    ddb_dir = Path(args.ddb_dir)
    hgv_dir = Path(args.hgv_dir)
    out_csv = Path(args.out)

    if not ddb_dir.exists():
        raise SystemExit(f"DDB dir not found: {ddb_dir}")
    if not hgv_dir.exists():
        raise SystemExit(f"HGV dir not found: {hgv_dir}")

    hgv_index = build_hgv_index(hgv_dir, logger)

    harvest_candidates(
        ddb_dir=ddb_dir,
        hgv_index=hgv_index,
        out_csv=out_csv,
        scan_commentary=args.scan_commentary,
        require_unit=args.require_unit,
        max_snippets=args.max_snippets,
        logger=logger,
        csv_encoding=args.encoding,
    )


if __name__ == "__main__":
    main()
