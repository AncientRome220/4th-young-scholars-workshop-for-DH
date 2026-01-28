#!/usr/bin/env python3
r"""
grain_candidate_harvester_v11a.py

Stage A (parallel) candidate harvester:
- Scans DDB EpiDoc XML files (edition [+ optional commentary]) to find *documents* that likely contain
  grain price information.
- Uses HGV meta EpiDoc directory to attach Title / Place / Date fields.

Parallelization:
- Uses ProcessPoolExecutor (multiprocessing) for faster XML parsing & regex scanning.
- Windows-safe (guarded by if __name__ == "__main__":).

Word-start matching (Option A) for cleaner hits:
- Grain:     r"\b(σιτ|πυρ|...)\w*"
- Unit:      r"\b(αρταβ|...)\w*"
- Currency:  r"\b(δραχμ|...)\w*"

Candidate logic (per block):
  grain_hit AND (num evidence if required) AND (money OR priceword) AND (unit if required)

Output:
- CSV with one row per candidate document, including up to 3 snippets.

Run:
  python grain_candidate_harvester_v11a.py ^
    --ddb-dir "C:\research\DDB_EpiDoc_XML" ^
    --hgv-dir "C:\research\HGV_meta_EpiDoc_XML" ^
    --out wheat_candidates_v11a.csv ^
    --workers 8

Notes:
- On Windows, always run from a terminal and keep the __main__ guard.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from lxml import etree

NS = {"tei": "http://www.tei-c.org/ns/1.0"}


# ----------------------------
# Text normalization
# ----------------------------
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


# ----------------------------
# File iteration
# ----------------------------
def iter_xml_files(root: Path):
    for fp in root.rglob("*.xml"):
        if fp.is_file():
            yield fp


# ----------------------------
# HGV metadata
# ----------------------------
@dataclass
class HgvMeta:
    title: str = ""
    place: str = ""
    date_text: str = ""
    date_when: str = ""
    date_notbefore: str = ""
    date_notafter: str = ""


def parse_hgv_meta(root: etree._Element) -> Tuple[str, HgvMeta]:
    """
    Extract ddb-hybrid id and a few useful metadata fields from HGV meta EpiDoc.
    """
    ddb_id = root.xpath('string(//tei:idno[@type="ddb-hybrid"][1])', namespaces=NS).strip()
    if not ddb_id:
        ddb_id = root.xpath("string(//tei:idno[1])", namespaces=NS).strip()

    title = root.xpath("string(//tei:titleStmt/tei:title[1])", namespaces=NS).strip()
    place = root.xpath("string(//tei:origPlace[1])", namespaces=NS).strip()
    date_text = root.xpath("string(//tei:origDate[1])", namespaces=NS).strip()

    date_when = root.xpath("string(//tei:origDate[1]/@when)", namespaces=NS).strip()
    date_nb = root.xpath("string(//tei:origDate[1]/@notBefore)", namespaces=NS).strip()
    date_na = root.xpath("string(//tei:origDate[1]/@notAfter)", namespaces=NS).strip()

    return ddb_id, HgvMeta(
        title=title,
        place=place,
        date_text=date_text,
        date_when=date_when,
        date_notbefore=date_nb,
        date_notafter=date_na,
    )


def build_hgv_index(hgv_dir: Path, logger: logging.Logger) -> Dict[str, HgvMeta]:
    idx: Dict[str, HgvMeta] = {}
    total, kept = 0, 0
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


# ----------------------------
# Patterns
# ----------------------------
def compile_patterns() -> Dict[str, re.Pattern]:
    """
    Edit vocab here. Patterns run on NORMALIZED text.
    Word-start matching (Option A) reduces false positives from substrings.
    """
    grain = re.compile(r"\b(σιτ|πυρ|κριθ|ζεα|ολυρ|σταχυ|αλευρ)\w*", re.UNICODE)
    unit = re.compile(r"\b(αρταβ|μεδιμν|χοινι|χοινιξ|μετρ|κοτυλ|καλαθ)\w*", re.UNICODE)
    money = re.compile(r"\b(δραχμ|οβολ|δηναρ|μνα|ταλαντ|σεστ|𐅵|χαλκ)\w*", re.UNICODE)

    # Price word: keep τιμή, but avoid a super-broad raw "τιμ" hit.
    priceword = re.compile(r"(τιμη|τιμ(?=[\.\)\(]))", re.UNICODE)

    digit = re.compile(r"[0-9]+")
    greek_numeral = re.compile(r"(?<!\w)([α-ωϛϝϟϡ]{1,6}[ʹʹ])(?!\w)", re.UNICODE)

    return {
        "grain": grain,
        "unit": unit,
        "money": money,
        "priceword": priceword,
        "digit": digit,
        "greek_numeral": greek_numeral,
    }


def get_ddb_id(root: etree._Element) -> str:
    ddb_id = root.xpath('string(//tei:idno[@type="ddb-hybrid"][1])', namespaces=NS).strip()
    if ddb_id:
        return ddb_id
    return root.xpath("string(//tei:idno[1])", namespaces=NS).strip()


def iter_blocks(div_node: etree._Element):
    blocks = div_node.xpath(".//tei:ab | .//tei:p | .//tei:seg | .//tei:l", namespaces=NS)
    for b in blocks:
        yield etree.QName(b).localname, b


def detect_number_1_2_3(block: etree._Element, raw_text: str, norm_text: str, pats: Dict[str, re.Pattern]):
    """
    Returns:
      has_number, number_types(set), num_tag_values(list), num_tag_texts(list)
    """
    number_types: Set[str] = set()
    num_vals: List[str] = []
    num_txts: List[str] = []

    # (1) <num value="...">
    num_nodes = block.xpath(".//tei:num[@value]", namespaces=NS)
    if num_nodes:
        for n in num_nodes:
            v = (n.get("value") or "").strip()
            if v:
                number_types.add("num_tag")
                num_vals.append(v)
            txt = safe_string(n)
            if txt:
                num_txts.append(txt)
        return True, number_types, num_vals, num_txts

    # (2) digits
    if pats["digit"].search(raw_text):
        number_types.add("digit")
        return True, number_types, num_vals, num_txts

    # (3) Greek numerals with keraia/prime (exclude ʼ/’ elision)
    if pats["greek_numeral"].search(norm_text):
        number_types.add("greek_numeral")
        return True, number_types, num_vals, num_txts

    return False, number_types, num_vals, num_txts


# ----------------------------
# Output row
# ----------------------------
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
    Number_Type_Summary: str
    Num_Tag_Values: str
    Num_Tag_Texts: str

    Snippet_1: str
    Snippet_2: str
    Snippet_3: str


# ----------------------------
# Parallel worker setup
# ----------------------------
_WORK_HGV_INDEX: Dict[str, HgvMeta] = {}
_WORK_PATS: Dict[str, re.Pattern] = {}
_WORK_DDB_DIR: Path | None = None
_WORK_SCAN_COMMENTARY: bool = False
_WORK_REQUIRE_NUMBER: bool = True
_WORK_REQUIRE_UNIT: bool = False
_WORK_MAX_SNIPPETS: int = 3


def _init_worker(hgv_index: Dict[str, HgvMeta], ddb_dir: str, scan_commentary: bool, require_number: bool, require_unit: bool, max_snippets: int):
    global _WORK_HGV_INDEX, _WORK_PATS, _WORK_DDB_DIR, _WORK_SCAN_COMMENTARY, _WORK_REQUIRE_NUMBER, _WORK_REQUIRE_UNIT, _WORK_MAX_SNIPPETS
    _WORK_HGV_INDEX = hgv_index
    _WORK_PATS = compile_patterns()
    _WORK_DDB_DIR = Path(ddb_dir)
    _WORK_SCAN_COMMENTARY = scan_commentary
    _WORK_REQUIRE_NUMBER = require_number
    _WORK_REQUIRE_UNIT = require_unit
    _WORK_MAX_SNIPPETS = max_snippets


def _process_one_ddb_xml(fp_str: str) -> Tuple[bool, CandidateRow | None]:
    """
    Returns (parsed_ok, CandidateRow or None).
    """
    fp = Path(fp_str)
    try:
        root = etree.parse(str(fp)).getroot()
    except Exception:
        return False, None

    ddb_id = get_ddb_id(root)
    if not ddb_id:
        return True, None

    meta = _WORK_HGV_INDEX.get(ddb_id, HgvMeta())
    rel_path = fp.relative_to(_WORK_DDB_DIR).as_posix() if _WORK_DDB_DIR else fp.name

    div_xpath = "//tei:div[@type='edition']"
    if _WORK_SCAN_COMMENTARY:
        div_xpath += " | //tei:div[@type='commentary']"
    divs = root.xpath(div_xpath, namespaces=NS)
    if not divs:
        return True, None

    block_match_count = 0
    snippets: List[str] = []

    grain_hits_all: Set[str] = set()
    unit_hits_all: Set[str] = set()
    money_hits_all: Set[str] = set()
    number_types_all: Set[str] = set()
    num_tag_values_all: Set[str] = set()
    num_tag_texts_all: Set[str] = set()
    priceword_hit = False

    for div in divs:
        for tag, block in iter_blocks(div):
            raw = safe_string(block)
            if not raw:
                continue
            raw = " ".join(raw.split())
            norm = normalize_for_search(raw)

            if not _WORK_PATS["grain"].search(norm):
                continue

            has_number, number_types, num_vals, num_txts = detect_number_1_2_3(block, raw, norm, _WORK_PATS)
            if _WORK_REQUIRE_NUMBER and not has_number:
                continue

            has_unit = bool(_WORK_PATS["unit"].search(norm))
            if _WORK_REQUIRE_UNIT and not has_unit:
                continue

            has_money = bool(_WORK_PATS["money"].search(norm))
            has_priceword = bool(_WORK_PATS["priceword"].search(norm))
            if not (has_money or has_priceword):
                continue

            # Collect evidence
            grain_hits_all.update(_WORK_PATS["grain"].findall(norm))
            if has_unit:
                unit_hits_all.update(_WORK_PATS["unit"].findall(norm))
            if has_money:
                money_hits_all.update(_WORK_PATS["money"].findall(norm))
            if has_priceword:
                priceword_hit = True

            number_types_all.update(number_types)
            if "num_tag" in number_types:
                num_tag_values_all.update([v for v in num_vals if v])
                num_tag_texts_all.update([t for t in num_txts if t])

            if len(snippets) < _WORK_MAX_SNIPPETS:
                snippets.append(raw[:300])

            block_match_count += 1

    if block_match_count == 0:
        return True, None

    # Score: lightweight quick prioritization
    score = 0
    if unit_hits_all:
        score += 1
    if money_hits_all:
        score += 1
    if priceword_hit:
        score += 1
    if "num_tag" in number_types_all:
        score += 1

    while len(snippets) < 3:
        snippets.append("")

    row = CandidateRow(
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
        Grain_Hits=";".join(sorted(grain_hits_all)),
        Unit_Hits=";".join(sorted(unit_hits_all)),
        Money_Hits=";".join(sorted(money_hits_all)),
        Priceword_Hit="yes" if priceword_hit else "no",
        Number_Type_Summary=";".join(sorted(number_types_all)),
        Num_Tag_Values=";".join(sorted(num_tag_values_all)),
        Num_Tag_Texts=";".join(sorted(num_tag_texts_all)),
        Snippet_1=snippets[0],
        Snippet_2=snippets[1],
        Snippet_3=snippets[2],
    )

    return True, row


# ----------------------------
# Parallel harvest
# ----------------------------
def harvest_candidates_parallel(
    ddb_dir: Path,
    hgv_index: Dict[str, HgvMeta],
    out_csv: Path,
    scan_commentary: bool,
    require_number: bool,
    require_unit: bool,
    max_snippets: int,
    workers: int,
    chunksize: int,
    logger: logging.Logger,
) -> None:
    files = [str(fp) for fp in iter_xml_files(ddb_dir)]
    total_files = len(files)
    if total_files == 0:
        logger.warning("No XML files found under: %s", ddb_dir)
        return

    logger.info("Stage A (parallel) scanning %d DDB XML files with %d workers...", total_files, workers)

    rows: List[CandidateRow] = []
    parsed_ok = 0
    processed = 0

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(hgv_index, str(ddb_dir), scan_commentary, require_number, require_unit, max_snippets),
    ) as ex:
        for ok, row in ex.map(_process_one_ddb_xml, files, chunksize=chunksize):
            processed += 1
            if ok:
                parsed_ok += 1
            if row is not None:
                rows.append(row)

            if processed % 2000 == 0:
                logger.info("Progress: %d/%d files processed; candidates so far: %d", processed, total_files, len(rows))

    if not rows:
        logger.warning("No candidates found. Nothing written.")
        return

    # Sort for nicer output: Score desc, then DDB_ID
    rows.sort(key=lambda r: (-r.Score, r.DDB_ID))

    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    logger.info("Stage A done. Parsed %d/%d DDB XML files. Candidates: %d", parsed_ok, total_files, len(rows))
    logger.info("Wrote: %s", out_csv.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage A grain-price candidate harvester (v10a) — PARALLEL.")
    parser.add_argument("--ddb-dir", required=True, help="Path to DDB EpiDoc XML directory")
    parser.add_argument("--hgv-dir", required=True, help="Path to HGV meta EpiDoc directory")
    parser.add_argument("--out", default="grain_candidates_v10a_parallel.csv", help="Output CSV path")

    parser.add_argument("--scan-commentary", action="store_true", help="Also scan <div type='commentary'>")
    parser.add_argument("--no-require-number", action="store_true", help="Do NOT require number evidence")
    parser.add_argument("--require-unit", action="store_true", help="Require unit term (precision↑ recall↓)")
    parser.add_argument("--max-snippets", type=int, default=3, help="How many sample snippets per papyrus")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1), help="Worker processes")
    parser.add_argument("--chunksize", type=int, default=25, help="Chunk size for executor.map")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("grain_candidate_harvester_v10a_parallel")

    ddb_dir = Path(args.ddb_dir)
    hgv_dir = Path(args.hgv_dir)
    out_csv = Path(args.out)

    if not ddb_dir.exists():
        raise SystemExit(f"DDB dir not found: {ddb_dir}")
    if not hgv_dir.exists():
        raise SystemExit(f"HGV dir not found: {hgv_dir}")

    hgv_index = build_hgv_index(hgv_dir, logger)

    harvest_candidates_parallel(
        ddb_dir=ddb_dir,
        hgv_index=hgv_index,
        out_csv=out_csv,
        scan_commentary=args.scan_commentary,
        require_number=not args.no_require_number,
        require_unit=args.require_unit,
        max_snippets=args.max_snippets,
        workers=args.workers,
        chunksize=max(1, args.chunksize),
        logger=logger,
    )


if __name__ == "__main__":
    main()
