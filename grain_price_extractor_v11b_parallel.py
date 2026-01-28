#!/usr/bin/env python3
r"""
grain_price_extractor_v11b_strict_numtag_fractions_wordstart_units_money_LINKED_TOPK_parallel.py

Stage B (STRICT <num @value>) mention extractor WITH linked grain↔(qty)↔price pairing + confidence scoring,
TOP-K candidate output, and MULTIPROCESSING (ProcessPoolExecutor) like v10a.

What this solves
----------------
Large EpiDoc textblocks can contain multiple unrelated numbers and money values.
Instead of exporting *all* money expressions found in a block, this extractor links
grain hits to nearby (qty, unit) and price expressions, assigns a confidence score,
and outputs the top-K candidate pairings per grain-hit.

Parallelism
-----------
This v11b version processes each candidate document in parallel with ProcessPoolExecutor.
The main process streams results to CSV so memory stays bounded.

Main outputs
------------
1) Linked CSV (--out):
   - One row per grain-hit candidate (up to --topk), with Candidate_Rank and Is_Primary
   - Optional emission controls:
       --emit-only-primary
       --emit-topk-and-primary-only

2) Optional mention debug CSV (--out-mentions):
   - Block-level rows with aggregated quantities/prices (old-style)

Usage example
-------------
python grain_price_extractor_v11b_parallel.py `
  --candidates grain_candidates_v11a.csv `
  --ddb-dir /path/to/ddb-epidoc-xml `
  --out linked_prices.csv `
  --out-mentions mentions_debug.csv `
  --workers 8 `
  --window-tokens 80 `
  --min-score 55 `
  --topk 3 `
  --global-assign `
  --emit-topk-and-primary-only

Notes
-----
- 𐅵 is treated ONLY as a fraction token (1/2), not as currency evidence.
- We do not force same-line matching; line markers contribute only a mild penalty.

"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from lxml import etree

NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# -----------------------------
# Normalization helpers
# -----------------------------


def strip_accents(s: str) -> str:
    if not s:
        return ""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def normalize_for_search(s: str) -> str:
    """
    Normalize text for robust regex matching:
    - NFKC normalize
    - collapse whitespace
    - remove accents/diacritics
    - casefold
    """
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


# -----------------------------
# XML traversal
# -----------------------------


def iter_blocks(div_node: etree._Element) -> Iterable[Tuple[str, etree._Element]]:
    blocks = div_node.xpath(".//tei:ab | .//tei:p | .//tei:seg | .//tei:l", namespaces=NS)
    for b in blocks:
        yield etree.QName(b).localname, b


def has_num_tag(block: etree._Element) -> Tuple[bool, List[str], List[str]]:
    """
    STRICT requirement: at least one <num value="..."> inside the block.
    Returns: (has_any, values, texts)
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


LB_MARK = "__LB__"
CB_MARK = "__CB__"
PB_MARK = "__PB__"


def block_text_with_num_values(elem: etree._Element) -> str:
    """
    Convert <num value="25">κε</num> -> "25" in the output string.
    Also inject line/page/column markers as tokens:
      <lb/> -> "__LB__"
      <cb/> -> "__CB__"
      <pb/> -> "__PB__"

    Markers enable a mild cross-line penalty during scoring (no hard filtering).
    """
    pieces: List[str] = []

    def walk(node: etree._Element):
        if node.text:
            pieces.append(node.text)

        for ch in node:
            ln = etree.QName(ch).localname

            if ln == "num":
                v = (ch.get("value") or "").strip()
                if v:
                    pieces.append(v)
                else:
                    t = safe_string(ch)
                    if t:
                        pieces.append(t)

            elif ln == "lb":
                pieces.append(f" {LB_MARK} ")

            elif ln == "cb":
                pieces.append(f" {CB_MARK} ")

            elif ln == "pb":
                pieces.append(f" {PB_MARK} ")

            else:
                walk(ch)

            if ch.tail:
                pieces.append(ch.tail)

    walk(elem)
    return " ".join(" ".join(pieces).split())


# -----------------------------
# Fractions
# -----------------------------


def fmt_fraction(fr: Fraction) -> str:
    if fr.denominator == 1:
        return str(fr.numerator)
    v = float(fr)
    return f"{v:.6f}".rstrip("0").rstrip(".")


FRAC_TOKEN_RE = re.compile(r"(?P<n>\d+)\s*/\s*(?P<d>\d+)")


def parse_int_and_fracs(int_str: Optional[str], fracs_str: Optional[str]) -> str:
    total = Fraction(0, 1)
    if int_str is not None and int_str != "":
        try:
            total += Fraction(int(int_str), 1)
        except Exception:
            pass

    if fracs_str:
        for m in FRAC_TOKEN_RE.finditer(fracs_str):
            n = int(m.group("n"))
            d = int(m.group("d"))
            if d != 0:
                total += Fraction(n, d)

    return fmt_fraction(total)


# -----------------------------
# Patterns
# -----------------------------


def compile_patterns() -> Dict[str, re.Pattern]:
    """
    NOTE:
      - 𐅵 is NOT treated as currency/money; it's a fraction sign and will be replaced with "1/2"
        only in the value-rendered text.
    """
    grain = re.compile(r"\b(σιτ|πυρ|κριθ|ζεα|ολυρ|σταχυ|αλευρ)\w*", re.UNICODE)
    unit = re.compile(r"\b(αρταβ|μεδιμν|χοινι|μετρ|κοτυλ|χοινιξ|γομφ|καλαθ)\w*", re.UNICODE)

    money = re.compile(r"\b(δραχμ|οβολ|δηναρ|μνα|ταλαντ|σεστ|ἀσσ|χαλκ)\w*", re.UNICODE)
    priceword = re.compile(r"(τιμη|τιμ(?=[\.\)\(]))", re.UNICODE)

    unit_core = r"(?:αρταβ\w*|μεδιμν\w*|χοιν\w*|μετρ\w*|κοτυλ\w*|χοινιξ\w*)"
    cur_core = r"(?:δραχμ\w*|οβολ\w*|δηναρ\w*|μνα\w*|ταλαντ\w*|σεστ\w*|ἀσσ\w*|χαλκ\w*)"

    int_plus_fracs = r"(?P<int>\d+)(?P<fracs>(?:\W+(?:\d+/\d+))*)"
    frac_only = r"(?P<fracs_only>\d+/\d+)"

    qty_num_unit = re.compile(rf"{int_plus_fracs}\W+(?P<unit>{unit_core})", re.UNICODE)
    qty_unit_num = re.compile(rf"(?P<unit>{unit_core})\W+{int_plus_fracs}", re.UNICODE)
    qty_frac_unit = re.compile(rf"{frac_only}\W+(?P<unit>{unit_core})", re.UNICODE)
    qty_unit_frac = re.compile(rf"(?P<unit>{unit_core})\W+{frac_only}", re.UNICODE)

    price_num_cur = re.compile(rf"{int_plus_fracs}\W+(?P<cur>{cur_core})", re.UNICODE)
    price_cur_num = re.compile(rf"(?P<cur>{cur_core})\W+{int_plus_fracs}", re.UNICODE)
    price_frac_cur = re.compile(rf"{frac_only}\W+(?P<cur>{cur_core})", re.UNICODE)
    price_cur_frac = re.compile(rf"(?P<cur>{cur_core})\W+{frac_only}", re.UNICODE)

    return {
        "grain": grain,
        "unit": unit,
        "money": money,
        "priceword": priceword,
        "qty_num_unit": qty_num_unit,
        "qty_unit_num": qty_unit_num,
        "qty_frac_unit": qty_frac_unit,
        "qty_unit_frac": qty_unit_frac,
        "price_num_cur": price_num_cur,
        "price_cur_num": price_cur_num,
        "price_frac_cur": price_frac_cur,
        "price_cur_frac": price_cur_frac,
    }


# -----------------------------
# Tokenization / positioning
# -----------------------------


@dataclass(frozen=True)
class Tokens:
    text: str
    tokens: List[str]
    starts: List[int]
    line_ids: List[int]

    def tok_at_char(self, charpos: int) -> int:
        lo, hi = 0, len(self.starts)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.starts[mid] <= charpos:
                lo = mid + 1
            else:
                hi = mid
        idx = lo - 1
        if idx < 0:
            return 0
        if idx >= len(self.tokens):
            return len(self.tokens) - 1
        return idx


def tokenize_with_positions(s: str) -> Tokens:
    toks: List[str] = []
    starts: List[int] = []
    line_ids: List[int] = []
    line = 0
    for m in re.finditer(r"\S+", s):
        tok = m.group(0)
        toks.append(tok)
        starts.append(m.start())
        if tok == LB_MARK:
            line += 1
        line_ids.append(line)
    return Tokens(text=s, tokens=toks, starts=starts, line_ids=line_ids)


# -----------------------------
# Events
# -----------------------------


@dataclass(frozen=True)
class GrainEvent:
    tok: int
    form: str


@dataclass(frozen=True)
class QtyEvent:
    tok: int
    value: str
    unit: str


@dataclass(frozen=True)
class PriceEvent:
    tok: int
    value: str
    cur: str


@dataclass(frozen=True)
class PricewordEvent:
    tok: int
    form: str


# -----------------------------
# Scoring
# -----------------------------


def prox(d: float, s: float) -> float:
    if d >= 1e9:
        return 0.0
    return math.exp(-d / s)


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def score_candidate(
    d_gp: int,
    d_gq: int,
    d_pw: int,
    span: int,
    k_price: int,
    k_qty: int,
    unit_strength: int,
    lines_crossed: int,
) -> float:
    unit_bonus = 8 if unit_strength == 2 else (3 if unit_strength == 1 else 0)
    ambiguity_pen = min(15, 5 * max(0, k_price - 1)) + min(8, 2 * max(0, k_qty - 1))
    line_pen = 2 * min(lines_crossed, 5)

    s = (
        10
        + 40 * prox(d_gp, 12)
        + 15 * prox(d_gq, 20)
        + 10 * prox(d_pw, 8)
        + 15 * prox(span, 25)
        + unit_bonus
        - ambiguity_pen
        - line_pen
    )
    return clamp(s)


# -----------------------------
# Output rows
# -----------------------------


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


@dataclass
class LinkedRow:
    DDB_ID: str
    Mention_ID: int
    Grain_Index: int
    Candidate_Rank: int
    Is_Primary: str
    Block_Tag: str

    Title: str
    Place: str
    Date_Text: str
    Date_When: str
    Date_NotBefore: str
    Date_NotAfter: str

    Grain_Form: str
    Qty_Value: str
    Qty_Unit: str
    Price_Value: str
    Price_Cur: str

    Score: float
    Dist_GP: int
    Dist_GQ: int
    Span_Toks: int
    Lines_Crossed: int
    Priceword_Near: str

    Ambiguous: str
    AltScore: str
    Alt_Qty: str
    Alt_Price: str

    Context_Window: str


# -----------------------------
# Candidate model + ranking
# -----------------------------


@dataclass(frozen=True)
class Candidate:
    grain: GrainEvent
    price: PriceEvent
    qty: Optional[QtyEvent]
    score: float
    d_gp: int
    d_gq: int
    d_pw: int
    span: int
    lines_crossed: int
    has_priceword_near: bool


def nearest_tok_distance(tok: int, events: List[int]) -> int:
    if not events:
        return int(1e9)
    return min(abs(tok - t) for t in events)


def make_context_window(value_tokens: List[str], lo_tok: int, hi_tok: int) -> str:
    out: List[str] = []
    for t in value_tokens[lo_tok:hi_tok]:
        if t == LB_MARK:
            out.append("\n")
        elif t in (CB_MARK, PB_MARK):
            out.append("\n")
        else:
            out.append(t)
    s = " ".join(out)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"[ \t]+", " ", s).strip()
    return s


def build_events(
    pats: Dict[str, re.Pattern],
    value_norm: str,
    tok_norm: Tokens,
) -> Tuple[List[GrainEvent], List[QtyEvent], List[PriceEvent], List[PricewordEvent]]:
    grains: List[GrainEvent] = []
    qtys: List[QtyEvent] = []
    prices: List[PriceEvent] = []
    pws: List[PricewordEvent] = []

    for m in pats["grain"].finditer(value_norm):
        t = tok_norm.tok_at_char(m.start())
        grains.append(GrainEvent(tok=t, form=m.group(0)))

    for m in pats["priceword"].finditer(value_norm):
        t = tok_norm.tok_at_char(m.start())
        pws.append(PricewordEvent(tok=t, form=m.group(0)))

    def add_qty(tok: int, n: str, u: str):
        qtys.append(QtyEvent(tok=tok, value=n, unit=u))

    for m in pats["qty_num_unit"].finditer(value_norm):
        t = tok_norm.tok_at_char(m.start())
        n = parse_int_and_fracs(m.groupdict().get("int"), m.groupdict().get("fracs"))
        add_qty(t, n, m.group("unit"))
    for m in pats["qty_unit_num"].finditer(value_norm):
        t = tok_norm.tok_at_char(m.start())
        n = parse_int_and_fracs(m.groupdict().get("int"), m.groupdict().get("fracs"))
        add_qty(t, n, m.group("unit"))
    for m in pats["qty_frac_unit"].finditer(value_norm):
        t = tok_norm.tok_at_char(m.start())
        n = parse_int_and_fracs(None, m.group("fracs_only"))
        add_qty(t, n, m.group("unit"))
    for m in pats["qty_unit_frac"].finditer(value_norm):
        t = tok_norm.tok_at_char(m.start())
        n = parse_int_and_fracs(None, m.group("fracs_only"))
        add_qty(t, n, m.group("unit"))

    def add_price(tok: int, n: str, c: str):
        prices.append(PriceEvent(tok=tok, value=n, cur=c))

    for m in pats["price_num_cur"].finditer(value_norm):
        t = tok_norm.tok_at_char(m.start())
        n = parse_int_and_fracs(m.groupdict().get("int"), m.groupdict().get("fracs"))
        add_price(t, n, m.group("cur"))
    for m in pats["price_cur_num"].finditer(value_norm):
        t = tok_norm.tok_at_char(m.start())
        n = parse_int_and_fracs(m.groupdict().get("int"), m.groupdict().get("fracs"))
        add_price(t, n, m.group("cur"))
    for m in pats["price_frac_cur"].finditer(value_norm):
        t = tok_norm.tok_at_char(m.start())
        n = parse_int_and_fracs(None, m.group("fracs_only"))
        add_price(t, n, m.group("cur"))
    for m in pats["price_cur_frac"].finditer(value_norm):
        t = tok_norm.tok_at_char(m.start())
        n = parse_int_and_fracs(None, m.group("fracs_only"))
        add_price(t, n, m.group("cur"))

    return grains, qtys, prices, pws


def generate_candidates_for_grain(
    grain: GrainEvent,
    prices: List[PriceEvent],
    qtys: List[QtyEvent],
    priceword_toks: List[int],
    tok_norm: Tokens,
    has_unit_sig: bool,
    window_tokens: int,
) -> List[Candidate]:
    local_prices = [p for p in prices if abs(p.tok - grain.tok) <= window_tokens]
    local_qtys = [q for q in qtys if abs(q.tok - grain.tok) <= window_tokens]

    k_price = len(local_prices)
    k_qty = len(local_qtys)

    cands: List[Candidate] = []
    for p in local_prices:
        d_pw = nearest_tok_distance(p.tok, priceword_toks)
        lines_crossed = abs(tok_norm.line_ids[grain.tok] - tok_norm.line_ids[p.tok]) if tok_norm.line_ids else 0

        qty_opts: List[Optional[QtyEvent]] = list(local_qtys) + [None]
        for q in qty_opts:
            d_gp = abs(grain.tok - p.tok)
            if q is None:
                d_gq = int(1e9)
                span = max(grain.tok, p.tok) - min(grain.tok, p.tok)
                unit_strength = 1 if has_unit_sig else 0
            else:
                d_gq = abs(grain.tok - q.tok)
                span = max(grain.tok, p.tok, q.tok) - min(grain.tok, p.tok, q.tok)
                unit_strength = 2

            score = score_candidate(
                d_gp=d_gp,
                d_gq=d_gq,
                d_pw=d_pw,
                span=span,
                k_price=k_price,
                k_qty=k_qty,
                unit_strength=unit_strength,
                lines_crossed=lines_crossed,
            )
            has_pw_near = (d_pw <= 8)
            cands.append(
                Candidate(
                    grain=grain,
                    price=p,
                    qty=q,
                    score=score,
                    d_gp=d_gp,
                    d_gq=d_gq if d_gq < 1e9 else -1,
                    d_pw=d_pw if d_pw < 1e9 else -1,
                    span=span,
                    lines_crossed=lines_crossed,
                    has_priceword_near=has_pw_near,
                )
            )
    return cands


def candidate_sort_key(c: Candidate) -> Tuple:
    """
    Best-first sort key.
    """
    has_qty = 1 if c.qty is not None else 0
    d_pw = c.d_pw if c.d_pw >= 0 else 999999
    # For sort descending: smaller span/distance should rank higher -> invert with negative.
    return (
        c.score,
        has_qty,
        -c.span,
        -c.d_gp,
        -d_pw,
        -c.price.tok,
    )


def rank_candidates_per_grain(
    grains: List[GrainEvent],
    prices: List[PriceEvent],
    qtys: List[QtyEvent],
    pws: List[PricewordEvent],
    tok_norm: Tokens,
    has_unit_sig: bool,
    window_tokens: int,
    min_score: float,
) -> Dict[int, List[Candidate]]:
    priceword_toks = [pw.tok for pw in pws]
    ranked: Dict[int, List[Candidate]] = {}

    for gi, g in enumerate(grains, start=1):
        cands = generate_candidates_for_grain(
            grain=g,
            prices=prices,
            qtys=qtys,
            priceword_toks=priceword_toks,
            tok_norm=tok_norm,
            has_unit_sig=has_unit_sig,
            window_tokens=window_tokens,
        )
        cands = [c for c in cands if c.score >= min_score]
        if not cands:
            continue
        cands.sort(key=candidate_sort_key, reverse=True)
        ranked[gi] = cands

    return ranked


def pick_primary_global(
    ranked: Dict[int, List[Candidate]],
    reuse_score: float = 80.0,
    reuse_max_dgp: int = 12,
    reuse_max_span: int = 18,
    fallback_score: float = 75.0,
) -> Dict[int, Candidate]:
    """
    Global greedy assignment to reduce reuse of a single money amount across many grains.
    Chooses ONE primary candidate per grain.
    """
    all_cands: List[Tuple[int, Candidate]] = []
    for gi, lst in ranked.items():
        for c in lst:
            all_cands.append((gi, c))

    all_cands.sort(key=lambda x: candidate_sort_key(x[1]), reverse=True)

    primary: Dict[int, Candidate] = {}
    used_price_toks: set[int] = set()

    for gi, c in all_cands:
        if gi in primary:
            continue
        if c.price.tok not in used_price_toks:
            primary[gi] = c
            used_price_toks.add(c.price.tok)
        else:
            if c.score >= reuse_score and c.d_gp <= reuse_max_dgp and c.span <= reuse_max_span:
                primary[gi] = c

    for gi, lst in ranked.items():
        if gi in primary:
            continue
        if not lst:
            continue
        if lst[0].score >= fallback_score:
            primary[gi] = lst[0]

    return primary


# -----------------------------
# Core extraction per doc
# -----------------------------


def extract_for_doc(
    xml_path: Path,
    ddb_id: str,
    meta: Dict[str, str],
    scan_commentary: bool,
    require_unit: bool,
    window_tokens: int,
    min_score: float,
    topk: int,
    global_assign: bool,
    emit_only_primary: bool,
    emit_topk_and_primary_only: bool,
    keep_block_chars: int = 2000,
    ambiguity_delta: float = 8.0,
) -> Tuple[List[LinkedRow], List[MentionRow]]:
    pats = WORKER_PATS or compile_patterns()

    linked: List[LinkedRow] = []
    mentions: List[MentionRow] = []

    try:
        root = etree.parse(str(xml_path)).getroot()
    except Exception:
        return linked, mentions

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

            # Gating
            if not pats["grain"].search(norm):
                continue

            has_num, num_vals, num_txts = has_num_tag(block)
            if not has_num:
                continue

            has_unit_sig = bool(pats["unit"].search(norm))
            if require_unit and not has_unit_sig:
                continue

            has_money_sig = bool(pats["money"].search(norm))
            has_priceword_sig = bool(pats["priceword"].search(norm))
            if not (has_money_sig or has_priceword_sig):
                continue

            grain_hits = sorted(set(m.group(0) for m in pats["grain"].finditer(norm)))
            unit_hits = sorted(set(m.group(0) for m in pats["unit"].finditer(norm)))
            money_hits = sorted(set(m.group(0) for m in pats["money"].finditer(norm)))
            priceword_hit = "yes" if has_priceword_sig else "no"

            # Value-rendered text + markers
            value_text = block_text_with_num_values(block)
            value_text = value_text.replace("𐅵", " 1/2 ")
            value_text = " ".join(value_text.split())
            value_norm = normalize_for_search(value_text)

            tok_norm = tokenize_with_positions(value_norm)
            tok_val = value_text.split()

            # Debug lists (old-style)
            quantities_dbg: List[str] = []
            prices_dbg: List[str] = []
            seenq = set()
            seenp = set()

            def add_qty_dbg(num_s: str, unit_s: str):
                item = f"{num_s} {unit_s}"
                if item not in seenq:
                    seenq.add(item)
                    quantities_dbg.append(item)

            def add_price_dbg(num_s: str, cur_s: str):
                item = f"{num_s} {cur_s}"
                if item not in seenp:
                    seenp.add(item)
                    prices_dbg.append(item)

            for m in pats["qty_num_unit"].finditer(value_norm):
                n = parse_int_and_fracs(m.groupdict().get("int"), m.groupdict().get("fracs"))
                add_qty_dbg(n, m.group("unit"))
            for m in pats["qty_unit_num"].finditer(value_norm):
                n = parse_int_and_fracs(m.groupdict().get("int"), m.groupdict().get("fracs"))
                add_qty_dbg(n, m.group("unit"))
            for m in pats["qty_frac_unit"].finditer(value_norm):
                n = parse_int_and_fracs(None, m.group("fracs_only"))
                add_qty_dbg(n, m.group("unit"))
            for m in pats["qty_unit_frac"].finditer(value_norm):
                n = parse_int_and_fracs(None, m.group("fracs_only"))
                add_qty_dbg(n, m.group("unit"))

            for m in pats["price_num_cur"].finditer(value_norm):
                n = parse_int_and_fracs(m.groupdict().get("int"), m.groupdict().get("fracs"))
                add_price_dbg(n, m.group("cur"))
            for m in pats["price_cur_num"].finditer(value_norm):
                n = parse_int_and_fracs(m.groupdict().get("int"), m.groupdict().get("fracs"))
                add_price_dbg(n, m.group("cur"))
            for m in pats["price_frac_cur"].finditer(value_norm):
                n = parse_int_and_fracs(None, m.group("fracs_only"))
                add_price_dbg(n, m.group("cur"))
            for m in pats["price_cur_frac"].finditer(value_norm):
                n = parse_int_and_fracs(None, m.group("fracs_only"))
                add_price_dbg(n, m.group("cur"))

            grains, qtys, prices, pws = build_events(pats, value_norm, tok_norm)

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
                    Number_Hit=f"num:{num_vals[0]}" if num_vals else "",
                    Number_Type="num_tag",
                    Num_Tag_Values=";".join([v for v in num_vals if v]),
                    Num_Tag_Texts=";".join([t for t in num_txts if t]),
                    Quantities=";".join(quantities_dbg),
                    Prices=";".join(prices_dbg),
                    Raw_Block=raw[:keep_block_chars],
                    Value_Block=value_text[:keep_block_chars],
                )
            )

            if not grains or not prices:
                continue

            ranked = rank_candidates_per_grain(
                grains=grains,
                prices=prices,
                qtys=qtys,
                pws=pws,
                tok_norm=tok_norm,
                has_unit_sig=has_unit_sig,
                window_tokens=window_tokens,
                min_score=min_score,
            )
            if not ranked:
                continue

            primary: Dict[int, Candidate] = {}
            if global_assign:
                primary = pick_primary_global(ranked=ranked)

            for gi, cand_list in ranked.items():
                if not cand_list:
                    continue

                prim = primary.get(gi, cand_list[0])

                # Build emission list
                if emit_only_primary:
                    try:
                        prim_rank = cand_list.index(prim) + 1
                    except ValueError:
                        prim_rank = 1
                    cand_iter = [(prim_rank, prim)]
                else:
                    K = max(1, topk)
                    cand_iter = list(enumerate(cand_list[:K], start=1))

                    if emit_topk_and_primary_only:
                        if prim not in [c for _, c in cand_iter]:
                            try:
                                prim_rank = cand_list.index(prim) + 1
                            except ValueError:
                                prim_rank = 1
                            cand_iter.append((prim_rank, prim))

                for rank, c in cand_iter:
                    is_primary = "yes" if c == prim else "no"

                    ambiguous = "no"
                    alt_score = ""
                    alt_qty = ""
                    alt_price = ""
                    # compare to next-ranked candidate in full list (not just emitted)
                    if 1 <= rank < len(cand_list):
                        nxt = cand_list[rank]  # rank is 1-based -> next is index=rank
                        if (c.score - nxt.score) < ambiguity_delta:
                            ambiguous = "yes"
                            alt_score = f"{nxt.score:.2f}"
                            alt_qty = f"{nxt.qty.value} {nxt.qty.unit}" if nxt.qty else ""
                            alt_price = f"{nxt.price.value} {nxt.price.cur}"

                    # Context window
                    lo = min(c.grain.tok, c.price.tok, c.qty.tok if c.qty else c.grain.tok)
                    hi = max(c.grain.tok, c.price.tok, c.qty.tok if c.qty else c.grain.tok)
                    pad = 8
                    lo2 = max(0, lo - pad)
                    hi2 = min(len(tok_val), hi + pad + 1)
                    ctx = make_context_window(tok_val, lo2, hi2)

                    linked.append(
                        LinkedRow(
                            DDB_ID=ddb_id,
                            Mention_ID=mention_id,
                            Grain_Index=gi,
                            Candidate_Rank=rank,
                            Is_Primary=is_primary,
                            Block_Tag=tag,
                            Title=meta.get("Title", ""),
                            Place=meta.get("Place", ""),
                            Date_Text=meta.get("Date_Text", ""),
                            Date_When=meta.get("Date_When", ""),
                            Date_NotBefore=meta.get("Date_NotBefore", ""),
                            Date_NotAfter=meta.get("Date_NotAfter", ""),
                            Grain_Form=c.grain.form,
                            Qty_Value=c.qty.value if c.qty else "",
                            Qty_Unit=c.qty.unit if c.qty else "",
                            Price_Value=c.price.value,
                            Price_Cur=c.price.cur,
                            Score=round(c.score, 2),
                            Dist_GP=c.d_gp,
                            Dist_GQ=c.d_gq,
                            Span_Toks=c.span,
                            Lines_Crossed=c.lines_crossed,
                            Priceword_Near="yes" if c.has_priceword_near else "no",
                            Ambiguous=ambiguous,
                            AltScore=alt_score,
                            Alt_Qty=alt_qty,
                            Alt_Price=alt_price,
                            Context_Window=ctx[:keep_block_chars],
                        )
                    )

    return linked, mentions


# -----------------------------
# Multiprocessing worker
# -----------------------------

WORKER_PATS: Optional[Dict[str, re.Pattern]] = None
WORKER_DDB_DIR: Optional[Path] = None
WORKER_SCAN_COMMENTARY: bool = False
WORKER_REQUIRE_UNIT: bool = False
WORKER_WINDOW_TOKENS: int = 80
WORKER_MIN_SCORE: float = 55.0
WORKER_TOPK: int = 1
WORKER_GLOBAL_ASSIGN: bool = False
WORKER_EMIT_ONLY_PRIMARY: bool = False
WORKER_EMIT_TOPK_AND_PRIMARY_ONLY: bool = False


def _init_worker(
    ddb_dir: str,
    scan_commentary: bool,
    require_unit: bool,
    window_tokens: int,
    min_score: float,
    topk: int,
    global_assign: bool,
    emit_only_primary: bool,
    emit_topk_and_primary_only: bool,
):
    global WORKER_PATS, WORKER_DDB_DIR
    global WORKER_SCAN_COMMENTARY, WORKER_REQUIRE_UNIT
    global WORKER_WINDOW_TOKENS, WORKER_MIN_SCORE, WORKER_TOPK
    global WORKER_GLOBAL_ASSIGN, WORKER_EMIT_ONLY_PRIMARY, WORKER_EMIT_TOPK_AND_PRIMARY_ONLY

    WORKER_PATS = compile_patterns()
    WORKER_DDB_DIR = Path(ddb_dir)
    WORKER_SCAN_COMMENTARY = scan_commentary
    WORKER_REQUIRE_UNIT = require_unit
    WORKER_WINDOW_TOKENS = window_tokens
    WORKER_MIN_SCORE = min_score
    WORKER_TOPK = topk
    WORKER_GLOBAL_ASSIGN = global_assign
    WORKER_EMIT_ONLY_PRIMARY = emit_only_primary
    WORKER_EMIT_TOPK_AND_PRIMARY_ONLY = emit_topk_and_primary_only


def _process_one_candidate(job: Tuple[str, str, Dict[str, str]]) -> Tuple[bool, List[dict], List[dict]]:
    """
    Worker entry point. Returns:
      (ok, linked_rows_as_dicts, mention_rows_as_dicts)
    """
    ddb_id, rel_path, meta = job
    try:
        assert WORKER_DDB_DIR is not None
        xml_path = WORKER_DDB_DIR / rel_path
        if not xml_path.exists():
            return False, [], []

        linked_rows, mention_rows = extract_for_doc(
            xml_path=xml_path,
            ddb_id=ddb_id,
            meta=meta,
            scan_commentary=WORKER_SCAN_COMMENTARY,
            require_unit=WORKER_REQUIRE_UNIT,
            window_tokens=WORKER_WINDOW_TOKENS,
            min_score=WORKER_MIN_SCORE,
            topk=WORKER_TOPK,
            global_assign=WORKER_GLOBAL_ASSIGN,
            emit_only_primary=WORKER_EMIT_ONLY_PRIMARY,
            emit_topk_and_primary_only=WORKER_EMIT_TOPK_AND_PRIMARY_ONLY,
        )
        return True, [asdict(r) for r in linked_rows], [asdict(r) for r in mention_rows]
    except Exception:
        return False, [], []


# -----------------------------
# Main
# -----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage B linked extractor v11b (STRICT num_tag, fraction-aware, TOPK, multiprocessing)."
    )
    parser.add_argument("--candidates", required=True, help="Stage A strict candidates CSV")
    parser.add_argument("--ddb-dir", required=True, help="Path to DDB EpiDoc XML directory")

    parser.add_argument(
        "--out",
        default="grain_linked_prices_v11b.csv",
        help="Output LINKED (grain↔qty↔price) CSV",
    )
    parser.add_argument(
        "--out-mentions",
        default="",
        help="Optional output block-level debug CSV (old-style mention rows)",
    )

    parser.add_argument("--scan-commentary", action="store_true", help="Also scan <div type='commentary'>")
    parser.add_argument("--require-unit", action="store_true", help="Require unit term (may miss some)")
    parser.add_argument("--window-tokens", type=int, default=80, help="Token window around each grain hit")
    parser.add_argument("--min-score", type=float, default=55.0, help="Minimum confidence score to output")
    parser.add_argument("--topk", type=int, default=1, help="Output top-K candidates per grain-hit (>=1)")
    parser.add_argument("--global-assign", action="store_true", help="Global greedy assignment to reduce price reuse")
    parser.add_argument(
        "--emit-only-primary",
        action="store_true",
        help="Emit only the primary linked row per grain-hit (ignores --topk for output)",
    )
    parser.add_argument(
        "--emit-topk-and-primary-only",
        action="store_true",
        help="Emit top-K candidates per grain-hit, and also include the primary candidate if it falls outside the top-K",
    )

    parser.add_argument("--workers", type=int, default=0, help="Number of worker processes (0=cpu_count)")
    parser.add_argument("--chunksize", type=int, default=10, help="ProcessPoolExecutor map chunksize")
    parser.add_argument("--max-docs", type=int, default=0, help="Process only N docs (0 = all)")
    parser.add_argument("--encoding", default="utf-8-sig", help="CSV encoding (default utf-8-sig for Excel)")
    parser.add_argument("--debug", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")
    logger = logging.getLogger("grain_price_extractor_v11b_parallel")

    cand_path = Path(args.candidates)
    ddb_dir = Path(args.ddb_dir)
    out_linked = Path(args.out)
    out_mentions = Path(args.out_mentions) if args.out_mentions else None

    if not cand_path.exists():
        raise SystemExit(f"Candidates CSV not found: {cand_path}")
    if not ddb_dir.exists():
        raise SystemExit(f"DDB dir not found: {ddb_dir}")

    df = pd.read_csv(cand_path).fillna("")
    if not {"DDB_ID", "XML_RelPath"}.issubset(df.columns):
        raise SystemExit("Candidates CSV must contain columns: DDB_ID, XML_RelPath")

    # Build jobs
    jobs: List[Tuple[str, str, Dict[str, str]]] = []
    for _, r in df.iterrows():
        ddb_id = str(r.get("DDB_ID", "")).strip()
        rel = str(r.get("XML_RelPath", "")).strip().replace("\\", "/")
        if not ddb_id or not rel:
            continue

        meta = {
            "Title": str(r.get("Title", "")),
            "Place": str(r.get("Place", "")),
            "Date_Text": str(r.get("Date_Text", "")),
            "Date_When": str(r.get("Date_When", "")),
            "Date_NotBefore": str(r.get("Date_NotBefore", "")),
            "Date_NotAfter": str(r.get("Date_NotAfter", "")),
        }
        jobs.append((ddb_id, rel, meta))

    if args.max_docs and args.max_docs > 0:
        jobs = jobs[: args.max_docs]

    total = len(jobs)
    if total == 0:
        logger.warning("No jobs found in candidates CSV.")
        return

    # Prepare CSV writers (streaming)
    linked_fields = [f.name for f in fields(LinkedRow)]
    mention_fields = [f.name for f in fields(MentionRow)]

    out_linked.parent.mkdir(parents=True, exist_ok=True)
    f_linked = out_linked.open("w", newline="", encoding=args.encoding)
    linked_writer = csv.DictWriter(f_linked, fieldnames=linked_fields)
    linked_writer.writeheader()

    f_mentions = None
    mention_writer = None
    if out_mentions is not None:
        out_mentions.parent.mkdir(parents=True, exist_ok=True)
        f_mentions = out_mentions.open("w", newline="", encoding=args.encoding)
        mention_writer = csv.DictWriter(f_mentions, fieldnames=mention_fields)
        mention_writer.writeheader()

    workers = args.workers if args.workers and args.workers > 0 else None
    logger.info("Stage B v11b (parallel) processing %d candidate docs with workers=%s ...", total, workers or "cpu_count")

    processed = 0
    ok_docs = 0
    linked_rows_written = 0
    mention_rows_written = 0

    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(
                str(ddb_dir),
                args.scan_commentary,
                args.require_unit,
                args.window_tokens,
                args.min_score,
                max(1, args.topk),
                args.global_assign,
                args.emit_only_primary,
                args.emit_topk_and_primary_only,
            ),
        ) as ex:
            for ok, linked_dicts, mention_dicts in ex.map(_process_one_candidate, jobs, chunksize=args.chunksize):
                processed += 1
                if ok:
                    ok_docs += 1

                for d in linked_dicts:
                    linked_writer.writerow(d)
                linked_rows_written += len(linked_dicts)

                if mention_writer is not None:
                    for d in mention_dicts:
                        mention_writer.writerow(d)
                    mention_rows_written += len(mention_dicts)

                if processed % 200 == 0:
                    logger.info(
                        "Progress: %d/%d docs; ok=%d; linked_rows=%d; mention_rows=%d",
                        processed,
                        total,
                        ok_docs,
                        linked_rows_written,
                        mention_rows_written,
                    )
    finally:
        f_linked.close()
        if f_mentions is not None:
            f_mentions.close()

    logger.info(
        "Done. Docs processed: %d/%d (ok=%d). Linked rows: %d. Debug mentions: %d.",
        processed,
        total,
        ok_docs,
        linked_rows_written,
        mention_rows_written,
    )
    logger.info("Wrote: %s", out_linked.resolve())
    if out_mentions is not None:
        logger.info("Wrote: %s", out_mentions.resolve())


if __name__ == "__main__":
    main()
