# -*- coding: utf-8 -*-
"""
v11b advanced cleaning + analysis pipeline (EXTENDED AGAIN)

Adds:
A) Place-based analysis:
   - canonicalize Place into buckets (Oxyrhynchite, Arsinoite, etc.)
   - trends per place (top-N by observations)

B) Smoothed trends + price index:
   - binning by YearBin
   - median + IQR
   - rolling median smoothing
   - price index relative to baseline period (e.g. 50–99 CE = 100)

C) Manual-review queue:
   - flags likely false positives and "needs review"
   - reasons column
   - prioritized sampling for human checking

Outputs written into OUTDIR:
 - v11b_PRIMARY_cleaned_advanced.csv
 - v11b_MAIN_filtered_advanced.csv
 - v11b_manual_review_queue.csv
 - subset_wheat_artaba_drachma_MAIN.csv
 - trend_wheat_artaba_drachma_all_places.csv
 - trend_wheat_artaba_drachma_top_places.csv
 - price_index_wheat_artaba_drachma.csv
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

INFILE = Path("linked_all_scores_4042.csv")   # <-- change if needed
OUTDIR = Path("v11b_outputs_advancedv2")
OUTDIR.mkdir(exist_ok=True)

# Confidence / filtering
MIN_SCORE_MAIN = 60
DROP_AMBIGUOUS_MAIN = False

YEAR_BIN = 25  # 25-year bins

# Manual review queue size
REVIEW_TOP_N = 600

# Group outlier robustness threshold
ROBUST_Z_THRESH = 4.0

# Trend smoothing window (in bins)
SMOOTH_WINDOW = 3  # rolling median over 3 bins

# Price-index baseline (inclusive range)
INDEX_BASELINE_START = 50
INDEX_BASELINE_END = 99

# Place trend lines
TOP_PLACES_TO_PLOT = 6
MIN_N_PER_BIN = 5


# ============================================================
# Conversion tables (EDIT THESE!)
# ============================================================

UNIT_TO_LITER = {
    "artaba": 38.8,     # approximate placeholder
    "choenix": 1.08,    # approximate placeholder
    "medimnos": 52.5,   # approximate placeholder
    "kotyle": 0.27,     # approximate placeholder
    "metretes": 39.0,   # approximate placeholder
}

CUR_TO_DRACHMA = {
    "drachma": 1.0,
    "obol": 1.0 / 6.0,
    "chalkous": 1.0 / 48.0,
    "denarius": 4.0,    
    "sestertius": 1.0,  
    "mina": 100.0,      
    "talent": 6000.0,   
}


# ============================================================
# NORMALIZATION HELPERS
# ============================================================

def strip_accents(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def norm_greek(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = strip_accents(s).casefold()
    s = " ".join(s.split())
    return s


def clean_context(s: str, max_len: int = 500) -> str:
    if s is None:
        return ""
    s = str(s).replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s).strip()
    return s[:max_len]


# ============================================================
# CANONICALIZATION: grain/unit/currency
# ============================================================

def canon_grain(form: str) -> str:
    g = norm_greek(form)
    if not g:
        return ""
    if g.startswith("πυρ"):
        return "wheat (pyros)"
    if g.startswith("σιτ"):
        return "grain (sitos)"
    if g.startswith("κριθ"):
        return "barley (krithē)"
    if g.startswith("ζε"):
        return "spelt/zea"
    if g.startswith("ολυρ"):
        return "emmer (olyra)"
    if g.startswith("αλευρ"):
        return "flour (aleuron)"
    if g.startswith("σταχυ"):
        return "ear/spike (stachys)"
    return g


def canon_unit(unit: str) -> str:
    u = norm_greek(unit)
    if not u:
        return ""
    if u.startswith("αρταβ"):
        return "artaba"
    if u.startswith("μεδιμν"):
        return "medimnos"
    if u.startswith("χοιν"):
        return "choenix"
    if u.startswith("κοτυλ"):
        return "kotyle"
    if u.startswith("μετρ"):
        return "metretes"
    return u


def canon_currency(cur: str) -> str:
    c = norm_greek(cur)
    if not c:
        return ""
    if c.startswith("δραχ"):
        return "drachma"
    if c.startswith("οβολ"):
        return "obol"
    if c.startswith("δηναρ"):
        return "denarius"
    if c.startswith("μνα"):
        return "mina"
    if c.startswith("ταλαν"):
        return "talent"
    if c.startswith("σεστ"):
        return "sestertius"
    if c.startswith("χαλκ"):
        return "chalkous"
    if c.startswith("ἀσσ") or c.startswith("ασσ"):
        return "as"
    return c


# ============================================================
# PLACE CANONICALIZATION
# ============================================================

PLACE_PATTERNS = [
    ("Oxyrhynchite", re.compile(r"oxyrh|οξυρ", re.I)),
    ("Arsinoite", re.compile(r"arsin|αρσιν|φαγιουμ|fayum", re.I)),
    ("Hermopolite", re.compile(r"hermop|ερμοπ", re.I)),
    ("Herakleopolite", re.compile(r"herakleop|ηρακλεοπ", re.I)),
    ("Memphite", re.compile(r"memph|μεμφ", re.I)),
    ("Panopolite", re.compile(r"panop|πανοπ|akhmim", re.I)),
    ("Thebaid", re.compile(r"theb|θηβ|λυκοπ|lycop", re.I)),
    ("Alexandria", re.compile(r"alex|αλεξανδρ", re.I)),
    ("Unknown", re.compile(r".*", re.I)),
]


def canon_place(place: str) -> str:
    p = norm_greek(place)
    if not p:
        return "Unknown"
    # sometimes Place has comma-separated details; keep full string but bucketize
    for label, pat in PLACE_PATTERNS:
        if pat.search(p):
            return label
    return "Unknown"


# ============================================================
# DATE PARSING
# ============================================================

def parse_year_from_iso(s: str) -> float:
    if s is None:
        return np.nan
    s = str(s).strip()
    if not s:
        return np.nan
    m = re.match(r"^([+-]?)\s*(\d{1,4})", s)
    if not m:
        return np.nan
    sign = -1 if m.group(1) == "-" else 1
    return float(sign * int(m.group(2)))


def infer_year(row: pd.Series) -> float:
    y = parse_year_from_iso(row.get("Date_When", ""))
    if np.isfinite(y):
        return y
    nb = parse_year_from_iso(row.get("Date_NotBefore", ""))
    na = parse_year_from_iso(row.get("Date_NotAfter", ""))
    if np.isfinite(nb) and np.isfinite(na):
        return (nb + na) / 2.0
    if np.isfinite(nb):
        return nb
    if np.isfinite(na):
        return na
    return np.nan


# ============================================================
# NUMERIC PARSING
# ============================================================

def to_float(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if not s:
        return np.nan
    if re.fullmatch(r"\d+/\d+", s):
        n, d = s.split("/")
        d = int(d)
        return float(int(n) / d) if d else np.nan
    try:
        return float(s)
    except:
        return np.nan


# ============================================================
# PRIMARY SELECTION
# ============================================================

def select_primary_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "Is_Primary" in df.columns:
        prim = df[df["Is_Primary"].astype(str).str.lower().eq("yes")]
        if len(prim) > 0:
            return prim.copy()

    if "Candidate_Rank" in df.columns:
        rank1 = df[df["Candidate_Rank"].astype(str) == "1"]
        if len(rank1) > 0:
            return rank1.copy()

    key_cols = [c for c in ["DDB_ID", "Mention_ID", "Grain_Index"] if c in df.columns]
    if key_cols and "Score" in df.columns:
        tmp = df.copy()
        tmp["ScoreNum"] = pd.to_numeric(tmp["Score"], errors="coerce")
        tmp = tmp.sort_values("ScoreNum", ascending=False)
        return tmp.drop_duplicates(key_cols, keep="first").drop(columns=["ScoreNum"])

    return df.copy()


# ============================================================
# CONVERSIONS
# ============================================================

def convert_currency_to_drachma(amount: float, cur_canon: str) -> float:
    if not np.isfinite(amount):
        return np.nan
    factor = CUR_TO_DRACHMA.get(cur_canon, None)
    if factor is None:
        return np.nan
    return amount * factor


def convert_qty_to_liters(qty: float, unit_canon: str) -> float:
    if not np.isfinite(qty):
        return np.nan
    factor = UNIT_TO_LITER.get(unit_canon, None)
    if factor is None:
        return np.nan
    return qty * factor


def add_standardized_prices(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Price_DrachmaEq"] = out.apply(lambda r: convert_currency_to_drachma(r["PriceNum"], r["Cur_Canon"]), axis=1)
    out["Qty_Liters"] = out.apply(lambda r: convert_qty_to_liters(r["QtyNum"], r["Unit_Canon"]), axis=1)

    out["Price_per_Liter_DrachmaEq"] = np.where(
        out["Qty_Liters"] > 0,
        out["Price_DrachmaEq"] / out["Qty_Liters"],
        np.nan,
    )

    artaba_L = UNIT_TO_LITER.get("artaba", None)
    if artaba_L and artaba_L > 0:
        out["Price_per_Artaba_DrachmaEq"] = out["Price_per_Liter_DrachmaEq"] * artaba_L
    else:
        out["Price_per_Artaba_DrachmaEq"] = np.nan

    out["log_Price_per_Liter"] = np.log10(out["Price_per_Liter_DrachmaEq"])
    out.loc[~np.isfinite(out["log_Price_per_Liter"]), "log_Price_per_Liter"] = np.nan

    return out


# ============================================================
# QUALITY TIERS & FILTERS
# ============================================================

def add_quality_tiers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    score = pd.to_numeric(out.get("Score", np.nan), errors="coerce")
    ambiguous = out.get("Ambiguous", "").astype(str).str.lower().eq("yes")

    out["Tier"] = np.select(
        [
            (score >= 65) & (~ambiguous),
            (score >= 60),
        ],
        ["A", "B"],
        default="C",
    )
    return out


def filter_main_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    score = pd.to_numeric(out.get("Score", np.nan), errors="coerce")
    out = out[score >= MIN_SCORE_MAIN]

    if DROP_AMBIGUOUS_MAIN and "Ambiguous" in out.columns:
        out = out[out["Ambiguous"].astype(str).str.lower().ne("yes")]

    return out


# ============================================================
# ROBUST OUTLIERS (within groups)
# ============================================================

def robust_zscore(x: pd.Series) -> pd.Series:
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if mad == 0 or not np.isfinite(mad):
        return pd.Series([np.nan] * len(x), index=x.index)
    return 0.6745 * (x - med) / mad


def add_group_outlier_flags(df: pd.DataFrame, value_col: str, out_col_prefix: str) -> pd.DataFrame:
    """
    Adds:
      {prefix}_GroupZ
      {prefix}_IsOutlier
    within groups (Grain_Canon, Unit_Canon, Cur_Canon)
    """
    out = df.copy()
    gcols = ["Grain_Canon", "Unit_Canon", "Cur_Canon"]

    zcol = f"{out_col_prefix}_GroupZ"
    fcol = f"{out_col_prefix}_IsOutlier"
    out[zcol] = np.nan

    for _, grp in out.groupby(gcols):
        z = robust_zscore(pd.to_numeric(grp[value_col], errors="coerce"))
        out.loc[grp.index, zcol] = z

    out[fcol] = out[zcol].abs() >= ROBUST_Z_THRESH
    return out


# ============================================================
# TRENDS + SMOOTHING + INDEX
# ============================================================

def rolling_median(s: pd.Series, window: int = 3) -> pd.Series:
    return s.rolling(window=window, center=True, min_periods=1).median()


def make_trend_table(df: pd.DataFrame, value_col: str, group_cols: list[str]) -> pd.DataFrame:
    """
    group_cols should include:
      - YearBin
      - optional Place_Canon, Grain_Canon, Unit_Canon, Cur_Canon
    """
    sub = df.dropna(subset=["YearBin", value_col]).copy()
    if len(sub) == 0:
        return pd.DataFrame()

    trend = sub.groupby(group_cols).agg(
        n=(value_col, "size"),
        median=(value_col, "median"),
        q25=(value_col, lambda x: np.nanpercentile(x, 25)),
        q75=(value_col, lambda x: np.nanpercentile(x, 75)),
    ).reset_index()

    trend = trend.sort_values(group_cols)
    # smoothed median only if YearBin is in group_cols
    if "YearBin" in group_cols:
        # apply smoothing per other groups
        other = [c for c in group_cols if c != "YearBin"]
        if other:
            trend["median_smooth"] = trend.groupby(other)["median"].transform(lambda s: rolling_median(s, SMOOTH_WINDOW))
        else:
            trend["median_smooth"] = rolling_median(trend["median"], SMOOTH_WINDOW)

    return trend


def compute_price_index(trend: pd.DataFrame) -> pd.DataFrame:
    """
    Compute index=100 for baseline period median level.
    Requires columns: YearBin, median (or median_smooth)
    """
    if len(trend) == 0:
        return trend

    out = trend.copy()
    y = out["YearBin"].astype(float)

    base = out[(y >= INDEX_BASELINE_START) & (y <= INDEX_BASELINE_END)]
    if len(base) == 0:
        out["price_index"] = np.nan
        return out

    base_level = np.nanmedian(base["median"])
    if not np.isfinite(base_level) or base_level == 0:
        out["price_index"] = np.nan
        return out

    out["price_index"] = 100.0 * out["median"] / base_level
    if "median_smooth" in out.columns and out["median_smooth"].notna().any():
        out["price_index_smooth"] = 100.0 * out["median_smooth"] / base_level
    else:
        out["price_index_smooth"] = np.nan

    return out


def plot_trend_lines(trend: pd.DataFrame, title: str, label_col: str | None = None,
                     ycol: str = "median_smooth", min_n: int = MIN_N_PER_BIN):
    """
    Plot smoothed median trend lines.
    If label_col is provided, plot one line per label value (top labels by total n).
    """
    if len(trend) == 0:
        print("No trend data:", title)
        return

    # filter bins by n
    tr = trend[trend["n"] >= min_n].copy()
    if len(tr) == 0:
        print(f"No bins with n >= {min_n}:", title)
        return

    plt.figure()

    if label_col is None:
        x = tr["YearBin"].astype(float)
        plt.plot(x, tr[ycol].astype(float), linewidth=2)
    else:
        # choose top places/labels by total n
        label_totals = tr.groupby(label_col)["n"].sum().sort_values(ascending=False)
        top_labels = list(label_totals.head(TOP_PLACES_TO_PLOT).index)

        for lab in top_labels:
            t2 = tr[tr[label_col] == lab].sort_values("YearBin")
            x = t2["YearBin"].astype(float)
            plt.plot(x, t2[ycol].astype(float), linewidth=2, label=str(lab))

        plt.legend()

    plt.xlabel(f"Year bin ({YEAR_BIN}y)")
    plt.ylabel(ycol)
    plt.title(title)
    plt.show()


# ============================================================
# MANUAL REVIEW QUEUE
# ============================================================

def build_review_queue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prioritize rows for manual inspection.

    Rules (reasons):
      - ambiguous extraction
      - low score
      - missing qty (can't compute unit price)
      - very large span or distance
      - group outlier in UnitPrice_raw
      - group outlier in Price_per_Liter_DrachmaEq
      - extreme unit price (top 1% within subset)
    """
    out = df.copy()

    score = pd.to_numeric(out.get("Score", np.nan), errors="coerce")
    ambiguous = out.get("Ambiguous", "").astype(str).str.lower().eq("yes")

    out["ReviewReason"] = ""

    def add_reason(mask, reason):
        out.loc[mask, "ReviewReason"] = out.loc[mask, "ReviewReason"].apply(
            lambda s: (s + "; " + reason).strip("; ").strip()
        )

    add_reason(ambiguous, "ambiguous")
    add_reason(score < 60, "low_score(<60)")

    add_reason(out["QtyNum"].isna() | (out["QtyNum"] <= 0), "missing_or_bad_qty")
    add_reason(out.get("Span_Toks", 0).astype(float) >= 60, "wide_span(>=60)")
    add_reason(out.get("Dist_GP", 0).astype(float) >= 35, "far_grain_price(>=35)")
    add_reason(out.get("Priceword_Near", "").astype(str).str.lower().eq("no"), "no_priceword_near")

    # outlier flags if present
    if "RAW_Group_IsOutlier" in out.columns:
        add_reason(out["RAW_Group_IsOutlier"].fillna(False), "group_outlier_raw_unitprice")
    if "STD_Group_IsOutlier" in out.columns:
        add_reason(out["STD_Group_IsOutlier"].fillna(False), "group_outlier_std_price_per_liter")

    # extreme unit price heuristic (global top 1%)
    if "UnitPrice_raw" in out.columns:
        up = pd.to_numeric(out["UnitPrice_raw"], errors="coerce")
        cutoff = up.quantile(0.99)
        add_reason(up >= cutoff, "unitprice_top1pct")

    # final priority score (bigger = review earlier)
    priority = np.zeros(len(out), dtype=float)
    priority += ambiguous.astype(float) * 3.0
    priority += (score < 60).astype(float) * 2.5
    priority += (out["QtyNum"].isna() | (out["QtyNum"] <= 0)).astype(float) * 2.0
    priority += (out.get("Span_Toks", 0).astype(float) >= 60).astype(float) * 1.5
    priority += (out.get("Dist_GP", 0).astype(float) >= 35).astype(float) * 1.2
    priority += (out.get("Priceword_Near", "").astype(str).str.lower().eq("no")).astype(float) * 1.0

    if "RAW_Group_IsOutlier" in out.columns:
        priority += out["RAW_Group_IsOutlier"].fillna(False).astype(float) * 2.0
    if "STD_Group_IsOutlier" in out.columns:
        priority += out["STD_Group_IsOutlier"].fillna(False).astype(float) * 2.0

    out["ReviewPriority"] = priority

    # keep only rows that have at least one reason
    out = out[out["ReviewReason"].str.len() > 0].copy()
    out = out.sort_values(["ReviewPriority", "Score"], ascending=[False, True])

    return out


# ============================================================
# MAIN
# ============================================================

def main():
    if not INFILE.exists():
        raise FileNotFoundError(f"Input CSV not found: {INFILE.resolve()}")

    df_raw = pd.read_csv(INFILE).fillna("")
    print("Loaded:", INFILE, "rows =", len(df_raw))

    # Primary selection
    df = select_primary_rows(df_raw)
    print("Primary rows:", len(df))

    # Canonicalize + numeric
    df["Grain_Canon"] = df["Grain_Form"].apply(canon_grain)
    df["Unit_Canon"] = df["Qty_Unit"].apply(canon_unit)
    df["Cur_Canon"] = df["Price_Cur"].apply(canon_currency)
    df["Place_Canon"] = df.get("Place", "").apply(canon_place)

    df["QtyNum"] = df["Qty_Value"].apply(to_float)
    df["PriceNum"] = df["Price_Value"].apply(to_float)

    df["UnitPrice_raw"] = np.where(df["QtyNum"] > 0, df["PriceNum"] / df["QtyNum"], np.nan)

    # Date fields
    df["Year"] = df.apply(infer_year, axis=1)
    df["YearBin"] = np.where(
        np.isfinite(df["Year"]),
        (np.floor(df["Year"] / YEAR_BIN) * YEAR_BIN).astype("Int64"),
        pd.NA,
    )

    # Context cleaning
    if "Context_Window" in df.columns:
        df["Context_Clean"] = df["Context_Window"].apply(clean_context)
        df["Context_OneLine"] = df["Context_Clean"].str.replace("\n", " ", regex=False)

    # Quality tiers + standardized prices
    df = add_quality_tiers(df)
    df = add_standardized_prices(df)

    # Export primary cleaned
    out_primary = OUTDIR / "v11b_PRIMARY_cleaned_advanced.csv"
    df.to_csv(out_primary, index=False, encoding="utf-8-sig")
    print("Saved:", out_primary)

    # Main filtered dataset
    df_main = filter_main_dataset(df)
    print("Main filtered rows:", len(df_main))

    # Outlier flags
    df_main = add_group_outlier_flags(df_main, value_col="UnitPrice_raw", out_col_prefix="RAW_Group")
    df_main = add_group_outlier_flags(df_main, value_col="Price_per_Liter_DrachmaEq", out_col_prefix="STD_Group")

    out_main = OUTDIR / "v11b_MAIN_filtered_advanced.csv"
    df_main.to_csv(out_main, index=False, encoding="utf-8-sig")
    print("Saved:", out_main)

    # Subset: wheat + artaba + drachma
    subset = df_main[
        (df_main["Grain_Canon"] == "wheat (pyros)") &
        (df_main["Unit_Canon"] == "artaba") &
        (df_main["Cur_Canon"] == "drachma")
    ].copy()

    out_subset = OUTDIR / "subset_wheat_artaba_drachma_MAIN.csv"
    subset.to_csv(out_subset, index=False, encoding="utf-8-sig")
    print("Saved subset:", out_subset, "rows =", len(subset))

    # Trend: all places combined
    trend_all = make_trend_table(
        subset,
        value_col="UnitPrice_raw",
        group_cols=["YearBin"],
    )
    out_trend_all = OUTDIR / "trend_wheat_artaba_drachma_all_places.csv"
    trend_all.to_csv(out_trend_all, index=False, encoding="utf-8-sig")
    print("Saved:", out_trend_all)

    # Trend: by place
    trend_place = make_trend_table(
        subset,
        value_col="UnitPrice_raw",
        group_cols=["Place_Canon", "YearBin"],
    )
    out_trend_place = OUTDIR / "trend_wheat_artaba_drachma_top_places.csv"
    trend_place.to_csv(out_trend_place, index=False, encoding="utf-8-sig")
    print("Saved:", out_trend_place)

    # Price index (baseline=50–99 CE)
    price_index = compute_price_index(trend_all)
    out_index = OUTDIR / "price_index_wheat_artaba_drachma.csv"
    price_index.to_csv(out_index, index=False, encoding="utf-8-sig")
    print("Saved:", out_index)

    # Manual review queue
    review = build_review_queue(df_main)
    review = review.head(REVIEW_TOP_N)
    out_review = OUTDIR / "v11b_manual_review_queue.csv"
    review.to_csv(out_review, index=False, encoding="utf-8-sig")
    print("Saved review queue:", out_review, "rows =", len(review))

    # ============================================================
    # PLOTS
    # ============================================================

    # 1) score histogram
    if "Score" in df.columns:
        s = pd.to_numeric(df["Score"], errors="coerce").dropna()
        plt.figure()
        plt.hist(s, bins=30)
        plt.xlabel("Score")
        plt.ylabel("Count")
        plt.title("Extraction confidence score distribution (primary rows)")
        plt.show()

    # 2) trend lines (all places)
    if len(trend_all) > 0:
        plot_trend_lines(
            trend_all,
            title="Wheat: drachma/artaba (median_smooth) — all places",
            label_col=None,
            ycol="median_smooth",
            min_n=MIN_N_PER_BIN,
        )

    # 3) trend lines (top places)
    if len(trend_place) > 0:
        plot_trend_lines(
            trend_place,
            title="Wheat: drachma/artaba (median_smooth) — top places",
            label_col="Place_Canon",
            ycol="median_smooth",
            min_n=MIN_N_PER_BIN,
        )

    # 4) price index plot
    if len(price_index) > 0 and "price_index_smooth" in price_index.columns:
        px = price_index.dropna(subset=["price_index_smooth", "YearBin"]).copy()
        if len(px) > 0:
            plt.figure()
            plt.plot(px["YearBin"].astype(float), px["price_index_smooth"].astype(float), linewidth=2)
            plt.xlabel(f"Year bin ({YEAR_BIN}y)")
            plt.ylabel("Price index (baseline=100)")
            plt.title(f"Wheat price index (drachma/artaba), baseline {INDEX_BASELINE_START}-{INDEX_BASELINE_END} CE")
            plt.show()

    # 5) scatter of subset (year vs unit price)
    scat = subset.dropna(subset=["Year", "UnitPrice_raw"])
    if len(scat) > 0:
        plt.figure()
        plt.scatter(scat["Year"], scat["UnitPrice_raw"], alpha=0.4)
        plt.xlabel("Year (approx)")
        plt.ylabel("Unit price (drachma/artaba)")
        plt.title("Wheat unit price scatter (main filtered)")
        plt.show()

    print("\nDONE. All outputs are in:", OUTDIR.resolve())


if __name__ == "__main__":
    main()
