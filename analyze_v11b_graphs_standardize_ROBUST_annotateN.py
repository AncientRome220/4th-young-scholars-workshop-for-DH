# -*- coding: utf-8 -*-
"""
analyze_v11b_graphs_standardize.py  (ROBUST)

Reads a v11b cleaned CSV and:
1) Converts currencies -> drachma-equivalent
2) Converts units -> artaba-equivalent (via liters proxy)
3) Produces summary charts + ROBUST trends by YearBin that are resistant to extreme values.

Why robust trends?
- Extracted prices are heavy-tailed and can include occasional extraction/conversion errors.
- Even median+IQR can be distorted when a bin has small n and one extreme value dominates.

Robust trend method:
- compute within-bin stats in log10(price) space
- trim outliers using MAD-based z (default 4.0)
- then compute median + IQR and plot (log y-axis by default)

USAGE
-----
python analyze_v11b_graphs_standardize.py --csv v11b_PRIMARY_cleaned_advanced.csv --outdir figs_std

Common:
  --min-score 65 --drop-ambiguous --year-min 0 --year-max 300
  --grain "wheat (pyros)"
  --mad-z 4.0 --min-n-bin 3

If you want to disable robust trimming:
  --mad-z 0

If you want linear y-axis (not recommended for prices):
  --linear-y
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# EDIT THESE CONVERSION TABLES
# ============================================================
# Currency -> drachma
# amount_in_drachma = amount_in_currency * CUR_TO_DRACHMA[currency]
CUR_TO_DRACHMA = {
    "drachma": 1.0,
    "obol": 1.0 / 6.0,
    "chalkous": 1.0 / 48.0,
    # Add if you decide a convention (examples):
    # "denarius": 4.0,
    # "sestertius": 1.0,
    # "mina": 100.0,
    # "talent": 6000.0,
}

# Units -> liters, then liters -> artaba using artaba liters.
# qty_artaba = (qty * UNIT_TO_LITER[unit]) / UNIT_TO_LITER["artaba"]
UNIT_TO_LITER = {
    "artaba": 38.8,     # placeholder; EDIT if needed
    "choenix": 1.08,    # placeholder
    "medimnos": 52.5,   # placeholder
    "kotyle": 0.27,     # placeholder
    "metretes": 39.0,   # placeholder
}


# ============================================================
# HELPERS
# ============================================================
def to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def savefig(outdir: Path, name: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    print("Saved:", path)


def bar_counts(df: pd.DataFrame, col: str, outdir: Path, topn: int = 15) -> None:
    if col not in df.columns:
        return
    counts = df[col].fillna("").replace("", "(blank)").value_counts().head(topn)
    plt.figure()
    counts.sort_values().plot(kind="barh")
    plt.xlabel("Count")
    plt.title(f"Top {topn}: {col}")
    savefig(outdir, f"counts_{col}.png")
    plt.close()


def hist_numeric(df: pd.DataFrame, col: str, outdir: Path, bins: int = 40) -> None:
    if col not in df.columns:
        return
    x = to_num(df[col]).dropna()
    if len(x) == 0:
        return
    plt.figure()
    plt.hist(x, bins=bins)
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.title(f"Histogram: {col}")
    savefig(outdir, f"hist_{col}.png")
    plt.close()


def hist_log(df: pd.DataFrame, col: str, outdir: Path, bins: int = 40, clip_quantile: float = 0.995) -> None:
    """
    Log histogram for positive values. Clips extreme tail for readability.
    """
    if col not in df.columns:
        return
    x = to_num(df[col])
    x = x[(x > 0) & np.isfinite(x)]
    if len(x) == 0:
        return
    cap = x.quantile(clip_quantile)
    x = x[x <= cap]
    lx = np.log10(x)
    plt.figure()
    plt.hist(lx, bins=bins)
    plt.xlabel(f"log10({col})  (clipped at q={clip_quantile})")
    plt.ylabel("Count")
    plt.title(f"Log histogram: {col}")
    savefig(outdir, f"hist_log10_{col}.png")
    plt.close()


def scatter_year_price(df: pd.DataFrame, xcol: str, ycol: str, outdir: Path, title: str, filename: str, ylog: bool) -> None:
    if xcol not in df.columns or ycol not in df.columns:
        return
    sub = df.copy()
    sub[xcol] = to_num(sub[xcol])
    sub[ycol] = to_num(sub[ycol])
    sub = sub.dropna(subset=[xcol, ycol])
    sub = sub[sub[ycol] > 0]  # for log friendliness
    if len(sub) == 0:
        return

    plt.figure()
    plt.scatter(sub[xcol], sub[ycol], alpha=0.4)
    plt.xlabel(xcol)
    plt.ylabel(ycol + (" [log]" if ylog else ""))
    plt.title(title)
    if ylog:
        plt.yscale("log")
    savefig(outdir, filename)
    plt.close()


# ============================================================
# STANDARDIZATION
# ============================================================
def standardize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      PriceNum, QtyNum
      Price_DrachmaEq_calc
      Qty_ArtabaEq_calc
      UnitPrice_Drachma_per_ArtabaEq_calc
    based on Cur_Canon, Unit_Canon, Price/Qty values.
    """
    out = df.copy()

    # Locate numeric columns
    if "PriceNum" not in out.columns:
        out["PriceNum"] = to_num(out["Price_Value"]) if "Price_Value" in out.columns else np.nan
    else:
        out["PriceNum"] = to_num(out["PriceNum"])

    if "QtyNum" not in out.columns:
        out["QtyNum"] = to_num(out["Qty_Value"]) if "Qty_Value" in out.columns else np.nan
    else:
        out["QtyNum"] = to_num(out["QtyNum"])

    # Currency conversion
    def price_to_drachma(row):
        amt = row["PriceNum"]
        cur = str(row.get("Cur_Canon", "")).strip()
        if not np.isfinite(amt):
            return np.nan
        f = CUR_TO_DRACHMA.get(cur, None)
        if f is None:
            return np.nan
        return float(amt) * float(f)

    out["Price_DrachmaEq_calc"] = out.apply(price_to_drachma, axis=1)

    # Unit conversion -> artaba
    artaba_L = UNIT_TO_LITER.get("artaba", None)
    if artaba_L is None or artaba_L <= 0:
        out["Qty_ArtabaEq_calc"] = np.nan
    else:
        def qty_to_artaba(row):
            q = row["QtyNum"]
            u = str(row.get("Unit_Canon", "")).strip()
            if not np.isfinite(q) or q <= 0:
                return np.nan
            uL = UNIT_TO_LITER.get(u, None)
            if uL is None:
                return np.nan
            return float(q) * float(uL) / float(artaba_L)

        out["Qty_ArtabaEq_calc"] = out.apply(qty_to_artaba, axis=1)

    out["UnitPrice_Drachma_per_ArtabaEq_calc"] = np.where(
        out["Qty_ArtabaEq_calc"] > 0,
        out["Price_DrachmaEq_calc"] / out["Qty_ArtabaEq_calc"],
        np.nan,
    )

    return out


# ============================================================
# ROBUST TREND (per YearBin)
# ============================================================
def robust_bin_summary(
    y: pd.Series,
    *,
    mad_z: float | None = 4.0,
    winsor: tuple[float, float] | None = None,
) -> dict:
    """
    Compute robust median + IQR for one bin.
    - Work in log10(y) space (y must be > 0)
    - MAD-z trimming removes extreme points
    """
    y = pd.to_numeric(y, errors="coerce")
    y = y[np.isfinite(y)]
    y = y[y > 0]
    if len(y) == 0:
        return {"n": 0, "median": np.nan, "q25": np.nan, "q75": np.nan}

    z = np.log10(y.to_numpy(dtype=float))

    if mad_z is not None:
        med = np.median(z)
        mad = np.median(np.abs(z - med))
        if np.isfinite(mad) and mad > 0:
            rz = 0.6745 * (z - med) / mad
            z = z[np.abs(rz) <= mad_z]

    if len(z) == 0:
        return {"n": 0, "median": np.nan, "q25": np.nan, "q75": np.nan}

    if winsor is not None:
        lo_q, hi_q = winsor
        lo = np.quantile(z, lo_q)
        hi = np.quantile(z, hi_q)
        z = np.clip(z, lo, hi)

    med_z = np.median(z)
    q25_z = np.quantile(z, 0.25)
    q75_z = np.quantile(z, 0.75)

    return {
        "n": int(len(z)),
        "median": float(10 ** med_z),
        "q25": float(10 ** q25_z),
        "q75": float(10 ** q75_z),
    }


def robust_trend_plot(
    df: pd.DataFrame,
    *,
    xcol: str,
    ycol: str,
    outdir: Path,
    min_n: int,
    mad_z: float | None,
    winsor: tuple[float, float] | None,
    ylog: bool,
    title: str,
    fig_name: str,
    table_name: str,
) -> pd.DataFrame:
    if xcol not in df.columns or ycol not in df.columns:
        return pd.DataFrame()

    sub = df.copy()
    sub[xcol] = to_num(sub[xcol])
    sub[ycol] = to_num(sub[ycol])
    sub = sub.dropna(subset=[xcol, ycol])

    rows = []
    for xb, g in sub.groupby(xcol):
        stats = robust_bin_summary(g[ycol], mad_z=mad_z, winsor=winsor)
        rows.append({xcol: xb, **stats})

    trend = pd.DataFrame(rows).sort_values(xcol)
    trend = trend[trend["n"] >= min_n]
    if len(trend) == 0:
        return trend

    # Save table
    outdir.mkdir(parents=True, exist_ok=True)
    trend.to_csv(outdir / table_name, index=False, encoding="utf-8-sig")
    print("Saved:", outdir / table_name)

    x = trend[xcol].to_numpy(dtype=float)
    med = trend["median"].to_numpy(dtype=float)
    q25 = trend["q25"].to_numpy(dtype=float)
    q75 = trend["q75"].to_numpy(dtype=float)

    plt.figure()
    plt.plot(x, med, linewidth=2)
    plt.fill_between(x, q25, q75, alpha=0.2)

    # Annotate bin sample size n on each point
    ns = trend["n"].to_numpy(dtype=int)
    for xi, yi, ni in zip(x, med, ns):
        if not np.isfinite(yi):
            continue
        y_text = yi * (1.12 if ylog else 1.02)
        plt.text(xi, y_text, str(int(ni)), ha="center", va="bottom", fontsize=8)

    plt.xlabel(xcol)
    plt.ylabel(ycol + (" [log]" if ylog else ""))
    plt.title(title)
    if ylog:
        plt.yscale("log")
    savefig(outdir, fig_name)
    plt.close()
    return trend


def place_trend_plot_robust(
    df: pd.DataFrame,
    *,
    place_col: str,
    xcol: str,
    ycol: str,
    outdir: Path,
    topn_places: int,
    min_n: int,
    mad_z: float | None,
    winsor: tuple[float, float] | None,
    ylog: bool,
    title: str,
    fig_name: str,
) -> None:
    if place_col not in df.columns or xcol not in df.columns or ycol not in df.columns:
        return

    sub = df.copy()
    sub[xcol] = to_num(sub[xcol])
    sub[ycol] = to_num(sub[ycol])
    sub = sub.dropna(subset=[place_col, xcol, ycol])
    if len(sub) == 0:
        return

    top_places = sub[place_col].value_counts().head(topn_places).index.tolist()

    plt.figure()
    for p in top_places:
        s = sub[sub[place_col] == p].copy()
        rows = []
        for xb, g in s.groupby(xcol):
            stats = robust_bin_summary(g[ycol], mad_z=mad_z, winsor=winsor)
            rows.append({xcol: xb, **stats})
        tr = pd.DataFrame(rows).sort_values(xcol)
        tr = tr[tr["n"] >= min_n]
        if len(tr) == 0:
            continue
        x = tr[xcol].to_numpy(dtype=float)
        med = tr["median"].to_numpy(dtype=float)
        plt.plot(x, med, linewidth=2, label=str(p))

    plt.xlabel(xcol)
    plt.ylabel(ycol + (" [log]" if ylog else ""))
    plt.title(title)
    if ylog:
        plt.yscale("log")
    plt.legend()
    savefig(outdir, fig_name)
    plt.close()


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input v11b cleaned CSV")
    ap.add_argument("--outdir", default="figs_v11b_std", help="Output directory for PNGs")

    ap.add_argument("--min-score", type=float, default=0.0, help="Filter: Score >= min-score")
    ap.add_argument("--drop-ambiguous", action="store_true", help="Filter: Ambiguous != yes")
    ap.add_argument("--year-min", type=float, default=None, help="Filter: Year >= year-min")
    ap.add_argument("--year-max", type=float, default=None, help="Filter: Year <= year-max")

    ap.add_argument("--grain", default="wheat (pyros)", help="Focus grain canon label")
    ap.add_argument("--min-n-bin", type=int, default=3, help="Min n per YearBin for trends (IMPORTANT)")
    ap.add_argument("--topn-places", type=int, default=6, help="Top places to plot")

    ap.add_argument("--mad-z", type=float, default=4.0, help="MAD-z trimming in each bin (0 disables)")
    ap.add_argument("--winsor", type=str, default="", help="Optional winsor like 0.01,0.99")
    ap.add_argument("--linear-y", action="store_true", help="Use linear y-axis (default is log for trends/scatter)")
    ap.add_argument("--export-extremes", type=int, default=50, help="Export top-N extreme standardized prices for inspection")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    ylog = not args.linear_y

    df = pd.read_csv(args.csv).fillna("")

    # Filters
    if "Score" in df.columns:
        df["ScoreNum"] = to_num(df["Score"])
        df = df[df["ScoreNum"] >= args.min_score]

    if args.drop_ambiguous and "Ambiguous" in df.columns:
        df = df[df["Ambiguous"].astype(str).str.lower().ne("yes")]

    if "Year" in df.columns:
        df["YearNum"] = to_num(df["Year"])
        if args.year_min is not None:
            df = df[df["YearNum"] >= args.year_min]
        if args.year_max is not None:
            df = df[df["YearNum"] <= args.year_max]

    print("Rows after filters:", len(df))

    # Standardize
    df = standardize_prices(df)

    # Global counts/hists
    if "Score" in df.columns:
        hist_numeric(df, "Score", outdir, bins=40)

    for c in ["Tier", "Grain_Canon", "Unit_Canon", "Cur_Canon", "Place_Canon"]:
        bar_counts(df, c, outdir, topn=12)

    # Raw and standardized distributions
    if "UnitPrice_raw" in df.columns:
        hist_log(df, "UnitPrice_raw", outdir, bins=45)
    hist_log(df, "UnitPrice_Drachma_per_ArtabaEq_calc", outdir, bins=45)

    # Focus grain subset (allow all convertible units/currencies)
    if "Grain_Canon" in df.columns:
        sub = df[df["Grain_Canon"] == args.grain].copy()
    else:
        sub = df.copy()

    print(f"Subset rows ({args.grain}):", len(sub))

    y_std = "UnitPrice_Drachma_per_ArtabaEq_calc"

    # Export extreme standardized prices for debugging (often reveals conversion/table issues)
    if args.export_extremes and y_std in sub.columns:
        tmp = sub.copy()
        tmp[y_std] = to_num(tmp[y_std])
        tmp = tmp.dropna(subset=[y_std])
        tmp = tmp[tmp[y_std] > 0].sort_values(y_std, ascending=False).head(args.export_extremes)
        (outdir / "tables").mkdir(parents=True, exist_ok=True)
        tmp.to_csv(outdir / "tables" / f"top_{args.export_extremes}_extreme_std_unitprices.csv", index=False, encoding="utf-8-sig")
        print("Saved:", outdir / "tables" / f"top_{args.export_extremes}_extreme_std_unitprices.csv")

    # Parse robust options
    mad_z = None if args.mad_z == 0 else float(args.mad_z)
    winsor = None
    if args.winsor.strip():
        lo, hi = args.winsor.split(",")
        winsor = (float(lo), float(hi))

    # Scatter: Year vs standardized unit price
    if "YearNum" in sub.columns and y_std in sub.columns:
        scatter_year_price(
            sub,
            xcol="YearNum",
            ycol=y_std,
            outdir=outdir,
            title=f"{args.grain}: standardized unit price scatter (drachma/artaba-eq)",
            filename=f"scatter_{args.grain.replace(' ', '_')}_std_unitprice_vs_year.png",
            ylog=ylog,
        )

    # Robust trend by YearBin + place comparison
    if "YearBin" in sub.columns and y_std in sub.columns:
        robust_trend_plot(
            sub,
            xcol="YearBin",
            ycol=y_std,
            outdir=outdir,
            min_n=args.min_n_bin,
            mad_z=mad_z,
            winsor=winsor,
            ylog=ylog,
            title=f"{args.grain}: ROBUST median+IQR (drachma/artaba-eq) by YearBin",
            fig_name=f"trend_{args.grain.replace(' ', '_')}_std_unitprice_by_yearbin_ROBUST.png",
            table_name=f"trend_{args.grain.replace(' ', '_')}_std_unitprice_by_yearbin_ROBUST.csv",
        )

        if "Place_Canon" in sub.columns:
            place_trend_plot_robust(
                sub,
                place_col="Place_Canon",
                xcol="YearBin",
                ycol=y_std,
                outdir=outdir,
                topn_places=args.topn_places,
                min_n=args.min_n_bin,
                mad_z=mad_z,
                winsor=winsor,
                ylog=ylog,
                title=f"{args.grain}: ROBUST median (drachma/artaba-eq) by place",
                fig_name=f"trend_{args.grain.replace(' ', '_')}_std_unitprice_by_place_ROBUST.png",
            )

    print("Done. If the trend still spikes, inspect the exported extremes CSV and adjust conversion tables.")


if __name__ == "__main__":
    main()
