import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("wheat_unit_prices_artaba_drachma_tier1.csv", encoding="utf-8-sig")

df = df.dropna(subset=["Bin25", "UnitPrice_Dr_per_Art"])
df = df[df["UnitPrice_Dr_per_Art"] > 0]

# Group by 25-year bins
summary = (
    df.groupby("Bin25")["UnitPrice_Dr_per_Art"]
    .agg(
        n="count",
        median="median",
        q25=lambda s: s.quantile(0.25),
        q75=lambda s: s.quantile(0.75),
    )
    .reset_index()
    .sort_values("Bin25")
)

plt.figure()

# Median line
plt.plot(summary["Bin25"], summary["median"], marker="o")

# IQR band (q25–q75)
plt.fill_between(summary["Bin25"], summary["q25"], summary["q75"], alpha=0.2)

plt.yscale("log")
plt.xlabel("25-year bin start year")
plt.ylabel("Median unit price (dr/artaba) [log scale]")
plt.title("Wheat price trend (Tier 1) — median and IQR")

plt.tight_layout()
plt.savefig("wheat_tier1_trend_bin25.png", dpi=300)
plt.show()
