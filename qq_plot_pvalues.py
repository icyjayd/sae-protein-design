import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

OUT = Path("outputs")
MONO_FILE = OUT / "latent_property_correlation_mono.csv"
REG_FILE = OUT / "latent_property_correlation_regular.csv"


def plot_qq(df, name="mono"):
    if "pval" not in df:
        print(f"No p-values found in {name} correlations.")
        return

    pvals = df["pval"].dropna().sort_values().values
    if len(pvals) < 2:
        print(f"Not enough p-values for Q-Q plot in {name} correlations.")
        return

    expected = np.linspace(0, 1, len(pvals), endpoint=False)[1:]  # skip 0
    pvals = pvals[:len(expected)]  # truncate to match expected size

    plt.figure(figsize=(6, 6))
    plt.plot(-np.log10(expected), -np.log10(pvals), 'o', markersize=3)
    plt.plot([0, max(-np.log10(expected))], [0, max(-np.log10(expected))], 'r--')
    plt.xlabel("Expected -log10(p)")
    plt.ylabel("Observed -log10(p)")
    plt.title(f"Q-Q Plot of {name.title()} Latent/Property p-values")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT / f"qqplot_latent_property_pvals_{name}.png")
    plt.close()

    print(f"[INFO] Saved Q-Q plot for {name} p-values to {OUT / f'qqplot_latent_property_pvals_{name}.png'}")


def main():
    if MONO_FILE.exists():
        df_mono = pd.read_csv(MONO_FILE)
        plot_qq(df_mono, "mono")

    if REG_FILE.exists():
        df_reg = pd.read_csv(REG_FILE)
        plot_qq(df_reg, "regular")


if __name__ == "__main__":
    main()
