"""Feature quality analysis for processed PathSnap candidate data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def descriptive_statistics(X: pd.DataFrame) -> pd.DataFrame:
    numeric = X.select_dtypes(include=np.number)
    rows = []
    for name, values in numeric.items():
        rows.append({
            "feature": name,
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "missing_pct": float(values.isna().mean() * 100),
            "unique_count": int(values.nunique(dropna=True)),
            "variance": float(values.var()),
        })
    return pd.DataFrame(rows)


def correlation_tables(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric = X.select_dtypes(include=np.number)
    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")
    pairs = []
    for i, left in enumerate(numeric.columns):
        for right in numeric.columns[i + 1:]:
            pairs.append({"feature_1": left, "feature_2": right,
                          "pearson": pearson.loc[left, right],
                          "spearman": spearman.loc[left, right]})
    return pearson, spearman, pd.DataFrame(pairs)


def save_feature_plots(X: pd.DataFrame, y: pd.Series, figures_dir: str | Path) -> None:
    figures = Path(figures_dir)
    figures.mkdir(parents=True, exist_ok=True)
    numeric = X.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(max(10, len(numeric.columns)), 8))
    numeric.hist(ax=ax, bins=40)
    fig.tight_layout()
    fig.savefig(figures / "feature_histograms.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(12, len(numeric.columns) * 1.1), 7))
    numeric.plot(kind="box", ax=ax, rot=70)
    ax.set_ylabel("Value")
    fig.tight_layout()
    fig.savefig(figures / "feature_boxplots.png", dpi=150)
    plt.close(fig)

    for name in numeric.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        for label in sorted(pd.unique(y)):
            numeric.loc[y == label, name].plot(kind="hist", bins=40, alpha=0.45,
                                               density=True, ax=ax, label=f"class_{label}")
        ax.set_title(name)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figures / f"distribution_{name}.png", dpi=120)
        plt.close(fig)

    corr = numeric.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=70, ha="right")
    ax.set_yticks(range(len(corr.columns)), corr.columns)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(figures / "feature_correlation_heatmap.png", dpi=150)
    plt.close(fig)
