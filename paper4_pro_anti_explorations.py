"""
Paper 4: Pro/Anti Time-series and Hypothesis-Driven Explorations

This script:
- Loads binarized pro/anti predictions per sector (thresholded CSVs)
- Loads binarized hypothesis (top7) predictions per sector (thresholded CSVs)
- Merges both with comment timestamps
- Produces time-series plots and at least 10 hypothesis-driven figures
- Saves each figure to paper4figures/gpt5_idea_explorations and emits a JSON sidecar
  describing the hypothesis, method, metrics, and key findings

Expected inputs (paths configurable via CLI args):
- paper4data/sectorwise_pro_anti_classifications_thresholded/{sector}_pro_anti_classifications_thresholded.csv
  Columns: 'sector','text' (or 'Document'), '{sector}_pro','{sector}_anti','{sector}_neither', '{sector}_predicted_label'
- paper4data/sectorwise_roberta_classifications_thresholded/{sector}_Roberta_classifications_thresholded.csv
  Columns: 'sector','text' (or 'Document'), '{sector}_{label_i}' for each hypothesis label
- paper4data/comments_metadata.csv (or user-specified):
  Columns: 'Document' (or 'text'), 'created_utc_' (datetime or str)

Sectors considered: transport, housing, food
"""

from __future__ import annotations

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------------
# Configuration dataclasses
# -------------------------------

@dataclass
class InputPaths:
    pro_anti_dir: str = "paper4data/sectorwise_pro_anti_classifications_thresholded"
    hypothesis_dir: str = "paper4data/sectorwise_roberta_classifications_thresholded"
    metadata_path: str = "paper4data/comments_metadata.csv"


@dataclass
class OutputPaths:
    figures_root: str = "paper4figures/gpt5_idea_explorations"


SECTORS: List[str] = ["transport", "housing", "food"]


# -------------------------------
# IO utilities
# -------------------------------

def ensure_output_dirs(paths: OutputPaths) -> None:
    os.makedirs(paths.figures_root, exist_ok=True)


def _normalize_text_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Document" in df.columns:
        return df.rename(columns={"Document": "text"})
    return df


def _normalize_time_column(df: pd.DataFrame) -> pd.DataFrame:
    time_col_candidates = ["created_utc_", "created_utc", "created"]
    for col in time_col_candidates:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            return df.rename(columns={col: "created_utc_"})
    raise ValueError(
        "Metadata file must contain a timestamp column among: created_utc_, created_utc, created"
    )


def load_metadata(metadata_path: str) -> pd.DataFrame:
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Metadata file not found at {metadata_path}. Provide a CSV with columns 'Document' or 'text' and 'created_utc_'."
        )
    meta = pd.read_csv(metadata_path)
    meta = _normalize_text_column(meta)
    meta = _normalize_time_column(meta)
    # Deduplicate on text/timestamp
    meta = meta.drop_duplicates(subset=["text", "created_utc_"])
    return meta


def load_pro_anti_thresholded(paths: InputPaths) -> pd.DataFrame:
    """Load thresholded pro/anti CSVs for all sectors and concatenate."""
    frames: List[pd.DataFrame] = []
    for sector in SECTORS:
        # Expected filename pattern
        csv_path = os.path.join(
            paths.pro_anti_dir, f"{sector}_pro_anti_classifications_thresholded.csv"
        )
        if not os.path.exists(csv_path):
            print(f"[WARN] Pro/anti file missing for {sector}: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df = _normalize_text_column(df)
        df["sector"] = sector
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No pro/anti thresholded CSVs found in {paths.pro_anti_dir} for sectors: {SECTORS}"
        )
    return pd.concat(frames, ignore_index=True)


def load_hypothesis_thresholded(paths: InputPaths) -> pd.DataFrame:
    """Load thresholded hypothesis (top7) CSVs for all sectors and concatenate."""
    frames: List[pd.DataFrame] = []
    for sector in SECTORS:
        csv_path = os.path.join(
            paths.hypothesis_dir, f"{sector}_Roberta_classifications_thresholded.csv"
        )
        if not os.path.exists(csv_path):
            print(f"[WARN] Hypothesis file missing for {sector}: {csv_path}")
            continue
        df = pd.read_csv(csv_path)
        df = _normalize_text_column(df)
        df["sector"] = sector
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No hypothesis thresholded CSVs found in {paths.hypothesis_dir} for sectors: {SECTORS}"
        )
    return pd.concat(frames, ignore_index=True)


def merge_all(pro_anti_df: pd.DataFrame, hyp_df: pd.DataFrame, meta_df: pd.DataFrame) -> pd.DataFrame:
    """Merge pro/anti, hypothesis, and metadata on 'text' and add month period."""
    # Left-join timestamps to both, then outer join on shared keys
    pro_anti_df = pd.merge(
        pro_anti_df, meta_df[["text", "created_utc_"]], on="text", how="left"
    )
    hyp_df = pd.merge(hyp_df, meta_df[["text", "created_utc_"]], on="text", how="left")

    # Outer merge pro/anti with hypothesis on sector+text+created_utc_
    merged = pd.merge(
        pro_anti_df,
        hyp_df,
        on=["sector", "text", "created_utc_"],
        how="outer",
        suffixes=("_proanti", "_hypo"),
    )

    # Normalize time and month
    merged["created_utc_"] = pd.to_datetime(merged["created_utc_"], errors="coerce")
    merged["month"] = merged["created_utc_"].dt.to_period("M")

    return merged


# -------------------------------
# Plot helpers
# -------------------------------

def _save_figure_with_json(
    fig: plt.Figure,
    out_dir: str,
    filename_stem: str,
    hypothesis: str,
    method: str,
    findings: Dict,
    extra: Optional[Dict] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    png_path = os.path.join(out_dir, f"{filename_stem}.png")
    pdf_path = os.path.join(out_dir, f"{filename_stem}.pdf")
    json_path = os.path.join(out_dir, f"{filename_stem}.json")

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "hypothesis": hypothesis,
        "method": method,
        "findings": findings,
    }
    if extra:
        payload.update(extra)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Period,)):
        return str(o)
    return str(o)


def _month_index(series: pd.Series) -> List[str]:
    return [str(p) for p in series]


def _rolling(series: pd.Series, window: int = 3) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


# -------------------------------
# Metric builders
# -------------------------------

def _sector_columns(df: pd.DataFrame, sector: str, suffix: str) -> List[str]:
    return [c for c in df.columns if c.startswith(f"{sector}_") and c.endswith(suffix)]


def compute_monthly_counts_pro_anti(merged: pd.DataFrame, sector: str) -> pd.DataFrame:
    """Return a DataFrame with columns month, pro_count, anti_count, neither_count for a sector."""
    sector_df = merged[merged["sector"] == sector].copy()
    # Identify columns for pro/anti/neither
    pro_col = f"{sector}_pro"
    anti_col = f"{sector}_anti"
    neither_col = f"{sector}_neither"
    for col in [pro_col, anti_col, neither_col]:
        if col not in sector_df.columns:
            sector_df[col] = 0

    counts = (
        sector_df.groupby("month")[[pro_col, anti_col, neither_col]]
        .sum(min_count=1)
        .rename(columns={pro_col: "pro", anti_col: "anti", neither_col: "neither"})
        .reset_index()
        .sort_values("month")
    )
    return counts


def compute_label_association(merged: pd.DataFrame, sector: str, min_support: int = 20) -> pd.DataFrame:
    """For each hypothesis label, compute pro and anti association rates within the sector."""
    sector_df = merged[merged["sector"] == sector].copy()
    # Hypothesis columns are those starting with f"{sector}_" and not pro/anti/neither
    ignore = {f"{sector}_pro", f"{sector}_anti", f"{sector}_neither", f"{sector}_predicted_label"}
    hyp_cols = [c for c in sector_df.columns if c.startswith(f"{sector}_") and c not in ignore]

    records = []
    for col in hyp_cols:
        # Comments where this hypothesis is present
        mask = sector_df[col] == 1
        n = int(mask.sum())
        if n < min_support:
            continue
        pro = int(sector_df.loc[mask, f"{sector}_pro"].sum())
        anti = int(sector_df.loc[mask, f"{sector}_anti"].sum())
        total = pro + anti
        pro_rate = pro / total if total > 0 else np.nan
        anti_rate = anti / total if total > 0 else np.nan
        records.append({
            "label": col.replace(f"{sector}_", ""),
            "n": n,
            "pro": pro,
            "anti": anti,
            "pro_rate": pro_rate,
            "anti_rate": anti_rate,
        })
    return pd.DataFrame(records).sort_values("pro_rate", ascending=False)


def compute_polarization_index(counts: pd.DataFrame) -> pd.DataFrame:
    """Polarization index defined as (pro + anti) / (pro + anti + neither)."""
    df = counts.copy()
    denom = df[["pro", "anti", "neither"]].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["polarization"] = (df["pro"] + df["anti"]) / denom.replace(0, np.nan)
    return df


def compute_support_ratio(counts: pd.DataFrame) -> pd.DataFrame:
    """Support ratio defined as pro / (pro + anti)."""
    df = counts.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["support_ratio"] = df["pro"] / (df["pro"] + df["anti"]).replace(0, np.nan)
    return df


def compute_net_support(counts: pd.DataFrame) -> pd.DataFrame:
    df = counts.copy()
    df["net_support"] = df["pro"] - df["anti"]
    return df


def compute_monthly_label_contributions(merged: pd.DataFrame, sector: str) -> pd.DataFrame:
    """For each hypothesis label and month, compute contributions to pro and anti counts."""
    sector_df = merged[merged["sector"] == sector].copy()
    ignore = {f"{sector}_pro", f"{sector}_anti", f"{sector}_neither", f"{sector}_predicted_label"}
    hyp_cols = [c for c in sector_df.columns if c.startswith(f"{sector}_") and c not in ignore]

    # Create records by month and label
    records = []
    for month, month_df in sector_df.groupby("month"):
        for col in hyp_cols:
            mask = month_df[col] == 1
            pro = int(month_df.loc[mask, f"{sector}_pro"].sum())
            anti = int(month_df.loc[mask, f"{sector}_anti"].sum())
            if pro + anti == 0:
                continue
            records.append({
                "month": month,
                "label": col.replace(f"{sector}_", ""),
                "pro": pro,
                "anti": anti,
            })
    return pd.DataFrame(records).sort_values(["month", "label"]) if records else pd.DataFrame()


def compute_month_of_year_profile(counts: pd.DataFrame) -> pd.DataFrame:
    df = counts.copy()
    df["moy"] = df["month"].astype(str).str[-2:].astype(int)
    return (
        df.groupby("moy")["pro", "anti", "neither"]  # type: ignore[index]
        .mean()
        .reset_index()
        .sort_values("moy")
    )


def _slope(series: pd.Series) -> float:
    if len(series) < 2:
        return 0.0
    x = np.arange(len(series))
    y = series.values
    # Linear regression slope
    denom = (x - x.mean()) ** 2
    denom_sum = denom.sum()
    if denom_sum == 0:
        return 0.0
    slope = ((x - x.mean()) * (y - y.mean())).sum() / denom_sum
    return float(slope)


def _top_n_changes(series: pd.Series, n: int = 5) -> List[Tuple[int, float]]:
    diffs = series.diff().fillna(0)
    order = np.argsort(-np.abs(diffs.values))[:n]
    return [(int(idx), float(diffs.iloc[idx])) for idx in order]


# -------------------------------
# Plot generators (10+ hypotheses)
# -------------------------------

def plot1_stacked_pro_anti(counts: pd.DataFrame, sector: str, out: OutputPaths) -> None:
    hyp = (
        f"H1: Support and opposition volumes for {sector} change over time; visualized via stacked area."
    )
    method = (
        "Sum monthly pro/anti/neither counts per sector; 3-month centered rolling average; stacked area."
    )
    months = _month_index(counts["month"])  # type: ignore[arg-type]
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    ax.stackplot(
        np.arange(len(months)),
        _rolling(counts["pro"]),
        _rolling(counts["anti"]),
        _rolling(counts["neither"]),
        labels=["Pro", "Anti", "Neither"],
        colors=["#2ca02c", "#d62728", "#7f7f7f"],
        alpha=0.95,
        linewidth=0.8,
        edgecolor="white",
    )
    ax.set_title(f"{sector.capitalize()}: Pro/Anti/Niether (3-mo RA)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Monthly counts (rolling)")
    ax.legend(loc="upper left")
    ax.set_xlim(0, len(months) - 1)
    findings = {
        "pro_slope": _slope(counts["pro"]),
        "anti_slope": _slope(counts["anti"]),
        "neither_slope": _slope(counts["neither"]),
        "periods": len(months),
    }
    _save_figure_with_json(
        fig,
        out.figures_root,
        f"{sector}_01_stacked_pro_anti",
        hyp,
        method,
        findings,
    )


def plot2_net_support(counts: pd.DataFrame, sector: str, out: OutputPaths) -> None:
    hyp = f"H2: Net support (pro - anti) for {sector} is changing over time."
    method = "Compute monthly (pro - anti) and plot with 3-month RA; annotate top jumps."
    df = compute_net_support(counts)
    months = _month_index(df["month"])  # type: ignore[arg-type]
    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
    y = _rolling(df["net_support"])
    ax.plot(np.arange(len(months)), y, color="#1f77b4", lw=2)
    ax.axhline(0, color="#999999", ls="--", lw=1)
    ax.set_title(f"{sector.capitalize()}: Net Support (Pro - Anti)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Net counts (rolling)")
    # Annotate top 3 changes
    top_changes = _top_n_changes(df["net_support"], n=3)
    for idx, delta in top_changes:
        ax.scatter(idx, y.iloc[idx], color="#ff7f0e")
        ax.text(idx, y.iloc[idx], f"Δ={delta:.0f}", fontsize=8)
    findings = {
        "net_support_slope": _slope(df["net_support"]),
        "top_changes": top_changes,
    }
    _save_figure_with_json(
        fig, out.figures_root, f"{sector}_02_net_support", hyp, method, findings
    )


def plot3_support_ratio(counts: pd.DataFrame, sector: str, out: OutputPaths) -> None:
    hyp = f"H3: Relative support fraction pro/(pro+anti) for {sector} shows trend/seasonality."
    method = "Compute monthly ratio and 3-month RA; line plot."
    df = compute_support_ratio(counts)
    months = _month_index(df["month"])  # type: ignore[arg-type]
    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
    y = _rolling(df["support_ratio"]) * 100
    ax.plot(np.arange(len(months)), y, color="#2ca02c", lw=2)
    ax.set_ylim(0, 100)
    ax.set_title(f"{sector.capitalize()}: Support Ratio (Pro / (Pro+Anti))")
    ax.set_xlabel("Month")
    ax.set_ylabel("Percent (%)")
    findings = {
        "mean_support_ratio_pct": float(np.nanmean(df["support_ratio"]) * 100.0),
        "support_ratio_slope": _slope(df["support_ratio"].fillna(0)),
    }
    _save_figure_with_json(
        fig, out.figures_root, f"{sector}_03_support_ratio", hyp, method, findings
    )


def plot4_polarization(counts: pd.DataFrame, sector: str, out: OutputPaths) -> None:
    hyp = f"H4: Polarization (Pro+Anti share) for {sector} varies over time."
    method = "Compute (pro+anti)/(pro+anti+neither); 3-month RA; line plot."
    df = compute_polarization_index(counts)
    months = _month_index(df["month"])  # type: ignore[arg-type]
    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
    y = _rolling(df["polarization"]) * 100
    ax.plot(np.arange(len(months)), y, color="#9467bd", lw=2)
    ax.set_ylim(0, 100)
    ax.set_title(f"{sector.capitalize()}: Polarization Index")
    ax.set_xlabel("Month")
    ax.set_ylabel("Percent (%)")
    findings = {
        "mean_polarization_pct": float(np.nanmean(df["polarization"]) * 100.0),
        "polarization_slope": _slope(df["polarization"].fillna(0)),
    }
    _save_figure_with_json(
        fig, out.figures_root, f"{sector}_04_polarization", hyp, method, findings
    )


def plot5_label_association(merged: pd.DataFrame, sector: str, out: OutputPaths) -> None:
    hyp = (
        f"H5: Certain hypothesis sub-labels are more associated with Pro (or Anti) in {sector}."
    )
    method = (
        "For labels with >=20 occurrences, compute pro_rate and anti_rate among comments with the label; bar chart."
    )
    assoc = compute_label_association(merged, sector, min_support=20)
    if assoc.empty:
        return
    assoc = assoc.head(12)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    ax.barh(assoc["label"], assoc["pro_rate"] * 100, color="#2ca02c", alpha=0.85, label="Pro %")
    ax.barh(assoc["label"], assoc["anti_rate"] * 100, color="#d62728", alpha=0.35, label="Anti %")
    ax.set_xlabel("Percent (%)")
    ax.set_title(f"{sector.capitalize()}: Pro/Anti Association by Label (Top by Pro%)")
    ax.legend()
    findings = {
        "top_labels_by_pro_rate": assoc[["label", "pro_rate", "n"]].to_dict(orient="records"),
    }
    _save_figure_with_json(
        fig, out.figures_root, f"{sector}_05_label_association", hyp, method, findings
    )


def plot6_top_contributors_growth(merged: pd.DataFrame, sector: str, out: OutputPaths) -> None:
    hyp = f"H6: Identify labels contributing most to monthly growth in Pro and Anti for {sector}."
    method = (
        "Compute per-month Pro/Anti counts per label; diff across months; show top +/- contributors as bars."
    )
    contrib = compute_monthly_label_contributions(merged, sector)
    if contrib.empty:
        return
    # Aggregate diffs across months to find overall contributors
    contrib_sorted = (
        contrib.groupby("label")["pro", "anti"].sum()  # type: ignore[index]
        .assign(net=lambda d: d["pro"] - d["anti"])  # type: ignore[index]
        .sort_values("net", ascending=False)
    )
    top_pos = contrib_sorted.head(7).reset_index()
    top_neg = contrib_sorted.tail(7).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200, sharey=True)
    axes[0].barh(top_pos["label"], top_pos["net"], color="#2ca02c")
    axes[0].set_title("Top Net Pro Contributors")
    axes[1].barh(top_neg["label"], top_neg["net"], color="#d62728")
    axes[1].set_title("Top Net Anti Contributors")
    for ax in axes:
        ax.set_xlabel("Net (Pro - Anti)")
    fig.suptitle(f"{sector.capitalize()}: Label Contributions to Net Support")
    findings = {
        "top_pro_contributors": top_pos.to_dict(orient="records"),
        "top_anti_contributors": top_neg.to_dict(orient="records"),
    }
    _save_figure_with_json(
        fig,
        out.figures_root,
        f"{sector}_06_label_contributors",
        hyp,
        method,
        findings,
    )


def plot7_neither_trend(counts: pd.DataFrame, sector: str, out: OutputPaths) -> None:
    hyp = f"H7: The share of 'Neither' evolves over time in {sector}, reflecting undecided discourse."
    method = "Plot monthly neither / (pro+anti+neither) with 3-month RA."
    denom = (counts[["pro", "anti", "neither"]].sum(axis=1)).replace(0, np.nan)
    ratio = (counts["neither"] / denom) * 100
    months = _month_index(counts["month"])  # type: ignore[arg-type]
    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
    ax.plot(np.arange(len(months)), _rolling(ratio), color="#7f7f7f", lw=2)
    ax.set_ylim(0, 100)
    ax.set_title(f"{sector.capitalize()}: Neither Share")
    ax.set_xlabel("Month")
    ax.set_ylabel("Percent (%)")
    findings = {"mean_neither_pct": float(np.nanmean(ratio))}
    _save_figure_with_json(
        fig, out.figures_root, f"{sector}_07_neither_share", hyp, method, findings
    )


def plot8_seasonality(counts: pd.DataFrame, sector: str, out: OutputPaths) -> None:
    hyp = f"H8: There is month-of-year seasonality in Pro/Anti/Neither for {sector}."
    method = "Group by month-of-year; average counts; line plot."
    moy = compute_month_of_year_profile(counts)
    if moy.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
    ax.plot(moy["moy"], moy["pro"], label="Pro", color="#2ca02c")
    ax.plot(moy["moy"], moy["anti"], label="Anti", color="#d62728")
    ax.plot(moy["moy"], moy["neither"], label="Neither", color="#7f7f7f")
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month of Year")
    ax.set_ylabel("Avg monthly counts")
    ax.legend()
    ax.set_title(f"{sector.capitalize()}: Seasonality")
    findings = {
        "peak_month_pro": int(moy.loc[moy["pro"].idxmax(), "moy"]),
        "peak_month_anti": int(moy.loc[moy["anti"].idxmax(), "moy"]),
    }
    _save_figure_with_json(
        fig, out.figures_root, f"{sector}_08_seasonality", hyp, method, findings
    )


def plot9_label_orientation_heatmap(merged: pd.DataFrame, sector: str, out: OutputPaths) -> None:
    hyp = f"H9: Hypothesis labels differ in their Pro vs Anti intensities in {sector}."
    method = (
        "Aggregate across time: compute total Pro and Anti per label; show normalized heatmap (Pro, Anti)."
    )
    contrib = compute_monthly_label_contributions(merged, sector)
    if contrib.empty:
        return
    totals = contrib.groupby("label")["pro", "anti"].sum().reset_index()  # type: ignore[index]
    totals[["pro", "anti"]] = totals[["pro", "anti"]].div(
        totals[["pro", "anti"]].sum(axis=1).replace(0, np.nan), axis=0
    )
    data = totals.set_index("label")["pro", "anti"].T  # type: ignore[index]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    sns.heatmap(data, cmap="RdYlGn", annot=False, cbar_kws={"label": "Share"}, ax=ax)
    ax.set_title(f"{sector.capitalize()}: Pro/Anti Share by Label")
    findings = {
        "most_pro_label": str(totals.loc[totals["pro"].idxmax(), "label"]),
        "most_anti_label": str(totals.loc[totals["anti"].idxmax(), "label"]),
    }
    _save_figure_with_json(
        fig, out.figures_root, f"{sector}_09_label_orientation_heatmap", hyp, method, findings
    )


def plot10_event_jumps(counts: pd.DataFrame, sector: str, out: OutputPaths) -> None:
    hyp = f"H10: There are spike months in Pro/Anti for {sector} indicating event-driven discourse."
    method = (
        "Compute z-scores of monthly changes; mark months with |z|>2 in Pro and Anti lines with rolling avg."
    )
    df = counts.copy()
    changes_pro = df["pro"].diff()
    changes_anti = df["anti"].diff()
    z_pro = (changes_pro - changes_pro.mean()) / (changes_pro.std(ddof=1) or 1)
    z_anti = (changes_anti - changes_anti.mean()) / (changes_anti.std(ddof=1) or 1)
    months = _month_index(df["month"])  # type: ignore[arg-type]
    fig, ax = plt.subplots(figsize=(10, 4), dpi=200)
    ax.plot(np.arange(len(months)), _rolling(df["pro"]), label="Pro", color="#2ca02c")
    ax.plot(np.arange(len(months)), _rolling(df["anti"]), label="Anti", color="#d62728")
    spikes_pro = np.where(np.abs(z_pro) > 2)[0]
    spikes_anti = np.where(np.abs(z_anti) > 2)[0]
    ax.scatter(spikes_pro, _rolling(df["pro"]).iloc[spikes_pro], color="#2ca02c", s=20)
    ax.scatter(spikes_anti, _rolling(df["anti"]).iloc[spikes_anti], color="#d62728", s=20)
    ax.set_title(f"{sector.capitalize()}: Event-like Jumps (|z|>2)")
    ax.legend()
    findings = {
        "spike_month_indices_pro": [int(i) for i in spikes_pro],
        "spike_month_indices_anti": [int(i) for i in spikes_anti],
    }
    _save_figure_with_json(
        fig, out.figures_root, f"{sector}_10_event_jumps", hyp, method, findings
    )


# -------------------------------
# Driver
# -------------------------------

def generate_all_plots(paths_in: InputPaths, paths_out: OutputPaths) -> None:
    """Load data, merge, and generate all hypothesis-driven plots per sector."""
    ensure_output_dirs(paths_out)

    # Load
    meta = load_metadata(paths_in.metadata_path)
    pro_anti = load_pro_anti_thresholded(paths_in)
    hypo = load_hypothesis_thresholded(paths_in)
    merged = merge_all(pro_anti, hypo, meta)

    # For each sector, build metrics and plots
    for sector in SECTORS:
        sector_counts = compute_monthly_counts_pro_anti(merged, sector)
        if sector_counts.empty:
            print(f"[WARN] No time-series counts for sector {sector}.")
            continue
        # 10 plots
        plot1_stacked_pro_anti(sector_counts, sector, paths_out)
        plot2_net_support(sector_counts, sector, paths_out)
        plot3_support_ratio(sector_counts, sector, paths_out)
        plot4_polarization(sector_counts, sector, paths_out)
        plot5_label_association(merged, sector, paths_out)
        plot6_top_contributors_growth(merged, sector, paths_out)
        plot7_neither_trend(sector_counts, sector, paths_out)
        plot8_seasonality(sector_counts, sector, paths_out)
        plot9_label_orientation_heatmap(merged, sector, paths_out)
        plot10_event_jumps(sector_counts, sector, paths_out)

    print(f"All figures saved under: {paths_out.figures_root}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pro/Anti time-series and hypothesis explorations")
    p.add_argument("--pro-anti-dir", dest="pro_anti_dir", default=InputPaths.pro_anti_dir)
    p.add_argument("--hypothesis-dir", dest="hypothesis_dir", default=InputPaths.hypothesis_dir)
    p.add_argument("--metadata", dest="metadata_path", default=InputPaths.metadata_path)
    p.add_argument("--outdir", dest="figures_root", default=OutputPaths.figures_root)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    inp = InputPaths(
        pro_anti_dir=args.pro_anti_dir,
        hypothesis_dir=args.hypothesis_dir,
        metadata_path=args.metadata_path,
    )
    outp = OutputPaths(figures_root=args.figures_root)
    generate_all_plots(inp, outp)


if __name__ == "__main__":
    main()


