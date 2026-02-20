"""
Paper4: Pro/Anti reporting — combined, high-aesthetic figure

This script:
- Loads df_joint_w_topics pickle for timestamps
- Builds merged_df by reading sectorwise hypothesis (top7) thresholded CSVs and merging timestamps
- Loads sectorwise pro/anti thresholded CSVs and merges with merged_df
- Produces ONE combined figure per run containing, for each sector: (1) stacked Pro/Anti/Neither with an inset of % contributions over time, (4) polarization, (6) improved net-contribution chart (diverging bars), (10) event-like jumps. Plots 2, 3, 5, 7, 8 (seasonality) are removed.
- Saves a concise summary CSV.

Outputs are saved under: paper4figures/gpt5_idea_explorations

Run:
  python paper4_pro_anti_reporting.py \
    --pickle ./data/df_joint_w_topics_annotation_paper1_0.pkl \
    --hypothesis-dir paper4data/sectorwise_roberta_classifications_thresholded \
    --pro-anti-dir paper4data/sectorwise_pro_anti_classifications_thresholded \
    --outdir paper4figures/gpt5_idea_explorations
"""

from __future__ import annotations

import os
import json
import argparse
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D


SECTORS: List[str] = ["transport", "housing", "food"]
# Global font scaling factor (applied across figures)
FONT_SCALING: float = 1.5
ALPHA = 1.0
SECTOR_DISPLAY = {
    "transport": "EVs",
    "housing": "solar",
    "food": "veganism",
}

# (Row background tint removed per request)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper4 Pro/Anti + Hypothesis reporting")
    p.add_argument("--pickle", dest="pickle_path", default="./data/df_joint_w_topics_annotation_paper1_0.pkl")
    p.add_argument("--hypothesis-dir", dest="hypothesis_dir", default="paper4data/sectorwise_roberta_classifications_thresholded")
    p.add_argument("--pro-anti-dir", dest="pro_anti_dir", default="paper4data/sectorwise_pro_anti_classifications_thresholded")
    p.add_argument("--outdir", dest="out_dir", default="paper4figures/gpt5_idea_explorations")
    return p.parse_args()


def ensure_dirs(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def load_df_joint_with_timestamps(pickle_path: str) -> pd.DataFrame:
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Pickle not found: {pickle_path}")
    df = pd.read_pickle(pickle_path)
    # Ensure created_utc_
    if "created_utc_" not in df.columns or df["created_utc_"].isna().any():
        if "created_utc" in df.columns:
            df["new_utc"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
            mask = df["created_utc_"].isna() if "created_utc_" in df.columns else pd.Series(True, index=df.index)
            df.loc[mask, "created_utc_"] = df.loc[mask, "new_utc"]
            if "new_utc" in df.columns:
                df = df.drop(columns=["new_utc"])
        else:
            raise ValueError("df_joint_w_topics must contain 'created_utc' or a valid 'created_utc_' column")
    # Normalize text column name to 'Document'
    if "Document" not in df.columns and "text" in df.columns:
        df = df.rename(columns={"text": "Document"})
    # Month period
    df["created_utc_"] = pd.to_datetime(df["created_utc_"], errors="coerce")
    df["month"] = df["created_utc_"].dt.to_period("M")
    return df


def load_hypothesis_thresholded(hypothesis_dir: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for sector in SECTORS:
        path = os.path.join(hypothesis_dir, f"{sector}_Roberta_classifications_thresholded.csv")
        if not os.path.exists(path):
            print(f"[WARN] Missing hypothesis file for {sector}: {path}")
            continue
        df = pd.read_csv(path)
        if "Document" not in df.columns and "text" in df.columns:
            df = df.rename(columns={"text": "Document"})
        df["sector"] = sector
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No hypothesis CSVs found in {hypothesis_dir}")
    return pd.concat(frames, ignore_index=True)


def load_pro_anti_thresholded(pro_anti_dir: str) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for sector in SECTORS:
        path = os.path.join(pro_anti_dir, f"{sector}_pro_anti_classifications_thresholded.csv")
        if not os.path.exists(path):
            print(f"[WARN] Missing pro/anti file for {sector}: {path}")
            continue
        df = pd.read_csv(path)
        if "Document" not in df.columns and "text" in df.columns:
            df = df.rename(columns={"text": "Document"})
        df["sector"] = sector
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No pro/anti CSVs found in {pro_anti_dir}")
    return pd.concat(frames, ignore_index=True)


def build_merged_df(df_joint: pd.DataFrame, hyp_df: pd.DataFrame) -> pd.DataFrame:
    # Merge timestamps into hypothesis results
    merged = pd.merge(
        hyp_df,
        df_joint[["Document", "created_utc_", "month"]],
        on="Document",
        how="left",
    )
    return merged


def merge_pro_anti(merged_df: pd.DataFrame, pro_anti_df: pd.DataFrame) -> pd.DataFrame:
    # Merge on sector + Document
    out = pd.merge(merged_df, pro_anti_df, on=["sector", "Document"], how="left")
    # Ensure time columns
    if "created_utc_" in out.columns:
        out["created_utc_"] = pd.to_datetime(out["created_utc_"], errors="coerce")
        out["month"] = out["created_utc_"].dt.to_period("M")
    return out


def save_fig_json(fig: plt.Figure, out_dir: str, stem: str, hypothesis: str, method: str, findings: Dict):
    png = os.path.join(out_dir, f"{stem}.png")
    pdf = os.path.join(out_dir, f"{stem}.pdf")
    jsn = os.path.join(out_dir, f"{stem}.json")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    with open(jsn, "w", encoding="utf-8") as f:
        json.dump({"hypothesis": hypothesis, "method": method, "findings": findings}, f, indent=2, default=str)


def rolling3(s: pd.Series) -> pd.Series:
    return s.rolling(window=3, center=True, min_periods=1).mean()


def slope(s: pd.Series) -> float:
    y = s.fillna(0).values
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)


def _remove_spines(ax: plt.Axes) -> None:
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3, width=0.8)


def _nature_like_style(scale: float = FONT_SCALING) -> None:
    sns.set_theme(context="talk", style="white")
    def s(v: float) -> int:
        return int(round(v * scale))
    plt.rcParams.update({
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": False,
        "axes.titlesize": s(11),
        "axes.labelsize": s(9),
        "xtick.labelsize": s(8),
        "ytick.labelsize": s(8),
        "legend.fontsize": s(10),
        "lines.linewidth": max(1.2, 2.0 * (scale / 1.0)),
    })


def year_tick_positions(month_periods: pd.Series, target_years: List[int]) -> tuple[list[int], list[str]]:
    """Return x positions and labels for the first month of each target year present.

    month_periods is a pandas Series of period[M].
    """
    try:
        years = month_periods.dt.year.values
    except Exception:
        # Fallback via string parsing
        years = month_periods.astype(str).str.slice(0, 4).astype(int).values
    ticks: list[int] = []
    labels: list[str] = []
    for y in target_years:
        idx = np.where(years == y)[0]
        if len(idx) > 0:
            ticks.append(int(idx[0]))
            labels.append(str(y))
    return ticks, labels


def monthly_counts_pro_anti(df_all: pd.DataFrame, sector: str) -> pd.DataFrame:
    sect = df_all[df_all["sector"] == sector].copy()
    for col in [f"{sector}_pro", f"{sector}_anti", f"{sector}_neither"]:
        if col not in sect.columns:
            sect[col] = 0
    out = (
        sect.groupby("month")[[f"{sector}_pro", f"{sector}_anti", f"{sector}_neither"]]
        .sum(min_count=1)
        .rename(columns={f"{sector}_pro": "pro", f"{sector}_anti": "anti", f"{sector}_neither": "neither"})
        .reset_index()
        .sort_values("month")
    )
    return out


def support_ratio(df_counts: pd.DataFrame) -> pd.Series:
    denom = (df_counts["pro"] + df_counts["anti"]).replace(0, np.nan)
    return (df_counts["pro"] / denom) * 100.0


def polarization(df_counts: pd.DataFrame) -> pd.Series:
    denom = (df_counts["pro"] + df_counts["anti"] + df_counts["neither"]).replace(0, np.nan)
    return ((df_counts["pro"] + df_counts["anti"]) / denom) * 100.0


def label_association(df_all: pd.DataFrame, sector: str, min_support: int = 20) -> pd.DataFrame:
    sect = df_all[df_all["sector"] == sector].copy()
    ignore = {f"{sector}_pro", f"{sector}_anti", f"{sector}_neither", f"{sector}_predicted_label"}
    hyp_cols = [c for c in sect.columns if c.startswith(f"{sector}_") and c not in ignore]
    rows = []
    for col in hyp_cols:
        mask = sect[col] == 1
        n = int(mask.sum())
        if n < min_support:
            continue
        pro = int(sect.loc[mask, f"{sector}_pro"].sum())
        anti = int(sect.loc[mask, f"{sector}_anti"].sum())
        tot = pro + anti
        pro_rate = pro / tot if tot > 0 else np.nan
        anti_rate = anti / tot if tot > 0 else np.nan
        rows.append({"label": col.replace(f"{sector}_", ""), "n": n, "pro": pro, "anti": anti, "pro_rate": pro_rate, "anti_rate": anti_rate})
    return pd.DataFrame(rows).sort_values("pro_rate", ascending=False)


def monthly_label_contrib(df_all: pd.DataFrame, sector: str) -> pd.DataFrame:
    sect = df_all[df_all["sector"] == sector].copy()
    ignore = {f"{sector}_pro", f"{sector}_anti", f"{sector}_neither", f"{sector}_predicted_label"}
    hyp_cols = [c for c in sect.columns if c.startswith(f"{sector}_") and c not in ignore]
    recs = []
    for month, mdf in sect.groupby("month"):
        for col in hyp_cols:
            mask = mdf[col] == 1
            pro = int(mdf.loc[mask, f"{sector}_pro"].sum())
            anti = int(mdf.loc[mask, f"{sector}_anti"].sum())
            neither = int(mdf.loc[mask, f"{sector}_neither"].sum())
            if pro + anti + neither == 0:
                continue
            recs.append({"month": month, "label": col.replace(f"{sector}_", ""), "pro": pro, "anti": anti, "neither": neither})
    
    if recs:
        df = pd.DataFrame(recs).sort_values(["month", "label"])
        # Ensure all required columns exist
        required_cols = ["pro", "anti", "neither", "label", "month"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0
        return df
    else:
        # Return empty DataFrame with correct columns
        return pd.DataFrame(columns=["month", "label", "pro", "anti", "neither"])


def print_and_save_table(df: pd.DataFrame, out_dir: str, stem: str, title: str | None = None) -> None:
    if title:
        print("\n" + title)
        print("-" * len(title))
    print(df.head(10))
    df.to_csv(os.path.join(out_dir, f"{stem}.csv"), index=False)
    print(f"Saved table: {os.path.join(out_dir, f'{stem}.csv')}")


def generate_all(df_all: pd.DataFrame, out_dir: str, font_scale: float = FONT_SCALING) -> None:
    _nature_like_style(font_scale)

    summary_rows: List[Dict] = []
    n_rows = len(SECTORS)
    n_cols = 2  # Col1 percentage share, Col2 contributions
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.5, 3.8 * n_rows), dpi=350, constrained_layout=True)
    try:
        fig.set_constrained_layout_pads(h_pad=0.15 * font_scale, w_pad=0.04, hspace=0.0, wspace=0.04)
    except Exception:
        pass
    if n_rows == 1:
        axes = np.array([axes])

    col_titles = [
        "% share over time",
        "motivators (%)",
    ]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, pad=8)

    # Updated color palette for Pro/Anti/Neither
    color_pro = "#43AA8B"      # green
    color_anti = "#F94144"     # red
    color_neither = "#FFFFBF"  # yellowish

    # Global legend handles using patches (Pro/Anti only; Neither not drawn in shares)
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color_pro, edgecolor='none', label="Pro"),
        Patch(facecolor=color_anti, edgecolor='none', label="Anti"),
    ]
    legend_labels = ["Pro", "Anti"]

    combined_summary = {"sectors": []}

    for i, sector in enumerate(SECTORS):
        counts = monthly_counts_pro_anti(df_all, sector)
        if counts.empty:
            print(f"[WARN] No counts for {sector}")
            continue

        # Save core table
        print_and_save_table(counts, out_dir, f"{sector}_monthly_counts", title=f"{sector.upper()} monthly pro/anti/neither")

        months = counts["month"].astype(str).tolist()
        x = np.arange(len(months))
        xticks, xlabels = year_tick_positions(counts["month"], [2012, 2015, 2018, 2021])

        sr = support_ratio(counts)
        pol = polarization(counts)
        summary_rows.append({
            "sector": sector,
            "pro_slope": slope(counts["pro"]),
            "anti_slope": slope(counts["anti"]),
            "net_support_slope": slope(counts["pro"] - counts["anti"]),
            "mean_support_ratio_pct": float(np.nanmean(sr)),
            "mean_polarization_pct": float(np.nanmean(pol)),
            "n_months": int(counts.shape[0]),
            "total_pro": int(counts["pro"].sum()),
            "total_anti": int(counts["anti"].sum()),
            "total_neither": int(counts["neither"].sum()),
        })

        # Col 1: Percentage share over time — stacked, show only Pro and Anti (denominator includes Neither)
        ax1 = axes[i, 0]
        denom_all = (counts["pro"] + counts["anti"] + counts["neither"]).replace(0, np.nan)
        pro_pct = (counts["pro"] / denom_all) * 100.0
        anti_pct = (counts["anti"] / denom_all) * 100.0
        neither_pct = (counts["neither"] / denom_all) * 100.0
        scale = (100.0 - neither_pct).fillna(0)
        # Rescale Pro and Anti so Pro% + Anti% = 100 - Neither%
        with np.errstate(divide='ignore', invalid='ignore'):
            pro_pct_rescaled = pro_pct * (scale / (pro_pct + anti_pct))
            anti_pct_rescaled = anti_pct * (scale / (pro_pct + anti_pct))
        ax1.stackplot(
            x,
            rolling3(pro_pct_rescaled),
            rolling3(anti_pct_rescaled),
            colors=[color_pro, color_anti],
            alpha=1.0,
            edgecolor="white",
            linewidth=0.5,
        )
        ax1.set_ylim(0, 100)
        ax1.set_yticks([0, 50, 100])
        if i == n_rows - 1:
            ax1.set_xticks(xticks); ax1.set_xticklabels(xlabels)
            ax1.set_xlabel("time")
        else:
            ax1.set_xticks(xticks); ax1.set_xticklabels(["" for _ in xlabels])
            ax1.set_xlabel("")
        _remove_spines(ax1)

        # Col 2: Diverging contributions — show Pro (right) and Anti (left) as percentages
        ax3 = axes[i, 1]
        contrib = monthly_label_contrib(df_all, sector)
        if not contrib.empty:
            # Ensure all required columns exist
            required_cols = ["pro", "anti", "neither", "label"]
            missing_cols = [col for col in required_cols if col not in contrib.columns]
            
            if not missing_cols:
                agg = contrib.groupby("label")[["pro", "anti", "neither"]].sum().reset_index()
                agg["total"] = agg["pro"] + agg["anti"] + agg["neither"]
                agg = agg[agg["total"] > 0]
                # Percent share among Pro+Anti+Neither for each label (new formula)
                agg["pro_pct"] = (agg["pro"] / agg["total"]) * 100.0
                agg["anti_pct"] = (agg["anti"] / agg["total"]) * 100.0
                agg = agg.sort_values("total", ascending=False).head(min(10, agg.shape[0]))

                labels = agg["label"].tolist()
                pro_vals = agg["pro_pct"].values.astype(float)
                anti_vals = -agg["anti_pct"].values.astype(float)  # negative for left
                y = np.arange(len(labels))

                pro_bars = ax3.barh(
                    y, pro_vals, color=color_pro, alpha=1.0,
                    edgecolor="#1f5e2a", linewidth=0.8, height=0.72, zorder=2
                )
                anti_bars = ax3.barh(
                    y, anti_vals, color=color_anti, alpha=1.0,
                    edgecolor="#7a231b", linewidth=0.8, height=0.72, zorder=2
                )
                ax3.set_yticks(y, labels)
                ax3.set_xlim(-102, 102)
                ax3.set_xticks([-100, -50, 0, 50, 100])
                ax3.axvline(0, color="#9a9a9a", lw=1)
                ax3.xaxis.grid(True, color="#e0e0e0", linestyle="-", linewidth=0.6, alpha=0.6)
                ax3.set_axisbelow(True)
                ax3.set_xlabel("Contributions (%)")
                label_fs = int(round(7.5 * font_scale))
                gap = 2.0
                for val, rect in zip(pro_vals, pro_bars):
                    ax3.text(rect.get_width() + gap, rect.get_y() + rect.get_height() / 2,
                             f"{val:.0f}%", va="center", ha="left", fontsize=label_fs, color="#222")
                for val, rect in zip(anti_vals, anti_bars):
                    # Shift annotation to the left by the height of the bar (negative direction)
                    ax3.text(rect.get_x() - gap - abs(rect.get_width()), rect.get_y() + rect.get_height() / 2,
                             f"{abs(val):.0f}%", va="center", ha="right", fontsize=label_fs, color="#222")
                _remove_spines(ax3)
            else:
                ax3.text(0.5, 0.5, "Missing required columns", ha="center", va="center", transform=ax3.transAxes)
                _remove_spines(ax3)
        else:
            ax3.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax3.transAxes)
            _remove_spines(ax3)

        combined_summary["sectors"].append({
            "sector": sector,
            "pro_slope": slope(counts["pro"]),
            "anti_slope": slope(counts["anti"]),
            "mean_polarization_pct": float(np.nanmean(pol)),
        })

        # Row label
        axes[i, 0].text(
            -0.15,
            0.5,
            SECTOR_DISPLAY.get(sector, sector.capitalize()),
            transform=axes[i, 0].transAxes,
            ha="right",
            va="center",
            fontsize=int(round(14 * font_scale)),
            rotation=90,
        )

        # No row background tint

    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.07))

    stem = "combined_pro_anti_reporting"
    fig.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=400, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    print_and_save_table(summary_df, out_dir, "summary_across_sectors", title="SUMMARY ACROSS SECTORS")
    with open(os.path.join(out_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
        json.dump({"findings": combined_summary}, f, indent=2, default=str)


def generate_event_jumps_figure(df_all: pd.DataFrame, out_dir: str, font_scale: float = FONT_SCALING) -> None:
    _nature_like_style(font_scale)

    color_pro = "#43AA8B"; color_anti = "#F94144"

    n_rows = len(SECTORS)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12.5, 2.8 * n_rows), dpi=350, constrained_layout=True)
    if n_rows == 1:
        axes = np.array([axes])

    summary = {"sectors": []}

    for i, sector in enumerate(SECTORS):
        counts = monthly_counts_pro_anti(df_all, sector)
        if counts.empty:
            print(f"[WARN] No counts for {sector}")
            continue
        months = counts["month"].astype(str).tolist()
        x = np.arange(len(months))
        xticks, xlabels = year_tick_positions(counts["month"], [2012, 2015, 2018, 2021])

        ch_pro = counts["pro"].diff(); ch_anti = counts["anti"].diff()
        z_pro = (ch_pro - ch_pro.mean()) / (ch_pro.std(ddof=1) or 1)
        z_anti = (ch_anti - ch_anti.mean()) / (ch_anti.std(ddof=1) or 1)
        spikes_pro = np.where(np.abs(z_pro.values) > 2)[0]
        spikes_anti = np.where(np.abs(z_anti.values) > 2)[0]

        ax = axes[i]
        rp = rolling3(counts["pro"]); ra = rolling3(counts["anti"]) 
        ax.plot(x, rp, label="Pro", color=color_pro)
        ax.plot(x, ra, label="Anti", color=color_anti)
        if len(spikes_pro) > 0:
            ax.scatter(spikes_pro, rp.iloc[spikes_pro], color=color_pro, s=12, zorder=3)
        if len(spikes_anti) > 0:
            ax.scatter(spikes_anti, ra.iloc[spikes_anti], color=color_anti, s=12, zorder=3)
        disp = SECTOR_DISPLAY.get(sector, sector.capitalize())
        ax.set_title(f"{disp}: Event-like jumps (|z|>2)")
        ax.set_xticks(xticks); ax.set_xticklabels(xlabels)
        ax.set_xlabel("")
        _remove_spines(ax)

        summary["sectors"].append({
            "sector": sector,
            "spike_months_pro": [str(m) for idx, m in enumerate(months) if idx in spikes_pro],
            "spike_months_anti": [str(m) for idx, m in enumerate(months) if idx in spikes_anti],
        })

    from matplotlib.patches import Patch
    fig.legend([
        Patch(facecolor=color_pro, edgecolor='none', label="Pro"),
        Patch(facecolor=color_anti, edgecolor='none', label="Anti"),
    ], ["Pro", "Anti"], loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.01))

    stem = "event_jumps"
    fig.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=400, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{stem}.pdf"), bbox_inches="tight")
    plt.close(fig)

    with open(os.path.join(out_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
        json.dump({"findings": summary}, f, indent=2, default=str)


def main() -> None:
    args = parse_args()
    ensure_dirs(args.out_dir)
    try:
        df_joint = load_df_joint_with_timestamps(args.pickle_path)
    except Exception as e:
        print(f"ERROR loading pickle: {e}")
        return
    try:
        hyp_df = load_hypothesis_thresholded(args.hypothesis_dir)
    except Exception as e:
        print(f"ERROR loading hypothesis CSVs: {e}")
        return
    merged = build_merged_df(df_joint, hyp_df)
    try:
        pro_anti_df = load_pro_anti_thresholded(args.pro_anti_dir)
    except Exception as e:
        print(f"ERROR loading pro/anti CSVs: {e}")
        return
    merged_all = merge_pro_anti(merged, pro_anti_df)
    generate_all(merged_all, args.out_dir, font_scale=FONT_SCALING)
    # Separate figure for event-like jumps
    try:
        generate_event_jumps_figure(merged_all, args.out_dir, font_scale=FONT_SCALING)
    except Exception as e:
        print(f"ERROR generating event jumps figure: {e}")
    print(f"\nAll outputs saved under: {args.out_dir}")


if __name__ == "__main__":
    main()


