# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Thermal-shift analysis pipeline for ovulation temperature data.
"""

import argparse
import logging
import pickle
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend – safe for CLI use
import matplotlib.pyplot as plt
from scipy.stats import shapiro, t, ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FECHA_REF: pd.Timestamp = pd.Timestamp("2025-01-01 00:00:00")

INTERVALO_MAX: list[int] = [21, 3]   # night maxima window: 21:00 → 03:59
INTERVALO_MIN: list[int] = [7, 15]   # day minima window:   07:00 → 15:59

DIAS_ANTES: list[int]   = [-3, -2, -1, 0]
DIAS_DESPUES: list[int] = [2, 3, 4, 5]

RANGE_VISUALIZ: list[float] = [36.25, 36.85]

FDR_METHOD: str = "fdr_bh"

# Number of top rows to print in the report
TOP_N: int = 10


# ===========================================================================
# I/O helpers
# ===========================================================================

def load_pickle(path: Path):
    """Deserialise and return the object stored at *path*."""
    logger.info("Loading pickle: %s", path)
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ===========================================================================
# 2.5 – Time-slot identification
# ===========================================================================

def compute_grand_summary(muestras: dict) -> pd.DataFrame:
    """
    Groups every hourly timestamp across all cycles, computes the mean and
    standard deviation of ``result`` at each timestamp, and returns the
    resulting summary DataFrame with columns:
    ``resultTimestamp``, ``mean``, ``std``.
    """
    all_series: list[pd.DataFrame] = []
    for datos in muestras.values():
        df = datos["serie"].copy()
        all_series.append(df[["resultTimestamp", "result"]])

    df_all = pd.concat(all_series, ignore_index=True)
    summary = (
        df_all.groupby("resultTimestamp")["result"]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["resultTimestamp", "mean", "std"]
    logger.info("Grand summary: %d hourly timestamps", len(summary))
    return summary


def plot_intervals(summary: pd.DataFrame, output_path: Path) -> None:
    """
    - Thin blue line for the full-day mean.
    - Blue ± SD band.
    - Dark-red overlay for the night maxima window (21:00–03:59).
    - Navy overlay for the day minima window (07:00–15:59).
    - Red dashed vertical line at ovulation (day 0).
    Saved at 600 dpi.
    """
    fecha_ref_day = FECHA_REF.floor("h")
    summary = summary.copy()
    summary["resultTimestamp"] = pd.to_datetime(summary["resultTimestamp"])
    summary["hour"] = summary["resultTimestamp"].dt.hour

    x_all = (summary["resultTimestamp"] - fecha_ref_day) / pd.Timedelta(days=1)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Full-day baseline
    ax.plot(x_all, summary["mean"], color="blue", linewidth=1, alpha=0.8,
            label="Mean Temperature")

    # ± SD band
    ax.fill_between(
        x_all,
        summary["mean"] - summary["std"],
        summary["mean"] + summary["std"],
        color="blue", alpha=0.2,
    )

    # Maxima window (night: hour >= 21 OR hour <= 3)
    mask_max = (
        (summary["hour"] >= INTERVALO_MAX[0]) |
        (summary["hour"] <= INTERVALO_MAX[1])
    )
    res_max = summary[mask_max].copy()
    res_max["delta"] = (
        res_max["resultTimestamp"].diff().dt.total_seconds().div(3600)
    )
    res_max["grupo"] = (res_max["delta"] > 1.5).cumsum()

    label_added = False
    for _, grp in res_max.groupby("grupo"):
        x_grp = (grp["resultTimestamp"] - fecha_ref_day) / pd.Timedelta(days=1)
        ax.plot(
            x_grp, grp["mean"],
            color="darkred", linewidth=2,
            label="Maxima (21:00–04:00)" if not label_added else "",
        )
        label_added = True

    # Minima window (day: 7 <= hour <= 15)
    mask_min = summary["hour"].between(INTERVALO_MIN[0], INTERVALO_MIN[1])
    res_min = summary[mask_min].copy()
    res_min["delta"] = (
        res_min["resultTimestamp"].diff().dt.total_seconds().div(3600)
    )
    res_min["grupo"] = (res_min["delta"] > 1.5).cumsum()

    label_added = False
    for _, grp in res_min.groupby("grupo"):
        x_grp = (grp["resultTimestamp"] - fecha_ref_day) / pd.Timedelta(days=1)
        ax.plot(
            x_grp, grp["mean"],
            color="navy", linewidth=2,
            label="Minima (07:00–15:00)" if not label_added else "",
        )
        label_added = True

    # Ovulation marker
    ax.axvline(x=0, color="red", linestyle="--", linewidth=1, alpha=0.7,
               label="Ovulation")

    ax.set_xticks(np.arange(-14, 15, 1))
    ax.set_ylim(RANGE_VISUALIZ)
    ax.set_xlabel("Days relative to Ovulation")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("Temperature Analysis: Maxima & Minima Intervals")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s", output_path)


# ===========================================================================
#Wilcoxon tests by daily means
# ===========================================================================

def _wilcoxon_row(
    dia_antes: int,
    dia_despues: int,
    valores_antes: np.ndarray,
    valores_despues: np.ndarray,
) -> dict:
    """
    Shared statistical block

    Parameters
    ----------
    dia_antes, dia_despues : int
        Relative day labels (for reporting only).
    valores_antes, valores_despues : np.ndarray
        Paired per-cycle temperature values.

    Returns
    -------
    dict with keys:
        dia_antes, dia_despues, n, wilcoxon_stat, wilcoxon_p,
        shapiro_p, ttest_p, mean_diff, std_err, ci_low, ci_high
    """
    diferencias = valores_despues - valores_antes
    n = len(diferencias)

    # Wilcoxon signed-rank test (one-sided: H1 before < after)
    stat_wil, p_wil = wilcoxon(valores_antes, valores_despues,
                                alternative="less")

    # Shapiro–Wilk normality of differences
    stat_shap, p_shap = shapiro(diferencias)

    # Paired t-test only when normality is not rejected
    if p_shap > 0.05:
        _, p_t = ttest_rel(valores_despues, valores_antes)
    else:
        p_t = np.nan

    # Mean difference and 95 % confidence interval
    mean_dif  = np.mean(diferencias)
    std_dif   = np.std(diferencias, ddof=1)
    alpha     = 0.05
    t_crit    = t.ppf(1 - alpha / 2, df=n - 1)
    std_err   = std_dif / np.sqrt(n)
    ci_low    = mean_dif - t_crit * std_err
    ci_high   = mean_dif + t_crit * std_err

    return {
        "dia_antes":    dia_antes,
        "dia_despues":  dia_despues,
        "n":            n,
        "wilcoxon_stat": stat_wil,
        "wilcoxon_p":   p_wil,
        "shapiro_p":    p_shap,
        "ttest_p":      p_t,
        "mean_diff":    mean_dif,
        "std_err":      std_err,
        "ci_low":       ci_low,
        "ci_high":      ci_high,
    }


def run_daily_tests(muestras: dict) -> pd.DataFrame:
    """

    For each (dia_antes, dia_despues) pair:
    1. Compute the daily mean temperature for each cycle by rounding relative
       days to the nearest integer.
    2. Pair the values across cycles that have both days.
    3. Run the statistical block (_wilcoxon_row).
    4. Apply Benjamini–Hochberg FDR correction across all comparisons.

    Returns a DataFrame with one row per comparison.
    """
    fecha_ref_day = FECHA_REF.floor("d")
    records: list[dict] = []

    for dia_antes in DIAS_ANTES:
        for dia_despues in DIAS_DESPUES:
            valores_antes: list[float]   = []
            valores_despues: list[float] = []

            for datos in muestras.values():
                df = datos["serie"].copy()
                df["dias_rel"]  = (
                    (df["resultTimestamp"] - fecha_ref_day) /
                    pd.Timedelta(days=1)
                )
                df["dia_entero"] = df["dias_rel"].round().astype(int)
                medias = df.groupby("dia_entero")["result"].mean()

                val_a = medias.get(dia_antes,  np.nan)
                val_d = medias.get(dia_despues, np.nan)
                if not (np.isnan(val_a) or np.isnan(val_d)):
                    valores_antes.append(val_a)
                    valores_despues.append(val_d)

            if len(valores_antes) < 2:
                logger.warning(
                    "Daily: skipping (%d, %d) – only %d valid pairs",
                    dia_antes, dia_despues, len(valores_antes),
                )
                continue

            row = _wilcoxon_row(
                dia_antes, dia_despues,
                np.array(valores_antes), np.array(valores_despues),
            )
            records.append(row)

    df_results = pd.DataFrame(records)
    df_results["adjusted_wilcoxon"] = multipletests(
        df_results["wilcoxon_p"], method=FDR_METHOD
    )[1]
    logger.info("Daily Wilcoxon tests: %d comparisons", len(df_results))
    return df_results


# ===========================================================================
# ===========================================================================

def _mean_for_time_slot(
    df: pd.DataFrame,
    inicio_hora: int,
    fin_hora: int,
    crosses_midnight: bool,
) -> pd.Series:
    """
    Extracts the subset of hourly readings that fall inside the
    [inicio_hora, fin_hora] window, handles midnight-crossing intervals,
    assigns each reading to its correct relative day, and returns the mean
    temperature per relative day as a Series indexed by integer day.

    For the maxima (night) interval (crosses_midnight=True) the post-midnight
    readings are assigned to the previous calendar day (e.g. 02:00 on day D
    belongs to the night of day D-1).  For the minima (day) interval
    (crosses_midnight=False) no adjustment is needed.
    """
    df = df.copy()
    fecha_ref_day = FECHA_REF.floor("d")
    df["dias_rel"] = (
        (df["resultTimestamp"] - fecha_ref_day) / pd.Timedelta(days=1)
    )
    df["hora"] = df["resultTimestamp"].dt.hour

    if crosses_midnight:
        filtro = (df["hora"] >= inicio_hora) | (df["hora"] <= fin_hora)
        df_f = df[filtro].copy()
        # Post-midnight hours belong to the previous night
        df_f.loc[df_f["hora"] <= inicio_hora, "dias_rel"] -= 1
        df_f["dia_entero"] = np.floor(df_f["dias_rel"]).astype(int) + 1
    else:
        filtro = (df["hora"] >= inicio_hora) & (df["hora"] <= fin_hora)
        df_f = df[filtro].copy()
        df_f["dia_entero"] = np.floor(df_f["dias_rel"]).astype(int)

    return df_f.groupby("dia_entero")["result"].mean()


def run_slot_tests(muestras: dict, slot_name: str,
                   inicio_hora: int, fin_hora: int) -> pd.DataFrame:
    """
    Identical statistical protocol to ``run_daily_tests`` but the per-cycle
    daily value comes from the mean over the specified hourly window.

    Parameters
    ----------
    muestras   : norm1 cycle dictionary
    slot_name  : descriptive label for logging ("maxima" or "minima")
    inicio_hora : start hour of the window (inclusive)
    fin_hora    : end hour of the window (inclusive); < inicio_hora → midnight crossing
    """
    crosses_midnight = fin_hora < inicio_hora
    records: list[dict] = []

    for dia_antes in DIAS_ANTES:
        for dia_despues in DIAS_DESPUES:
            valores_antes: list[float]   = []
            valores_despues: list[float] = []

            for datos in muestras.values():
                df = datos["serie"].copy()
                medias = _mean_for_time_slot(
                    df, inicio_hora, fin_hora, crosses_midnight
                )
                val_a = medias.get(dia_antes,  np.nan)
                val_d = medias.get(dia_despues, np.nan)
                if not (np.isnan(val_a) or np.isnan(val_d)):
                    valores_antes.append(val_a)
                    valores_despues.append(val_d)

            if len(valores_antes) < 2:
                logger.warning(
                    "%s slot: skipping (%d, %d) – only %d valid pairs",
                    slot_name, dia_antes, dia_despues, len(valores_antes),
                )
                continue

            row = _wilcoxon_row(
                dia_antes, dia_despues,
                np.array(valores_antes), np.array(valores_despues),
            )
            records.append(row)

    df_results = pd.DataFrame(records)
    df_results["adjusted_wilcoxon"] = multipletests(
        df_results["wilcoxon_p"], method=FDR_METHOD
    )[1]
    logger.info(
        "%s slot Wilcoxon tests: %d comparisons", slot_name, len(df_results)
    )
    return df_results


# ===========================================================================
# Reporting
# ===========================================================================

REPORT_COLS = [
    "dia_antes", "dia_despues", "n",
    "wilcoxon_p", "adjusted_wilcoxon",
    #"shapiro_p", "ttest_p",
    "mean_diff", "std_err", 
    #"ci_low", "ci_high",
]


def _section(title: str, df: pd.DataFrame, buf: StringIO) -> None:
    """Write a formatted section to *buf*."""
    buf.write("=" * 70 + "\n")
    buf.write(f"  {title}\n")
    buf.write("=" * 70 + "\n\n")

    buf.write(f"Total comparisons: {len(df)}\n\n")

    buf.write(f"Top {TOP_N} by Wilcoxon p-value:\n")
    top = df.sort_values("wilcoxon_p").head(TOP_N)[REPORT_COLS]
    buf.write(
        top.to_string(
            index=False,
            float_format=lambda v: f"{v:.4f}" if not np.isnan(v) else "NaN",
        )
    )
    buf.write("\n\n")

    buf.write("All results sorted by Wilcoxon p-value:\n")
    all_sorted = df.sort_values("wilcoxon_p")[REPORT_COLS]
    buf.write(
        all_sorted.to_string(
            index=False,
            float_format=lambda v: f"{v:.4f}" if not np.isnan(v) else "NaN",
        )
    )
    buf.write("\n\n")


def build_report(
    df_daily: pd.DataFrame,
    df_maxima: pd.DataFrame,
    df_minima: pd.DataFrame,
) -> str:
    """Compose the full text report"""
    buf = StringIO()
    buf.write("THERMAL-SHIFT STATISTICAL ANALYSIS REPORT\n")
    buf.write(f"Normalisation: norm1  |  FDR method: {FDR_METHOD}\n")
    buf.write(
        f"Day-before grid: {DIAS_ANTES}  |  "
        f"Day-after grid: {DIAS_DESPUES}\n\n"
    )

    _section(
        "2.6 – Daily means  "
        "(H1: mean temperature BEFORE ovulation < AFTER)",
        df_daily, buf,
    )
    _section(
        f"2.7 – Maxima window ({INTERVALO_MAX[0]}:00–{INTERVALO_MAX[1]:02d}:00)  "
        "(H1: before < after)",
        df_maxima, buf,
    )
    _section(
        f"2.7 – Minima window ({INTERVALO_MIN[0]}:00–{INTERVALO_MIN[1]:02d}:00)  "
        "(H1: before < after)",
        df_minima, buf,
    )
    return buf.getvalue()


# ===========================================================================
# End-to-end pipeline
# ===========================================================================

def run_analysis_pipeline(
    norm1_pkl: Path,
    output_dir: Path,
) -> None:
    """
    Execute the full analysis pipeline and persist all output artefacts.

    Artefacts written
    -----------------
    Intervals.png         – 2.5 combined intervals figure
    analysis_report.txt   – 2.6 + 2.7 full text report
    results_daily.csv     – 2.6 full result table
    results_maxima.csv    – 2.7 maxima window result table
    results_minima.csv    – 2.7 minima window result table

    Parameters
    ----------
    norm1_pkl  : Path to muestras_ovul_horas_norm1.pkl
    output_dir : Directory where all output files are written
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    muestras = load_pickle(norm1_pkl)
    logger.info("Loaded %d cycles from norm1 pickle", len(muestras))

    # ------------------------------------------------------------------
    # Time-slot identification and figure
    # ------------------------------------------------------------------
    logger.info("=== – Time-slot identification ===")
    summary = compute_grand_summary(muestras)
    plot_intervals(summary, output_dir / "Intervals.png")

    # ------------------------------------------------------------------
    # Daily Wilcoxon tests
    # ------------------------------------------------------------------
    logger.info("===  – Wilcoxon tests (daily means) ===")
    df_daily = run_daily_tests(muestras)
    df_daily.to_csv(output_dir / "results_daily.csv", index=False)
    logger.info("Saved: %s", output_dir / "results_daily.csv")

    # ------------------------------------------------------------------
    # Slot-based Wilcoxon tests
    # ------------------------------------------------------------------
    logger.info("===  – Wilcoxon tests (time-slot means) ===")
    df_maxima = run_slot_tests(
        muestras,
        slot_name="maxima",
        inicio_hora=INTERVALO_MAX[0],
        fin_hora=INTERVALO_MAX[1],
    )
    df_maxima.to_csv(output_dir / "results_maxima.csv", index=False)
    logger.info("Saved: %s", output_dir / "results_maxima.csv")

    df_minima = run_slot_tests(
        muestras,
        slot_name="minima",
        inicio_hora=INTERVALO_MIN[0],
        fin_hora=INTERVALO_MIN[1],
    )
    df_minima.to_csv(output_dir / "results_minima.csv", index=False)
    logger.info("Saved: %s", output_dir / "results_minima.csv")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    report_text = build_report(df_daily, df_maxima, df_minima)
    report_path = output_dir / "analysis_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Saved: %s", report_path)

    # Echo key results to stdout
    print("\n" + report_text)
    logger.info("Analysis pipeline complete. Artefacts in: %s", output_dir)


# ===========================================================================
# CLI entry point
# ===========================================================================

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Thermal-shift analysis pipeline (notebooks 2.5–2.7). "
            "Uses only the norm1-normalised hourly cycle data."
        )
    )
    parser.add_argument(
        "--norm1_pkl",
        type=str,
        default="../results/muestras_ovul_horas_norm1.pkl",
        help="Path to muestras_ovul_horas_norm1.pkl "
             "(default: ./muestras_ovul_horas_norm1.pkl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../results/",
        help="Directory for output artIfacts (default: current directory)",
    )
    args = parser.parse_args(argv)

    run_analysis_pipeline(
        norm1_pkl=Path(args.norm1_pkl),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
