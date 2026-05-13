# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Classification dataset construction for ovulation temperature data.

Outputs
-------
  MaxRange.csv          – Feature matrix (57 rows × 140 columns + index)
  construction_report.txt – Text report: filter stats, shape, removed IDs
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

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
#
# ---------------------------------------------------------------------------
FECHA_REF: pd.Timestamp = pd.Timestamp("2025-01-01 00:00:00").floor("D")

# Day window relative to ovulation (both endpoints inclusive)
INTERVALO_DIAS: list[int] = [-14, 5]

# Nightly maxima hour window (crosses midnight: 21:00 → 03:59)
INTERVALO_HORAS: list[int] = [21, 3]

# Number of feature columns kept (notebook's .iloc[:, :140])
N_FEATURE_COLS: int = 140

# IQR fence multiplier for std-dev outlier filter
IQR_FENCE: float = 1.5


# ===========================================================================
# I/O helpers
# ===========================================================================

def load_pickle(path: Path):
    """Deserialise and return the object stored at *path*."""
    logger.info("Loading pickle: %s", path)
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ===========================================================================
# Step 1 – High-variance filter
# ===========================================================================

def filter_high_variance(muestras: dict) -> tuple[dict, pd.DataFrame]:
    """

    Returns
    -------
    muestras_filtered : dict
        Subset of *muestras* with high-variance cycles removed.
    df_std : pd.DataFrame
        Full std table with columns ``serie_id``, ``std``,
        ``q1``, ``q3``, ``iqr``, ``lim_sup``, ``kept``.
    """
    records = [
        (sid, datos["serie"]["result"].std())
        for sid, datos in muestras.items()
    ]
    df_std = pd.DataFrame(records, columns=["serie_id", "std"])

    q1     = df_std["std"].quantile(0.25)
    q3     = df_std["std"].quantile(0.75)
    iqr    = q3 - q1
    lim_sup = q3 + IQR_FENCE * iqr

    df_std["q1"]      = q1
    df_std["q3"]      = q3
    df_std["iqr"]     = iqr
    df_std["lim_sup"] = lim_sup
    df_std["kept"]    = df_std["std"] <= lim_sup

    series_ok = df_std.loc[df_std["kept"], "serie_id"].tolist()
    muestras_filtered = {sid: muestras[sid] for sid in series_ok}

    removed = set(muestras.keys()) - set(series_ok)
    logger.info(
        "IQR std filter: kept %d / %d cycles (removed: %s)",
        len(muestras_filtered), len(muestras), sorted(removed),
    )
    return muestras_filtered, df_std


# ===========================================================================
# Step 2 – Time-window extraction
# ===========================================================================

def _filter_hours(df: pd.DataFrame, h0: int, h1: int) -> pd.DataFrame:
    """

    Returns rows whose hour falls inside [h0, h1], handling the midnight-
    crossing case (h0 > h1) exactly as the original notebook.
    """
    horas = df["resultTimestamp"].dt.hour
    if h0 <= h1:
        return df[(horas >= h0) & (horas <= h1)]
    else:
        return df[(horas >= h0) | (horas <= h1)]


def extract_windows(muestras_filtered: dict) -> dict:
    """

    For each retained cycle:
    1. Keep only hourly readings within INTERVALO_DIAS relative to ovulation.
    2. Keep only readings in the nightly maxima window INTERVALO_HORAS.
    3. Retain the series only if the filtered DataFrame is non-empty.

    Returns
    -------
    dict mapping serie_id → {"serie": filtered_df, "ovul": FECHA_REF}
    """
    h0, h1 = INTERVALO_HORAS
    d0, d1 = INTERVALO_DIAS
    muestras_positivas: dict = {}

    for sid, datos in muestras_filtered.items():
        df = datos["serie"].copy()

        # Day filter (note: notebook uses d1 + 1 as the upper Timedelta bound)
        tiempo_rel = df["resultTimestamp"] - FECHA_REF
        filtro_dias = (
            (tiempo_rel >= pd.Timedelta(days=d0)) &
            (tiempo_rel <= pd.Timedelta(days=d1 + 1))
        )
        df_f = df[filtro_dias].copy()

        # Hour filter
        df_f = _filter_hours(df_f, h0, h1)

        if not df_f.empty:
            muestras_positivas[sid] = {
                "serie": df_f.reset_index(drop=True),
                "ovul":  FECHA_REF,
            }

    logger.info(
        "Window extraction: %d series retained", len(muestras_positivas)
    )
    return muestras_positivas


# ===========================================================================
# Step 3 – Fixed-width feature matrix
# ===========================================================================

def _compute_num_horas() -> int:
    h0, h1 = INTERVALO_HORAS
    d0, d1 = INTERVALO_DIAS
    if h0 <= h1:
        num_horas_intervalo = h1 - h0 + 1
    else:
        num_horas_intervalo = (24 - h0) + (h1 + 1)
    return (d1 - d0 + 1) * num_horas_intervalo + 1


def build_feature_matrix(muestras_positivas: dict) -> pd.DataFrame:
    """

    Flattens each series into a row of temperature values.  Column names are
    ``t_0``, ``t_1``, … ``t_{N_FEATURE_COLS-1}``.

    The notebook generates up to ``num_horas`` columns then truncates to
    N_FEATURE_COLS (140) via ``.iloc[:, :140]``.  Each series extracted with
    the current parameters contains exactly 141 values so the truncation
    always drops the last one.

    The DataFrame index is set to the cycle identifiers (serie_id) so that
    MaxRange.csv preserves them in the first column.
    """
    num_horas = _compute_num_horas()
    col_names = [f"t_{i}" for i in range(num_horas)]

    rows: list[np.ndarray] = []
    index: list[str] = []

    for sid, datos in muestras_positivas.items():
        df = datos["serie"].sort_values("resultTimestamp")
        rows.append(df["result"].values)
        index.append(sid)

    df_matrix = pd.DataFrame(rows, columns=col_names, index=index)
    df_matrix.index.name = "serie_id"

    # Truncate to N_FEATURE_COLS exactly as the notebook does
    df_matrix = df_matrix.iloc[:, :N_FEATURE_COLS]

    logger.info(
        "Feature matrix shape: %d rows × %d columns",
        *df_matrix.shape,
    )
    return df_matrix


# ===========================================================================
# Report
# ===========================================================================

def build_report(
    n_original: int,
    df_std: pd.DataFrame,
    df_matrix: pd.DataFrame,
) -> str:
    """Compose a concise text report of the construction pipeline."""
    q1      = df_std["q1"].iloc[0]
    q3      = df_std["q3"].iloc[0]
    iqr_val = df_std["iqr"].iloc[0]
    lim_sup = df_std["lim_sup"].iloc[0]
    removed = df_std.loc[~df_std["kept"], "serie_id"].tolist()

    lines = [
        "CLASSIFICATION DATASET CONSTRUCTION REPORT",
        f"Source: muestras_ovul_horas_norm1.pkl   |   norm1 normalisation",
        "",
        "=" * 60,
        "  Step 1 – High-variance IQR filter",
        "=" * 60,
        f"  Cycles before filter : {n_original}",
        f"  Q1 (std)             : {q1:.4f}",
        f"  Q3 (std)             : {q3:.4f}",
        f"  IQR                  : {iqr_val:.4f}",
        f"  Upper fence (Q3+1.5×IQR) : {lim_sup:.4f}",
        f"  Removed ({len(removed)}) : {sorted(removed)}",
        f"  Cycles after filter  : {n_original - len(removed)}",
        "",
        "=" * 60,
        "  Step 2 – Time-window parameters",
        "=" * 60,
        f"  Day range (relative to ovulation) : {INTERVALO_DIAS}",
        f"  Hour window (maxima, night)        : {INTERVALO_HORAS[0]}:00–{INTERVALO_HORAS[1]:02d}:59",
        f"  Hours per day in window            : {_compute_num_horas() // (INTERVALO_DIAS[1] - INTERVALO_DIAS[0] + 1)}",
        f"  Total time-points per series       : {_compute_num_horas()} (truncated to {N_FEATURE_COLS})",
        "",
        "=" * 60,
        "  Step 3 – Feature matrix (MaxRange.csv)",
        "=" * 60,
        f"  Shape  : {df_matrix.shape[0]} rows × {df_matrix.shape[1]} columns",
        f"  Columns: t_0 … t_{N_FEATURE_COLS - 1}",
        "",
        "  Per-feature descriptive statistics (first 5 / last 5 columns):",
    ]

    desc = df_matrix.describe().T
    preview_cols = list(df_matrix.columns[:5]) + list(df_matrix.columns[-5:])
    lines.append(
        desc.loc[preview_cols, ["mean", "std", "min", "max"]].to_string(
            float_format=lambda v: f"{v:.4f}"
        )
    )
    lines += [
        "",
        "  Series in dataset:",
        "  " + ", ".join(df_matrix.index.tolist()),
        "",
    ]

    return "\n".join(lines)


# ===========================================================================
# End-to-end pipeline
# ===========================================================================

def run_construction_pipeline(
    norm1_pkl: Path,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Execute the full construction pipeline and persist all output artefacts.

    Artefacts written
    -----------------
    MaxRange.csv              – Feature matrix (N_cycles × N_FEATURE_COLS)
    construction_report.txt   – Text report of all pipeline steps

    Parameters
    ----------
    norm1_pkl  : Path to muestras_ovul_horas_norm1.pkl
    output_dir : Directory where all output files are written

    Returns
    -------
    df_matrix : pd.DataFrame  – the exported feature matrix
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    muestras = load_pickle(norm1_pkl)
    n_original = len(muestras)
    logger.info("Loaded %d cycles from norm1 pickle", n_original)

    # ------------------------------------------------------------------
    # Step 1 – IQR std filter
    # ------------------------------------------------------------------
    logger.info("=== Step 1 – High-variance IQR filter ===")
    muestras_filtered, df_std = filter_high_variance(muestras)

    # ------------------------------------------------------------------
    # Step 2 – Time-window extraction
    # ------------------------------------------------------------------
    logger.info("=== Step 2 – Time-window extraction ===")
    muestras_positivas = extract_windows(muestras_filtered)

    # ------------------------------------------------------------------
    # Step 3 – Feature matrix
    # ------------------------------------------------------------------
    logger.info("=== Step 3 – Feature matrix construction ===")
    df_matrix = build_feature_matrix(muestras_positivas)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    csv_path = output_dir / "MaxRange.csv"
    df_matrix.to_csv(csv_path)
    logger.info("Saved: %s  (%d × %d)", csv_path, *df_matrix.shape)

    report_text = build_report(n_original, df_std, df_matrix)
    report_path = output_dir / "construction_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    logger.info("Saved: %s", report_path)

    print("\n" + report_text)
    logger.info("Construction pipeline complete. Artefacts in: %s", output_dir)

    return df_matrix


# ===========================================================================
# CLI entry point
# ===========================================================================

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Classification dataset construction pipeline (notebook 3.1). "
            "Filters cycles by std-dev IQR, extracts the nightly maxima "
            "window, and exports MaxRange.csv."
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
        help="Directory for output artefacts (default: current directory)",
    )
    args = parser.parse_args(argv)

    run_construction_pipeline(
        norm1_pkl=Path(args.norm1_pkl),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
