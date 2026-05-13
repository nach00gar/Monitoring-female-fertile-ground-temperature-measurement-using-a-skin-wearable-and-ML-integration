# -*- coding: utf-8 -*-
from __future__ import annotations
"""
LSSVM ovulation-phase validation pipeline.
Reads MaxRange.csv (output of construction.py).

Outputs
-------
  Thesis_Final_Labeled.png  – main thesis figure (panels A, B, C)
  validation_report.txt     – text report with metrics and confusion matrices

Usage
-----
    python validation_lssvm.py --maxrange_csv ./MaxRange.csv --output_dir .
"""

import argparse
import logging
from collections import defaultdict
from io import StringIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from LSSVMRegression import LSSVMRegression

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour map  (cells 11 / 33 / 39)
# ---------------------------------------------------------------------------
colors_cm = [(1, 1, 1), (1, 0, 0)]
cmap = mcolors.LinearSegmentedColormap.from_list("white_red", colors_cm)


# ===========================================================================
# Cell 9 – Dataset construction
# ===========================================================================

def build_dataset(df: pd.DataFrame):
    n_muestras, n_columnas = df.shape
    n_instantes = 7
    dias = list(range(-14, 6))
    ventana_dias = 10
    ventana_cols = ventana_dias * n_instantes
    ventanas = list(range(-4, 6))

    X_list, y_list, group_list = [], [], []

    for i_muestra in range(n_muestras):
        serie = df.iloc[i_muestra].values
        for clase, end_day in enumerate(ventanas):
            end_idx   = (end_day - dias[0] + 1) * n_instantes
            start_idx = end_idx - ventana_cols
            if start_idx >= 0 and end_idx <= n_columnas:
                X_list.append(serie[start_idx:end_idx])
                y_list.append(clase)
                group_list.append(i_muestra)

    X      = np.vstack(X_list)
    y      = np.array(y_list)
    groups = np.array(group_list)

    logger.info("X shape: %s", X.shape)
    logger.info("Class distribution:\n%s", pd.Series(y).value_counts().sort_index())
    return X, y, groups


# ===========================================================================
# Cell 21 – Patient map from samples list
# ===========================================================================

def build_patient_map(samples: list):
    """Cell 21 – groups sample indices by patient prefix."""
    patient_map = defaultdict(list)
    for i, s in enumerate(samples):
        prefix, num = s.split("_")
        patient_map[prefix].append((i, int(num)))
    return dict(patient_map)


# ===========================================================================
# Cell 22 – sequential_sample_splitter
# ===========================================================================

def sequential_sample_splitter(X, patient_map, instances_per_sample=10):
    """
    Validación para TODAS las muestras:
    - Test: Muestra actual 'i' de cualquier paciente.
    - Train: Todo el dataset EXCEPTO la muestra actual y muestras futuras del mismo paciente.
    """
    total_instances = X.shape[0]
    all_indices = np.arange(total_instances)

    for patient_id, samples in patient_map.items():
        for i in range(len(samples)):
            current_sample_info = samples[i]
            current_global_idx  = current_sample_info[0]

            start_test   = current_global_idx * instances_per_sample
            end_test     = start_test + instances_per_sample
            test_indices = np.arange(start_test, min(end_test, total_instances))

            future_samples      = samples[i + 1:]
            future_indices_list = []
            for f_sample in future_samples:
                f_idx   = f_sample[0]
                f_start = f_idx * instances_per_sample
                f_end   = f_start + instances_per_sample
                future_indices_list.extend(range(f_start, min(f_end, total_instances)))

            indices_to_exclude = np.concatenate([test_indices, future_indices_list]).astype(int)
            train_indices      = np.setdiff1d(all_indices, indices_to_exclude)

            yield train_indices, test_indices, patient_id, current_global_idx


# ===========================================================================
# Cell 23 – valida_modelo_secuencial_regresion
# ===========================================================================

def valida_modelo_secuencial_regresion(modelo, parametros, X, y, patient_map):
    accs_global        = []
    maes_global        = []
    rmses_global       = []
    me_global          = []
    matrices_conf      = []
    resultados_detallados = {}

    print("Iniciando Validación Secuencial (Walk-Forward por paciente)...")

    splitter = sequential_sample_splitter(X, patient_map)

    for train_idx, test_idx, pat_id, samp_id in splitter:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        grid = GridSearchCV(
            modelo,
            parametros,
            cv=inner_cv,
            n_jobs=-1,
            scoring='neg_mean_squared_error',
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        y_pred_reg = best_model.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred_reg)
        rmse = root_mean_squared_error(y_test, y_pred_reg)
        maes_global.append(mae)
        rmses_global.append(rmse)

        y_pred = np.clip(np.round(y_pred_reg), 0, 9).astype(int)

        acc = accuracy_score(y_test, y_pred)
        accs_global.append(acc)

        aciertos = np.sum(y_test == y_pred)
        resultados_detallados[f"{pat_id}_MuestraGlobal{samp_id}"] = aciertos

        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
        matrices_conf.append(cm)

        logger.info(
            "patient=%s  cycle=%d  acc=%.3f  MAE=%.3f  gamma=%.4g  sigma=%.4g",
            pat_id, samp_id, acc, mae,
            grid.best_params_["gamma"], grid.best_params_["sigma"],
        )

    print(f"\n--- Resultados Validación Secuencial ---")
    print(f"Total validaciones: {len(accs_global)}")
    print(f"Accuracy medio: {np.mean(accs_global):.3f}")
    print(f"MAE medio: {np.mean(maes_global):.3f}")
    print(f"RMSE medio: {np.mean(rmses_global):.3f}")

    if len(matrices_conf) > 0:
        cm_mean = np.sum(matrices_conf, axis=0)
        cm_df = pd.DataFrame(
            cm_mean,
            index=[f"true_{i}" for i in np.unique(y)],
            columns=[f"pred_{i}" for i in np.unique(y)],
        )
        print("\nMatriz de confusión acumulada:")
        print(cm_df)

    return {
        "accs_global":       accs_global,
        "mae_global":        maes_global,
        "rmse_global":       rmses_global,
        "matriz_conf_media": cm_df if len(matrices_conf) > 0 else None,
    }


# ===========================================================================
# Cell 12 – metricas_por_rango
# ===========================================================================

def metricas_por_rango(cm):
    """
    Para k=1..5:
      - Considera solo muestras reales >= k.
      - Acierto = predicción >= 1.
    """
    clases        = cm.index.tolist()
    clases_reales = [1, 2, 3, 4, 5]
    pred_validas  = [1, 2, 3, 4, 5]
    resultados    = []

    for k in clases_reales:
        reales    = [c for c in clases_reales if c >= k]
        total     = cm.loc[reales, :].sum().sum()
        aciertos  = cm.loc[reales, pred_validas].sum().sum()
        porcentaje = aciertos / total if total != 0 else 0
        resultados.append([k, total, aciertos, porcentaje])

    return pd.DataFrame(
        resultados,
        columns=["k", "Total_muestras", "Aciertos_pred>=1", "Porcentaje_acierto"],
    )


# ===========================================================================
# Cell 17 – metricas_completas_por_clase
# ===========================================================================

def metricas_completas_por_clase(cm):
    """
    Analiza cada clase real de forma independiente:
    - Clases >= 1: Tasa de detección (Predicción >= 1)
    - Clases <= 0: Tasa de acierto 'bajo umbral' (Predicción < 1)
    """
    clases_reales       = sorted(cm.index.tolist())
    resultados          = []
    cols_pred_evento    = [c for c in cm.columns if c >= 1]
    cols_pred_no_evento = [c for c in cm.columns if c < 1]

    for k in clases_reales:
        fila_k = cm.loc[k, :]
        total  = fila_k.sum()
        if total == 0:
            continue
        if k >= 1:
            aciertos     = fila_k[cols_pred_evento].sum()
            tipo_metrica = "Detección (Pred >= 1)"
        else:
            aciertos     = fila_k[cols_pred_no_evento].sum()
            tipo_metrica = "Correcto Negativo (Pred < 1)"

        tasa = aciertos / total
        resultados.append([k, tipo_metrica, total, aciertos, tasa])

    return pd.DataFrame(
        resultados,
        columns=["Clase_Real", "Tipo_Metrica", "Total_Muestras", "Aciertos", "Tasa"],
    )


# ===========================================================================
# Cell 39 – procesar_matriz_final_tesis  →  Thesis_Final_Labeled.png
# ===========================================================================

def procesar_matriz_final_tesis(
    res_lssvm,
    neg_idx=list(range(5)),
    pos_idx=list(range(5, 10)),
    output_path: Path = Path("Thesis_Final_Labeled.png"),
):
    # --- 1. CÁLCULOS ---
    cm = res_lssvm["matriz_conf_media"].copy()
    labels = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    cm.index   = labels
    cm.columns = labels

    # Phase success rates (bar panel)
    phase_success_rates = []
    for true_label in labels:
        row_data      = cm.loc[true_label]
        total_samples = row_data.sum()
        if total_samples == 0:
            rate = 0
        elif true_label <= 0:
            correct_preds = row_data[row_data.index <= 0].sum()
            rate = correct_preds / total_samples
        else:
            correct_preds = row_data[row_data.index >= 1].sum()
            rate = correct_preds / total_samples
        phase_success_rates.append(rate)

    bar_colors = ['#bdc3c7' if x <= 0 else '#e74c3c' for x in labels]

    # Binary aggregation
    TN = cm.iloc[neg_idx, :].iloc[:, neg_idx].sum().sum()
    FP = cm.iloc[neg_idx, :].iloc[:, pos_idx].sum().sum()
    FN = cm.iloc[pos_idx, :].iloc[:, neg_idx].sum().sum()
    TP = cm.iloc[pos_idx, :].iloc[:, pos_idx].sum().sum()

    cm_binary = pd.DataFrame(
        [[TN, FP], [FN, TP]],
        index=["PreOv", "PostOv"],
        columns=["PreOv", "PostOv"],
    )

    accuracy  = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) != 0 else 0

    # --- 2. PLOTTING ---
    fig = plt.figure(figsize=(16, 10), dpi=300)
    gs  = gridspec.GridSpec(2, 2, height_ratios=[3, 1.2], width_ratios=[1.6, 1])
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[0, :], width_ratios=[3, 1.2], wspace=0.02
    )

    # Panel A – multiclass confusion matrix
    ax_cm = fig.add_subplot(gs_top[0])
    im = ax_cm.imshow(cm, cmap=cmap, aspect='auto')
    ax_cm.text(-0.09, 1.05, 'A', transform=ax_cm.transAxes,
               fontsize=20, fontweight='bold', va='top', ha='right')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm.iloc[i, j]
            color_txt = "white" if val > cm.max().max() / 2 else "black"
            ax_cm.text(j, i, int(val), ha='center', va='center',
                       fontsize=9, color=color_txt)
    ax_cm.set_xticks(range(len(labels)))
    ax_cm.set_xticklabels(labels)
    ax_cm.set_yticks(range(len(labels)))
    ax_cm.set_yticklabels(labels)
    ax_cm.set_ylabel("True Cycle Day",      fontsize=12, fontweight='bold')
    ax_cm.set_xlabel("Predicted Cycle Day", fontsize=12, fontweight='bold')
    ax_cm.set_title("Multiclass Confusion Matrix", fontsize=14,
                    fontweight='bold', pad=15)
    ax_cm.axhline(y=4.5, color='black', linestyle='--', linewidth=1, alpha=0.6)
    ax_cm.axvline(x=4.5, color='black', linestyle='--', linewidth=1, alpha=0.6)

    # Panel B – bar chart
    ax_bar = fig.add_subplot(gs_top[1], sharey=ax_cm)
    ax_bar.text(0.05, 1.05, 'B', transform=ax_bar.transAxes,
                fontsize=20, fontweight='bold', va='top', ha='right')
    ax_bar.set_title("Ovulation Detection", fontsize=14, fontweight='bold', pad=15)
    y_pos = range(len(labels))
    bars  = ax_bar.barh(y_pos, phase_success_rates, align='center',
                        color=bar_colors, edgecolor='black', height=0.8)
    for bar in bars:
        width = bar.get_width()
        ax_bar.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                    f'{width:.1%}', va='center', ha='left',
                    fontsize=10, fontweight='bold', color='black')
    ax_bar.set_xlim(0, 1.0)
    ax_bar.set_xlabel("Ovulation Detection Rate\n(Correct Binary Classification)",
                      fontsize=10, fontweight='bold')
    ax_bar.tick_params(axis='y', left=False, labelleft=False)
    ax_bar.spines['left'].set_visible(False)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)

    # Panel C – binary confusion matrix
    ax_bin = fig.add_subplot(gs[1, 0])
    ax_bin.text(-0.1, 1.1, 'C', transform=ax_bin.transAxes,
                fontsize=20, fontweight='bold', va='top', ha='right')
    ax_bin.imshow(cm_binary, cmap=cmap, aspect='auto')
    for i in range(2):
        for j in range(2):
            ax_bin.text(j, i, cm_binary.iloc[i, j],
                        ha='center', va='center',
                        fontsize=14, fontweight='bold', color="black")
    ax_bin.set_xticks([0, 1])
    ax_bin.set_xticklabels(["PreOv", "PostOv"])
    ax_bin.set_yticks([0, 1])
    ax_bin.set_yticklabels(["PreOv (True)", "PostOv (True)"], rotation=90, va="center")
    ax_bin.set_title("Aggregated Binary Matrix", fontsize=14,
                     fontweight='bold', pad=10)

    # Metrics text box
    ax_text = fig.add_subplot(gs[1, 1])
    ax_text.axis('off')
    box_props = dict(boxstyle='round,pad=1', facecolor='#f8f9fa',
                     alpha=0.8, edgecolor='#bdc3c7')
    metric_str = (
        r"$\bf{AGGREGATED\ BINARY\ METRICS}$" + "\n"
        "--------------------------\n"
        f"Accuracy:        {accuracy:.4f}\n"
        f"Sensitivity:     {recall:.4f}\n"
        f"Precision:       {precision:.4f}"
    )
    ax_text.text(0.5, 0.5, metric_str, transform=ax_text.transAxes,
                 fontsize=12, va='center', ha='center',
                 bbox=box_props, family='monospace')

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=600, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", output_path)

    return fig


# ===========================================================================
# Text report
# ===========================================================================

def build_report(res_test2: dict) -> str:
    cm = res_test2["matriz_conf_media"].copy()
    cm.index   = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    cm.columns = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

    df_metricas  = metricas_completas_por_clase(cm)
    df_por_rango = metricas_por_rango(cm)

    neg_idx = list(range(5))
    pos_idx = list(range(5, 10))
    TN = cm.iloc[neg_idx, :].iloc[:, neg_idx].sum().sum()
    FP = cm.iloc[neg_idx, :].iloc[:, pos_idx].sum().sum()
    FN = cm.iloc[pos_idx, :].iloc[:, neg_idx].sum().sum()
    TP = cm.iloc[pos_idx, :].iloc[:, pos_idx].sum().sum()
    accuracy  = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) != 0 else 0

    buf = StringIO()
    buf.write("LSSVM WALK-FORWARD VALIDATION REPORT\n")
    buf.write("=" * 60 + "\n")
    buf.write(f"Total validaciones : {len(res_test2['accs_global'])}\n")
    buf.write(f"Accuracy medio     : {np.mean(res_test2['accs_global']):.4f}\n")
    buf.write(f"MAE medio          : {np.mean(res_test2['mae_global']):.4f}\n")
    buf.write(f"RMSE medio         : {np.mean(res_test2['rmse_global']):.4f}\n\n")

    buf.write("=" * 60 + "\n")
    buf.write("Aggregated binary metrics (Pre- vs Post-ovulation)\n")
    buf.write("=" * 60 + "\n")
    buf.write(f"  Accuracy   : {accuracy:.4f}\n")
    buf.write(f"  Precision  : {precision:.4f}\n")
    buf.write(f"  Sensitivity: {recall:.4f}\n\n")

    buf.write("=" * 60 + "\n")
    buf.write("Multiclass confusion matrix\n")
    buf.write("=" * 60 + "\n")
    buf.write(cm.to_string())
    buf.write("\n\n")

    buf.write("=" * 60 + "\n")
    buf.write("metricas_completas_por_clase\n")
    buf.write("=" * 60 + "\n")
    buf.write(df_metricas.to_string(index=False))
    buf.write("\n\n")

    buf.write("=" * 60 + "\n")
    buf.write("metricas_por_rango\n")
    buf.write("=" * 60 + "\n")
    buf.write(df_por_rango.to_string(index=False))
    buf.write("\n")

    return buf.getvalue()


# ===========================================================================
# Pipeline
# ===========================================================================

def run_pipeline(maxrange_csv: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load (cell 2) ---
    logger.info("Loading %s", maxrange_csv)
    df = pd.read_csv(maxrange_csv, index_col=0)
    print("Dimensiones del DataFrame:", df.shape)
    print("\nPrimeras filas:")
    print(df.head())
    print("\nVista parcial (5 filas x 10 columnas):")
    print(df.iloc[:5, :10])

    # --- Dataset (cell 9) ---
    X, y, groups = build_dataset(df)

    # --- StandardScaler (cell 15) – kept for parity; raw X passed to validation ---
    s  = StandardScaler()
    Xs = s.fit_transform(X)   # Xs available but cell 24 uses X

    # --- Parameters (cell 16) ---
    lssvm = LSSVMRegression()
    parameters = {
        'kernel': ['rbf'],
        'gamma':  [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
        'sigma':  [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0],
    }

    # --- Patient map (cell 21) – built from CSV index to avoid hardcoding ---
    samples     = df.index.tolist()
    patient_map = build_patient_map(samples)

    # --- Validation (cell 24) – passes raw X exactly as the notebook ---
    lssvm      = LSSVMRegression()
    res_test2  = valida_modelo_secuencial_regresion(lssvm, parameters, X, y, patient_map)

    # --- Outputs (cells 30, 31, 36, 39) ---
    print("\nmetricas_completas_por_clase:")
    cm_labeled = res_test2["matriz_conf_media"].copy()
    cm_labeled.index   = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    cm_labeled.columns = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    print(metricas_completas_por_clase(cm_labeled))

    print(f"\nnp.mean(mae_global): {np.mean(res_test2['mae_global']):.4f}")

    print("\nmetricas_por_rango:")
    print(metricas_por_rango(cm_labeled))

    # Figure (cell 39)
    procesar_matriz_final_tesis(
        res_test2,
        output_path=output_dir / "Thesis_Final_Labeled.png",
    )

    # Text report
    report = build_report(res_test2)
    rpath  = output_dir / "validation_report.txt"
    rpath.write_text(report, encoding="utf-8")
    logger.info("Saved: %s", rpath)
    print("\n" + report)

    logger.info("Done. Artefacts in: %s", output_dir)


# ===========================================================================
# CLI
# ===========================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="LSSVM walk-forward validation (PruebaFinalLSSVM-ForRepo.ipynb)."
    )
    parser.add_argument("--maxrange_csv", default="../results/MaxRange.csv",
                        help="Path to MaxRange.csv")
    parser.add_argument("--output_dir",   default="../results/",
                        help="Output directory")
    args = parser.parse_args(argv)
    run_pipeline(Path(args.maxrange_csv), Path(args.output_dir))


if __name__ == "__main__":
    main()
