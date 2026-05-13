# -*- coding: utf-8 -*-
"""
Preprocessing pipeline for ovulation temperature data.

"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import trim_mean
from statsmodels.nonparametric.smoothers_lowess import lowess

logger = logging.getLogger(__name__)


DIAS_VENTANA: int = 14          # 1.3 – days before AND after ovulation in window
DIAS_REQUERIDOS_ANTES: int = 2  # 1.4 – min days before ovulation with data
DIAS_REQUERIDOS_DESPUES: int = 5  # 1.4 – min days after ovulation with data
UMBRAL_DATOS_POR_DIA: int = 1   # 1.4 – at least 1 reading per required day
FECHA_REF: pd.Timestamp = pd.Timestamp('2025-01-01 00:00:00')  # alignment origin

TEMP_MIN: float = 35.7
TEMP_MAX: float = 37.7

# Pattern strings (1.2)
OVUL_POS_PATTERN: str = r"ov|Ov|OV|test|Test|TEST|pos|Pos|POS"
OVUL_NEG_PATTERN: str = (
    r"neg|Neg|NEG|ei ole|ei olnud|pole|ei tuvastanud"
    r"|mitte ovulats|positiivset testi ei ole saanud"
)


# ===========================================================================
# Step 1.1 – Load & filter raw temperature data
# ===========================================================================

def load_temperature_data(raw_csv_path: Path) -> pd.DataFrame:
    """Load medResultsTu.csv, keep relevant columns, drop NaN rows."""
    logger.info("Loading raw temperature data from %s", raw_csv_path)
    df = pd.read_csv(
        raw_csv_path, header=0, sep=';', low_memory=False,
        usecols=['prodId', 'result', 'resultTimestamp', 'Tu'],
    )
    before = len(df)
    df = df.dropna()
    logger.info("Dropped NaN: %d → %d rows", before, len(df))
    return df


def filter_temperature_range(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only readings in (TEMP_MIN, TEMP_MAX)."""
    mask = (df['result'] > TEMP_MIN) & (df['result'] < TEMP_MAX)
    out = df.loc[mask].dropna().copy()
    logger.info("Temperature filter [%.1f, %.1f]: %d rows retained", TEMP_MIN, TEMP_MAX, len(out))
    return out


# ===========================================================================
# Step 1.2 – Load ovulation annotations
# ===========================================================================

def load_ovulation_notes(notes_csv_path: Path) -> pd.DataFrame:
    """
    Parse medNotes.csv: identify positive ovulation tests, convert UNIX
    timestamps, remove intra-day duplicates.

    Returns DataFrame with columns: prodId, timestamp (Timestamp floored to day).
    """
    logger.info("Loading ovulation notes from %s", notes_csv_path)
    med_notes = pd.read_csv(notes_csv_path, header=0, sep=';')
    mask_pos = med_notes['givenComment'].str.contains(OVUL_POS_PATTERN, case=False, na=False)
    mask_neg = med_notes['givenComment'].str.contains(OVUL_NEG_PATTERN, case=False, na=False)

    ovul_notes = med_notes.loc[mask_pos & ~mask_neg, ['givenTimestamp', 'prodId']].copy()
    ovul_notes = ovul_notes.dropna()
    ovul_notes = ovul_notes.rename(columns={'givenTimestamp': 'timestamp'})
    ovul_notes['timestamp'] = pd.to_datetime(ovul_notes['timestamp'], unit='s').dt.floor('D')

    # Remove duplicates on same patient-day
    ovul_notes['_date'] = ovul_notes['timestamp'].dt.date
    ovul_notes = ovul_notes.drop_duplicates(subset=['prodId', '_date'], keep='first')
    ovul_notes = ovul_notes.drop(columns=['_date']).reset_index(drop=True)

    logger.info("Found %d unique ovulation events", len(ovul_notes))
    return ovul_notes


# ===========================================================================
# Step 1.3 – Cycle segmentation 
# ===========================================================================

def segment_cycles(
    temp_filtered: pd.DataFrame,
    ovul_notes: pd.DataFrame,
    dias_ventana: int = DIAS_VENTANA,
    fecha_ref: pd.Timestamp = FECHA_REF,
) -> Dict[str, Dict]:
    """
    Notebook 1.3 – Visualización y separación de las series por ciclo.

    For each patient×ovulation event:
    1. Filter temperature readings within the ±dias_ventana window.
    2. Round timestamps to the nearest even minute (as in original notebook).
    3. Remove intra-minute duplicates.
    4. Align all timestamps so that the ovulation date maps to fecha_ref.
    5. Left-merge onto a complete 2-min grid (introducing NaN for gaps).

    Returns
    -------
    dict: {"{Tu}_{i}": {"serie": DataFrame(resultTimestamp, result), "ovul": Timestamp}}
    """
    temp = temp_filtered.copy()
    temp['resultTimestamp'] = pd.to_datetime(temp['resultTimestamp'], unit='s', errors='coerce')
    temp = temp.dropna(subset=['resultTimestamp'])

    margen = pd.Timedelta(days=dias_ventana)

    # Complete 2-min grid centred on fecha_ref (same as notebook)
    tiempos_completos = pd.date_range(
        start=fecha_ref - margen,
        end=fecha_ref + margen,
        freq='2min',
    ).to_frame(index=False, name='resultTimestamp')

    muestras_ovul: Dict[str, Dict] = {}

    for tu_id in temp['Tu'].unique():
        muestra = temp[temp['Tu'] == tu_id].copy()
        ovul = ovul_notes[ovul_notes['prodId'].isin(muestra['prodId'].unique())]

        i = 1
        for fecha_ovul in ovul['timestamp']:
            fecha_ovul = pd.Timestamp(fecha_ovul)

            # Filter window
            muestra_ovul = muestra[
                (muestra['resultTimestamp'] >= fecha_ovul - margen) &
                (muestra['resultTimestamp'] <= fecha_ovul + margen)
            ].copy()

            # Round to nearest even minute (drop seconds/microseconds, floor minute to even)
            muestra_ovul['resultTimestamp'] = muestra_ovul['resultTimestamp'].apply(
                lambda x: x - pd.Timedelta(
                    minutes=x.minute % 2,
                    seconds=x.second,
                    microseconds=x.microsecond,
                )
            )

            # Drop duplicate timestamps
            muestra_ovul = muestra_ovul.drop_duplicates(subset='resultTimestamp', keep='first')

            # Align: shift so ovulation day → fecha_ref
            muestra_ovul['resultTimestamp'] = muestra_ovul['resultTimestamp'] + (fecha_ref - fecha_ovul)

            # Left-merge onto complete grid (gaps become NaN)
            muestra_alineada = tiempos_completos.merge(
                muestra_ovul[['resultTimestamp', 'result']],
                on='resultTimestamp',
                how='left',
            )

            name = f"{tu_id}_{i}"
            muestras_ovul[name] = {'serie': muestra_alineada, 'ovul': fecha_ovul}
            i += 1

    logger.info("Segmented %d cycles", len(muestras_ovul))
    return muestras_ovul


# ===========================================================================
# Step 1.4 – Filter incomplete cycles 
# ===========================================================================

def filter_incomplete_cycles(
    muestras_ovul: Dict[str, Dict],
    dias_antes: int = DIAS_REQUERIDOS_ANTES,
    dias_despues: int = DIAS_REQUERIDOS_DESPUES,
    fecha_ref: pd.Timestamp = FECHA_REF,
) -> Dict[str, Dict]:
    """
    Notebook 1.4 – Filtrado de series temporales en función de valores faltantes.

    Keep only cycles that have at least 1 non-NaN reading on every calendar
    day in the range [-dias_antes, +dias_despues] relative to fecha_ref.
    """
    fecha_ref_h = fecha_ref.floor('h')
    series_filtradas: Dict[str, Dict] = {}

    for id_muestra, muestra in muestras_ovul.items():
        serie = muestra['serie'].copy()

        # Relative day (integer)
        serie['dia_relativo'] = (
            (serie['resultTimestamp'] - fecha_ref_h) / pd.Timedelta(days=1)
        ).astype(int)

        conteo_por_dia = serie.dropna(subset=['result']).groupby('dia_relativo').size()
        dias_necesarios = list(range(-dias_antes, dias_despues + 1))

        if all(dia in conteo_por_dia.index for dia in dias_necesarios):
            series_filtradas[id_muestra] = muestra

    logger.info(
        "Incomplete-cycle filter: kept %d / %d cycles",
        len(series_filtradas), len(muestras_ovul),
    )
    return series_filtradas


# ===========================================================================
# Step 1.5 – Artifact smoothing 
# ===========================================================================

def corregir_series(
    muestras_ovul: Dict[str, Dict],
    umbral_cambio: float = 0.3,
    factor_lowess: float = 0.5,
    ventana_estudio: int = 60,
    ventana_antes: int = 30,
    ventana_despues: int = 30,
    ampliar_correccion: int = 8,
    suavizado_lateral: int = 30,
) -> Dict[str, Dict]:
    """
    Notebook 1.5 – corregir_series (exact translation).

    Detects abrupt temperature spikes using sliding windows and flattens
    them to the average of the surrounding context. Border regions are
    smoothed with LOWESS.

    All window parameters are in **minutes** (internally divided by 2
    because timestamps are every 2 min).

    Note: Called **twice** in the original pipeline on the same data:
        muestras_corregidas = corregir_series(muestras_ovul)
        muestras_corregidas = corregir_series(muestras_corregidas)
    """
    muestras_corregidas: Dict[str, Dict] = {}

    # Convert minutes → 2-min sample units
    ventana_estudio   = ventana_estudio   // 2
    ventana_antes     = ventana_antes     // 2
    ventana_despues   = ventana_despues   // 2
    ampliar_correccion = ampliar_correccion // 2
    suavizado_lateral = suavizado_lateral // 2

    for clave, entrada in muestras_ovul.items():
        serie = entrada['serie'].copy()
        serie = serie.sort_values('resultTimestamp').reset_index(drop=True)

        temps = serie['result'].copy()
        times = serie['resultTimestamp']

        minutos = ((times - times.iloc[0]) / pd.Timedelta(minutes=1)).values
        temps_array = temps.values.astype('float64')

        i = ventana_antes + ventana_estudio // 2

        while i < len(temps_array) - (ventana_despues + ventana_estudio // 2):
            if np.isnan(temps_array[i]):
                i += 1
                continue

            idx_ventana_antes = np.arange(
                i - (ventana_antes + ventana_estudio // 2),
                i - ventana_estudio // 2,
            )
            idx_ventana_antes = [idx for idx in idx_ventana_antes if not np.isnan(temps_array[idx])]
            if len(idx_ventana_antes) < 5:
                i += 1
                continue

            idx_ventana_despues = np.arange(
                i + ventana_estudio // 2,
                i + ventana_despues + ventana_estudio // 2,
            )
            idx_ventana_despues = [idx for idx in idx_ventana_despues if not np.isnan(temps_array[idx])]
            if len(idx_ventana_despues) < 5:
                i += 1
                continue

            idx_ventana_estudio = np.arange(i - ventana_estudio // 2, i + ventana_estudio // 2)
            idx_ventana_estudio = [idx for idx in idx_ventana_estudio if not np.isnan(temps_array[idx])]
            if len(idx_ventana_estudio) < 5:
                i += 1
                continue

            media_antes   = np.mean([temps_array[idx] for idx in idx_ventana_antes])
            media_despues = np.mean([temps_array[idx] for idx in idx_ventana_despues])
            min_ventana   = np.min([temps_array[idx] for idx in idx_ventana_estudio])
            max_ventana   = np.max([temps_array[idx] for idx in idx_ventana_estudio])

            diferencia_antes   = max(abs(media_antes   - min_ventana), abs(media_antes   - max_ventana))
            diferencia_despues = max(abs(media_despues - min_ventana), abs(media_despues - max_ventana))

            if diferencia_antes > umbral_cambio and diferencia_despues > umbral_cambio:
                idx_inicio = max(0, i - ventana_estudio // 2 - ampliar_correccion)
                idx_fin    = min(len(temps_array), i + ventana_estudio // 2 + ampliar_correccion)
                idx_aplanado = np.arange(idx_inicio, idx_fin)
                idx_aplanado = [idx for idx in idx_aplanado if not np.isnan(temps_array[idx])]

                media_aplanado = (media_antes + media_despues) / 2
                for idx in idx_aplanado:
                    temps_array[idx] = media_aplanado

                suavizar_margen = suavizado_lateral // 2
                for borde in [idx_inicio, idx_fin - 1]:
                    idx_borde_ini = max(0, borde - suavizar_margen)
                    idx_borde_fin = min(len(temps_array), borde + suavizar_margen)
                    x_borde = minutos[idx_borde_ini:idx_borde_fin]
                    y_borde = temps_array[idx_borde_ini:idx_borde_fin]

                    mask_valid_borde = ~np.isnan(y_borde)
                    x_valid = x_borde[mask_valid_borde]
                    y_valid = y_borde[mask_valid_borde]

                    if len(y_valid) >= 5:
                        try:
                            y_suav = lowess(y_valid, x_valid, frac=factor_lowess, return_sorted=False)
                            j = 0
                            for k in range(idx_borde_ini, idx_borde_fin):
                                if not np.isnan(temps_array[k]):
                                    temps_array[k] = y_suav[j]
                                    j += 1
                        except Exception as e:
                            logger.warning("LOWESS error in %s at index %d: %s", clave, borde, e)

                i += ventana_estudio
            else:
                i += 1

        serie['result'] = temps_array
        muestras_corregidas[clave] = {'serie': serie, 'ovul': entrada['ovul']}

    logger.info("corregir_series applied to %d cycles", len(muestras_corregidas))
    return muestras_corregidas


def smooth_cycles(muestras_ovul: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Run corregir_series **twice** as done in notebook 1.5:
        muestras_corregidas = corregir_series(muestras_ovul)
        muestras_corregidas = corregir_series(muestras_corregidas)
    """
    out = corregir_series(muestras_ovul)
    out = corregir_series(out)
    logger.info("Two-pass smoothing complete")
    return out


# ===========================================================================
# Step 1.6 – Adaptive imputation
# ===========================================================================

def imputar_faltantes_adaptativo(
    muestras_corregidas: Dict[str, Dict],
    ventana_contexto_base: int = 80,
    suavizado_bordes_base: int = 0,
    factor_lowess: float = 0.3,
    proportiontocut_base: float = 0.2,
    max_proportiontocut: float = 0.4,
    min_valores_contexto: int = 3,
) -> Dict[str, Dict]:
    """
    Notebook 1.6 – imputar_faltantes_adaptativo (exact translation).

    For each NaN segment, estimate its value from trimmed means of the
    surrounding context. Context window and trim fraction scale with the
    length of the gap. Falls back to the global median for isolated segments.
    """
    imputadas: Dict[str, Dict] = {}

    for clave, entrada in muestras_corregidas.items():
        serie  = entrada['serie'].copy()
        temps  = serie['result'].copy()
        times  = serie['resultTimestamp']

        minutos     = ((times - times.iloc[0]) / pd.Timedelta(minutes=1)).values
        temps_array = temps.values.astype(float)

        isnan   = np.isnan(temps_array)
        cambios = np.diff(isnan.astype(int))
        inicios = np.where(cambios == 1)[0] + 1
        finales = np.where(cambios == -1)[0] + 1

        if isnan[0]:
            inicios = np.insert(inicios, 0, 0)
        if isnan[-1]:
            finales = np.append(finales, len(temps_array))

        mediana_global = np.nanmedian(temps_array)

        for inicio, fin in zip(inicios, finales):
            largo_tramo = fin - inicio
            factor      = min(2.5, max(1.0, largo_tramo / 10))

            ventana_contexto  = int((ventana_contexto_base * factor) // 2)
            suavizado_bordes  = int((suavizado_bordes_base  * factor) // 2)
            proportiontocut   = min(max_proportiontocut, proportiontocut_base * factor)

            t_ini = minutos[inicio]
            t_fin = minutos[fin - 1]

            t_pre_ini  = t_ini - ventana_contexto
            t_post_fin = t_fin + ventana_contexto

            idx_antes   = (minutos >= t_pre_ini)  & (minutos < t_ini)  & (~np.isnan(temps_array))
            idx_despues = (minutos > t_fin)        & (minutos <= t_post_fin) & (~np.isnan(temps_array))

            if np.sum(idx_antes) >= min_valores_contexto and np.sum(idx_despues) >= min_valores_contexto:
                media_antes   = trim_mean(temps_array[idx_antes],   proportiontocut=proportiontocut)
                media_despues = trim_mean(temps_array[idx_despues], proportiontocut=proportiontocut)
                valor_imputado = (media_antes + media_despues) / 2
            else:
                valor_imputado = mediana_global

            temps_array[inicio:fin] = valor_imputado

            # LOWESS smoothing at borders
            for borde in [inicio, fin - 1]:
                idx_ini = max(0, borde - suavizado_bordes)
                idx_fin = min(len(temps_array), borde + suavizado_bordes)

                x_vals = minutos[idx_ini:idx_fin]
                y_vals = temps_array[idx_ini:idx_fin]

                mask_valid = ~np.isnan(y_vals)
                if np.sum(mask_valid) >= 5:
                    try:
                        y_suav = lowess(
                            y_vals[mask_valid], x_vals[mask_valid],
                            frac=factor_lowess, return_sorted=False,
                        )
                        j = 0
                        for k in range(idx_ini, idx_fin):
                            if not np.isnan(temps_array[k]):
                                temps_array[k] = y_suav[j]
                                j += 1
                    except Exception as e:
                        logger.warning("LOWESS failed in %s, gap %d-%d: %s", clave, inicio, fin, e)

        serie['result'] = temps_array
        imputadas[clave] = {'serie': serie, 'ovul': entrada['ovul']}

    logger.info("Adaptive imputation applied to %d cycles", len(imputadas))
    return imputadas


def impute_cycles(muestras_suaviz: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Notebook 1.6 full sequence:
    1. imputar_faltantes_adaptativo()
    2. corregir_series() – second smoothing pass after imputation
       (defaults: ampliar_correccion=0, suavizado_lateral=20)

    Saves as muestras_ovul_imput.pkl in the original pipeline.
    """
    imputadas = imputar_faltantes_adaptativo(muestras_suaviz)
    imputadas = corregir_series(imputadas, ampliar_correccion=0, suavizado_lateral=20)
    logger.info("Imputation + post-smoothing complete")
    return imputadas


# ===========================================================================
# Step 2.1 – Hourly means
# ===========================================================================

def compute_hourly_means(muestras_ovul: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Notebook 2.2 – Calcular medias horarias.

    Floors resultTimestamp to the hour and computes mean per hour.
    Output saved as muestras_ovul_horas.pkl.
    """
    muestras_ovul_horas: Dict[str, Dict] = {}

    for key, data in muestras_ovul.items():
        df = data['serie'].copy()
        df['resultTimestamp'] = df['resultTimestamp'].dt.floor('h')
        df_horas = df.groupby('resultTimestamp', as_index=False)['result'].mean()
        muestras_ovul_horas[key] = {'serie': df_horas, 'ovul': data['ovul']}

    logger.info("Hourly means computed for %d cycles", len(muestras_ovul_horas))
    return muestras_ovul_horas


# ===========================================================================
# Step 2.2 – Normalisation norm1: shift each series mean to 36.5 °C
# ===========================================================================

def normalise_cycles_norm1(muestras_ovul_horas: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Notebook 2.2 – Normalisation "norm1" (exact original code):

        df["result"] = df["result"] - mean + 36.5

    Centres each series around 36.5 °C by removing the personal mean offset.
    Saved as muestras_ovul_horas_norm1.pkl.
    """
    muestras_norm1: Dict[str, Dict] = {}

    for id_, datos in muestras_ovul_horas.items():
        df   = datos['serie'].copy()
        mean = df['result'].mean()
        df['result'] = df['result'] - mean + 36.5
        muestras_norm1[id_] = {'serie': df, 'ovul': datos['ovul']}

    logger.info("Norm1 (mean → 36.5 °C) applied to %d cycles", len(muestras_norm1))
    return muestras_norm1


# ===========================================================================
# I/O helpers
# ===========================================================================

def save_pickle(obj, path: Path) -> None:
    """Serialize obj to path using pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as fh:
        pickle.dump(obj, fh)
    logger.info("Saved: %s", path)


def load_pickle(path: Path):
    """Deserialise and return the object at path."""
    with open(path, 'rb') as fh:
        return pickle.load(fh)


# ===========================================================================
# End-to-end pipeline
# ===========================================================================

def run_preprocessing_pipeline(
    raw_temp_csv: Path,
    med_notes_csv: Path,
    output_dir: Path,
) -> Dict[str, Dict]:
    """
    Execute the full preprocessing pipeline and persist all intermediate
    pickle artefacts.

    Artefacts written
    -----------------
    tempNotes.csv
    tempNotes_filt.csv
    ovulNotes.csv
    muestras_ovul_filt.pkl   ← 1.3
    muestras_ovul_filt2.pkl  ← 1.4
    muestras_ovul_suaviz.pkl ← 1.5 (corregir x2)
    muestras_ovul_imput.pkl  ← 1.6 (impute + corregir)
    muestras_ovul_horas.pkl  ← 2.1
    muestras_ovul_horas_norm1.pkl ← 2.2

    Returns
    -------
    muestras_ovul_horas_norm1 dict (input to dataset.py)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1.1 – Load & filter temperature
    temp_raw  = load_temperature_data(raw_temp_csv)
    temp_raw.to_csv(output_dir / 'tempNotes.csv', sep=';', index=False)
    temp_filt = filter_temperature_range(temp_raw)
    temp_filt.to_csv(output_dir / 'tempNotes_filt.csv', sep=';', index=False)

    # 1.2 – Ovulation notes
    ovul_notes = load_ovulation_notes(med_notes_csv)
    ovul_notes.to_csv(output_dir / 'ovulNotes.csv', sep=';', index=False)

    # 1.3 – Segment cycles
    muestras_filt = segment_cycles(temp_filt, ovul_notes)
    save_pickle(muestras_filt, output_dir / 'muestras_ovul_filt.pkl')

    # 1.4 – Filter incomplete
    muestras_filt2 = filter_incomplete_cycles(muestras_filt)
    save_pickle(muestras_filt2, output_dir / 'muestras_ovul_filt2.pkl')

    # 1.5 – Smooth (corregir_series x2)
    muestras_suaviz = smooth_cycles(muestras_filt2)
    save_pickle(muestras_suaviz, output_dir / 'muestras_ovul_suaviz.pkl')

    # 1.6 – Impute + second smooth
    muestras_imput = impute_cycles(muestras_suaviz)

    # 2.1 – Hourly means
    muestras_horas = compute_hourly_means(muestras_imput)

    # 2.2 – Normalise (norm1)
    muestras_norm1 = normalise_cycles_norm1(muestras_horas)

    ids_series_repetidas = ["AN011_3","TU074_3"]
    for id in ids_series_repetidas:
        del muestras_imput[id]
        del muestras_norm1[id]
        del muestras_horas[id]
    
    save_pickle(muestras_horas, output_dir / 'muestras_ovul_horas.pkl')
    save_pickle(muestras_imput, output_dir / 'muestras_ovul_imput.pkl')
    save_pickle(muestras_norm1, output_dir / 'muestras_ovul_horas_norm1.pkl')

    logger.info("Preprocessing pipeline complete. Artefacts in %s", output_dir)
    return muestras_norm1

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")

    parser.add_argument(
        "--rawtemp",
        type=str,
        required=True,
        help="Path to raw temperature CSV"
    )
    parser.add_argument(
        "--mednotes",
        type=str,
        required=True,
        help="Path to medical/ovulation notes CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory"
    )

    args = parser.parse_args()

    run_preprocessing_pipeline(
        raw_temp_csv=Path(args.rawtemp),
        med_notes_csv=Path(args.mednotes),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()