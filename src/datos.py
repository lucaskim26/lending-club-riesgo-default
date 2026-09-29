"""
Carga del dataset de Lending Club (Kaggle: wordsforthewise/lending-club).

El archivo original (`accepted_2007_to_2018Q4.csv.gz`, ~2,26M préstamos x 151
columnas) no entra cómodo en memoria en una notebook estándar. Este módulo:

1. `cargar_resumen`: lee solo unas pocas columnas de todos los préstamos, para
   describir el dataset completo y decidir qué cosechas están maduras.
2. `cargar_cosechas_maduras`: lee el archivo completo por bloques, se queda con
   las cosechas maduras (préstamos que ya tuvieron tiempo de terminar) y las
   guarda en parquet para no repetir la lectura.
"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA_RAW = RAIZ / "data" / "accepted_2007_to_2018Q4.csv.gz"
RUTA_MADURAS = RAIZ / "data" / "prestamos_cosechas_maduras.parquet"
RUTA_RESUMEN = RAIZ / "data" / "resumen_todos_los_prestamos.parquet"

# Última cosecha (mes de originación) en la que prácticamente todos los
# préstamos ya terminaron en el snapshot observado. El nombre del archivo
# delimita las originaciones (2007-2018), no la fecha exacta de actualización.
# Se justifica con la tabla de % de préstamos resueltos por cosecha y plazo
# (notebook 02, Sección 2.2).
ULTIMA_COSECHA_MADURA = {
    36: pd.Timestamp("2015-12-01"),
    60: pd.Timestamp("2013-12-01"),
}


def _parsear_fecha(serie):
    return pd.to_datetime(serie, format="%b-%Y", errors="coerce")


def _plazo_meses(serie):
    return serie.str.extract(r"(\d+)", expand=False).astype(float)


def _es_cosecha_madura(df):
    fecha = _parsear_fecha(df["issue_d"])
    plazo = _plazo_meses(df["term"])
    return ((plazo == 36) & (fecha <= ULTIMA_COSECHA_MADURA[36])) | (
        (plazo == 60) & (fecha <= ULTIMA_COSECHA_MADURA[60])
    )


def cargar_resumen(ruta=RUTA_RAW, destino=RUTA_RESUMEN, forzar=False):
    """Todos los préstamos, solo las columnas necesarias para describir el dataset."""
    if destino.exists() and not forzar:
        return pd.read_parquet(destino)
    df = pd.read_csv(
        ruta,
        usecols=["id", "issue_d", "term", "loan_status", "last_pymnt_d"],
        dtype=str,
    )
    df["issue_d"] = _parsear_fecha(df["issue_d"])
    df["last_pymnt_d"] = _parsear_fecha(df["last_pymnt_d"])
    df["plazo_meses"] = _plazo_meses(df["term"])
    df.to_parquet(destino, index=False)
    return df


def cargar_cosechas_maduras(ruta=RUTA_RAW, destino=RUTA_MADURAS, forzar=False):
    """Préstamos de cosechas maduras con las 151 columnas originales.

    La primera vez lee el CSV completo por bloques (tarda unos minutos) y
    guarda el resultado en parquet; las siguientes lee directamente el parquet.
    Todas las columnas se leen como texto y se convierten a numéricas las que
    lo son, para que los tipos no dependan de cada bloque.
    """
    if destino.exists() and not forzar:
        return pd.read_parquet(destino)

    # Los bloques se guardan como texto de pyarrow (mucho más compacto que el
    # texto de Python), para reducir el uso de memoria al juntarlos.
    bloques = []
    for bloque in pd.read_csv(ruta, dtype=str, chunksize=200_000):
        bloques.append(bloque[_es_cosecha_madura(bloque)].astype("string[pyarrow]"))
    df = pd.concat(bloques, ignore_index=True)
    del bloques

    for col in df.columns:
        texto = df[col].astype(object).where(df[col].notna(), None)
        convertida = pd.to_numeric(texto, errors="coerce")
        # Una columna es numérica si todo valor no vacío se pudo convertir
        if convertida.notna().sum() == texto.notna().sum():
            df[col] = convertida
        else:
            df[col] = texto

    df.to_parquet(destino, index=False)
    return df
