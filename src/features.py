"""
Clasificación de variables y transformaciones para el modelo de PD.

- Las listas de variables documentan la clasificación ex-ante / ex-post
  justificada en el notebook 02 (Sección 3).
- `construir_features` aplica transformaciones fila a fila, sin parámetros
  aprendidos de los datos: se puede aplicar igual a train y a test.
- `crear_preprocesador` devuelve las transformaciones que sí aprenden de los
  datos (percentiles, medianas, medias, categorías); se ajusta solo con train.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Clasificación de las 151 columnas del dataset
# ---------------------------------------------------------------------------

# Información del solicitante y de su historial en el buró de crédito al
# momento de la solicitud: candidatas a feature del modelo.
EX_ANTE = [
    # Préstamo pedido y datos declarados
    "loan_amnt", "term", "purpose", "title", "desc",
    "emp_title", "emp_length", "home_ownership", "annual_inc",
    "zip_code", "addr_state", "dti",
    "application_type", "annual_inc_joint", "dti_joint", "verification_status_joint",
    # Buró de crédito: historial básico (disponible desde 2007)
    "fico_range_low", "fico_range_high", "earliest_cr_line", "delinq_2yrs",
    "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "collections_12_mths_ex_med", "acc_now_delinq", "chargeoff_within_12_mths",
    "delinq_amnt", "pub_rec_bankruptcies", "tax_liens",
    # Buró de crédito: detalle extendido (reportado desde mediados de 2012)
    "mths_since_last_major_derog", "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim",
    "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy", "bc_util",
    "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl", "mort_acc", "mths_since_recent_bc", "mths_since_recent_bc_dlq",
    "mths_since_recent_inq", "mths_since_recent_revol_delinq", "num_accts_ever_120_pd",
    "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats",
    "num_tl_120dpd_2m", "num_tl_30dpd", "num_tl_90g_dpd_24m", "num_tl_op_past_12m",
    "pct_tl_nvr_dlq", "percent_bc_gt_75", "tot_hi_cred_lim", "total_bal_ex_mort",
    "total_bc_limit", "total_il_high_credit_limit",
    # Buró de crédito: detalle reportado recién desde fines de 2015
    "open_acc_6m", "open_act_il", "open_il_12m", "open_il_24m", "mths_since_rcnt_il",
    "total_bal_il", "il_util", "open_rv_12m", "open_rv_24m", "max_bal_bc", "all_util",
    "inq_fi", "total_cu_tl", "inq_last_12m",
    # Co-solicitante (solicitudes conjuntas, desde 2017)
    "revol_bal_joint", "sec_app_fico_range_low", "sec_app_fico_range_high",
    "sec_app_earliest_cr_line", "sec_app_inq_last_6mths", "sec_app_mort_acc",
    "sec_app_open_acc", "sec_app_revol_util", "sec_app_open_act_il",
    "sec_app_num_rev_accts", "sec_app_chargeoff_within_12_mths",
    "sec_app_collections_12_mths_ex_med", "sec_app_mths_since_last_major_derog",
]

# Output del propio modelo de riesgo de Lending Club: disponibles al otorgar,
# pero no son características del solicitante. Se reservan como benchmark.
BENCHMARK_LC = ["grade", "sub_grade", "int_rate", "installment"]

# Datos de la originación decididos por Lending Club o por los inversores, no
# por el solicitante. No se usan como feature.
# `verification_status` también es una decisión de Lending Club: verificaba el
# ingreso con más frecuencia a los solicitantes que su propio modelo consideraba
# riesgosos, así que en parte es un output de su scoring (ver notebook, 6.4).
ORIGINACION = [
    "issue_d", "funded_amnt", "funded_amnt_inv", "initial_list_status",
    "disbursement_method", "policy_code", "verification_status",
]

# Solo existen después de otorgado el préstamo: usarlas sería data leakage.
# `total_rec_prncp`, `recoveries` y `collection_recovery_fee` se usan para
# calcular el LGD empírico (no como feature).
EX_POST = [
    "pymnt_plan", "out_prncp", "out_prncp_inv", "total_pymnt", "total_pymnt_inv",
    "total_rec_prncp", "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt", "next_pymnt_d",
    "last_credit_pull_d", "last_fico_range_high", "last_fico_range_low",
    "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
    "deferral_term", "hardship_amount", "hardship_start_date", "hardship_end_date",
    "payment_plan_start_date", "hardship_length", "hardship_dpd",
    "hardship_loan_status", "orig_projected_additional_accrued_interest",
    "hardship_payoff_balance_amount", "hardship_last_payment_amount",
    "debt_settlement_flag", "debt_settlement_flag_date", "settlement_status",
    "settlement_date", "settlement_amount", "settlement_percentage", "settlement_term",
]

IDENTIFICADORES = ["id", "member_id", "url"]

TARGET = ["loan_status"]

# ---------------------------------------------------------------------------
# Features del modelo
# ---------------------------------------------------------------------------

# Buró extendido con cobertura completa desde 2013 (ver notebook, Sección 5.2)
BURO_EXTENDIDO = [
    "tot_coll_amt", "tot_cur_bal", "total_rev_hi_lim", "acc_open_past_24mths",
    "avg_cur_bal", "bc_open_to_buy", "bc_util", "mo_sin_old_il_acct",
    "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op", "mo_sin_rcnt_tl", "mort_acc",
    "mths_since_recent_bc", "mths_since_recent_inq", "num_accts_ever_120_pd",
    "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_sats", "num_bc_tl", "num_il_tl",
    "num_op_rev_tl", "num_rev_accts", "num_rev_tl_bal_gt_0", "num_sats",
    "num_tl_90g_dpd_24m", "num_tl_op_past_12m", "pct_tl_nvr_dlq",
    "percent_bc_gt_75", "tot_hi_cred_lim", "total_bal_ex_mort", "total_bc_limit",
    "total_il_high_credit_limit", "num_tl_120dpd_2m", "num_tl_30dpd",
]

# Variables originales del buró básico que pasan sin cambios
BURO_BASICO = [
    "loan_amnt", "delinq_2yrs", "inq_last_6mths", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "pub_rec_bankruptcies", "tax_liens",
    "collections_12_mths_ex_med", "acc_now_delinq", "chargeoff_within_12_mths",
    "delinq_amnt",
]

NUMERICAS_BASE = BURO_BASICO + [
    "dti",
    # Construidas en `construir_features`
    "fico_promedio", "log_annual_inc", "antiguedad_crediticia_anios",
    "emp_length_anios", "prestamo_sobre_ingreso", "revol_bal_sobre_ingreso",
]

BINARIAS = [
    "tuvo_mora_previa", "tuvo_registro_publico", "emp_length_faltante",
    "buro_extendido_disponible",
]

# Variables del buró extendido cuyo faltante, aun con el buró disponible, es
# informativo: sin consultas recientes, sin cuentas en cuotas, sin tarjetas.
# El resto de los faltantes es estructural (período previo a 2012) y lo
# captura `buro_extendido_disponible`.
FALTANTE_INFORMATIVO = ["mths_since_recent_inq", "mo_sin_old_il_acct", "bc_util"]

CATEGORICAS = ["home_ownership", "purpose", "addr_state"]

# Resultado de las reglas de selección aplicadas sobre train (el notebook,
# Sección 8, las recalcula y verifica que coincidan con estas listas):
# - Casi-constantes: > 99% de los valores informados son iguales.
DESCARTE_CASI_CONSTANTES = [
    "num_tl_120dpd_2m", "delinq_amnt", "num_tl_30dpd", "acc_now_delinq",
    "chargeoff_within_12_mths",
]
# - Redundantes: |correlación de Spearman| > 0,9 con otra variable. De cada par
#   (de mayor a menor correlación) se conserva la de mayor cobertura y, si la
#   cobertura difiere en menos de 1 punto, la de mayor AUC univariado.
DESCARTE_REDUNDANTES = ["num_sats", "tot_cur_bal", "num_rev_tl_bal_gt_0"]


def features_modelo():
    """Lista final de features numéricas, binarias y categóricas."""
    descartadas = set(DESCARTE_CASI_CONSTANTES + DESCARTE_REDUNDANTES)
    numericas = [c for c in NUMERICAS_BASE + BURO_EXTENDIDO if c not in descartadas]
    binarias = [c for c in BINARIAS if c not in descartadas]
    return numericas, binarias, CATEGORICAS


# ---------------------------------------------------------------------------
# Transformaciones fila a fila (sin parámetros aprendidos)
# ---------------------------------------------------------------------------

EMP_LENGTH_ANIOS = {
    "< 1 year": 0, "1 year": 1, "2 years": 2, "3 years": 3, "4 years": 4,
    "5 years": 5, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9,
    "10+ years": 10,
}

# Hasta marzo de 2008 Lending Club registraba "nunca tuvo el evento" como 0 en
# las variables "meses desde...", y después como vacío (ver notebook, Sección 5.3).
FIN_CODIFICACION_CERO = pd.Timestamp("2008-03-01")


def construir_features(df):
    """Limpieza y variables derivadas. Devuelve un DataFrame nuevo."""
    out = pd.DataFrame(index=df.index)
    fecha = pd.to_datetime(df["issue_d"], format="%b-%Y")

    # --- Corrección de valores imposibles ---
    # dti = 999 es un valor centinela (sin ingreso verificable), no un ratio real
    dti = df["dti"].where(df["dti"] <= 100)
    ingreso = df["annual_inc"].where(df["annual_inc"] > 0)

    meses_mora = df["mths_since_last_delinq"].copy()
    meses_registro = df["mths_since_last_record"].copy()
    codificacion_vieja = fecha <= FIN_CODIFICACION_CERO
    meses_mora[codificacion_vieja & (meses_mora == 0)] = np.nan
    meses_registro[codificacion_vieja & (meses_registro == 0)] = np.nan

    # --- Variables originales que pasan sin cambios ---
    for col in BURO_BASICO + BURO_EXTENDIDO:
        out[col] = df[col]
    out["dti"] = dti

    # --- Variables derivadas ---
    out["fico_promedio"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    out["log_annual_inc"] = np.log(ingreso)
    inicio_credito = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y")
    out["antiguedad_crediticia_anios"] = (fecha - inicio_credito).dt.days / 365.25
    out["emp_length_anios"] = df["emp_length"].map(EMP_LENGTH_ANIOS)
    out["prestamo_sobre_ingreso"] = df["loan_amnt"] / ingreso
    out["revol_bal_sobre_ingreso"] = df["revol_bal"] / ingreso

    # --- Flags: el faltante es informativo ---
    out["tuvo_mora_previa"] = (meses_mora.notna() | (df["delinq_2yrs"] > 0)).astype(int)
    out["tuvo_registro_publico"] = (meses_registro.notna() | (df["pub_rec"] > 0)).astype(int)
    out["emp_length_faltante"] = df["emp_length"].isna().astype(int)
    out["buro_extendido_disponible"] = df["tot_cur_bal"].notna().astype(int)

    # --- Categóricas ---
    # OTHER / NONE / ANY suman < 0,03% de los préstamos
    out["home_ownership"] = df["home_ownership"].where(
        df["home_ownership"].isin(["RENT", "MORTGAGE", "OWN"]), "OTHER"
    )
    for col in ["purpose", "addr_state"]:
        out[col] = df[col]

    return out


def calcular_perdida(df):
    """Pérdida realizada de los préstamos en default (NaN para los pagados).

    Devuelve tres columnas:
    - ead: exposición al momento del default = monto financiado - capital cobrado.
    - lgd: 1 - recupero neto / EAD, acotado a [0, 1]. El recupero neto son los
      recuperos posteriores al charge-off menos el costo de cobranza.
    - perdida_sobre_monto: (EAD - recupero neto) / monto financiado, acotada a
      [0, 1]. Es la pérdida observada expresada como % del monto prestado, la
      misma unidad que la prima de riesgo.
    """
    en_default = df["loan_status"] == "Charged Off"
    ead = df["funded_amnt"] - df["total_rec_prncp"]
    recupero_neto = df["recoveries"] - df["collection_recovery_fee"]
    return pd.DataFrame({
        "ead": ead,
        "lgd": (1 - recupero_neto / ead.where(ead > 0)).clip(0, 1),
        "perdida_sobre_monto": ((ead - recupero_neto) / df["funded_amnt"]).clip(0, 1),
    }, index=df.index).where(en_default, axis=0)


# ---------------------------------------------------------------------------
# Transformaciones con parámetros aprendidos (se ajustan solo con train)
# ---------------------------------------------------------------------------

class Winsorizador(BaseEstimator, TransformerMixin):
    """Recorta cada columna a los percentiles [inferior, superior] de train."""

    def __init__(self, inferior=0.005, superior=0.995):
        self.inferior = inferior
        self.superior = superior

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.limites_ = X.quantile([self.inferior, self.superior])
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        return X.clip(self.limites_.iloc[0], self.limites_.iloc[1], axis=1)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self.feature_names_in_
        return np.asarray(input_features, dtype=object)


def crear_preprocesador(numericas, binarias, categoricas):
    """Preprocesamiento para modelos lineales (Logistic Regression).

    - Numéricas: winsorización (0,5% / 99,5%), imputación por mediana y
      estandarización. Solo las de `FALTANTE_INFORMATIVO` suman un indicador
      de faltante.
    - Binarias: sin cambios.
    - Categóricas: one-hot; las categorías con < 0,5% de los préstamos se
      agrupan en una sola ("infrequent"), también las no vistas en train.

    Los modelos de árboles (Random Forest, Gradient Boosting) no necesitan
    winsorizar ni estandarizar; en la Entrega 03 se usa solo la parte
    categórica para ellos.
    """
    def pipeline_numericas(con_indicador):
        return Pipeline([
            ("winsorizar", Winsorizador()),
            ("imputar", SimpleImputer(strategy="median", add_indicator=con_indicador)),
            ("estandarizar", StandardScaler()),
        ])

    con_indicador = [c for c in numericas if c in FALTANTE_INFORMATIVO]
    sin_indicador = [c for c in numericas if c not in FALTANTE_INFORMATIVO]
    codificador = OneHotEncoder(
        handle_unknown="infrequent_if_exist", min_frequency=0.005, sparse_output=False
    )
    preprocesador = ColumnTransformer([
        ("num", pipeline_numericas(False), sin_indicador),
        ("num_faltante", pipeline_numericas(True), con_indicador),
        ("bin", "passthrough", binarias),
        ("cat", codificador, categoricas),
    ], verbose_feature_names_out=False)
    return preprocesador.set_output(transform="pandas")
