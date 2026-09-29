"""Evaluación de probabilidades de default y bandas para la Entrega 03.

Estas funciones reciben predicciones ya calculadas; no ajustan modelos,
calibradores, cortes ni políticas de aprobación. Definir esas decisiones con
train/validación y congelarlas antes de evaluar test.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)


def _vector(valores, nombre):
    """Convertir sin alinear índices ni admitir matrices accidentalmente."""
    try:
        vector = np.asarray(valores, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{nombre} debe ser un vector numérico.") from exc
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{nombre} debe ser un vector no vacío de una dimensión.")
    return vector


def _validar_predicciones(y, prob_default, perdida_sobre_monto=None):
    entradas = [("y", y), ("prob_default", prob_default)]
    if perdida_sobre_monto is not None:
        entradas.append(("perdida_sobre_monto", perdida_sobre_monto))

    series = [valor for _, valor in entradas if isinstance(valor, pd.Series)]
    if series and any(not s.index.equals(series[0].index) for s in series[1:]):
        raise ValueError("Las Series deben tener el mismo índice en el mismo orden.")

    vectores = [_vector(valor, nombre) for nombre, valor in entradas]
    if any(len(v) != len(vectores[0]) for v in vectores[1:]):
        raise ValueError("Todos los vectores deben tener la misma longitud.")
    target, prob = vectores[:2]
    if not np.isin(target, [0, 1]).all():
        raise ValueError("y debe contener solo 0 (pagado) y 1 (default).")
    if not np.isfinite(prob).all() or ((prob < 0) | (prob > 1)).any():
        raise ValueError("prob_default debe contener probabilidades finitas en [0, 1].")

    perdida = None
    if perdida_sobre_monto is not None:
        perdida = vectores[2].copy()
        pagados = target == 0
        # calcular_perdida devuelve NaN para pagados; la pérdida de principal
        # de esos préstamos es cero al promediar sobre toda la banda.
        perdida[pagados & np.isnan(perdida)] = 0
        if not np.isfinite(perdida).all() or ((perdida < 0) | (perdida > 1)).any():
            raise ValueError(
                "perdida_sobre_monto debe estar en [0, 1]; "
                "solo se admite NaN en préstamos pagados."
            )
        if (perdida[pagados] != 0).any():
            raise ValueError("La pérdida de principal de los préstamos pagados debe ser cero.")
    return target, prob, perdida


def metricas_pd(y, prob_default):
    """Métricas de discriminación y calidad probabilística, sin umbral.

    ``y`` es binario (default=1) y ``prob_default`` es su probabilidad, en [0, 1].
    Requiere ambas clases porque ROC-AUC y KS no se definen con una sola.
    Las Series deben compartir índice y orden; los arrays/listas se interpretan
    por posición. No se realiza alineación automática de pandas.

    Devuelve ``roc_auc``, ``average_precision``, ``ks``, ``brier`` y ``log_loss``.
    Average precision es la suma de precisiones ponderadas por incrementos de
    recall, no el área trapezoidal de la curva precisión-recall. KS es bilateral
    (máxima separación absoluta entre distribuciones); leerlo junto con ROC-AUC
    porque también puede ser alto con scores invertidos. Brier y log loss son
    reglas de puntuación propias: menores valores indican mejores probabilidades,
    pero no sustituyen la inspección de calibración por bandas. Log loss usa el
    recorte numérico de scikit-learn en las probabilidades exactamente 0 o 1.
    """
    target, prob, _ = _validar_predicciones(y, prob_default)
    if np.unique(target).size != 2:
        raise ValueError("metricas_pd requiere préstamos pagados y en default.")
    fpr, tpr, _ = roc_curve(target, prob)
    return {
        "roc_auc": float(roc_auc_score(target, prob)),
        "average_precision": float(average_precision_score(target, prob)),
        "ks": float(np.max(np.abs(tpr - fpr))),
        "brier": float(brier_score_loss(target, prob)),
        "log_loss": float(log_loss(target, prob, labels=[0, 1])),
    }


def tabla_por_banda(
    y, prob_default, cortes, *, perdida_sobre_monto=None, severidad=None
):
    """Resumir bandas definidas fuera del conjunto que se evalúa.

    ``cortes`` debe ser una secuencia estrictamente creciente de 0 a 1, incluidos
    ambos extremos. Cada banda incluye su límite inferior y excluye el superior,
    salvo la última, que incluye 1. Se conservan las bandas vacías: cantidad y
    proporción son cero; las medias y tasas son NaN. No se calculan cuantiles.

    ``perdida_sobre_monto`` es la pérdida realizada por préstamo como fracción
    del monto financiado (la columna homónima de ``calcular_perdida``). NaN en
    pagados se convierte a cero; un default sin pérdida observada es un error.
    ``severidad`` es un escalar en [0, 1], estimado solo con defaults de train
    como el promedio de esa fracción. Si se informa, prima_media = PD media ×
    severidad. Es una pérdida esperada de toda la vida del préstamo; no una tasa
    anual. Esta función no puede verificar de qué muestra proviene el escalar.

    Las proporciones y medias ponderan cada préstamo por igual, no por capital.
    La proporción de una banda no es una tasa de aprobación. Las Series deben
    compartir índice y orden; los arrays/listas se interpretan por posición.
    """
    target, prob, perdida = _validar_predicciones(y, prob_default, perdida_sobre_monto)
    limites = _vector(cortes, "cortes")
    if (
        limites.size < 2
        or not np.isfinite(limites).all()
        or limites[0] != 0
        or limites[-1] != 1
        or (np.diff(limites) <= 0).any()
    ):
        raise ValueError("cortes debe crecer estrictamente desde 0 hasta 1.")

    if severidad is not None:
        try:
            valor = np.asarray(severidad, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("severidad debe ser un escalar finito en [0, 1].") from exc
        if valor.ndim != 0 or not np.isfinite(valor) or not 0 <= valor <= 1:
            raise ValueError("severidad debe ser un escalar finito en [0, 1].")
        severidad = float(valor)

    n_bandas = len(limites) - 1
    # Cortes internos exactos pasan a la banda de la derecha; 1 queda en la última.
    banda = np.minimum(np.searchsorted(limites, prob, side="right") - 1, n_bandas - 1)
    datos = pd.DataFrame({"banda": banda, "pd": prob, "default": target})
    grupos = datos.groupby("banda")
    tabla = grupos.agg(
        cantidad=("pd", "size"),
        pd_media=("pd", "mean"),
        tasa_default_observada=("default", "mean"),
    ).reindex(range(n_bandas))
    tabla["cantidad"] = tabla["cantidad"].fillna(0).astype(int)
    tabla.insert(1, "proporcion", tabla["cantidad"] / len(target))
    tabla.insert(0, "limite_inferior", limites[:-1])
    tabla.insert(1, "limite_superior", limites[1:])
    tabla.insert(2, "incluye_limite_superior", np.arange(n_bandas) == n_bandas - 1)
    if perdida is not None:
        tabla["perdida_media_observada"] = (
            pd.Series(perdida).groupby(banda).mean().reindex(range(n_bandas))
        )
    if severidad is not None:
        tabla["prima_media"] = tabla["pd_media"] * severidad
    return tabla
