# Propuesta de evaluación para E03 y E04

El objetivo es comprobar si las PD permiten ordenar el riesgo y estimar pérdidas
con probabilidades adecuadas. Las herramientas de este PR preparan esa evaluación;
todavía no hay modelos entrenados ni resultados nuevos sobre Lending Club.

## Qué queda implementado

- `features.estimar_severidad(train)` promedia `perdida_sobre_monto` entre defaults
  de train. Evita sustituir la media del producto por el producto de las medias.
  La exportación de E02 guarda el valor en `data/severidad_train.json`.
- `evaluacion.metricas_pd(y, prob_default)` devuelve ROC-AUC, average precision,
  KS, Brier y log loss. Average precision es la suma ponderada de precisiones
  sobre incrementos de recall, no el área trapezoidal de la curva PR.
- `evaluacion.tabla_por_banda(...)` recibe cortes explícitos y muestra volumen,
  PD media, default observado y, opcionalmente, pérdida observada y prima media.
  Conserva las bandas vacías con métricas indefinidas, sin inventar tasas.
- Pruebas sintéticas verifican el cálculo de severidad, la diferencia entre
  ranking y calibración, los límites de las bandas y el tratamiento de pérdidas.

## Protocolo propuesto

1. Mantener el test de septiembre–diciembre de 2015 reservado. Comparar modelos,
   hiperparámetros y ventanas de entrenamiento dentro de train mediante folds
   ordenados por fecha. Ajustar imputación, escalado, selección supervisada de
   features y severidad exclusivamente con el entrenamiento de cada fold.
2. Comparar regresión logística con regularización y un modelo de boosting.
   Agregar Random Forest si aporta una comparación útil dentro del tiempo del TP.
   Comparar con `sub_grade` sobre los mismos préstamos de evaluación; su AUC de
   train no es la referencia contra la cual comparar el AUC de test del modelo.
3. Evaluar discriminación y calibración. Mostrar la curva PD media vs. default
   observado junto con Brier y log loss; estos dos scores no aíslan por sí solos
   la calibración. Si hace falta un calibrador, ajustarlo en una porción temporal
   de train distinta de aquella usada para entrenar el clasificador. Nunca usar
   test para ajustarlo. Revisar la calibración si se usan pesos de clase.
4. Definir cortes de bandas y umbral de aprobación usando validación y criterios
   de negocio. Congelarlos antes de test. No redefinir las bandas para que sus
   tasas observadas en test resulten monótonas.
5. Evaluar una vez la configuración seleccionada en test y reportar métricas,
   tablas por banda y sensibilidad entre cosechas dentro de train. Para una
   política de aprobación, reportar proporción aprobada, defaults evitados,
   buenos pagadores rechazados y pérdida de los préstamos retenidos.

Las etiquetas finales de originaciones de 2015 contienen información conocida
años después. Este protocolo evalúa cohortes retrospectivamente, no reconstruye
lo que se sabía al otorgar en 2015. Un backtest operativo requiere una fecha de
disponibilidad de cada etiqueta y snapshots históricos, o una ventana conservadora
de maduración justificable. `last_pymnt_d` no es una fecha fiable de conocimiento
del desenlace. Agregar un corte temporal simple no resuelve esta limitación.

## Uso de las herramientas

Ejemplo sintético ejecutable desde la raíz del repositorio. Los cortes son solo
ilustrativos; no constituyen bandas recomendadas para Lending Club.

```python
import numpy as np
import pandas as pd

from src.features import estimar_severidad
from src.evaluacion import metricas_pd, tabla_por_banda

train = pd.DataFrame({
    "default": [0, 1, 1, 0],
    "perdida_sobre_monto": [np.nan, 0.2, 0.6, np.nan],
})
severidad = estimar_severidad(train)  # 0.4, estimada antes de evaluar

y = pd.Series([0, 1, 0, 1])
prob_default = pd.Series([0.05, 0.20, 0.15, 0.60])
perdida = pd.Series([np.nan, 0.3, np.nan, 0.8])
print(metricas_pd(y, prob_default))
print(tabla_por_banda(
    y, prob_default, cortes=[0, 0.1, 0.3, 1],
    perdida_sobre_monto=perdida, severidad=severidad,
))
```

Las Series deben tener índices coincidentes y en el mismo orden. Los arrays se
interpretan por posición. Las bandas incluyen el límite inferior y excluyen el
superior, salvo la última que incluye 1. La proporción es la fracción de préstamos
en la banda; solo se convierte en tasa de aprobación al definir qué bandas aceptar.

`perdida_media_observada` incluye todos los préstamos de la banda: los pagados
aportan cero y los defaults su pérdida/monto. Un faltante en un default es un
error, no un cero. Tanto esa pérdida como `prima_media` son promedios por préstamo.
Para medir pérdida monetaria de cartera, ponderar por monto financiado usando el
mismo criterio en observado y esperado; no comparar un promedio simple con un
cociente de totales ponderado.

## Decisiones para discutir con el equipo

| Tema | Propuesta inicial | Evidencia pendiente |
|---|---|---|
| Ventana de train | Comparar todo train con datos desde 2012/2013 | Estabilidad entre folds y cobertura del buró |
| Calibración | Empezar sin calibrador y comparar si hay desajustes | Curva de calibración, Brier y log loss en validación |
| Bandas y aprobación | Fijar cortes con validación y un costo explícito de errores | Volumen aprobado, pérdida y buenos pagadores rechazados |
| Severidad | Media fija de pérdida/monto dado default | Error por banda/cosecha y sensibilidad a recuperos abiertos |
| Horizonte temporal | Declarar evaluación retrospectiva | Datos necesarios para reconstruir disponibilidad histórica |
| Prima anual (E04) | Mantener por ahora pérdida esperada de vida completa | Supuestos sobre saldo, tiempos de default, recuperos y descuento |

La tasa base, el costo de fondeo y otros componentes de pricing no se estiman con
estas funciones. La falta de variables macroeconómicas tampoco convierte por sí
sola una PD histórica en una PD calibrada *through-the-cycle*.

## Referencias metodológicas

- [Calibración de probabilidades en scikit-learn](https://scikit-learn.org/stable/modules/calibration.html).
- [Definición de average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
- [Validación de sistemas de rating, Comité de Basilea](https://www.bis.org/publ/bcbs_wp14.pdf).
