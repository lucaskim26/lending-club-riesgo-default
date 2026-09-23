# Modelo de riesgo de default crediticio — Lending Club

**Ciencia de Datos Aplicada · ITBA**
Gianluca Giannine Lizarraga · Timoteo Harrington · Lucas Kim

Sistema de decisión para el otorgamiento de préstamos, basado en datos históricos de Lending Club (plataforma estadounidense de préstamos peer-to-peer). El objetivo es estimar la probabilidad de default (PD) de un solicitante a partir de información disponible al momento de la solicitud, traducirla en bandas de riesgo, y estimar una pérdida esperada como porcentaje del monto prestado (PD × severidad media dado default) para apoyar la decisión de otorgamiento. Esa pérdida corresponde a toda la vida del préstamo; su conversión a una prima anual se aborda en E04.

## Estado del proyecto

| Entrega | Contenido | Estado |
|---|---|---|
| E01 — Propuesta de proyecto | Problema de negocio, objetivos, alcance | ✅ Entregado |
| E02 — Recopilación y preparación de datos | Dataset, calidad de datos, leakage, split temporal, EDA, LGD, transformaciones | ✅ Este repo |
| E03 — Modelado de la solución | Baseline, Random Forest, LightGBM/XGBoost, evaluación, bandas | Pendiente |
| E04 — Despliegue y presentación | Pricing, informe final | Pendiente |

## Estructura del repositorio

```
├── data/                    # Datos (no versionados; se generan con los pasos de abajo)
├── figures/                 # Gráficos generados por el notebook
├── docs/
│   └── propuesta_e03.md     # Protocolo de evaluación y decisiones para debatir
├── notebooks/
│   └── 02_recopilacion_preparacion_datos.ipynb   # Entregable 02
├── scripts/
│   └── download_data.py     # Descarga el dataset de Kaggle
├── src/
│   ├── datos.py             # Carga del dataset y filtro de cosechas maduras
│   ├── features.py          # Clasificación de variables, transformaciones, LGD
│   ├── evaluacion.py        # Métricas de PD y diagnóstico por bandas
│   └── graficos.py          # Estilo de los gráficos
├── tests/                   # Pruebas con datos sintéticos, sin Kaggle
├── requirements.txt
└── README.md
```

## Cómo leer el notebook sin ejecutarlo

El notebook conserva las tablas y gráficos de la ejecución original y se puede leer directamente en GitHub sin instalar nada ni descargar los datos. Las dos celdas de código modificadas en esta revisión (diagnóstico de fecha y exportación final) quedan sin salida hasta volver a ejecutarlas; no se presentan resultados nuevos sin ejecutar. Los gráficos también están en `figures/`.

## Cómo reproducir

```bash
git clone https://github.com/lucaskim26/lending-club-riesgo-default.git
cd lending-club-riesgo-default
python -m venv .venv
source .venv/bin/activate   # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Para descargar los datos hace falta un token de la API de Kaggle: en kaggle.com → Settings → API → *Create New Token*, y guardarlo como indica Kaggle (por ejemplo, en `~/.kaggle/access_token`). Después:

```bash
python scripts/download_data.py
jupyter notebook notebooks/02_recopilacion_preparacion_datos.ipynb
```

La primera ejecución del notebook lee el CSV completo por bloques y guarda parquets intermedios en `data/` (tarda unos minutos y usa ~5 GB de RAM en el pico); las siguientes ejecuciones leen los parquets.

Para comprobar el cálculo de severidad y las herramientas de evaluación sin descargar datos:

```bash
python -m unittest discover -s tests -v
```

La exportación final del notebook guarda `train.parquet`, `test.parquet` y `severidad_train.json`. Este último contiene la media de `perdida_sobre_monto` entre defaults de train; se mantiene fija al evaluar test. En validación temporal debe reestimarse usando solo la parte de entrenamiento de cada fold.

## Dataset

[*Lending Club Loan Data*](https://www.kaggle.com/datasets/wordsforthewise/lending-club) en Kaggle: préstamos otorgados por Lending Club entre 2007 y 2018 (2,26 millones de préstamos, 151 variables). La ejecución guardada registra pagos hasta marzo de 2019: el período de originación no es la fecha de actualización de los resultados. La fecha máxima de pago tampoco identifica por sí sola la fecha exacta de extracción del dataset.

## Metodología (resumen)

1. **Target**: `loan_status` → binario (`Charged Off` = 1, `Fully Paid` = 0), excluyendo préstamos en curso.
2. **Cosechas maduras**: excluir los préstamos en curso sesga la tasa de default de las cosechas recientes (censura). Se usan solo cosechas en las que prácticamente todos los préstamos ya terminaron, y solo el plazo de 36 meses (2007-2015; 618 mil préstamos).
3. **Clasificación de variables**: las 151 columnas se clasifican en ex-ante (disponibles al solicitar), ex-post (excluidas por *data leakage*), benchmark de Lending Club (`grade`, `sub_grade`, `int_rate`) y datos de originación, que incluyen las decisiones de Lending Club como `verification_status`.
4. **Split temporal**: train (jun-2007 a ago-2015) y test (sep-dic 2015) por fecha de originación, apenas definida la población y *antes* de todo el EDA. El diagnóstico de calidad y el EDA se hacen solo sobre train. Es una evaluación retrospectiva por cohortes; no simula un entrenamiento en septiembre de 2015, cuando muchos desenlaces de train aún no se conocían.
5. **LGD**: calculado empíricamente sobre la exposición al momento del default, con los préstamos de train en default. Se guarda además la pérdida realizada de cada préstamo (train y test) para validar la prima por banda en la E04.
6. **Transformaciones**: fila a fila en `construir_features`; las que aprenden de los datos (winsorización, imputación, estandarización, one-hot) en un pipeline de scikit-learn ajustado solo con train.
7. **Output final del sistema**: PD × media(LGD × EAD/monto | default), usando directamente la pérdida sobre monto de cada default de train. Multiplicar LGD promedio por EAD/monto promedio no es equivalente en general. El baseline usa una severidad fija por solicitante y expresa pérdida esperada de vida completa; la anualización, tasa base y otros componentes requieren supuestos adicionales.
8. **Evaluación propuesta para E03**: discriminación (ROC-AUC, average precision y KS), calidad de probabilidades (Brier y log loss) y calibración por bandas. `src/evaluacion.py` implementa los cálculos; el entrenamiento, la calibración del modelo y la selección de cortes siguen pendientes. Ver el [protocolo y las decisiones abiertas](docs/propuesta_e03.md).

## Autores

Gianluca Giannine Lizarraga · Timoteo Harrington · Lucas Kim — ITBA, 2026
