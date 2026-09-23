# Modelo de riesgo de default crediticio — Lending Club

**Ciencia de Datos Aplicada · ITBA**
Gianluca Giannine Lizarraga · Timoteo Harrington · Lucas Kim

Sistema de decisión para el otorgamiento de préstamos, basado en datos históricos de Lending Club (plataforma estadounidense de préstamos peer-to-peer). El objetivo es estimar la probabilidad de default (PD) de un solicitante a partir de información disponible al momento de la solicitud, traducirla en bandas de riesgo, y estimar una prima de riesgo de default (PD × LGD) para apoyar la decisión de otorgamiento.

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
├── notebooks/
│   └── 02_recopilacion_preparacion_datos.ipynb   # Entregable 02
├── scripts/
│   └── download_data.py     # Descarga el dataset de Kaggle
├── src/
│   ├── datos.py             # Carga del dataset y filtro de cosechas maduras
│   ├── features.py          # Clasificación de variables, transformaciones, LGD
│   └── graficos.py          # Estilo de los gráficos
├── requirements.txt
└── README.md
```

## Cómo leer el notebook sin ejecutarlo

El notebook está guardado con todos sus resultados (tablas y gráficos), así que se puede leer directamente en GitHub sin instalar nada ni descargar los datos. Los gráficos también están en `figures/`.

## Cómo reproducir

```bash
git clone <url-del-repo>
cd lending-club-riesgo-default
pip install -r requirements.txt
```

Para descargar los datos hace falta un token de la API de Kaggle: en kaggle.com → Settings → API → *Create New Token*, y guardarlo como indica Kaggle (por ejemplo, en `~/.kaggle/access_token`). Después:

```bash
python scripts/download_data.py
jupyter notebook notebooks/02_recopilacion_preparacion_datos.ipynb
```

La primera ejecución del notebook lee el CSV completo por bloques y guarda parquets intermedios en `data/` (tarda unos minutos y usa ~5 GB de RAM en el pico); las siguientes ejecuciones leen los parquets.

## Dataset

[*Lending Club Loan Data*](https://www.kaggle.com/datasets/wordsforthewise/lending-club) en Kaggle: todos los préstamos otorgados por Lending Club entre 2007 y 2018 (2,26 millones de préstamos, 151 variables), con fecha de corte a fines de 2018.

## Metodología (resumen)

1. **Target**: `loan_status` → binario (`Charged Off` = 1, `Fully Paid` = 0), excluyendo préstamos en curso.
2. **Cosechas maduras**: excluir los préstamos en curso sesga la tasa de default de las cosechas recientes (censura). Se usan solo cosechas en las que prácticamente todos los préstamos ya terminaron, y solo el plazo de 36 meses (2007-2015; 618 mil préstamos).
3. **Clasificación de variables**: las 151 columnas se clasifican en ex-ante (disponibles al solicitar), ex-post (excluidas por *data leakage*), benchmark de Lending Club (`grade`, `sub_grade`, `int_rate`) y datos de originación, que incluyen las decisiones de Lending Club como `verification_status`.
4. **Split temporal**: train (jun-2007 a ago-2015) y test (sep-dic 2015) por fecha de originación, apenas definida la población y *antes* de todo el EDA. El diagnóstico de calidad y el EDA se hacen solo sobre train.
5. **LGD**: calculado empíricamente sobre la exposición al momento del default, con los préstamos de train en default. Se guarda además la pérdida realizada de cada préstamo (train y test) para validar la prima por banda en la E04.
6. **Transformaciones**: fila a fila en `construir_features`; las que aprenden de los datos (winsorización, imputación, estandarización, one-hot) en un pipeline de scikit-learn ajustado solo con train.
7. **Output final del sistema**: una prima de riesgo de default (PD × LGD × EAD/monto) por solicitante, no una tasa de interés final. La tasa base y el resto de la estructura de tasa (plazo, iliquidez) quedan fuera del alcance de este TP.

## Autores

Gianluca Giannine Lizarraga · Timoteo Harrington · Lucas Kim — ITBA, 2026
