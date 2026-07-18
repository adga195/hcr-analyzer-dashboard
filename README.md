# HCR Analyzer — Dashboard de Streamlit

Dashboard del PIDA con dos secciones: análisis exploratorio (a partir de métricas ya
agregadas, sin datos a nivel de registro) y predicción de cancelación de reservas con
el modelo ya entrenado.

## Estructura de archivos

```
streamlit_dashboard/
├── app.py
├── requirements.txt
├── model_artifacts/
│   ├── model.joblib
│   ├── power_transformer.joblib
│   ├── scaler.joblib
│   ├── feature_columns.json
│   ├── threshold.json
│   └── form_ranges.json
└── eda_artifacts/
    ├── kpis_globales.json
    ├── grupo_mes_weekend.csv
    ├── boxplot_por_clase.json
    ├── matriz_correlacion.csv
    └── distribucion_dia_llegada.json
```

## Correr en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Desplegar en Streamlit Community Cloud

1. Sube esta carpeta completa (incluyendo `model_artifacts/` y `eda_artifacts/`) a un repositorio de GitHub.
2. Entra a [share.streamlit.io](https://share.streamlit.io), clic en **"New app"**.
3. Selecciona el repositorio y como archivo principal indica `app.py`.
4. Clic en **"Deploy"**.
