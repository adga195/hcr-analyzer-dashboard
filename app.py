import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="HCR Analyzer",
    page_icon="🏨",
    layout="wide"
)

COLUMNAS_NUMERICAS = ['total_amount_before_tax', 'days_booked_prior_to_arrival',
                      'duration_of_stay', 'booking_hour']
COLUMNAS_CATEGORICAS = ['booking_day_of_week', 'arrival_day_of_week', 'arrival_month']

DIAS = {0: 'Lunes', 1: 'Martes', 2: 'Miércoles', 3: 'Jueves',
        4: 'Viernes', 5: 'Sábado', 6: 'Domingo'}
MESES = {1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril', 5: 'Mayo', 6: 'Junio',
         7: 'Julio', 8: 'Agosto', 9: 'Septiembre', 10: 'Octubre',
         11: 'Noviembre', 12: 'Diciembre'}
MESES_ABREV = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun',
               7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}

# EL DATASET DE ENTRENAMIENTO CUBRE ÚNICAMENTE ENERO-JUNIO DE UN AÑO FIJO.
# EL DROPDOWN DE AÑO SE DEJA PREPARADO PARA CUANDO SE INCORPOREN MÁS AÑOS DE DATOS.
ANIOS_DISPONIBLES = [2026]
MESES_DEL_DATASET = list(range(1, 7))  # ENERO A JUNIO

COLOR_TEAL = 'teal'
COLOR_CORAL = 'coral'
COLOR_STEELBLUE = 'steelblue'


# ==============================================================================
# CARGA DE ARTEFACTOS (CACHEADA — NO SE VUELVE A ENTRENAR NADA EN VIVO)
# ==============================================================================
@st.cache_resource
def cargar_artefactos():
    modelo = joblib.load("model_artifacts/model.joblib")
    power_transformer = joblib.load("model_artifacts/power_transformer.joblib")
    scaler = joblib.load("model_artifacts/scaler.joblib")

    with open("model_artifacts/feature_columns.json") as f:
        columnas_modelo = json.load(f)
    with open("model_artifacts/threshold.json") as f:
        info_umbral = json.load(f)
    with open("model_artifacts/form_ranges.json") as f:
        rangos = json.load(f)

    return modelo, power_transformer, scaler, columnas_modelo, info_umbral, rangos


@st.cache_data
def cargar_metricas_eda():
    """Carga únicamente métricas ya agregadas — nunca datos a nivel de registro."""
    with open("eda_artifacts/kpis_globales.json") as f:
        kpis_globales = json.load(f)
    grupo_mes_weekend = pd.read_csv("eda_artifacts/grupo_mes_weekend.csv")
    with open("eda_artifacts/boxplot_por_clase.json") as f:
        boxplot_por_clase = json.load(f)
    matriz_correlacion = pd.read_csv("eda_artifacts/matriz_correlacion.csv", index_col=0)
    with open("eda_artifacts/distribucion_dia_llegada.json") as f:
        distribucion_dia_llegada = json.load(f)
    return kpis_globales, grupo_mes_weekend, boxplot_por_clase, matriz_correlacion, distribucion_dia_llegada


try:
    modelo, power_transformer, scaler, columnas_modelo, info_umbral, rangos = cargar_artefactos()
    kpis_globales, grupo_mes_weekend, boxplot_por_clase, matriz_correlacion, distribucion_dia_llegada = cargar_metricas_eda()
    ARTEFACTOS_OK = True
except FileNotFoundError as e:
    ARTEFACTOS_OK = False
    ERROR_ARTEFACTOS = str(e)


def predecir_cancelacion(monto, anticipacion, duracion, hora_reserva,
                          dia_reserva, dia_llegada, mes_llegada):
    """Arma una fila con las mismas transformaciones del entrenamiento y predice."""
    fila = pd.DataFrame([{
        'total_amount_before_tax': monto,
        'days_booked_prior_to_arrival': anticipacion,
        'duration_of_stay': duracion,
        'booking_hour': hora_reserva,
        'booking_is_weekend': int(dia_reserva >= 5),
        'arrival_is_weekend': int(dia_llegada >= 5),
        'booking_day_of_week': dia_reserva,
        'arrival_day_of_week': dia_llegada,
        'arrival_month': mes_llegada,
    }])

    # TRANSFORMACIÓN YEO-JOHNSON DEL MONTO, CON EL MISMO TRANSFORMADOR AJUSTADO EN ENTRENAMIENTO
    fila['total_amount_before_tax'] = power_transformer.transform(
        fila[['total_amount_before_tax']]
    ).flatten()

    # ONE-HOT ENCODING Y ALINEACIÓN CON LAS COLUMNAS EXACTAS DEL ENTRENAMIENTO
    fila_encoded = pd.get_dummies(
        fila, columns=COLUMNAS_CATEGORICAS, drop_first=True, dtype=int
    )
    fila_encoded = fila_encoded.reindex(columns=columnas_modelo, fill_value=0)

    # ESCALAMIENTO DE LAS VARIABLES NUMÉRICAS CONTINUAS CON EL SCALER DE ENTRENAMIENTO
    fila_encoded[COLUMNAS_NUMERICAS] = scaler.transform(fila_encoded[COLUMNAS_NUMERICAS])

    probabilidad = modelo.predict_proba(fila_encoded)[0, 1]
    return probabilidad


# ==============================================================================
# ENCABEZADO
# ==============================================================================
st.title("🏨 HCR Analyzer")
st.caption("Hotel Channel Reservation Analyzer — Análisis y predicción de cancelaciones · Canal Alfa")

if not ARTEFACTOS_OK:
    st.error(
        "No se encontraron los artefactos necesarios. Verifica que las carpetas "
        "`model_artifacts/` y `eda_artifacts/` estén junto a `app.py`. "
        "Genera ambas ejecutando `train_and_export.py` con tu dataset real.\n\n"
        f"Detalle técnico: {ERROR_ARTEFACTOS}"
    )
    st.stop()

tab_eda, tab_pred = st.tabs(["📊 Análisis Exploratorio", "🔮 Predicción de Cancelación"])

# ==============================================================================
# TAB 1 — ANÁLISIS EXPLORATORIO
# ==============================================================================
with tab_eda:
    st.caption(
        "Esta sección se construye a partir de métricas ya agregadas durante el "
        "entrenamiento (conteos, promedios, cuartiles) — la aplicación no carga "
        "ni almacena reservas individuales."
    )
    st.sidebar.header("Filtros del análisis")

    anio_sel = st.sidebar.selectbox("Año", options=ANIOS_DISPONIBLES)

    # SE RESTRINGE A ENERO-JUNIO, LA VENTANA REAL CUBIERTA POR EL DATASET DE ENTRENAMIENTO,
    # AUNQUE LA TABLA AGREGADA CONTUVIERA ALGÚN OTRO MES
    meses_disponibles = sorted(
        m for m in grupo_mes_weekend['arrival_month'].unique().tolist()
        if m in MESES_DEL_DATASET
    )
    meses_sel = st.sidebar.multiselect(
        "Mes de llegada",
        options=meses_disponibles,
        default=meses_disponibles,
        format_func=lambda m: f"{MESES.get(m, m)} {anio_sel}"
    )

    solo_fin_de_semana = st.sidebar.selectbox(
        "Llegada en fin de semana",
        options=["Todas", "Solo fin de semana", "Solo entre semana"]
    )

    # FILTRADO SOBRE LA TABLA AGREGADA (24 FILAS: MES x FIN DE SEMANA), NO SOBRE REGISTROS
    grupo_filtrado = grupo_mes_weekend[grupo_mes_weekend['arrival_month'].isin(meses_sel)].copy()
    if solo_fin_de_semana == "Solo fin de semana":
        grupo_filtrado = grupo_filtrado[grupo_filtrado['arrival_is_weekend'] == 1]
    elif solo_fin_de_semana == "Solo entre semana":
        grupo_filtrado = grupo_filtrado[grupo_filtrado['arrival_is_weekend'] == 0]

    if grupo_filtrado.empty or grupo_filtrado['n_reservas'].sum() == 0:
        st.warning("No hay datos agregados para los filtros seleccionados.")
    else:
        n_reservas = int(grupo_filtrado['n_reservas'].sum())
        n_canceladas = int(grupo_filtrado['n_canceladas'].sum())
        tasa_cancelacion = n_canceladas / n_reservas

        # PROMEDIOS PONDERADOS POR EL TAMAÑO DE CADA GRUPO — EXACTOS, NO APROXIMADOS,
        # PORQUE EL PROMEDIO SÍ ES UNA OPERACIÓN AGREGABLE ENTRE GRUPOS (A DIFERENCIA DE LA MEDIANA)
        pesos = grupo_filtrado['n_reservas']
        anticipacion_prom = np.average(grupo_filtrado['anticipacion_promedio'], weights=pesos)
        monto_prom = np.average(grupo_filtrado['monto_promedio'], weights=pesos)
        duracion_prom = np.average(grupo_filtrado['duracion_promedio'], weights=pesos)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Reservas analizadas", f"{n_reservas:,}")
        col2.metric("Tasa de cancelación", f"{tasa_cancelacion:.1%}")
        col3.metric("Anticipación promedio", f"{anticipacion_prom:.0f} días")
        col4.metric("Monto promedio (USD)", f"${monto_prom:,.0f}")

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Distribución de la variable objetivo")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(['Completada', 'Cancelada'],
                   [n_reservas - n_canceladas, n_canceladas],
                   color=[COLOR_TEAL, COLOR_CORAL])
            ax.set_ylabel('Cantidad')
            fig.tight_layout()
            st.pyplot(fig)

        with c2:
            st.subheader(f"Proporción de cancelación por mes de llegada ({anio_sel})")
            por_mes = grupo_filtrado.groupby('arrival_month').agg(
                n_reservas=('n_reservas', 'sum'),
                n_canceladas=('n_canceladas', 'sum')
            ).reset_index()
            por_mes['proporcion_cancelada'] = por_mes['n_canceladas'] / por_mes['n_reservas']
            por_mes['proporcion_completada'] = 1 - por_mes['proporcion_cancelada']
            etiquetas_mes = [f"{MESES_ABREV.get(m, m)}\n{anio_sel}" for m in por_mes['arrival_month']]

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar(etiquetas_mes, por_mes['proporcion_completada'],
                   color=COLOR_TEAL, label='Completada')
            ax.bar(etiquetas_mes, por_mes['proporcion_cancelada'],
                   bottom=por_mes['proporcion_completada'], color=COLOR_CORAL, label='Cancelada')
            ax.set_xlabel('Mes de llegada')
            ax.set_ylabel('Proporción')
            ax.legend(loc='lower right')
            fig.tight_layout()
            st.pyplot(fig)

        st.markdown("---")
        st.caption(
            "Los siguientes gráficos reflejan el dataset completo con el que se entrenó "
            "el modelo — no se ven afectados por los filtros de la barra lateral, ya que "
            "se precalcularon una sola vez durante el entrenamiento."
        )

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("Anticipación de reserva por clase")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bxp(boxplot_por_clase['days_booked_prior_to_arrival'], patch_artist=True,
                   boxprops=dict(facecolor=COLOR_STEELBLUE, alpha=0.6))
            ax.set_ylabel('Días de anticipación')
            fig.tight_layout()
            st.pyplot(fig)

        with c4:
            st.subheader("Monto de reserva por clase")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bxp(boxplot_por_clase['total_amount_before_tax'], patch_artist=True,
                   boxprops=dict(facecolor=COLOR_STEELBLUE, alpha=0.6))
            ax.set_ylabel('Monto antes de impuestos (USD)')
            fig.tight_layout()
            st.pyplot(fig)

        c5, c6 = st.columns(2)
        with c5:
            st.subheader("Correlación entre variables numéricas")
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(matriz_correlacion, annot=True, cmap='Blues', fmt='.2f', ax=ax)
            fig.tight_layout()
            st.pyplot(fig)

        with c6:
            st.subheader("Llegadas por día de la semana")
            dias_ordenados = sorted(distribucion_dia_llegada.keys(), key=int)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.bar([DIAS[int(d)] for d in dias_ordenados],
                   [distribucion_dia_llegada[d] for d in dias_ordenados],
                   color=COLOR_STEELBLUE)
            ax.set_ylabel('Cantidad de reservas')
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            fig.tight_layout()
            st.pyplot(fig)

        with st.expander("⚠️ Nota sobre limitaciones del dataset"):
            st.markdown(
                "El dataset cubre únicamente reservas de un periodo acotado. Las reservas con "
                "fecha de llegada cercana al borde de esa ventana solo pueden tener desenlace "
                "de *cancelada* (ya que aún no ha ocurrido su check-in), lo que sobre-representa "
                "la cancelación en los meses de llegada más recientes. Interpretar el patrón "
                "estacional con esta consideración en mente."
            )

# ==============================================================================
# TAB 2 — PREDICCIÓN DE CANCELACIÓN
# ==============================================================================
with tab_pred:
    st.subheader("Simulador de riesgo de cancelación")
    st.write(
        f"Modelo: **{info_umbral['modelo_ganador']}** · "
        f"ROC-AUC en prueba: **{info_umbral['roc_auc_test']:.3f}** · "
        f"Umbral de decisión: **{info_umbral['best_threshold']:.2f}**"
    )

    with st.form("formulario_prediccion"):
        col_a, col_b = st.columns(2)

        with col_a:
            monto = st.number_input(
                "Monto de la reserva antes de impuestos (USD)",
                min_value=0.0,
                max_value=float(rangos['total_amount_before_tax']['max']) * 2,
                value=float(rangos['total_amount_before_tax']['default']),
                step=10.0
            )
            anticipacion = st.number_input(
                "Días de anticipación (creación → llegada)",
                min_value=0,
                max_value=int(rangos['days_booked_prior_to_arrival']['max']) * 2,
                value=int(rangos['days_booked_prior_to_arrival']['default']),
                step=1
            )
            duracion = st.number_input(
                "Duración de la estancia (noches)",
                min_value=0,
                max_value=int(rangos['duration_of_stay']['max']) * 3,
                value=int(rangos['duration_of_stay']['default']),
                step=1
            )
            hora_reserva = st.slider(
                "Hora del día en que se realiza la reserva", 0, 23, 12
            )

        with col_b:
            dia_reserva = st.selectbox(
                "Día de la semana en que se realiza la reserva",
                options=list(DIAS.keys()), format_func=lambda d: DIAS[d]
            )
            dia_llegada = st.selectbox(
                "Día de la semana de llegada (check-in)",
                options=list(DIAS.keys()), format_func=lambda d: DIAS[d]
            )
            mes_llegada = st.selectbox(
                "Mes de llegada",
                options=list(MESES.keys()), format_func=lambda m: MESES[m]
            )

        enviado = st.form_submit_button("Calcular probabilidad de cancelación", type="primary")

    if enviado:
        probabilidad = predecir_cancelacion(
            monto, anticipacion, duracion, hora_reserva,
            dia_reserva, dia_llegada, mes_llegada
        )
        umbral = info_umbral['best_threshold']
        es_riesgo_alto = probabilidad >= umbral

        st.markdown("---")
        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            st.metric("Probabilidad de cancelación", f"{probabilidad:.1%}")
            if es_riesgo_alto:
                st.error(f"⚠️ Riesgo alto de cancelación (umbral = {umbral:.2f})")
            else:
                st.success(f"✅ Riesgo bajo de cancelación (umbral = {umbral:.2f})")

        with col_res2:
            fig, ax = plt.subplots(figsize=(6, 1.2))
            color_barra = COLOR_CORAL if es_riesgo_alto else COLOR_TEAL
            ax.barh([0], [probabilidad], color=color_barra, height=0.5)
            ax.barh([0], [1], color='lightgray', height=0.5, zorder=0)
            ax.barh([0], [probabilidad], color=color_barra, height=0.5, zorder=1)
            ax.axvline(umbral, color='gray', linestyle='--', linewidth=1.5)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel('Probabilidad de cancelación')
            fig.tight_layout()
            st.pyplot(fig)

        if es_riesgo_alto:
            st.info(
                "**Acción recomendada:** considerar contacto proactivo con el huésped, "
                "confirmar la reserva con anticipación, o evaluar políticas de overbooking "
                "moderado para esta franja de riesgo."
            )
        else:
            st.info("**Acción recomendada:** ninguna acción especial requerida; riesgo dentro de lo esperado.")

st.sidebar.markdown("---")
st.sidebar.caption("HCR Analyzer — Proyecto Integrador de Dominio Autónomo (PIDA)")
