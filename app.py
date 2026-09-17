# -*- coding: utf-8 -*-
"""
Despliegue_Clasificacion - app.py

- Cargamos el modelo
- Capturamos los datos del paciente (sidebar)
- Preparamos los datos: dummies + reindex de columnas
- Aplicamos el modelo para la predicción
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st

# ---------------------------------------------------------
# Configuración general de la página
# ---------------------------------------------------------
st.set_page_config(
    page_title="Riesgo Cardíaco",
    page_icon="🫀",
    layout="wide"
)

# ---------------------------------------------------------
# Cargamos el modelo
# ---------------------------------------------------------
filename = 'modelo-cla.pkl'
modelo, labelencoder, variables, min_max_scaler = pickle.load(open(filename, 'rb'))

# ---------------------------------------------------------
# Encabezado
# ---------------------------------------------------------
st.title("🫀 Predicción de riesgo de ataque al corazón (stroke)")
st.caption("Modelo Random Forest entrenado con validación cruzada estratificada")

# ---------------------------------------------------------
# Captura de datos en el sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("Datos del paciente")

    age = st.slider('Edad', min_value=0, max_value=100, value=30, step=1)
    avg_glucose_level = st.number_input(
        'Nivel promedio de glucosa',
        min_value=40.0, max_value=300.0, value=100.0, step=0.1
    )

    st.divider()

    hypertension = st.radio('Hipertensión', ['No', 'Yes'], horizontal=True)
    heart_disease = st.radio('Enfermedad cardíaca', ['No', 'Yes'], horizontal=True)
    ever_married = st.radio('Alguna vez casado/a', ['No', 'Yes'], horizontal=True)

    smoking_status = st.selectbox(
        'Estado de fumador',
        ["'never smoked'", 'Unknown', "'formerly smoked'", 'smokes']
    )

    st.divider()
    predecir = st.button("🔍 Predecir riesgo", use_container_width=True)

# ---------------------------------------------------------
# Dataframe con los datos capturados
# ---------------------------------------------------------
datos = [[age, avg_glucose_level, hypertension, heart_disease, ever_married, smoking_status]]
data = pd.DataFrame(
    datos,
    columns=['age', 'avg_glucose_level', 'hypertension', 'heart_disease', 'ever_married', 'smoking_status']
)

# Mostramos un resumen de lo capturado, en dos columnas
col_izq, col_der = st.columns(2)
with col_izq:
    st.metric("Edad", f"{age} años")
    st.metric("Glucosa promedio", f"{avg_glucose_level:.1f}")
with col_der:
    st.write("**Hipertensión:**", hypertension)
    st.write("**Enfermedad cardíaca:**", heart_disease)
    st.write("**Alguna vez casado/a:**", ever_married)
    st.write("**Fumador:**", smoking_status)

st.divider()

# ---------------------------------------------------------
# Preparación de datos + predicción (solo al presionar el botón)
# ---------------------------------------------------------
if predecir:
    data_preparada = data.copy()

    # En despliegue drop_first se mantiene igual que en el entrenamiento
    data_preparada = pd.get_dummies(
        data_preparada, columns=['smoking_status'], drop_first=False, dtype=int
    )  # 3 o más categorías
    data_preparada = pd.get_dummies(
        data_preparada, columns=['hypertension', 'heart_disease', 'ever_married'], drop_first=True, dtype=int
    )  # 2 categorías

    # Se adicionan las columnas faltantes
    data_preparada = data_preparada.reindex(columns=variables, fill_value=0)

    # El modelo final es Random Forest (basado en árboles), por lo tanto NO requiere normalización
    # data_preparada[['age','avg_glucose_level']] = min_max_scaler.transform(data_preparada[['age','avg_glucose_level']])

    # Predicción
    Y_pred = modelo.predict(data_preparada)
    Y_pred = labelencoder.inverse_transform(Y_pred)  # decodificamos 0/1 -> etiqueta original

    resultado = Y_pred[0]
    data['Prediccion'] = Y_pred

    # Resultado destacado
    if str(resultado).strip().lower() == 'yes':
        st.error(f"⚠️ Riesgo ALTO de stroke (predicción: {resultado})")
    else:
        st.success(f"✅ Riesgo BAJO de stroke (predicción: {resultado})")

    with st.expander("Ver detalle de la predicción"):
        st.dataframe(data, use_container_width=True)

    st.info("El modelo tiene un accuracy aproximado del 85% (validación cruzada)")
else:
    st.write("Completa los datos en el panel izquierdo y presiona **Predecir riesgo**.")

