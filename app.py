# -*- coding: utf-8 -*-
"""
Despliegue_Clasificacion - app.py
 
- Cargamos el modelo
- Capturamos los datos del paciente (individual, en el sidebar) o cargamos un dataset (varios pacientes)
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
 
COLUMNAS_ESPERADAS = ['age', 'avg_glucose_level', 'hypertension', 'heart_disease', 'ever_married', 'smoking_status']
 
 
def preparar_y_predecir(data: pd.DataFrame) -> pd.DataFrame:
    """Recibe un dataframe con las columnas originales y devuelve el mismo
    dataframe con la columna 'Prediccion' agregada."""
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
 
    Y_pred = modelo.predict(data_preparada)
    Y_pred = labelencoder.inverse_transform(Y_pred)  # decodificamos 0/1 -> etiqueta original
 
    resultado = data.copy()
    resultado['Prediccion'] = Y_pred
    return resultado
 
 
# ---------------------------------------------------------
# Encabezado
# ---------------------------------------------------------
st.title("🫀 Predicción de riesgo de ataque al corazón (stroke)")
st.caption("Modelo Random Forest entrenado con validación cruzada estratificada")
 
tab_individual, tab_dataset = st.tabs(["🧍 Predicción individual", "📁 Cargar dataset"])
 
# ===========================================================
# TAB 1: Predicción individual (captura manual en el sidebar)
# ===========================================================
with tab_individual:
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
 
    datos = [[age, avg_glucose_level, hypertension, heart_disease, ever_married, smoking_status]]
    data = pd.DataFrame(datos, columns=COLUMNAS_ESPERADAS)
 
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
 
    if predecir:
        resultado = preparar_y_predecir(data)
        prediccion = resultado.loc[0, 'Prediccion']
 
        if str(prediccion).strip().lower() == 'yes':
            st.error(f"⚠️ Riesgo ALTO de stroke (predicción: {prediccion})")
        else:
            st.success(f"✅ Riesgo BAJO de stroke (predicción: {prediccion})")
 
        with st.expander("Ver detalle de la predicción"):
            st.dataframe(resultado, use_container_width=True)
 
        st.info("El modelo tiene un accuracy aproximado del 85% (validación cruzada)")
    else:
        st.write("Completa los datos en el panel izquierdo y presiona **Predecir riesgo**.")
 
# ===========================================================
# TAB 2: Cargar dataset (predicción de varios pacientes a la vez)
# ===========================================================
with tab_dataset:
    st.subheader("Cargar un archivo con varios pacientes")
    st.write(
        "El archivo debe tener exactamente estas columnas: "
        f"`{', '.join(COLUMNAS_ESPERADAS)}`"
    )
 
    archivo = st.file_uploader(
        "Sube un archivo CSV o Excel",
        type=["csv", "xlsx", "xls"]
    )
 
    if archivo is not None:
        try:
            if archivo.name.endswith(".csv"):
                data_subida = pd.read_csv(archivo)
            else:
                data_subida = pd.read_excel(archivo)
 
            st.write("Vista previa de los datos cargados:")
            st.dataframe(data_subida.head(), use_container_width=True)
 
            columnas_faltantes = [c for c in COLUMNAS_ESPERADAS if c not in data_subida.columns]
 
            if columnas_faltantes:
                st.error(
                    "Al archivo le faltan estas columnas: "
                    f"{', '.join(columnas_faltantes)}"
                )
            else:
                if st.button("🔍 Predecir para todo el dataset", use_container_width=True):
                    resultado = preparar_y_predecir(data_subida[COLUMNAS_ESPERADAS])
 
                    st.success(f"Se generaron predicciones para {len(resultado)} registros.")
                    st.dataframe(resultado, use_container_width=True)
 
                    conteo = resultado['Prediccion'].value_counts()
                    st.bar_chart(conteo)
 
                    csv_resultado = resultado.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "⬇️ Descargar resultados en CSV",
                        data=csv_resultado,
                        file_name="predicciones_stroke.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
    else:
        st.info("Aún no has cargado ningún archivo.")
