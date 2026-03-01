
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Predicción del Peso al Nacer",
    layout="centered"
)

# ===============================
# Cargar modelo
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("modelo_final_regresion_lineal.pkl")

model = load_model()

st.title("Predicción del Peso al Nacer")

# ===============================
# Inputs del usuario
# ===============================
edad_ = st.number_input("Edad de la madre", 10, 60, 25)
sem_gest = st.number_input("Semanas de gestación", 20.0, 45.0, 38.0)

tipo_ss = st.selectbox(
    "Tipo de seguridad social",
    ["E", "I", "N", "P", "S"]
)

niv_edu = st.selectbox(
    "Nivel educativo",
    ["2", "3", "4", "5", "SD"]
)

mapa_comunas = {
    "Aranjuez": "comuna_Aranjuez",
    "Belen": "comuna_Belen",
    "Buenos Aires": "comuna_Buenos Aires",
    "Corregimiento de San Cristobal": "comuna_Corregimiento de San Cristobal",
    "Doce de Octubre": "comuna_Doce de Octubre",
    "El Poblado": "comuna_El Poblado",
    "Guayabal": "comuna_Guayabal",
    "La Candelaria": "comuna_La Candelaria",
    "Manrique": "comuna_Manrique",
    "Popular": "comuna_Popular",
    "Robledo": "comuna_Robledo",
    "San Antonio de Prado": "comuna_San Antonio de Prado",
    "San Javier": "comuna_San Javier",
    "Santa Elena": "comuna_Santa_Elena",
    "Sin informacion": "comuna_Sin_Info"
}

comuna_usuario = st.selectbox("Comuna", list(mapa_comunas.keys()))
# ===============================
# Columnas EXACTAS del modelo
# ===============================
columnas_modelo = [
    "edad_", "sem_gest",

    "tipo_ss__E", "tipo_ss__I", "tipo_ss__N", "tipo_ss__P", "tipo_ss__S",

    "niv_edu_ma_2", "niv_edu_ma_3", "niv_edu_ma_4",
    "niv_edu_ma_5", "niv_edu_ma_SD",

    "comuna_Aranjuez",
    "comuna_Belen",
    "comuna_Buenos Aires",
    "comuna_Corregimiento de San Cristobal",
    "comuna_Doce de Octubre",
    "comuna_El Poblado",
    "comuna_Guayabal",
    "comuna_La Candelaria",
    "comuna_Manrique",          
    "comuna_Popular",
    "comuna_Robledo",
    "comuna_San Antonio de Prado",
    "comuna_San Javier",
    "comuna_Santa_Elena",       
    "comuna_Sin_Info"           
]

# ===============================
# Fila base en ceros
# ===============================
data = dict.fromkeys(columnas_modelo, 0)

# Numéricas
data["edad_"] = edad_
data["sem_gest"] = sem_gest

# One-hot seguro
if f"tipo_ss__{tipo_ss}" in data:
    data[f"tipo_ss__{tipo_ss}"] = 1

if f"niv_edu_ma_{niv_edu}" in data:
    data[f"niv_edu_ma_{niv_edu}"] = 1

if f"comuna_{comuna}" in data:
    data[f"comuna_{comuna}"] = 1

df = pd.DataFrame([data])

# ===============================
# Predicción
# ===============================
if st.button("Predecir"):
    try:
        pred = model.predict(df)[0]
        st.success(f"### Peso estimado: {pred:.2f} gramos")
    except Exception as e:
        st.error(f"Error en la predicción: {e}")
