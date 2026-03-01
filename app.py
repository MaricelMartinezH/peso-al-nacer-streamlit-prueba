
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

comuna = st.selectbox(
    "Comuna",
    [
        "Aranjuez", "Belen", "Buenos_Aires",
        "Corregimiento_de_San_Cristobal",
        "Doce_de_Octubre", "El_Poblado",
        "Guayabal", "La_Candelaria",
        "Popular", "Robledo",
        "San_Antonio_de_Prado",
        "San_Javier", "Santa_Elena",
        "Sin_Info"
    ]
)

# ===============================
# Columnas EXACTAS del modelo
# ===============================
columnas_modelo = [
    "edad_", "sem_gest",
    "tipo_ss__E", "tipo_ss__I", "tipo_ss__N", "tipo_ss__P", "tipo_ss__S",
    "niv_edu_ma_2", "niv_edu_ma_3", "niv_edu_ma_4",
    "niv_edu_ma_5", "niv_edu_ma_SD",
    "comuna_Aranjuez", "comuna_Belen", "comuna_Buenos_Aires",
    "comuna_Corregimiento_de_San_Cristobal",
    "comuna_Doce_de_Octubre", "comuna_El_Poblado",
    "comuna_Guayabal", "comuna_La_Candelaria",
    "comuna_Popular", "comuna_Robledo",
    "comuna_San_Antonio_de_Prado",
    "comuna_San_Javier", "comuna_Santa_Elena",
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
