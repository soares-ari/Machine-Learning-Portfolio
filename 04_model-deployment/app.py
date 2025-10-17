import streamlit as st
import pandas as pd
import joblib
from custom_transformers import CleanBrandTransformer


st.set_page_config(page_title="Car Efficiency Predictor", page_icon="🚗", layout="centered")
st.title("🚗 Previsão de Eficiência de Carros (mpg)")
st.markdown("Pipeline com **pré-processamento + modelo** (scikit-learn).")

# Carregar artefatos
pipe = joblib.load("pipeline.pkl")
schema = joblib.load("schema.joblib")

num_features = schema["num_features"]
cat_features = schema["cat_features"]
pass_features = schema["pass_features"]

# Entradas do usuário
st.sidebar.header("🔧 Parâmetros do veículo")
cubicinches = st.sidebar.slider("Cilindrada (cubicinches)", 60.0, 500.0, 200.0, 1.0)
hp          = st.sidebar.slider("Potência (hp)", 40, 250, 100, 1)
weightlbs   = st.sidebar.slider("Peso (lbs)", 1000, 5000, 2500, 10)
time_to_60  = st.sidebar.slider("0–60 mph (s)", 5.0, 25.0, 10.0, 0.1)
brand       = st.sidebar.selectbox("Marca (origem)", ["us", "japan", "europe"])
cylinders   = st.sidebar.selectbox("Cilindros", [3,4,5,6,8])
year        = st.sidebar.slider("Ano", 1970, 1982, 1976)

# Montar DataFrame
X_input = pd.DataFrame({
    "cubicinches": [cubicinches],
    "hp": [hp],
    "weightlbs": [weightlbs],
    "time-to-60": [time_to_60],
    "brand": [brand],
    "cylinders": [cylinders],
    "year": [year],
})

if st.button("Prever eficiência (mpg)"):
    pred = pipe.predict(X_input)[0]
    st.success(f"Eficiência estimada: **{pred:.2f} mpg**")
