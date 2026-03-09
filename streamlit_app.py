# We ensure proper path handling in Python
from IMG_Classifier import Definitions
import streamlit as st

from IMG_Classifier.src.ModelController import ModelController

### Setup and configuration

st.set_page_config(
    layout="centered", page_title="Clasificador de ODS", page_icon="🌍"
)

### My vars

ctrl = ModelController()

### My UI starting here

st.title("Clasificador de Objetivos de Desarrollo Sostenible")

st.write(
    "Ingrese un texto y el modelo clasificará a qué Objetivo de Desarrollo "
    "Sostenible (ODS) pertenece."
)

texto = st.text_area("Ingrese el texto")

if st.button("Clasificar"):

    if texto.strip() == "":
        st.warning("Por favor ingrese un texto")
    else:
        pred = ctrl.predict_text(texto)

        st.success(f"ODS predicho: {pred}")