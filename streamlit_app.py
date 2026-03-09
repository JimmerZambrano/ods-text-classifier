# We ensure proper path handling in Python
from IMG_Classifier import Definitions
import streamlit as st
from IMG_Classifier.src.ModelController import ModelController

st.set_page_config(
    layout="centered", page_title="Clasificador de ODS", page_icon="🌍"
)

@st.cache_resource
def load_model():
    return ModelController()

ctrl = load_model()

ODS_NAMES = {
    1: "Fin de la pobreza",
    2: "Hambre cero",
    3: "Salud y bienestar",
    4: "Educación de calidad",
    5: "Igualdad de género",
    6: "Agua limpia y saneamiento",
    7: "Energía asequible y no contaminante",
    8: "Trabajo decente y crecimiento económico",
    9: "Industria, innovación e infraestructura",
    10: "Reducción de desigualdades",
    11: "Ciudades y comunidades sostenibles",
    12: "Producción y consumo responsables",
    13: "Acción por el clima",
    14: "Vida submarina",
    15: "Vida de ecosistemas terrestres",
    16: "Paz, justicia e instituciones sólidas",
    17: "Alianzas para lograr los objetivos"
}

st.title("Clasificador de Objetivos de Desarrollo Sostenible")

st.write(
    "Ingrese un texto y el modelo clasificará a qué Objetivo de Desarrollo "
    "Sostenible (ODS) pertenece."
)

texto = st.text_area(
    "Ingrese el texto",
    placeholder="Ejemplo: Programas educativos para mejorar el acceso a educación de calidad en comunidades rurales."
)

if st.button("Clasificar"):

    if texto.strip() == "":
        st.warning("Por favor ingrese un texto")
    else:
        pred = ctrl.predict_text(texto)
        st.success(f"ODS predicho: {pred} - {ODS_NAMES.get(pred, '')}")