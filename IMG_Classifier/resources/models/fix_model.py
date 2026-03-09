import pathlib
import sys
import os
import joblib

# Fix para cargar modelos entrenados en Linux
pathlib.PosixPath = pathlib.WindowsPath

# agregar raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from IMG_Classifier.src.DataPreprocessing import TextPreprocessor

# registrar clase para pickle
sys.modules["__main__"].TextPreprocessor = TextPreprocessor

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "ods_text_classifier_pipeline.joblib")

print("Cargando modelo...")
model = joblib.load(model_path)

print("Buscando TextPreprocessor dentro del pipeline...")

for name, step in model.named_steps.items():
    print("Paso encontrado:", name, type(step))

    if isinstance(step, TextPreprocessor):
        print("Quitando spaCy del paso:", name)
        step.nlp = None

print("Guardando modelo corregido...")
joblib.dump(model, model_path)

print("Modelo re-serializado correctamente")