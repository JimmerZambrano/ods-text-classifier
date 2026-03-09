import pathlib
import sys
import os

# Fix paths when loading models saved in Linux
pathlib.PosixPath = pathlib.WindowsPath

# agregar raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import joblib
from IMG_Classifier.src.DataPreprocessing import TextPreprocessor

# registrar la clase para pickle
sys.modules["__main__"].TextPreprocessor = TextPreprocessor

BASE_DIR = os.path.dirname(__file__)
model_path = os.path.join(BASE_DIR, "ods_text_classifier_pipeline.joblib")

# cargar modelo
model = joblib.load(model_path)

# volver a guardarlo
joblib.dump(model, model_path)

print("Modelo re-serializado correctamente")