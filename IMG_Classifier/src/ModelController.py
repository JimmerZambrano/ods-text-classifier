import pathlib
import platform

# Fix cross-platform pickle paths (Windows ↔ Linux)
if platform.system() != "Windows":
    pathlib.WindowsPath = pathlib.PosixPath
else:
    pathlib.PosixPath = pathlib.WindowsPath

import joblib
import os
import sys
import gdown

from IMG_Classifier.src.DataPreprocessing import TextPreprocessor


class ModelController:

    def __init__(self):

        # Registrar la clase personalizada para que joblib pueda deserializarla
        sys.modules["__main__"].TextPreprocessor = TextPreprocessor

        MODEL_ID = "1ilRZddw1cMuLOCMjI8x14p8MUZlqfrFy"
        MODEL_PATH = "ods_text_classifier_pipeline.joblib"

        # Descargar modelo si no existe
        if not os.path.exists(MODEL_PATH):
            url = f"https://drive.google.com/uc?id={MODEL_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)

        # Cargar modelo
        self.model = joblib.load(MODEL_PATH)


    def predict_text(self, text):

        prediction = self.model.predict([text])[0]

        return prediction