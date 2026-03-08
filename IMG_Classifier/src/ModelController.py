import joblib
import os
import sys
import pathlib

from src.DataPreprocessing import TextPreprocessor


class ModelController:

    def __init__(self):

        # arreglar paths de Colab (Linux) en Windows
        pathlib.PosixPath = pathlib.WindowsPath

        # registrar la clase para joblib
        sys.modules["__main__"].TextPreprocessor = TextPreprocessor

        model_path = os.path.join(
            "IMG_Classifier",
            "resources",
            "models",
            "ods_text_classifier_pipeline.joblib"
        )

        self.model = joblib.load(model_path)


    def predict_text(self, text):

        prediction = self.model.predict([text])[0]

        return prediction