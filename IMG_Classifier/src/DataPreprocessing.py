import numpy as np
import pandas as pd

class DataPreprocessing:

    def __init__(self):
        print("DataPreprocessing.__init__ ->")
        self.MAX_PIXEL_COUNT = 784

    def transform(self, df):
        print("DataPreprocessing.transform ->")
        return None

    def get_columns(self):
        print("DataPreprocessing.get_columns ->")
        #TO-DO: Genera los nombres de las columnas en una lista para la variable res
        res = []

        return set(res)

    def get_cat_name(self, index):
        print("DataPreprocessing.get_cat_name ->")
        if index < 0 or index > 1:
            return ""
        return self.get_categories()[index]
    
from sklearn.base import BaseEstimator, TransformerMixin
import spacy
from nltk.corpus import stopwords as nltk_stopwords


class TextPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.nlp = spacy.load("es_core_news_sm", disable=["parser","ner","textcat"])

        spacy_stopwords = self.nlp.Defaults.stop_words
        nltk_stopwords_es = set(nltk_stopwords.words("spanish"))

        words_to_keep = {"no", "ni", "sin"}

        self.custom_stopwords = (spacy_stopwords | nltk_stopwords_es) - words_to_keep


    def preprocess_text(self, text):

        doc = self.nlp(text.lower())

        tokens = []

        for token in doc:

            if not token.is_alpha:
                continue

            lemma = token.lemma_.strip()

            if lemma in self.custom_stopwords or len(lemma) < 2:
                continue

            tokens.append(lemma)

        return " ".join(tokens)


    def transform(self, X):

        return [self.preprocess_text(text) for text in X]


    def fit(self, X, y=None):
        return self
