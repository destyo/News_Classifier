import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from clean_func import preprocessing



class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Sklearn custom transformer to preprocess text data
    args: 
    params:
    """
    def __init__(self,remove_numbers=False):
        self.remove_numbers = remove_numbers

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.map(lambda x : preprocessing(x, self.remove_numbers))
        return X