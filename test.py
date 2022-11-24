import numpy as np
import pickle
import streamlit as st
import sklearn
import ModelSave
from sklearn.svm import SVR
from joblib import load,dump
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import PolynomialFeatures
from joblib import load,dump
from sklearn.svm import SVR
from sklearn import metrics
import pandas as pd
import numpy as np
import sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import sklearn

if __name__=='__main__':
    parameters = {
        "kernel": ["rbf"],
        "C": [1, 10, 100, 1000],
        "gamma": [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    }
    grid = GridSearchCV(SVR(), parameters, cv=5, verbose=2)
    grid.fit(xTrain, yTrain)
