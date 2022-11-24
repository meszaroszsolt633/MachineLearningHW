import numpy as np
import pickle
import streamlit as st
import sklearn
import ModelSave
from sklearn.svm import SVR
from joblib import load,dump
from sklearn.model_selection import GridSearchCV




def culmenlength_prediction(input):
    inputnp = np.asarray(input)
    inputreshaped = inputnp.reshape(1, -1)

    #converted=poly_conv_loaded.fit_transform(input)
    #print(converted)
    scaled=scaler_loaded.transform(inputreshaped)
    prediction = SVR_model_loaded.predict(scaled)
    print('Culmen length in mm:', prediction[0])
    return ('Culmen length in mm:', prediction[0])


def main():
    st.title('Penguin culmen length prediction')

    speciesstring = st.text_input('Species: Adelie, Chinstrap, Gentoo')
    if (speciesstring == 'Adelie'):
        species = 0
    if (speciesstring == 'Chinstrap'):
        species = 1
    if (speciesstring == 'Gentoo'):
        species = 2
    islandstring = st.text_input('Island: Torgersen, Biscoe, Dream')
    if (islandstring == 'Dream'):
        island = 0
    if (islandstring == 'Biscoe'):
        island = 1
    if (islandstring == 'Torgersen'):
        island = 2
    culmen_depth_mm = st.text_input('Culmen depth mm')
    flipper_length_mm = st.text_input('Flipper length mm')
    body_mass_g = st.text_input('Body mass')
    sex = st.slider('Sex: 1-Female 2-Male', 1, 2, 1)
    prediction = ''
    if st.button('Penguin culmen length prediction:'):
        prediction = culmenlength_prediction(
            [sex, species, island, culmen_depth_mm, flipper_length_mm, body_mass_g])

    st.success(prediction)


if __name__ == '__main__':
    poly_conv_loaded = load("poly_conv.joblib")
    SVR_model_loaded = load("SVR_Model.joblib")
    scaler_loaded = load("std_scaler.joblib")
    main()
    #print(culmenlength_prediction([0, 0, 1, 17, 165, 4500]))
    #print(culmenlength_prediction([1, 2, 2, 15, 135, 2500]))
    #print(culmenlength_prediction([1, 1, 1, 18, 185, 3500]))
    #main()


