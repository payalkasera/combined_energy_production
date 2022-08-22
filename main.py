# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 09:41:04 2022

@author: HP
"""

from re import S
from tkinter import E
import pandas as pd
import streamlit as st
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

st.title('Model Deployment')
st.sidebar.header('User Input Parameters')


def user_input_features():
    TEMPERATURE = st.sidebar.number_input('temperature')
    EXHAUST_VACUUM = st.sidebar.number_input('exhaust_vacuum')
    AMB_PRESSURE = st.sidebar.number_input('amb_pressure')
    R_HUMIDITY = st.sidebar.number_input("r_humidity")
       
    data = {'temperature':TEMPERATURE,
            'exhaust_vacuum':EXHAUST_VACUUM,
            'amb_pressure':AMB_PRESSURE,
            'r_humidity':R_HUMIDITY}
    
    features = pd.DataFrame(data,index=[0])
    return features


df = user_input_features()
st.subheader('User Input Parameters')
st.write(df)



energy_pred = pd.read_csv('energy_production.csv',sep=';')

# to drop duplicates
energy_pred=energy_pred.drop_duplicates().reset_index(drop=True)

# to remove outliers
energy_pred.drop(energy_pred.index[energy_pred['amb_pressure']<999],inplace=True, axis=0)
energy_pred.drop(energy_pred.index[energy_pred['amb_pressure']>1028],inplace=True,axis=0)
energy_pred.drop(energy_pred.index[energy_pred['r_humidity']<32],inplace=True,axis=0)

X = energy_pred.drop(labels='energy_production',axis=1)
X.head()
X=preprocessing.normalize(X)
y = energy_pred['energy_production']

regressor = LinearRegression()
regressor.fit(X,y)

df=preprocessing.normalize(df)


prediction = regressor.predict(df)

st.subheader('Predicted result')
st.write(prediction)
st.snow()
