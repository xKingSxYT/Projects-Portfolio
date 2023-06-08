import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm 
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.datasets import make_regression
from sklearn import metrics
import math
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

st.title('Creacion variables entrenamiento y pruebas')

def import_data():
    tipo_archivo = st.selectbox('Selecciona el tipo de archivo que deseas importar', ['CSV', 'XLSX'])

    if 'CSV' in tipo_archivo:
        csv = st.file_uploader("Importe su archivo CSV")
        df = pd.read_csv(csv)
    elif 'XLSX' in tipo_archivo:
        xlsx = st.file_uploader("Importe su archivo XLSX")
        df = pd.read_excel(xlsx)
    else:
        st.warning("Por favor, selecciona un archivo CSV.")

    return df

train_data = None
test_data = None
train_y = None
test_y = None

def dependiente(df):
    global train_data, test_data, train_y, test_y
    dependiente_key = 'dependiente_selectbox'
    dependiente = st.selectbox('Selecciona variable dependiente', df.columns.tolist(),key=dependiente_key)
    st.write('Crea los dataset de entrenamiento y pruebas')
    test_size = st.slider('Elige el tamano de tu prueba: ', min_value=0.1, max_value=1.0)

    if st.sidebar.button('Crear Datasets'):
        y = df[dependiente]
        df=df.drop(dependiente,axis=1)
        train_data, test_data,train_y,test_y = train_test_split(df,y, test_size=test_size, random_state=42)
        st.dataframe(train_data.head(5))
        st.dataframe(test_data.head(5))
        st.dataframe(train_y.head(5))
        st.dataframe(test_y.head(5))

        train_data.to_csv('train_data.csv', index=False)
        test_data.to_csv('test_data.csv', index=False)
        train_y.to_csv('train_y.csv', index=False)
        test_y.to_csv('test_y.csv', index=False)

        st.sidebar.write('Cambios guardados correctamente.')

def main():

    st.write('Importa tu archivo')

    if 'df' not in st.session_state:
        st.session_state.df = import_data()

    if st.sidebar.button('Reiniciar'):
        st.session_state.df = import_data()
    st.dataframe(st.session_state.df.head(5))

    if train_data is not None:
        st.dataframe(train_data.head(5))
    
    if test_data is not None:
        st.dataframe(test_data.head(5))

    if train_y is not None:
        st.dataframe(train_y.head(5))
    
    if test_y is not None:
        st.dataframe(test_y.head(5))
    
    dependiente(st.session_state.df)

if __name__ == '__main__':
    if 'changes_applied' not in st.session_state:
        st.session_state.changes_applied = False
    main()



      


