import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm 
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
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

st.title('Elaboracion del modelo')

def import_data():
    tipo_archivo = st.sidebar.selectbox('Importa tus datasets de entrenamiento y prueba', ['CSV', 'XLSX'])

    if 'CSV' in tipo_archivo:
        ruta_X_train = st.sidebar.file_uploader("Selecciona el archivo de entrenamiento (X)", type="csv")
        ruta_X_test = st.sidebar.file_uploader("Selecciona el archivo de prueba (X)", type="csv")
        ruta_y_train = st.sidebar.file_uploader("Selecciona el archivo de entrenamiento (y)", type="csv")
        ruta_y_test = st.sidebar.file_uploader("Selecciona el archivo de prueba (y)", type="csv")
    elif 'XLSX' in tipo_archivo:
        ruta_X_train = st.sidebar.file_uploader("Selecciona el archivo de entrenamiento (X)", type="xlsx")
        ruta_X_test = st.sidebar.file_uploader("Selecciona el archivo de prueba (X)", type="xlsx")
        ruta_y_train = st.sidebar.file_uploader("Selecciona el archivo de entrenamiento (y)", type="xlsx")
        ruta_y_test = st.sidebar.file_uploader("Selecciona el archivo de prueba (y)", type="xlsx")
    else:
        st.error('Tipo de archivo no válido')

    X_train = pd.read_csv(ruta_X_train) if ruta_X_train else None
    X_test = pd.read_csv(ruta_X_test) if ruta_X_test else None
    y_train = pd.read_csv(ruta_y_train) if ruta_y_train else None
    y_test = pd.read_csv(ruta_y_test) if ruta_y_test else None

    st.write('Dataset (X) de entrenamiento')
    st.dataframe(X_train.head(2))
    st.write('Dataset (X) de prueba')
    st.dataframe(X_test.head(2))
    st.write('Dataset (y) de entrenamiento')
    st.dataframe(y_train.head(2))
    st.write('Dataset (y) de prueba')
    st.dataframe(y_test.head(2)) 


    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = import_data()

st.subheader('Modelos')
st.write('OLS Regresion Lineal: Hace una seleccion de las variables para la regresion lineal mediante losp-value.')
st.write('Red Neuronal Artificial: construye un modelo mediante aprendizaje.')
st.write('Eleccion Mejor Modelo: Hace un analisis modelando con diferentes algoritmos.')
modelo=st.selectbox('Seleccion de metodologia para modelado: ',['OLS Regresion Lineal','Red Neuronal Artificial','Eleccion Mejor Modelo'])



if modelo == 'OLS Regresion Lineal':
    x=X_train
    y=y_train
    x2=sm.add_constant(x)
    model=sm.OLS(y,x2).fit()
    
    while len (x2.columns) >1:
        p_values=model.pvalues[1:]
        max_p_value=p_values.max()
        if max_p_value>0.05:
            max_p_index=p_values.idxmax()
            x2=x2.drop(max_p_index,axis=1)
            model=sm.OLS(y,x2).fit()
        else:
            break
    st.write(model.summary()) 

    X_train=X_train.loc[:,x2.columns.drop('const')]
    X_test=X_test.loc[:,x2.columns.drop('const')]

    sc=StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)

    lr=LinearRegression()
    lr.fit(X_train,y_train)
    y_pred=lr.predict(X_test)
    st.header(f'La r2 es: {r2_score(y_test,y_pred).round(4)*100}%')

    predict=lr.predict(X_test)
    predict_df = pd.DataFrame(np.ravel(predict), columns=['Predicciones'])
    predict_df['Predicciones'] = predict_df['Predicciones'].astype(int)  
    comparasion_df = pd.concat([pd.DataFrame(y_test, columns=['SalePrice']), predict_df], axis=1)
    st.table(comparasion_df.head(10))

    mae = metrics.mean_absolute_error(y_test, predict)
    mse = metrics.mean_squared_error(y_test, predict)
    rmse = np.sqrt(mse)
    explained_variance = metrics.explained_variance_score(y_test, predict)

    st.write("MAE:", mae)
    st.write("MSE:", mse)
    st.write("RMSE:", rmse)










      


