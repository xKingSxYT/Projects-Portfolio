import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score,mean_squared_error
from sklearn import metrics
from sklearn.neural_network import MLPRegressor
from keras.optimizers import Adam,RMSprop
from keras.models import Sequential
from keras.layers import Dense

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

def EvaluationMetric(Xt, yt, yp, disp="on"):
    MSE = round(mean_squared_error(y_true=yt, y_pred=yp), 4)
    RMSE = (np.sqrt(MSE))
    R2 = (r2_score(y_true=yt, y_pred=yp))
    Adjusted_R2 = (1 - (1 - r2_score(yt, yp)) * ((Xt.shape[0] - 1) / (Xt.shape[0] - Xt.shape[1] - 1)))
    
    if disp == "on":
        st.write("MSE:", MSE)
        st.write("RMSE:", RMSE)
        st.write("R2:", R2)
        st.write("Adjusted R2:", Adjusted_R2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(len(yp[:100])), y=yp[:100], name="Predicción"))
    fig.add_trace(go.Scatter(x=np.arange(len(yt[:100])), y=np.array(yt)[:100, 0], name="Real"))
    fig.update_layout(title='Precio Real y Predicción del modelo')
    st.plotly_chart(fig)

    return (MSE, RMSE, R2, Adjusted_R2)

def search_best_model(X_train, y_train, modelos, hiperparametros, cv=5, scoring='r2'):
    mejores_modelos = {}
    
    for nombre_modelo, modelo in modelos.items():
        params = hiperparametros[nombre_modelo]
        
        grid_search = GridSearchCV(estimator=modelo, param_grid=params, cv=cv, scoring=scoring)
        grid_search.fit(X_train, y_train)
        
        mejor_modelo = grid_search.best_estimator_
        mejor_puntaje = grid_search.best_score_
        
        mejores_modelos[nombre_modelo] = {'Modelo': mejor_modelo, 'Puntaje': mejor_puntaje}
        
    return mejores_modelos



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

    sc=MinMaxScaler()
    sc.fit(X_train)
    sc.fit(X_test)
    sc.fit(y_train)
    sc.fit(y_test)
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

if modelo == "Red Neuronal Artificial":
    capas = st.number_input('Introduzca el número de capas para su red:', min_value=1, max_value=99, value=1, step=1)
    nodos_por_capa = []
    for i in range(capas):
        nodos = st.number_input(f"Número de nodos para la Capa {i+1}:", min_value=1, max_value=512, value=64, step=1)
        nodos_por_capa.append(nodos)

    model = MLPRegressor(hidden_layer_sizes=tuple(nodos_por_capa), max_iter=1000)
    
    x = X_train
    y = y_train
    x2 = sm.add_constant(x)
    modelo = sm.OLS(y, x2).fit()

    while len (x2.columns) >1:
        p_values=modelo.pvalues[1:]
        max_p_value=p_values.max()
        if max_p_value>0.05:
            max_p_index=p_values.idxmax()
            x2=x2.drop(max_p_index,axis=1)
            modelo=sm.OLS(y,x2).fit()
        else:
            break

    X_train=X_train.loc[:,x2.columns.drop('const')]
    X_test=X_test.loc[:,x2.columns.drop('const')]


    sc = MinMaxScaler()
    sc.fit(X_train)
    sc.fit(X_test)
    sc.fit(y_train)
    sc.fit(y_test)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model.fit(X_train, y_train)

    predict = model.predict(X_test)
    predict_df = pd.DataFrame(np.ravel(predict), columns=['Predicciones'])
    predict_df['Predicciones'] = predict_df['Predicciones'].astype(int)  
    comparasion_df = pd.concat([pd.DataFrame(y_test, columns=['SalePrice']), predict_df], axis=1)
    st.table(comparasion_df.head(10))
    
    EvaluationMetric(X_test, y_test, predict)

if modelo == 'Eleccion Mejor Modelo':
    
    modelos={
        'Regresion Lineal':LinearRegression(),
        'Arbol de Decision':DecisionTreeRegressor(),
        'Random Forest':RandomForestRegressor(),
        'Gradient Boosting Regressor':GradientBoostingRegressor(),
        'Ada Boost Regressor':AdaBoostRegressor(),
    }

    hiperparametros={
        'Regresion Lineal':{},
        'Arbol de Decision':{'max_depth':[10,15]},
        'Random Forest':{'n_estimators':[100,500],
                         'max_depth':[10,15]},
        'Gradient Boosting Regressor':{'learning_rate': [0.2,0.5],
                                       'n_estimators': [100,500],
                                       'max_depth': [5,10],
                                       'subsample': [1.0,2.0],
                                       'loss': ['ls'] },
        'Ada Boost Regressor':{'n_estimators':[100,500],
                               'learning_rate':[1.0,2.0],
                               'loss':['linear']},
    }
    x = X_train
    y = y_train
    x2 = sm.add_constant(x)
    modelo = sm.OLS(y, x2).fit()

    while len (x2.columns) >1:
        p_values=modelo.pvalues[1:]
        max_p_value=p_values.max()
        if max_p_value>0.05:
            max_p_index=p_values.idxmax()
            x2=x2.drop(max_p_index,axis=1)
            modelo=sm.OLS(y,x2).fit()
        else:
            break

    X_train=X_train.loc[:,x2.columns.drop('const')]
    X_test=X_test.loc[:,x2.columns.drop('const')]

    sc = MinMaxScaler()
    sc.fit(X_train)
    sc.fit(X_test)
    sc.fit(y_train)
    sc.fit(y_test)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    mejor_modelo = search_best_model(X_train, y_train, modelos, hiperparametros)

    for nombre_modelo, modelo in mejor_modelo.items():
        st.write(nombre_modelo)
        st.write('Mejor puntaje (R2): ', modelo['Puntaje'])
        st.write('Mejores hiperparámetros: ', modelo['Modelo'].get_params())









      


