import os
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
import statsmodels.api as sm
from scipy import stats
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector

st.title('Generador de modelos')

st.write('Importa tu archivo')

tipo_archivo=st.selectbox('Selecciona el tipo de archivo que deseas importar',
             ['CSV','XLSX'])    
if 'CSV' in tipo_archivo:
        csv=st.file_uploader("Importe su archivo CSV")
        df=pd.read_csv(csv)
elif 'XLSX' in tipo_archivo:
        xlsx=st.file_uploader("Importe su archivo XLSX")
        df=pd.read_excel(xlsx)

st.dataframe(df.head(5))
#st.write(list(set(df.dtypes.tolist())))
st.subheader('Seleccion de variable dependiente')
var_dep=st.multiselect('Selecciona tu variable dependiente',df.columns)

columns=[col for col in df.columns if col not in var_dep]
y=df[var_dep]

st.subheader('Descripcion y distribucion de la variable dependiente')

#y=df.iloc[:,-1].dropna()
x=df.iloc[:,:-1]
x=x.iloc[:,1:]
st.write(y.describe())

def plot1(y):
        fig1=plt.figure(figsize=(9,8))
        plt.hist(y,color='g',
                    bins=100,
                    alpha=0.4)
        plt.title("Ditribucion de precios")
        plt.xlabel=("Precio")
        plt.ylabel=("Conteo")
        return fig1

price_plot=plot1(y)
st.pyplot(price_plot)

st.header('Estadisticos para el modelo')
st.subheader('Seleccion variables')

variables=st.selectbox('Seleccion el metodo para analizar sus variables',
                       ['OSL (Solo variables numericas)','Completo'])
if 'OSL (Solo variables numericas)' in variables:
        x.dropna()
        x.reset_index(drop=True)
        x=x.select_dtypes(include=['float64','int64']).fillna(0)
        x2=sm.add_constant(x)
        est=sm.OLS(y,x2)
        est2=est.fit()
        st.write(est2.summary())
        
        
if 'Completo' in variables:
        x.dropna()
        x.reset_index(drop=True)
        cat_cols=x.select_dtypes(include='object').columns
        x[cat_cols]=x[cat_cols].apply(lambda x:x.astype('category')) 
        x[cat_cols]=x[cat_cols].apply(lambda x:x.cat.codes)
        x=x[cat_cols]
        column_trans = ColumnTransformer(transformers=
            [('num', MinMaxScaler(), selector(dtype_exclude="object")),
            ('cat', OrdinalEncoder(), selector(dtype_include="object"))],
            remainder='drop')
        clf = RandomForestClassifier(random_state=42, n_jobs=6, class_weight='balanced')
        pipeline = Pipeline([('prep',column_trans),
                     ('clf', clf)])
        pipeline.fit(x,y)
        feat_list = []
        total_importance = 0
        for feature in zip(x.columns, pipeline['clf'].feature_importances_):
            feat_list.append(feature)
            total_importance += feature[1]
        included_feats = []
        for feature in zip(x.columns, pipeline['clf'].feature_importances_):
            if feature[1] > .00:
                included_feats.append(feature[0])
        df_x=pd.DataFrame(feat_list,columns=['Variable',
                        'Importancia']).sort_values(by='Importancia',
                                                    ascending=False)
        df_x['Sumatorio']=df_x['Importancia'].cumsum()
        st.dataframe(df_x)

                
drop=st.multiselect('Selecciona las variables que deseas eliminar de tu modelo',x.columns)

x_1=x.drop(drop,axis=1)
x_1

st.header('Preparacion test y entrenamiento')
test=st.slider('Selecciona el tamaño de tus pruebas:',min_value=0.1,max_value=1.0)

X_train,X_test,y_train,y_test=train_test_split(x_1,y,test_size=test,random_state=0)

st.subheader('Dataset de entrenamiento')
st.write('Variables independientes')
AgGrid(X_train.head(10))
st.write('Variable dependiente')
st.table(y_train.head(10))
st.subheader('Dataset de prueba')
st.write('Variables independientes')
AgGrid(X_test.tail(10))
st.write('Variable dependiente')
st.table(y_test.head(10))

sc=StandardScaler()
sc.fit(X_train)
X_train=sc.transform(X_train)
X_test=sc.transform(X_test)

btn_lr=st.button('Regresion Lineal')
btn_rf=st.button('Random Forest')

if btn_lr:
    lr=LinearRegression()
    lr.fit(X_train,y_train)
    y_pred_lr=lr.predict(X_test)
    st.write(r2_score(y_test,y_pred_lr))

n=st.number_input('Ingrese el numero de estimadores que desea realizar',min_value=1,max_value=None)    
if btn_rf:
    rf=RandomForestRegressor(n_estimators=n)
    rf.fit(X_train,y_train)
    y_pred_rf=rf.predict(X_test)
    st.write(r2_score(y_test,y_pred_rf))
