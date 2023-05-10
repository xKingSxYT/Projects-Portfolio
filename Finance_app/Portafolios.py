import streamlit as st
import pandas as pd
import pandas_datareader.data as web
from datetime import date
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
import mplcyberpunk
import numpy as np
from prophet import Prophet
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import quantstats as qs
import statistics
import streamlit.components.v1 as components
import csv
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting 

st.title("CONFIGURE SU PORTAFOLIO")
prediction_days=365
start_date = st.sidebar.date_input("Start date",date(2010, 1, 1))
end_date = st.sidebar.date_input("End date",date(2023, 1, 31))
ticker_list=pd.read_csv('Tickers.txt')


#################### SELECCION DE ACCIONES Y GRAFICO#################################
options = st.multiselect('Nombre de la accion', ticker_list)# Select ticker symbol
stocks=yf.download(options,start=start_date,end=end_date)['Adj Close']
st.write('Informacion de las Acciones')
stocks_df=pd.DataFrame(stocks)
st.table(stocks_df.tail(10))
def plot1(stocks):
    fig=plt.figure(figsize=(12,12))
    #plt.plot(stocks/stocks.iloc[0]*10) 
    plt.plot(stocks) 
    plt.ylabel('Precio USD')
    plt.xlabel('Fecha')
    plt.legend(stocks_df)
    plt.title('Precio Historico Acciones del Portafolio')
    plt.xticks(rotation=45,ha='right')
    return fig

stock_plot=plot1(stocks)
st.pyplot(stock_plot)


############ VOLUMEN ##############################################
log_retornos=np.log(stocks/stocks.shift(1))
st.write("La rentabilidad esperada es:")
log_ret_data=pd.DataFrame({'Rentabilidad':(log_retornos).mean()*252})
st.table(log_ret_data)
st.write("Matriz de covarianza")
st.table(log_retornos.cov()*252)
#retornos
num_stocks=len(options)
volumen=np.random.random(num_stocks)
volumen/=np.sum(volumen)
varianza=np.dot(volumen.T,np.dot(log_retornos.cov()*252,volumen))
#np.sum(volumen*log_retornos.mean())*250
#varianza
volatilidad=np.sqrt(varianza)
#volatilidad
rf=0.01 #Factor de riesgo
retornos_portafolio=[]
varianza_portafolio=[]
volatilidad_portafolio=[]
riesgo_portafolio=[]



#### ESTADISTICOS ###############################################
retornos_portafolio=[]
varianza_portafolio=[]
volatilidad_portafolio=[]
sr=[]
volumen_portafolio=[]

st.write("Selecciona cuantas simulaciones Montecarlo desera realizar")
mc=st.number_input("Simulaciones que deasea realizar: ",min_value=1,max_value=100000)
for x in range(mc):
    volumen=np.random.random(len(options))
    volumen=np.round((volumen/np.sum(volumen)),3)
    volumen_portafolio.append(volumen)
    retornos_portafolio.append(np.sum(volumen*log_retornos.mean())*252)
    volatilidad_portafolio.append(np.sqrt(np.dot(volumen.T,np.dot(log_retornos.cov()*252,volumen))))
sr=((np.sum(volumen*log_retornos.mean())*252)-rf)/volatilidad_portafolio

retornos_portafolio=np.array(retornos_portafolio)
volatilidad_portafolio=np.array(volatilidad_portafolio)
sr=np.array(sr)

portafolio_df=[retornos_portafolio,volatilidad_portafolio,sr]
portafolio_metrics=pd.DataFrame(portafolio_df).T
portafolio_metrics.columns=['Retorno','Volatilidad','Sharpe']

option_1=st.selectbox('Que busca en su portafolio',
                      ['Rendimiento Maximo','Riesgo Minimo','Sharpe Ratio Maximo'])

st.write('Porcentaje de inversion para cada accion')
volumen_data=pd.DataFrame({'% Accion':volumen},index=options)
st.table(volumen_data)

riesgo_min=portafolio_metrics.iloc[portafolio_metrics['Volatilidad'].idxmin()]
r_max=portafolio_metrics.iloc[portafolio_metrics['Retorno'].idxmax()]
sr_max=portafolio_metrics.iloc[portafolio_metrics['Sharpe'].idxmax()]

riesgo_min_data=pd.DataFrame({'%':riesgo_min})
r_max_data=pd.DataFrame({'%':r_max})
sr_max_data=pd.DataFrame({'%':sr_max})

st.write(f'Su consulta fue: {option_1}')
if 'Rendimiento Maximo' in option_1:
    st.table(r_max_data)
if 'Riesgo Minimo' in option_1:
    st.table(riesgo_min_data)
if 'Sharpe Ratio Maximo' in option_1:
    st.table(sr_max_data)

r_optimo=portafolio_metrics.iloc[((portafolio_metrics['Retorno']-rf)/portafolio_metrics['Volatilidad']).idxmax()]
r_optimo_data=pd.DataFrame({'%':r_optimo})



def plot2(portafolio):
    fig1=plt.figure(figsize=(12,8))
    plt.scatter(portafolio_metrics['Volatilidad'],portafolio_metrics['Retorno'],c=portafolio_metrics['Sharpe'],cmap='RdYlBu',marker='o',s=10,alpha=0.3)
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(riesgo_min[1],riesgo_min[0],color='r',marker='*',s=500)
    plt.scatter(r_optimo[1],r_optimo[0],color='g',marker='*',s=500)
    plt.xlabel('Volatilidad')
    plt.ylabel('Rendimiento')
    plt.title('Frontera de eficiencia')
    return fig1

portafolio_plot=plot2(portafolio_metrics)
st.pyplot(portafolio_plot)



