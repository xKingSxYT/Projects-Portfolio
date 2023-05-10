# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 00:52:59 2023

@author: Pablo
"""

import streamlit as st
import pandas as pd
import pandas_datareader.data as web
import datetime 
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

st.markdown('''
# Stock Price Web App
Shown are the stock price data for query companies!

- App built by xKingSx 
''')

st.write('---')

#if "my_input" not in st.session_state:
#    st.session_state["my_input"]=""

#my_input=st.text("""Quieres hacer el analisis de un portafolios?""")
#Portafolio=st.button("Portafolio")

#if Portafolio:
#    st.session_state["my_input"]=my_input


# Sidebar

prediction_days=365

with st.sidebar:
    st.subheader('CHOOSE STOCK AND DATE')
    start_date = st.sidebar.date_input("Start date", datetime.date(2010, 1, 1))
    end_date = st.sidebar.date_input("End date", datetime.date(2023, 1, 31))
    ticker_list=pd.read_csv('Tickers.txt')
    #ticker_list = pd.read_excel('Tickers.xlsx',usecols="A",na_values='NA')
    #ticker_list=ticker_list.drop([ticker_list.index[0],ticker_list.index[1],ticker_list.index[2]])
    #ticker_list=ticker_list.sort_values('Yahoo Stock Tickers')
    tickerSymbol = st.sidebar.selectbox('Nombre de la accion', ticker_list) # Select ticker symbol
    periods=st.number_input('Forecast',value=prediction_days,min_value=1,max_value=5000)
    'Built by xKingSx'
    'Using Prophet Forecast'
    'Yahoo finance database'

tickerData = yf.Ticker(tickerSymbol) # Get ticker data
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)

def get_data(tickerData, start_date, end_date):
    data=yf.download(tickerSymbol,start_date,end_date)
    return data

data=get_data(tickerData, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
st.dataframe(data,height=(400),width=(800))
sc=MinMaxScaler(feature_range=(0,1))

scaled_data=sc.fit_transform(data['Close'].values.reshape(-1,1))

def get_levels(data):
    
    def isSupport(data,i):
        support=data['low'][i]<data['low'][i-1] and data['low'][i]<data['low'][i+1] and data['low'][i+1]<data['low'][i+2] and data['low'][i-1]<data['low'][-2]
        return support
    

    def isResistance(data,i):
        resistance=data['high'][i]<data['high'][i-1] and data['high'][i]<data['high'][i+1] and data['high'][i+1]<data['high'][i+2] and data['high'][i-1]<data['high'][i-2]
        return resistance
   
    def isFarFromLevel(l,levels,s):
        level=np.sum([abs(l-x[0])<s for x in levels])
        return level==0
    
    data.rename(columns={'High':'high','Low':'low'}, inplace=True)
    s=np.mean(data['high']-data['low'])
    levels=[]
    for i in range(2,data.shape[0]-2):
        if isSupport(data, i):
            levels.append((i,data['low'][i]))
        elif isResistance(data, i):
            levels.append((i,data['high'][i]))
    
    filter_levels=[]
    for i in range(2,data.shape[0]-2):
        if isSupport(data,i):
            l=data['low'][i]
            if isFarFromLevel(l, levels, s):
                filter_levels.append((i,l))
        elif isResistance(data, i):
            l=data['high'][i]
            if isFarFromLevel(l, levels, s):
                filter_levels.append((i,l))
                
    return filter_levels


def plot_close_price(data):
    levels=get_levels(data)
    data_levels=pd.DataFrame(levels,columns=['index','close'])
    data_levels.set_index('index',inplace=True)
    max_level=data_levels.idxmax()
    min_level=data_levels.idxmin()
    
    ratios=[0,0.236, 0.382, 0.5 , 0.618, 0.786,1]
    
    if min_level.close > max_level.close:
        trend = 'down'
        fib_levels = [data.Close.iloc[max_level.close] - (data.Close.iloc[max_level.close] - data.Close.iloc[min_level.close]) * ratio for ratio in ratios]
        idx_level = max_level
    else:
        trend = 'up'
        fib_levels = [data.Close.iloc[min_level.close] + (data.Close.iloc[max_level.close] - data.Close.iloc[min_level.close]) * ratio for ratio in ratios]
        idx_level = min_level
    
    fig0=plt.figure(figsize=(10,6))
    plt.plot(data.index, data.Close, color='dodgerblue', linewidth=1)
    mplcyberpunk.add_glow_effects()
    for level, ratio in zip(fib_levels, ratios):
        plt.hlines(level, xmin=data.index[0], xmax=data.index[-1], colors='snow', linestyles='dotted',linewidth=0.9,label="{:.1f}%".format(ratio*100) )

    plt.ylabel('Precio USD')
    plt.xticks(rotation=45,  ha='right')
    plt.grid(True,color='black', linestyle='-', linewidth=0.4)
    
    return fig0
    
stock_fibo_plot=plot_close_price(data)
                  
def returns (data):
    data['returns']=np.log(data['Close']).diff()
    return data 
stock_rend=returns(data)['Close']

def volatilidad(data):
    data['volatilidad']=data.returns.rolling(12).std()
    return data
stock_vol=volatilidad(data)

def plot_volatilidad(data):
    data_plot=data
    fig=plt.figure(figsize=(10,6))
    plt.plot(data_plot.index, data_plot.returns, color='dodgerblue', linewidth=0.5)
    plt.plot(data_plot.index, data_plot.volatilidad, color='darkorange', linewidth=1)
    mplcyberpunk.add_glow_effects()
    plt.ylabel('% Porcentaje')
    plt.xticks(rotation=45,  ha='right')
    plt.grid(True,color='gray', linestyle='-', linewidth=0.2)
    plt.legend(('Retornos Diarios', 'Volatilidad Móvil'), frameon=False)
    return fig

stock_plot_vol=plot_volatilidad(data)

def plot_prophet(data,n_forecast=prediction_days):
    data=data.reset_index().copy()
    data.rename(columns={'Date':'ds','Close':'y'}, inplace=True)
    m = Prophet()
    if data["ds"].dt.tz:
        data["ds"]=data["ds"].dt.tz_convert(None)
    m.fit(data)
    future = m.make_future_dataframe(periods=n_forecast)
    forecast = m.predict(future)
    fig1 = m.plot(forecast,figsize=(10,6))
    mplcyberpunk.add_glow_effects()
    plt.grid(True,color='gray', linestyle='-', linewidth=0.4)
    plt.xticks(rotation=45,  ha='right')
    plt.ylabel('Precio de Cierre')
    plt.plot(forecast.ds, forecast.yhat, color='darkorange', linewidth=0.5)
    plt.legend(('Prediccion','Precio'),frameon=False)
    return fig1

stock=stock_vol['returns']
stats=pd.DataFrame(data=[[qs.stats.avg_return(stock),qs.stats.max_drawdown(stock),qs.stats.win_loss_ratio(stock),qs.stats.sharpe(stock)]],
                   columns=['avg.return','max drawdown','win_loss ratio','sharpe ratio'])


stock_forecast_plot=plot_prophet(data)

st.title("Stock Analysis")

st.subheader('Precio de Cierre-Niveles Fibonacci')
st.pyplot(stock_fibo_plot)

st.subheader('Forecast a un Año - Prophet')
st.pyplot(stock_forecast_plot)

st.subheader('Retornos Diarios')
st.pyplot(stock_plot_vol)

st.subheader('Stock Stats and ratios')
st.write(stats)

st.subheader('Full report')
components.iframe( "file:///C:/Codigos/Stock.html" )
