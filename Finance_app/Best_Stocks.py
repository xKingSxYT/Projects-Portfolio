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

start_date = st.sidebar.date_input("Start date",date(2010, 1, 1))
end_date = st.sidebar.date_input("End date",date(2023, 1, 31))
df=pd.read_csv('Tickers.txt')

stocks=yf.download('ABT	ABBV	ABMD	ACN	ATVI	ADBE	AMD	AAP	AES	AFL	A	APD	AKAM	ALK	ALB	ARE	ALXN	ALGN	ALLE	ALL	AMZN	AMCR	AEE	AAL	AEP	AXP	AIG	AMT	AWK	AMP	ABC	AME	AMGN	APH	ADI	ANSS	ANTM	AON	AOS	APA	AIV	AAPL	AMAT	APTV	ADM	ANET	AJG	AIZ	ATO	ADSK	ADP	AZO	AVB	AVY	AVGO	BKR	BLL	BAC	BK	BAX	BDX	BRK.B	BBY	BIO	BIIB	BLK	BA	BKNG	BWA	BXP	BSX	BMY	BR	BF.B	BEN	CHRW	COG	CDNS	CPB	COF	CAH	CCL	CARR	CAT	CBOE	CBRE	CDW	CE	CNC	CNP	CTL	CERN	CF	CHTR	CVX	CMG	CB	CHD	CI	CINF	CTAS	CSCO	C	CFG	CTXS	CLX	CME	CMS	CTSH	CL	CMCSA	CMA	CAG	CXO	COP	COO	CPRT	CTVA	COST	COTY	CCI	CSX	CMI	CVS	CRM	DHI	DHR	DRI	DVA	DE	DAL	DVN	DXCM	DLR	DFS	DISCA	DISCK	DISH	DG	DLTR	D	DPZ	DOV	DOW	DTE	DUK	DRE	DD	DXC	DGX	DIS	ED	ETFC	EMN	ETN	EBAY	ECL	EIX	EW	EA	EMR	ETR	EOG	EFX	EQIX	EQR	ESS	EL	EVRG	ES	EXC	EXPE	EXPD	EXR	FANG	FFIV	FB	FAST	FRT	FDX	FIS	FITB	FE	FRC	FISV	FLT	FLIR	FLS	FMC	F	FTNT	FTV	FBHS	FOXA	FOX	FCX	FTI	GOOGL	GOOG	GLW	GPS	GRMN	GD	GE	GIS	GM	GPC	GILD	GL	GPN	GS	GWW	HRB	HAL	HBI	HIG	HAS	HCA	HSIC	HSY	HES	HPE	HLT	HFC	HOLX	HD	HON	HRL	HST	HWM	HPQ	HUM	HBAN	HII	IT	IEX	IDXX	INFO	ITW	ILMN	INCY	IR	INTC	ICE	IBM	IP	IPG	IFF	INTU	ISRG	IVZ	IPGP	IQV	IRM	JKHY	J	JBHT	JNJ	JCI	JPM	JNPR	KMX	KO	KSU	K	KEY	KEYS	KMB	KIM	KMI	KLAC	KSS	KHC	KR	LNT	LB	LHX	LH	LRCX	LW	LVS	LEG	LDOS	LEN	LLY	LNC	LIN	LYV	LKQ	LMT	L	LOW	LYB	LUV	MMM	MO	MTB	MRO	MPC	MKTX	MAR	MMC	MLM	MAS	MA	MKC	MXIM	MCD	MCK	MDT	MRK	MET	MTD	MGM	MCHP	MU	MSFT	MAA	MHK	MDLZ	MNST	MCO	MS	MOS	MSI	MSCI	MYL	NDAQ	NOV	NTAP	NFLX	NWL	NEM	NWSA	NWS	NEE	NLSN	NKE	NI	NBL	NSC	NTRS	NOC	NLOK	NCLH	NRG	NUE	NVDA	NVR	NOW	ORLY	OXY	ODFL	OMC	OKE	ORCL	OTIS	O	PEAK	PCAR	PKG	PH	PAYX	PAYC	PYPL	PNR	PBCT	PEP	PKI	PRGO	PFE	PM	PSX	PNW	PXD	PNC	PPG	PPL	PFG	PG	PGR	PLD	PRU	PEG	PSA	PHM	PVH	PWR	QRVO	QCOM	RE	RL	RJF	RTX	REG	REGN	RF	RSG	RMD	RHI	ROK	ROL	ROP	ROST	RCL	SCHW	STZ	SJM	SPGI	SBAC	SLB	STX	SEE	SRE	SHW	SPG	SWKS	SLG	SNA	SO	SWK	SBUX	STT	STE	SYK	SIVB	SYF	SNPS	SYY	T	TAP	TMUS	TROW	TTWO	TPR	TGT	TEL	TDY	TFX	TXN	TXT	TMO	TIF	TJX	TSCO	TT	TDG	TRV	TFC	TWTR	TYL	TSN	TSLA	UDR	ULTA	USB	UAA	UA	UNP	UAL	UNH	UPS	URI	UHS	UNM	VFC	VLO	VAR	VTR	VRSN	VRSK	VZ	VRTX	VIAC	V	VNO	VMC	WRB	WAB	WMT	WBA	WM	WAT	WEC	WFC	WELL	WST	WDC	WU	WRK	WY	WHR	WMB	WLTW	WYNN	XRAY	XOM	XEL	XRX	XLNX	XYL	YUM	ZBRA	ZBH	ZION	ZTS',start=start_date,end=end_date)['Adj Close']


r=np.log(stocks/stocks.shift(1))
r_total=r.sum()
ra=r.mean()*252
r_total_data=pd.DataFrame({'Anualizado':ra,
                           'Total':r_total})

st.title('XKINGSX STOCK APP')
st.header('Analisis Rendmiento de las acciones')

options=st.selectbox('Mostrar las acciones segun: ',
                         ['Rendimiento Total Maximo','Rendimiento Total Minimo',
                          'Rendimiento Anualizado Maximo','Rendimiento Anualizado Minimo'])

n=st.slider('Seleccione el numero de acciones que desea visualizar'
                ,min_value=1
                ,max_value=25)


max_r=r_total_data.sort_values(by='Total',ascending=False).head(n)
max_ra=r_total_data.sort_values(by='Anualizado',ascending=False).head(n)
min_r=r_total_data.sort_values(by='Total',ascending=True).head(n)
min_ra=r_total_data.sort_values(by='Anualizado',ascending=True).head(n)

st.write(f'La tabla esta ordenado segun: {options}')
if 'Rendimiento Total Maximo' in options:
    st.table(max_r)
if 'Rendimiento Anualizado Maximo' in options:
    st.table(max_ra)
if 'Rendimiento Total Minimo' in options:
    st.table(min_r)
if 'Rendimiento Anualizado Minimo' in options:
    st.table(min_ra)