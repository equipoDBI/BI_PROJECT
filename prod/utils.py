import joblib
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score
from scipy.ndimage import shift
# Importamos la librería nueva
import yfinance as yf


def obtenerData(instrumentoFinanciero, fechaInicio, fechaFin):
    IF_df = yf.download(instrumentoFinanciero,
                        start=fechaInicio, end=fechaFin)
    IF_df.columns += "_" + instrumentoFinanciero
    GLD_data = yf.download('GLD', start=fechaInicio, end=fechaFin)
    GLD_data.columns += "_GLD"
    SLV_data = yf.download('SLV', start=fechaInicio, end=fechaFin)
    SLV_data.columns += "_SLV"
    COPX_data = yf.download('COPX', start=fechaInicio, end=fechaFin)
    COPX_data.columns += "_COPX"
    GSPC_data = yf.download('^GSPC', start=fechaInicio, end=fechaFin)
    GSPC_data.columns += "_GSPC"
    IXIC_data = yf.download('^IXIC', start=fechaInicio, end=fechaFin)
    IXIC_data.columns += "_IXIC"
    DJI_data = yf.download('^DJI', start=fechaInicio, end=fechaFin)
    DJI_data.columns += "_DJI"
    PEN_X_data = yf.download('PEN=X', start=fechaInicio, end=fechaFin)
    PEN_X_data.columns += "_PEN_X"
    BZ_F_data = yf.download('BZ=F', start=fechaInicio, end=fechaFin)
    BZ_F_data.columns += "_BZ_F"
    df = pd.merge(IF_df, GLD_data, on='Date')
    df = pd.merge(df, SLV_data, on='Date')
    df = pd.merge(df, COPX_data, on='Date')
    df = pd.merge(df, GSPC_data, on='Date')
    df = pd.merge(df, IXIC_data, on='Date')
    df = pd.merge(df, DJI_data, on='Date')
    df = pd.merge(df, PEN_X_data, on='Date')
    df = pd.merge(df, BZ_F_data, on='Date')
    df = df.drop(['Volume_PEN_X'], axis=1)
    return df


def transformarDataEntradaADataPrediccion(instrumentoFinanciero, fechaInicio, fechaFin, modelo):
    df = obtenerData(instrumentoFinanciero, fechaInicio, fechaFin)
    nombreCloseInstrumentoFinanciero = "Close_" + instrumentoFinanciero
    escala = StandardScaler()
    if modelo == "SVR":
        df = df.dropna(how='any')
        X = escala.fit_transform(
            df.drop([nombreCloseInstrumentoFinanciero], axis=1))
        y = df[nombreCloseInstrumentoFinanciero]
    if modelo == "SVC":
        df['Return'] = df[nombreCloseInstrumentoFinanciero].pct_change()
        df['Return'] = df['Return'].fillna(0)
        df['Trend'] = np.where(df['Return'] > 0.00, 1, 0)
        df['Trend'] = df['Trend'].shift(-1)
        df['Trend'] = df['Trend'].fillna(0)
        df = df.dropna(how='any')
        X = escala.fit_transform(df.drop(['Trend', 'Return'], axis=1))
        y = df['Trend']
    return X, y


def obtenerModelo(instrumentoFinanciero, fechaInicio, fechaFin, modelo):
    X, y = transformarDataEntradaADataPrediccion(
        instrumentoFinanciero, fechaInicio, fechaFin, modelo)
    if modelo == "SVC":
        modeloEntrenado = SVC().fit(X, y)
    if modelo == "SVR":
        modeloEntrenado = SVR().fit(X, y)
    joblib.dump(modeloEntrenado, modelo + '.pkl')
    pklModelo = joblib.load("prod/" + modelo + instrumentoFinanciero + '.pkl')
    return pklModelo


def pct_change_numpy(arreglo):
    arreglo_pct_change = []
    for i in range(len(arreglo)):
        arreglo_pct_change[i] = (arreglo[i] - arreglo[i-1])/arreglo[i]


def hacerPrediccion(instrumentoFinanciero, fechaInicioEntrenamiento, fechaFinEntrenamiento, fechaInicioPrediccion, fechaFinPrediccion, modelo):
    modeloEntrenado = obtenerModelo(instrumentoFinanciero,
                                    fechaInicioEntrenamiento, fechaFinEntrenamiento, modelo)
    X, y = transformarDataEntradaADataPrediccion(
        instrumentoFinanciero, fechaInicioPrediccion, fechaFinPrediccion, modelo)
    resultado = modeloEntrenado.predict(X)
    texto = []
    listaFechas = [(fechaInicioPrediccion + timedelta(days=d)).strftime("%Y-%m-%d")
                   for d in range((fechaFinPrediccion - fechaInicioPrediccion).days + 1)]
    if modelo == "SVC":
        for i in range(0, resultado.size):
            texto.append("*   Compra acciones el día " +
                         listaFechas[i] if resultado[i] > 0 else "*   No compres acciones el día " + listaFechas[i])
    if modelo == "SVR":
        resultado = pd.Series(resultado)
        retorno = resultado.pct_change()
        retorno = retorno.fillna(0)
        trend = np.where(retorno > 0.00, 1, 0)
        trend = shift(trend, -1, cval=np.NaN)
        trend = np.nan_to_num(trend)
        for i in range(0, trend.size):
            texto.append("*   Compra acciones el día " +
                         listaFechas[i] if trend[i] > 0 else "*   No compres acciones el día " + listaFechas[i])
    return texto


def plotHeatMap(dataframe):
    fig, ax = plt.subplots()
    sns.heatmap(dataframe.corr(), ax=ax)
    return fig
