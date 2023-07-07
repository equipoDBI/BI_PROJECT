import joblib
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
#Importamos la librería nueva
import yfinance as yf

def loadModelo(ruta):
    modelo = joblib.load(ruta)
    return modelo
def trainingData():
    fechaInicio = '2018-01-01'
    fechaFin = '2022-12-31'
    BVN_df = yf.download('BVN', start = fechaInicio, end = fechaFin)
    #Añadimos la terminación _BVN a cada columna
    BVN_df.columns += "_BVN"
    GLD_data = yf.download('GLD', start = fechaInicio, end = fechaFin)
    GLD_data.columns += "_GLD"
    SLV_data = yf.download('SLV', start = fechaInicio, end = fechaFin)
    SLV_data.columns += "_SLV"
    COPX_data = yf.download('COPX', start = fechaInicio, end = fechaFin)
    COPX_data.columns += "_COPX"
    GSPC_data = yf.download('^GSPC', start = fechaInicio, end = fechaFin)
    GSPC_data.columns += "_GSPC"
    IXIC_data = yf.download('^IXIC', start = fechaInicio, end = fechaFin)
    IXIC_data.columns += "_IXIC"
    DJI_data = yf.download('^DJI', start = fechaInicio, end = fechaFin)
    DJI_data.columns += "_DJI"
    PEN_X_data = yf.download('PEN=X', start = fechaInicio, end = fechaFin)
    PEN_X_data.columns += "_PEN_X"
    BZ_F_data = yf.download('BZ=F', start = fechaInicio, end = fechaFin)
    BZ_F_data.columns += "_BZ_F"
    df = pd.merge(BVN_df, GLD_data, on = 'Date')
    df = pd.merge(df, SLV_data, on = 'Date')
    df = pd.merge(df, COPX_data, on = 'Date')
    df = pd.merge(df, GSPC_data, on = 'Date')
    df = pd.merge(df, IXIC_data, on = 'Date')
    df = pd.merge(df, DJI_data, on = 'Date')
    df = pd.merge(df, PEN_X_data, on = 'Date')
    df = pd.merge(df, BZ_F_data, on = 'Date')
    df = df.drop(['Volume_PEN_X'], axis=1)

    return df.sample(10)

def plotHeatMap(dataframe):
    fig, ax = plt.subplots()
    sns.heatmap(dataframe.corr(), ax=ax)
    return fig
def loadDataBetweenDates(fechaInicio, fechaFin):
    BVN_df = yf.download('BVN', start = fechaInicio, end = fechaFin)
    #Añadimos la terminación _BVN a cada columna
    BVN_df.columns += "_BVN"
    GLD_data = yf.download('GLD', start = fechaInicio, end = fechaFin)
    GLD_data.columns += "_GLD"
    SLV_data = yf.download('SLV', start = fechaInicio, end = fechaFin)
    SLV_data.columns += "_SLV"
    COPX_data = yf.download('COPX', start = fechaInicio, end = fechaFin)
    COPX_data.columns += "_COPX"
    GSPC_data = yf.download('^GSPC', start = fechaInicio, end = fechaFin)
    GSPC_data.columns += "_GSPC"
    IXIC_data = yf.download('^IXIC', start = fechaInicio, end = fechaFin)
    IXIC_data.columns += "_IXIC"
    DJI_data = yf.download('^DJI', start = fechaInicio, end = fechaFin)
    DJI_data.columns += "_DJI"
    PEN_X_data = yf.download('PEN=X', start = fechaInicio, end = fechaFin)
    PEN_X_data.columns += "_PEN_X"
    BZ_F_data = yf.download('BZ=F', start = fechaInicio, end = fechaFin)
    BZ_F_data.columns += "_BZ_F"
    df = pd.merge(BVN_df, GLD_data, on = 'Date')
    df = pd.merge(df, SLV_data, on = 'Date')
    df = pd.merge(df, COPX_data, on = 'Date')
    df = pd.merge(df, GSPC_data, on = 'Date')
    df = pd.merge(df, IXIC_data, on = 'Date')
    df = pd.merge(df, DJI_data, on = 'Date')
    df = pd.merge(df, PEN_X_data, on = 'Date')
    df = pd.merge(df, BZ_F_data, on = 'Date')
    df = df.drop(['Volume_PEN_X'], axis=1)
    escala = StandardScaler()
    df = escala.fit_transform(df)
    return df

def makePredition(fechaInicio, fechaFin):
    modelo = loadModelo('svc_bvn.pkl')
    df= loadDataBetweenDates(fechaInicio, fechaFin)
    result=predict(modelo,df)
    len = result.size
    return "Compra acciones este dia" if result[len-1]>0 else "No compres acciones este dia"



def noNormalizedData(fechaInicio, fechaFin):
    BVN_df = yf.download('BVN', start = fechaInicio, end = fechaFin)
    #Añadimos la terminación _BVN a cada columna
    BVN_df.columns += "_BVN"
    GLD_data = yf.download('GLD', start = fechaInicio, end = fechaFin)
    GLD_data.columns += "_GLD"
    SLV_data = yf.download('SLV', start = fechaInicio, end = fechaFin)
    SLV_data.columns += "_SLV"
    COPX_data = yf.download('COPX', start = fechaInicio, end = fechaFin)
    COPX_data.columns += "_COPX"
    GSPC_data = yf.download('^GSPC', start = fechaInicio, end = fechaFin)
    GSPC_data.columns += "_GSPC"
    IXIC_data = yf.download('^IXIC', start = fechaInicio, end = fechaFin)
    IXIC_data.columns += "_IXIC"
    DJI_data = yf.download('^DJI', start = fechaInicio, end = fechaFin)
    DJI_data.columns += "_DJI"
    PEN_X_data = yf.download('PEN=X', start = fechaInicio, end = fechaFin)
    PEN_X_data.columns += "_PEN_X"
    BZ_F_data = yf.download('BZ=F', start = fechaInicio, end = fechaFin)
    BZ_F_data.columns += "_BZ_F"
    df = pd.merge(BVN_df, GLD_data, on = 'Date')
    df = pd.merge(df, SLV_data, on = 'Date')
    df = pd.merge(df, COPX_data, on = 'Date')
    df = pd.merge(df, GSPC_data, on = 'Date')
    df = pd.merge(df, IXIC_data, on = 'Date')
    df = pd.merge(df, DJI_data, on = 'Date')
    df = pd.merge(df, PEN_X_data, on = 'Date')
    df = pd.merge(df, BZ_F_data, on = 'Date')
    df = df.drop(['Volume_PEN_X'], axis=1)
    return df.tail(5)

def predict(model, data):
    df = model.predict(data)
    return df
