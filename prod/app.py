import yfinance as yf
import streamlit as st
from utils import *
import datetime
from PIL import Image
from streamlit_option_menu import option_menu
streamlit_style = """
			<style>
			@import url("https://fonts.googleapis.com/css2?family=Poppins:wght@300&display=swap");
			html, body, [class*="css"]  {
			    font-family: 'Poppins', sans-serif;
			}
            .css-vk3wp9, .css-1544g2n, .css-6qob1r {
                background-image: linear-gradient(#7bf6f7,#7bf6f7);
                color: white;
            }
			</style>
			"""
st.markdown(streamlit_style, unsafe_allow_html=True)
with st.sidebar:
    image = Image.open('prod/BusinessIntelligence.png')
    st.image(image)
    selected = option_menu(
        menu_title="Menú",
        options=["Inicio", "LSTM", "SVC", "SVR", "Clustering K-Means"],
        icons=["house-fill", "1-circle-fill",
               "2-circle-fill", "3-circle-fill", "4-circle-fill"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "rgba(0, 147, 171, 0.458)"},
            "nav-link-selected": {"background-color": "#ffd124", "color": "black", "font-weight": "300"},
        },
    )
if selected == "Inicio":
    st.title("Sistema de Inteligencia para Bolsa de Valores - Equipo D")
    image = Image.open('prod/MachineLearningStocksMarket.png')
    st.image(image)
    st.markdown('**Curso:** Inteligencia de Negocios')
    st.markdown('**Docente Nombrado:** Ernesto Cancho')
    st.markdown('**Integrantes:**')
    st.markdown('*   Hurtado Santos, Estiven Salvador - 20200135')
    st.markdown('*   López Terrones, Ximena Xiomy - 20200020')
    st.markdown('*   Llactahuaman Muguerza, Anthony Joel - 20200091')
    st.markdown('*   Mondragón Zúñiga, Rubén Alberto - 20200082')
    st.markdown('*   Morales Robladillo, Nicole Maria - 20200136')
    st.markdown('*   Aquije Vásquez, Carlos Adrian - 19200319')
    st.markdown('*   Cespedes Flores, Sebastian - 1820025')
if selected == "LSTM":
    st.title("Modelo de predicción LSTM")
    tickerSymbol = 'GOOGL'
    tickerData = yf.Ticker(tickerSymbol)
    tickerDf = tickerData.history(
        period='1d', start='2010-5-31', end='2020-5-31')
    st.line_chart(tickerDf.Close)
    st.line_chart(tickerDf.Volume)
if selected == "SVC":
    st.title("Modelo de predicción SVC")
    st.header('Modelo Predictivo para BUENAVENTURA con SVC')
    st.markdown('En este apartado se explicará el modelo predictivo usado para realizar un sistema de recomendación para las acciones de Buenaventura')
    datForTraining = trainingData()
    st.table(datForTraining)
    st.header('HeatMap')
    st.write(plotHeatMap(datForTraining))
    initDate = datetime.date(2023, 1, 1)
    endDate = st.date_input(
        "Seleccione la fecha de inicio",
        datetime.date(2023, 1, 11))
    st.write('Fecha Fin: ', endDate + datetime.timedelta(days=1))
    st.table(noNormalizedData(initDate, endDate + datetime.timedelta(days=1)))
    container = st.container()
    container.write("La predicción es: ")
    st.write(makePredition(initDate, endDate))
if selected == "SVR":
    st.title("Modelo de predicción SVR")
if selected == "Clustering K-Means":
    st.title("Modelo de predicción Clustering K-Means")
