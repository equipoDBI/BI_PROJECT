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
    st.markdown('**Equipo:** Equipo D')
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
if selected == "SVC":
    st.title("Modelo de predicción SVC")
    st.header('1. Entrenamiento')
    st.subheader('1.1. Ingreso de datos')
    instrumentoFinancieroSVC = st.text_input(
        'Instrumento Financiero', 'BVN', key="placeholder")
    fechaInicioEntrenamientoSVC = st.date_input(
        "Fecha inicio para el entrenamiento", datetime.date(2018, 1, 1))
    fechaFinEntrenamientoSVC = st.date_input(
        "Fecha fin para el entrenamiento", datetime.date(2022, 1, 1))
    dataParaEntrenamientoSVC = obtenerData(
        instrumentoFinancieroSVC, fechaInicioEntrenamientoSVC, fechaFinEntrenamientoSVC)
    st.subheader('1.2. Visualización')
    st.table(dataParaEntrenamientoSVC.head(10))
    st.subheader('1.3. HeatMap')
    st.write(plotHeatMap(dataParaEntrenamientoSVC))
    st.header('2. Predicción')
    st.subheader('2.1. Ingreso de datos')
    fechaInicioPrediccionSVC = st.date_input(
        "Fecha inicio para la predicción",
        datetime.date(2023, 6, 8))
    fechaFinPrediccionSVC = st.date_input(
        "Fecha fin para la predicción",
        datetime.date(2023, 7, 7))
    dataParaPrediccionSVC = obtenerData(
        instrumentoFinancieroSVC, fechaInicioPrediccionSVC, fechaFinPrediccionSVC)
    st.subheader('2.2. Visualización')
    st.table(dataParaPrediccionSVC.tail(10))
    st.subheader('2.3. Predicciones')
    containerSVC = st.container()
    resultadosSVC = hacerPrediccion(instrumentoFinancieroSVC, fechaInicioEntrenamientoSVC,
                                    fechaFinEntrenamientoSVC, fechaInicioPrediccionSVC, fechaFinPrediccionSVC, selected)
    for i in resultadosSVC:
        st.write(i)
if selected == "SVR":
    st.title("Modelo de predicción SVR")
    st.header('1. Entrenamiento')
    st.subheader('1.1. Ingreso de datos')
    instrumentoFinancieroSVR = st.text_input(
        'Instrumento Financiero', 'BVN', key="placeholder")
    fechaInicioEntrenamientoSVR = st.date_input(
        "Fecha inicio para el entrenamiento", datetime.date(2018, 1, 1))
    fechaFinEntrenamientoSVR = st.date_input(
        "Fecha fin para el entrenamiento", datetime.date(2022, 1, 1))
    dataParaEntrenamientoSVR = obtenerData(
        instrumentoFinancieroSVR, fechaInicioEntrenamientoSVR, fechaFinEntrenamientoSVR)
    st.subheader('1.2. Visualización')
    st.table(dataParaEntrenamientoSVR.head(10))
    st.subheader('1.3. HeatMap')
    st.write(plotHeatMap(dataParaEntrenamientoSVR))
    st.header('2. Predicción')
    st.subheader('2.1. Ingreso de datos')
    fechaInicioPrediccionSVR = st.date_input(
        "Fecha inicio para la predicción",
        datetime.date(2023, 6, 8))
    fechaFinPrediccionSVR = st.date_input(
        "Fecha fin para la predicción",
        datetime.date(2023, 7, 7))
    dataParaPrediccionSVR = obtenerData(
        instrumentoFinancieroSVR, fechaInicioPrediccionSVR, fechaFinPrediccionSVR)
    st.subheader('2.2. Visualización')
    st.table(dataParaPrediccionSVR.tail(10))
    st.subheader('2.3. Predicciones')
    containerSVR = st.container()
    resultadosSVR = hacerPrediccion(instrumentoFinancieroSVR, fechaInicioEntrenamientoSVR,
                                    fechaFinEntrenamientoSVR, fechaInicioPrediccionSVR, fechaFinPrediccionSVR, selected)
    for i in resultadosSVR:
        st.write(i)
if selected == "Clustering K-Means":
    st.title("Modelo de predicción Clustering K-Means")
