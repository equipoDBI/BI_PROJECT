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
        options=["Inicio", "LSTM", "SVC", "SVR"],
        icons=["house-fill", "1-circle-fill",
               "2-circle-fill", "3-circle-fill"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "rgba(0, 147, 171, 0.458)"},
            "nav-link-selected": {"background-color": "#ffd124", "color": "black", "font-weight": "300"},
        },
    )
definiciones = {
    'LSTM': 'LSTM es la abreviatura de "Long Short-Term Memory" (Memoria a Largo Plazo de Corto Plazo), que es un tipo de red neuronal recurrente (RNN, por sus siglas en inglés) especializada en el procesamiento de secuencias de datos. Las redes LSTM se utilizan ampliamente en tareas de aprendizaje automático y procesamiento del lenguaje natural debido a su capacidad para manejar dependencias a largo plazo en las secuencias. A diferencia de las redes neuronales recurrentes tradicionales, las LSTM están diseñadas para superar el problema de la desaparición o explosión del gradiente, lo cual ocurre cuando se propagan los errores a través de múltiples pasos de tiempo en una RNN.',
    'SVC': 'Support Vector Classification (Clasificación de Vectores de Soporte): SVC es un algoritmo de aprendizaje automático utilizado para la clasificación de datos. Es una técnica de aprendizaje supervisado que se basa en la teoría de los vectores de soporte. El objetivo del SVC es encontrar un hiperplano en un espacio de alta dimensión que clasifique de manera óptima los datos en diferentes categorías.',
    'SVR': 'SVR significa "Support Vector Regression" (Regresión de Vectores de Soporte). SVR es un algoritmo de aprendizaje automático utilizado para realizar tareas de regresión, es decir, predecir valores numéricos continuos en lugar de clasificar datos en categorías. Similar al algoritmo de clasificación de vectores de soporte (SVC), SVR también se basa en la teoría de los vectores de soporte. El objetivo de SVR es encontrar una función de regresión óptima que se ajuste a los datos de entrenamiento mientras minimiza el error de predicción.'
}
if selected == "Inicio":
    st.title("Sistema de Inteligencia para Bolsa de Valores - Equipo D")
    image = Image.open('prod/MachineLearningStocksMarket.png')
    st.image(image)
    st.markdown('**Curso:** Inteligencia de Negocios')
    st.markdown('**Docente Líder del Proyecto:** Mg. Ing. Ernesto Cancho-Rodríguez, MBA de la George Washington University ecr@gwu.edu')
    st.markdown('**Equipo:** Equipo D')
    st.markdown('**Integrantes:**')
    st.markdown('*   Hurtado Santos, Estiven Salvador - 20200135')
    st.markdown('*   López Terrones, Ximena Xiomy - 20200020')
    st.markdown('*   Llactahuaman Muguerza, Anthony Joel - 20200091')
    st.markdown('*   Mondragón Zúñiga, Rubén Alberto - 20200082')
    st.markdown('*   Morales Robladillo, Nicole Maria - 20200136')
    st.markdown('*   Aquije Vásquez, Carlos Adrian - 19200319')
    st.markdown('*   Cespedes Flores, Sebastian - 1820025')
elif selected == "LSTM":
    st.title("Modelo de predicción " + selected)
    st.markdown(definiciones[selected])
    st.header('1. Entrenamiento')
    st.subheader('1.1. Parámetros para el entrenamiento')
    st.markdown('Para el siguiente sistema de recomendación de inversión en la Bolsa de Valores, se considera invertir en 5 Empresas mineras:')
    st.markdown('*   BHP Billiton Limited (BHP)')
    st.markdown('*   Minera Buenaventura (BVN)')
    st.markdown('*   Fortuna Silver Mines (FSM)')
    st.markdown('*   Barrick Gold Corporation (GOLD)')
    st.markdown('*   Minera Southern Copper (SCCO)')
    st.markdown('**Empresa en la cual se piensa invertir**')
    instrumentoFinanciero = st.selectbox(
        'Seleccione la empresa en la cual se piensa invertir', ('BHP', 'BVN', 'FSM', 'GOLD', 'SCCO'))
    st.markdown('**Fecha inicio para el entrenamiento**')
    st.markdown(datetime.date(2018, 1, 1))
    st.markdown('**Fecha fin para el entrenamiento**')
    st.markdown(datetime.date(2022, 1, 1))
    st.markdown('**Inputs para la predicción: Múltiples**')
    st.markdown('*   Mercado de Commodities:')
    st.markdown('Oro (GLD)')
    st.markdown('Plata (SLV)')
    st.markdown('Cobre (COPX)')
    st.markdown('*   Índices:')
    st.markdown('Índice SP500 (^GSPC)')
    st.markdown('Índice NASDAQ (^IXIC)')
    st.markdown('Índice Dow Jones (^DJI)')
    st.markdown('*   Mercado de divisas y criptomonedas:')
    st.markdown('US Dollar vs Peruvian Nuevo Sol (PEN=X)')
    st.markdown('Brent Crude Oil Last Day Financial (BZ=F)')
    fechaInicioEntrenamiento = datetime.date(2018, 1, 1)
    fechaFinEntrenamiento = datetime.date(2022, 1, 1)
    dataParaEntrenamiento = obtenerData(
        instrumentoFinanciero, fechaInicioEntrenamiento, fechaFinEntrenamiento)
    st.subheader('1.2. Visualización de los datos de entrenamiento')
    st.markdown('En stock trading, el alto y el bajo se refieren a los precios máximos y mínimos en un período de tiempo determinado. Apertura y cierre son los precios a los que comenzó y terminó una acción trading en el mismo período. El volumen es la cantidad total de trading actividad: factor de valores ajustados en acciones corporativas como dividendos, división de acciones y emisión de nuevas acciones.')
    st.table(dataParaEntrenamiento.head(10))
    st.subheader('1.3. HeatMap')
    st.markdown('Un Heatmap (mapa de calor en español) es una representación visual de datos en forma de una matriz de colores en una cuadrícula. Se utiliza para resaltar patrones y variaciones en los datos mediante el uso de colores para representar valores numéricos o de otro tipo. En un Heatmap, cada celda de la cuadrícula está asociada a un valor, y el color utilizado para esa celda indica el nivel o la magnitud de ese valor. Los colores pueden variar desde tonos fríos (como azules o verdes) para valores bajos, hasta tonos cálidos (como rojos o amarillos) para valores altos. De esta manera, se puede visualizar fácilmente la distribución y la intensidad de los valores en la cuadrícula. En esta gráfica se relacionan los inputs de la predicción para ver que tan correlacionados están los unos con los otros. Más relacionados se representan con tonos muy claros o muy oscuros (extremos).')
    st.write(plotHeatMap(dataParaEntrenamiento))
    st.subheader(
        '1.4. Correlación de los inputs de la predicción + Return con la etiqueta Trend')
    st.markdown('*   **Retorno:** El retorno, también conocido como retorno financiero, en sus términos más simples, es el dinero ganado o perdido en una inversión durante un período de tiempo. Un retorno puede expresarse nominalmente como el cambio en el valor en dólares de una inversión a lo largo del tiempo. Un retorno también se puede expresar como un porcentaje derivado de la relación entre la ganancia y la inversión. Los retornos también se pueden presentar como resultados netos (después de tarifas, impuestos e inflación) o retonos brutos que no representan nada más que el cambio de precio. Para este sistema, se considera al retorno como el cambio porcentual del precio de cierre (Close) con respecto al precio del día anterior.')
    st.markdown('*   **Trend:** La etiqueta Trend está representada por 1 o 0, donde 1 significa que se debería de invertir y 0 que no. Se calcula considerando que se debería invertir cuando el retorno es mayor a 0.')
    st.markdown('*   **Coeficiente de correlación:** Un valor menor que 0 indica que existe una correlación negativa, es decir, que las dos variables están asociadas en sentido inverso. Cuando el valor en una sea muy alto, el valor en la otra será muy bajo. Por otro lado, un valor mayor que 0 indica que existe una correlación positiva. En este caso las variables estarían asociadas en sentido directo. Cuanto más cerca de +1, más alta es su asociación.')
    st.markdown('De la tabla se interpreta que las entradas más relacionadas a la etiqueta Return son los primeros y últimos registros de la tabla.')
    st.table(obtenerCorrelacionDeReturnConLosInputsLSTM(
        instrumentoFinanciero, fechaInicioEntrenamiento, fechaFinEntrenamiento))
    st.header('2. Predicción')
    st.subheader('2.1. Ingreso de datos')
    fechaInicioPrediccion = datetime.date(2022, 6, 8)
    fechaFinPrediccion = st.date_input(
        "Fecha fin para la predicción",
        datetime.date(2023, 7, 7))
    dataParaPrediccion = obtenerData(
        instrumentoFinanciero, fechaInicioPrediccion, fechaFinPrediccion)
    st.subheader(
        '2.2. Visualización de los datos reales en el intervalo de tiempo que se busca predecir')
    st.table(dataParaPrediccion.tail(10))
    st.subheader(
        '2.3. Variación del retorno')
    st.pyplot(obtenerGraficaRetornoLSTM(instrumentoFinanciero,
                                    fechaInicioPrediccion, fechaFinPrediccion))
    st.subheader('2.4. Retorno acumulado VS Retorno estratégico')
    st.markdown("Amarillo = Retorno estratégico")
    st.markdown("Verde = Retorno acumulado")
    st.pyplot(obtenerGraficaRetornoAcumuladoVSEstrategicoParaLSTM(
        instrumentoFinanciero, fechaInicioPrediccion, fechaFinPrediccion, selected))
    st.subheader('2.5. Predicciones')
    container = st.container()
    resultados = hacerPrediccion(
        instrumentoFinanciero, fechaInicioPrediccion, fechaFinPrediccion, selected)
    for i in resultados:
        st.write(resultados)
else:
    st.title("Modelo de predicción " + selected)
    st.markdown(definiciones[selected])
    st.header('1. Entrenamiento')
    st.subheader('1.1. Parámetros para el entrenamiento')
    st.markdown('Para el siguiente sistema de recomendación de inversión en la Bolsa de Valores, se considera invertir en 5 Empresas mineras:')
    st.markdown('*   BHP Billiton Limited (BHP)')
    st.markdown('*   Minera Buenaventura (BVN)')
    st.markdown('*   Fortuna Silver Mines (FSM)')
    st.markdown('*   Barrick Gold Corporation (GOLD)')
    st.markdown('*   Minera Southern Copper (SCCO)')
    st.markdown('**Empresa en la cual se piensa invertir**')
    instrumentoFinanciero = st.selectbox(
        'Seleccione la empresa en la cual se piensa invertir', ('BHP', 'BVN', 'FSM', 'GOLD', 'SCCO'))
    st.markdown('**Fecha inicio para el entrenamiento**')
    st.markdown(datetime.date(2018, 1, 1))
    st.markdown('**Fecha fin para el entrenamiento**')
    st.markdown(datetime.date(2022, 1, 1))
    st.markdown('**Inputs para la predicción: Múltiples**')
    st.markdown('*   Mercado de Commodities:')
    st.markdown('Oro (GLD)')
    st.markdown('Plata (SLV)')
    st.markdown('Cobre (COPX)')
    st.markdown('*   Índices:')
    st.markdown('Índice SP500 (^GSPC)')
    st.markdown('Índice NASDAQ (^IXIC)')
    st.markdown('Índice Dow Jones (^DJI)')
    st.markdown('*   Mercado de divisas y criptomonedas:')
    st.markdown('US Dollar vs Peruvian Nuevo Sol (PEN=X)')
    st.markdown('Brent Crude Oil Last Day Financial (BZ=F)')
    fechaInicioEntrenamiento = datetime.date(2018, 1, 1)
    fechaFinEntrenamiento = datetime.date(2022, 1, 1)
    dataParaEntrenamiento = obtenerData(
        instrumentoFinanciero, fechaInicioEntrenamiento, fechaFinEntrenamiento)
    st.subheader('1.2. Visualización de los datos de entrenamiento')
    st.markdown('En stock trading, el alto y el bajo se refieren a los precios máximos y mínimos en un período de tiempo determinado. Apertura y cierre son los precios a los que comenzó y terminó una acción trading en el mismo período. El volumen es la cantidad total de trading actividad: factor de valores ajustados en acciones corporativas como dividendos, división de acciones y emisión de nuevas acciones.')
    st.table(dataParaEntrenamiento.head(10))
    st.subheader('1.3. HeatMap')
    st.markdown('Un Heatmap (mapa de calor en español) es una representación visual de datos en forma de una matriz de colores en una cuadrícula. Se utiliza para resaltar patrones y variaciones en los datos mediante el uso de colores para representar valores numéricos o de otro tipo. En un Heatmap, cada celda de la cuadrícula está asociada a un valor, y el color utilizado para esa celda indica el nivel o la magnitud de ese valor. Los colores pueden variar desde tonos fríos (como azules o verdes) para valores bajos, hasta tonos cálidos (como rojos o amarillos) para valores altos. De esta manera, se puede visualizar fácilmente la distribución y la intensidad de los valores en la cuadrícula. En esta gráfica se relacionan los inputs de la predicción para ver que tan correlacionados están los unos con los otros. Más relacionados se representan con tonos muy claros o muy oscuros (extremos).')
    st.write(plotHeatMap(dataParaEntrenamiento))
    st.subheader(
        '1.4. Correlación de los inputs de la predicción + Return con la etiqueta Trend')
    st.markdown('*   **Retorno:** El retorno, también conocido como retorno financiero, en sus términos más simples, es el dinero ganado o perdido en una inversión durante un período de tiempo. Un retorno puede expresarse nominalmente como el cambio en el valor en dólares de una inversión a lo largo del tiempo. Un retorno también se puede expresar como un porcentaje derivado de la relación entre la ganancia y la inversión. Los retornos también se pueden presentar como resultados netos (después de tarifas, impuestos e inflación) o retonos brutos que no representan nada más que el cambio de precio. Para este sistema, se considera al retorno como el cambio porcentual del precio de cierre (Close) con respecto al precio del día anterior.')
    st.markdown('*   **Trend:** La etiqueta Trend está representada por 1 o 0, donde 1 significa que se debería de invertir y 0 que no. Se calcula considerando que se debería invertir cuando el retorno es mayor a 0.')
    st.markdown('*   **Coeficiente de correlación:** Un valor menor que 0 indica que existe una correlación negativa, es decir, que las dos variables están asociadas en sentido inverso. Cuando el valor en una sea muy alto, el valor en la otra será muy bajo. Por otro lado, un valor mayor que 0 indica que existe una correlación positiva. En este caso las variables estarían asociadas en sentido directo. Cuanto más cerca de +1, más alta es su asociación.')
    st.markdown('De la tabla se interpreta que las entradas más relacionadas a la etiqueta Trend son los primeros y últimos registros de la tabla.')
    st.table(obtenerCorrelacionDeTrendConLosInputs(
        instrumentoFinanciero, fechaInicioEntrenamiento, fechaFinEntrenamiento))
    st.header('2. Predicción')
    st.subheader('2.1. Ingreso de datos')
    fechaInicioPrediccion = st.date_input(
        "Fecha inicio para la predicción",
        datetime.date(2023, 6, 8))
    fechaFinPrediccion = st.date_input(
        "Fecha fin para la predicción",
        datetime.date(2023, 7, 7))
    dataParaPrediccion = obtenerData(
        instrumentoFinanciero, fechaInicioPrediccion, fechaFinPrediccion)
    st.subheader(
        '2.2. Visualización de los datos reales en el intervalo de tiempo que se busca predecir')
    st.table(dataParaPrediccion.tail(10))
    st.subheader(
        '2.3. Variación del retorno')
    st.pyplot(obtenerGraficaRetorno(instrumentoFinanciero,
                                    fechaInicioPrediccion, fechaFinPrediccion))
    st.subheader('2.4. Retorno acumulado VS Retorno estratégico')
    st.markdown("Amarillo = Retorno estratégico")
    st.markdown("Verde = Retorno acumulado")
    st.pyplot(obtenerGraficaRetornoAcumuladoVSEstrategico(
        instrumentoFinanciero, fechaInicioPrediccion, fechaFinPrediccion, selected))
    st.subheader('2.5. Predicciones')
    container = st.container()
    resultados = hacerPrediccion(
        instrumentoFinanciero, fechaInicioPrediccion, fechaFinPrediccion, selected)
    for i in resultados:
        st.write(i)
