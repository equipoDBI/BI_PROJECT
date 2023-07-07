import streamlit as st
from utils import *
import datetime
from PIL import Image

with st.sidebar:
    
    image = Image.open('FONDO.png')
    st.image(image, caption='SEMANA 15')
    st.title('EQUIPO D')
    st.header('CURSO: INTELIGENCIA DE NEGOCIOS')
    st.subheader('DOCENTE NOMBRADO: ERNESTO CANCHO')
    st.markdown('**Integrantes:**')    
    st.markdown('*    Hurtado Santos, Estiven Salvador - 20200135')
    st.markdown('*   López Terrones, Ximena Xiomy - 20200020')
    st.markdown('*   Llactahuaman Muguerza, Anthony Joel - 20200091')
    st.markdown('*   Mondragón Zúñiga, Rubén Alberto - 20200082')
    st.markdown('*   Morales Robladillo, Nicole Maria - 20200136')
    st.markdown('*   Aquije Vásquez, Carlos Adrian - 19200319')
    st.markdown('*   Cespedes Flores, Sebastian - 1820025')    


st.header('Modelo Predictivo para BUENAVENTURA con SVC')
st.markdown('En este apartado se explicará el modelo predictivo usado para realizar un sistema de recomendación para las acciones de Buenaventura')
datForTraining=trainingData()
st.table(datForTraining)
st.header('HeatMap')
st.write(plotHeatMap(datForTraining))



initDate = datetime.date(2023, 1, 1)


endDate = st.date_input(
    "Seleccione la fecha de inicio",
    datetime.date(2023, 1, 12))
st.write('Fecha Fin: ', endDate)


st.table(noNormalizedData(initDate, endDate))


container = st.container()
container.write("La predicción es: ")
st.write(makePredition(initDate, endDate))

# Now insert some more in the container
