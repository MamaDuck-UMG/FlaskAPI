#import streamlit as st #funciones generales de streamlit
from bokeh.plotting import figure as grafica #para mostrar graficas de lineas
import plotnine as p9 #pip install plotnine, para graficas de puntos y de lineas
from bokeh.models import ColumnDataSource#para importar datos de tablas
from PIL import Image #para abrir imagenes
import numpy as np#para arrays
import pandas as pd#para dataframes
import streamlit.components.v1 as components#para importar y exportar elementos de archivos
import pydeck as pdk#para los mapas 
import datetime#libreria para usar formatos de fechas 
import json#libreria para usar json
import matplotlib.pyplot as plt
from pyvis import network as net
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  
from queue import PriorityQueue #libreria colas de prioridad
import math #Para los infinitos
import warnings
from flask import Flask,jsonify,request, render_template #render template es para mostrar paginas html o js en un endpoint
warnings.filterwarnings('ignore')
##funciones en crudo

with open('ejemplo.json') as file:
    datajson = json.load(file)#cargamos el archivo a una variable
    jnodes=datajson.get("nodes")
    jn1=jnodes[0]#esta variable guarda todo el json, recordar que su acceso es similara a diccionaros de dict o listas



strindices=[]#este string va a guardar el nombre de los nodos (las funciones get del diccionario solo aceptan strings)
for i in range(len(jn1)):
  strindices.append(str(i+1))#esta ya puede utilizarse para otras funciones

def obtencionCoords():#obtengo las coordenadas de latitud y longitud de todos los nodos
  #lat y lon 1 es para coordenadas de llamadas
  #lat  lon2 son para coordenadas de NODOS
  list1=list()
  list2=list()
  for s in strindices:
    for i in range(len(jn1.get(str(s))[0]['history'])):
      auxlat=jn1.get(str(s))[0]['history'][i]['localization'][0]['lat']
      auxlon=jn1.get(str(s))[0]['history'][i]['localization'][0]['long']
      list1.append([auxlat,auxlon])

    list2.append([jn1.get(str(s))[0]['localization'][0]['lat'],jn1.get(str(s))[0]['localization'][0]['long']])
  a1=np.array(list1)
  a2=np.array(list2)
  #print(a1.reshape(-2,2))
  df1 = pd.DataFrame(
  a1,
  columns=['lat', 'lon'])
  #return df1, df2
  #print(a1.reshape(-2,2))
  df2 = pd.DataFrame(
  a2,
  columns=['lat', 'lon'])
  #return df1, df2
  return df1,df2


#RUTINA DE NODOS
class Nodo:#estructura de Nodo para recorrido de nodos
    w=None
    h = None  # heuristica
    f = math.inf  # f es infinito

    def __init__(self, tag, conexiones, padre, mamaduck, personasconect, tiempoactivos,adyacentes):  # inicializador
        self.tag = tag
        self.conexiones = conexiones
        self.padre = padre
        self.mamaduck = mamaduck
        self.tiempoactivos = tiempoactivos
        self.adyacentes = adyacentes#adyacentes=adyacentes primarios
        if padre is not None:  # si es un hijo
            self.g = padre.g + 1
        else:  # si es padre
            self.g = 0
            self.f = 0  # su f debe valer 0
        self.h = personasconect
    def expand(self, adyacentes):  # nos va a decir como explorar el grafo, obtiene la raiz en primera instancia
        ady = []  # aqui guardaremos los adyacentes que son objetos del tipo nodo
        for i in adyacentes:
            tag,conections,parent,mamaduck,people,timeactive,list1=DataJSONtoGraph(i)
            ady.append(Nodo(tag, conections, self, mamaduck, people, timeactive,list1))
        return ady  # retornamos los adyacentes o hijos del nodo expandido




    #funcion realizada el primer parcial de IA: recorrido de nodos por prioridad
def recorridonodos(n1):  # recibe una cadena de valores
    n0 = n1  # lo cofiguro como un nodo tipo puzzle
    Q = PriorityQueue()  # Q es una cola de prioridad
    aux = 0  # otro indice secundario de prioridad para Q
    Q.put((n0.f, aux, n0))  # la cola de prioridad se manejará mediante los f(n)
    visitados = []
    visitadoschart = []
    while not Q.empty():  # mientras la cola no este vacía
        u = Q.get()  # con el metodo get se guarda en u pero se quita de la cola el elemento
        u = u[2]  # porque una cola de prioridad almacena tuplas de prioridad,contador,nodo
        if u.tag not in visitados:
            visitados.append(u.tag)  # para evitar volverlo a visitar
            visitadoschart.append(u)
        ady = u.expand(u.adyacentes)  # expand me genera una lista de adyacencia, con heuristica y señalando a su padre, establece costo de 1 al generar neuvo nodo
        for v in ady:  # explorar los vecinos
            if v.tag not in visitados:  # si todavia no esta en visitados
                fp = v.h + v.g  # cálculo de funciones
                if fp < v.f:
                    v.f = fp
                    aux = aux + 1  # debo tener un entero antes de insertar un nodo en prioridad
                    Q.put((-(v.f), aux,
                            v))  # lo colocamos en la cola, para que en cada ciclo se evite agregar uno repetido
    return visitados, visitadoschart #visitados son solo los valores, visitadoschart son objetos nodo

def functionWhyPriority(result):#muestra detalles de los nodos visitados
    #para saber por qué se eligio un numero como prioridad
    chart=list()#una lista con tupla de valores para una tabla
    ismd=""#variable para preguntar: Is Mama Duck?
    for u in result:
        if u.mamaduck==True:
            ismd="Sí"
        else:
            ismd="No"
    chart.append([u.tag,u.conexiones,ismd,u.h,u.adyacentes])
    df=pd.DataFrame(chart)#Creo un dataframe de Pandas para ilustrar la info en una tabla
    df.columns = ["No. de nodo", "No. nodos conectados", "Es Mamaduck?","Llamadas recibidas","Adyacentes"]
    #st.write(df)
    return df
def DataJSONtoGraph(numnodo):#funcion para extraer datos de acuerdo con el formato del algoritmo de recorrido
    #exclusivo para algoritmo de recorrido
    ismamaduck=False
    numcalls=0
    activetime=None
    numcalls=len(jn1.get(numnodo)[0]['history'])#ya esta listo numcalls
    numConected=len(jn1.get(numnodo)[0]['conections'])
    if numnodo=='1':
        ismamaduck=True
    adys=jn1.get(numnodo)[0]['conections']
    return int(numnodo),numConected,None,ismamaduck,numcalls,activetime,adys
##funciones en crudo
def routinemap():
        #st.header("Mapa de nodos y emergencias")
    #st.write("Hexagonos: ubicacion de nodos, Círculo verde: casos de emergencia")
    df1,df2=obtencionCoords()
    # if st.checkbox('Mostrar tabla de coordenadas de emergencia'):
    #   st.write(df1)
    #  if st.checkbox('Mostrar tabla de coordenadas de Nodos'):
    #    st.write(df2)
    view_state=pdk.ViewState(
        latitude=20.63494981128319,#lat y lon inicial 
        longitude=-103.40648023281342,
        zoom=16,
        pitch=40.5,
        bearing=-27.36
    )
    layers1=pdk.Layer(
            'HexagonLayer',#puntos en forma de hexagono, es para nodos
            data=df2,#aqui obtengo datos de lat y lon
            get_position='[lon, lat]',
            radius=3,
            elevation_scale=4,
            elevation_range=[0, 10],
            pickable=True,
            extruded=True,
            auto_highlight=True,
            coverage=1
        )
    layers2= pdk.Layer(
            'ScatterplotLayer',#puntos, es para emergencias
            data=df1,#aqui obtengo datos de lat y lon
            get_position='[lon, lat]',
            get_color='[100, 230, 0, 160]',
            get_radius=2,
        )

    # Render
    r = pdk.Deck(layers=[layers1,layers2], initial_view_state=view_state)
    r.to_html('templates/mapa.html')
    return render_template('mapa.html')

def routinenodes():
    
    #st.title("Nodos receptores Mama Duck")
    #st.write("A continución se muestra la distribución de nodos de Mama Duck, sus interconexiones e información")
    g=net.Network(height='400px', width='60%')
    colacolores=['teal', 'yellowgreen','purple','blue']#para que muestre colores distintos en cada nodo, hare un pop()
    for i in range(len(jn1)):
    #issue: no detecta los saltos de linea
        g.add_node(i+1,title=jn1.get(str(strindices[i]))[0].get('name')+
    """\n"""+ """Status: """+jn1.get(str(strindices[i]))[0].get('status')+ """
    Risk index:"""+str(jn1.get(str(strindices[i]))[0].get('risks')[0]),color=colacolores.pop(),borderWidthSelected=3,labelHighlightBold=True)
    for sti in strindices:
        for s in range(len(jn1.get(sti)[0]['conections'])):
            aux=jn1.get(sti)[0]['conections']             
            g.add_edge(int(sti),int(aux[s]),color='black')
    #guardo grafico
    g.save_graph('templates/graph.html')#en un archivo
    #g.show('graph.html')
    #HtmlFile=open('/templates/graph.html','r',encoding='utf-8')
    #sourceCode=HtmlFile.read()
    #components.html(sourceCode,height=400,width=1500)
    #st.header("Monitoreo de nodos")
    #aqui empieza la implementacion de recorrido de Grafo
    tag,conections,parent,mamaduck,people,timeactive,list1=DataJSONtoGraph('1')#con el nodo 1 se inicia para empezar a
    #crear los demas grafos
    n1=Nodo(tag,conections,parent,mamaduck,people,timeactive,list1)
    res,restab=recorridonodos(n1)#iniciamos el algoritmo de A estrella en una variable de objeto
    print("El Nodo: %d "%res[1],"requiere ser monitoreado prioritariamente.")#en res 1 está el nodo elegido como prioritario, esta función imprimirá el porqué primero este y porque los demas
    #st.header("Orden de prioridad")
    print(res)
    #if st.button("Más detalles..."):
        #if not restab and not res:
        #st.write("Primero presione en el botón de Monitoreo de nodos para presentarle detalles")
        #else:
    df=functionWhyPriority(restab)
    df.to_html('templates/analisisgrafo.html')
    return render_template('graph.html')
#https://ichi.pro/es/paneles-en-python-para-principiantes-y-todos-los-demas-que-usan-dash-264571154097732

#https://www.youtube.com/watch?v=N9yHClPGWG4



app=Flask(__name__)
@app.route("/main")
def index():
    #return render_template('bienvenido.html')#la plantilla que tenga de bienvenida en templates
    #image = Image.open('duck.png')#abro el ícono de Mama Duck
    print("Bienvenido a Mama Duck")#encabezado
    var=open('templates/output_file.txt')
    print(var.read())
    #st.image(image, caption='Mama Duck',width=80)#subo la imagen con su tamaño y pie de foto
    #importamos el json que es actualmente un archivo de ejemplo
    return str(len(jn1.get("1")[0]['history']))
#flask logica streamlit api rest
#paquete dash python
#D3JS javascript
@app.route("/nodes",methods=['GET'])
def analytics():
    return routinenodes()

@app.route("/map", methods=['GET'])
def map():
    return routinemap()
@app.route("/profile", methods=['GET'])
def profile():
    routinenodes()
    routinemap()
  #  h=dash.Dash(dev_tools_hot_reload=True)
   # h.scripts.config.serve_locally = True
 #   h.config['suppress_callback_exceptions'] = True
   # h.layout=html.Div(children=[html.H1(children='Bienvenido a Mama Duck'),
   #             html.Div(children='''Nodos conectados'''),
    #            ])
   # dcc.Dropdown(
   #     id='Tabla Nodos',
    #    options=[{'label': e, 'value': e} for e in b],
    #    value='Tabla Nodos',
    #    clearable=False
  #  ),
    
   # dash_table.DataTable(id='Tabla Nodos')
    return render_template("profilef.html")#este html fue creado manualmente para unir las dos paginas
    #como si fueran componentes
    #h.run_server()

@app.route("/train", methods=['GET'])

#este modelo entrenará todos los métodos y cada uno los guardará en un joblib
def train():

    return "prueba"






#main
app.run(host="0.0.0.0")

#https://www.iartificial.net/regresion-polinomica-en-python-con-scikit-learn/

#https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html#sphx-glr-auto-examples-linear-model-plot-polynomial-interpolation-py

#https://colab.research.google.com/drive/1rNIUjVRzQnFcvojyzCcGQqc2r0UcGUzD#scrollTo=5xEo0-arRaZz
#https://machinelearningmastery.com/make-predictions-scikit-learn/