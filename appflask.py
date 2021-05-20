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
import matplotlib.pyplot as plt
from joblib import dump, load

from sklearn.linear_model import Ridge#regularizacion o penalizacion del modelo
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from flask import Flask,jsonify,request, render_template #render template es para mostrar paginas html o js en un endpoint
warnings.filterwarnings('ignore')
grado=10
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

def rellenofaltantes(x,y):
    x_plot = np.linspace(0,23,num=24)
    for numero in range(0,len(x_plot)):
        if numero not in x:
            x.append(numero)
            y.append(0)
    return x,y

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
@app.route("/upload_data", methods=['POST'])

#este modelo entrenará todos los métodos y cada uno los guardará en un joblib
def upload_data():

    return "prueba"




@app.route("/trainNumEmergencias", methods=['GET'])

#este modelo entrenará todos los métodos y cada uno los guardará en un joblib
def trainPR():

    for numeronodo in strindices:
        _,_,c,d,_,_,_=obtencionlistasJS(str(numeronodo))#obtendre todas las variables que regresa en el orden documentado
        c,d=rellenofaltantes(c,d)
        regresionPolinomialNumEmergencias(c,d,"trainPR-%s"%str(numeronodo),grado)
    response={
        'message':'Todos los modelos de regresion polinomial se han creado',
        'carpeta':'modelos'
    }
    return jsonify(response)

@app.route("/predictNumEmergencias", methods=['POST'])
def predictPR():
    #descerializar
    req=request.get_json(force=True)
    #datos del json POST
        #node:1
        #hour:16
    nodouser=req['node']
    horauser=req['hour']
    if horauser<0 or horauser>23:
        res={
        'message':'Error, hora ingresada no válida'
    }
        print(res['message'])
        return jsonify(res)
        
    #DAR MANTENIMIENTO SI SE INCLUYEN MAS NODOS
    if nodouser in strindices: #validar que esta en los indices del json
        if nodouser==1 or nodouser=='1':
            modeloPR=load('modelos/trainPR-1.joblib')
        elif nodouser==2 or nodouser=='2':
            modeloPR=load('modelos/trainPR-2.joblib')
        elif nodouser==3 or nodouser=='3':
            modeloPR=load('modelos/trainPR-3.joblib')
        elif nodouser==4 or nodouser=='4':
            modeloPR=load('modelos/trainPR-4.joblib')
    else:
        res={
        'message':'Error, el nodo no existe en los registros'
    }
        print(res['message'])
        return jsonify(res)

    #save body data
    res_hora=np.array(horauser).reshape(-1,1)

    x_plot = np.linspace(0,23,24)

    X_plot = x_plot[:, np.newaxis]

    y_resplot= modeloPR.predict(X_plot)
    ypred=modeloPR.predict(res_hora)
    _,_,x,y,_,_,_=obtencionlistasJS(str(nodouser))

    plt.scatter(np.array(x), np.array(y), color='navy', s=30, marker='o', label="Datos historicos del nodo %s"%str(nodouser))
    plt.plot(x_plot, y_resplot, color='teal', linewidth=2,label="Polinomial grado %d" % grado)
    plt.scatter(res_hora,ypred,color="red",label="Resultado seleccionado por el usuario= hora %s:00"%horauser)
    print("Para las",horauser,"con el grado",grado,"se pronostican ",int(np.around(ypred)[0]), "llamadas de emergencia")
    plt.legend(loc='upper left')
    plt.title('Prediccion modelo polinomial',None,'center')
    coef=0.0
    if float(nodouser)>2:
        coef=(float(nodouser)*0.05)+(float(nodouser)-3)*0.39
    plt.text(1.5,-0.85+coef,"Para las %s en nodo: %s"%(horauser,nodouser)+", se pronostican %s "%int(np.around(ypred)[0])+"llamadas de emergencia")

    plt.savefig('graficas/PR_result_%s_%s.png'%(nodouser,horauser),metadata={"title":"Prediccion de modelo polinomial"})
    plt.close()

    response={
        'result':"Para las %s en nodo: %s"%(horauser,nodouser)+", se pronostican %s "%int(np.around(ypred)[0])+"llamadas de emergencia"
    }
    return jsonify(response)



def regresionPolinomialNumEmergencias(X, Y, namefile,grado):#se hace una prediccion con regresion lineal
#dado un conjunto de datos de abcisas, ordenadas, numero de hora ingresado y número de nodo en que se realiza
    X = np.array(X).reshape(-1, 1)
    # cada elemento solo tiene un feature
    Y = np.array(Y)#.reshape(-1, 1)
    #X = X[:, np.newaxis]
    #Y = Y[:, np.newaxis]
    #plt.scatter(x, y, color='navy', s=30, marker='o', label="puntos de training")
    modeloPR = make_pipeline(PolynomialFeatures(grado), Ridge())#lasso
    #make pipeline: tuberia de datos, se multiplican los features por el grado, agrupaciones de metodos
    modeloPR.fit(X, Y)
    dump(modeloPR,'modelos/%s.joblib'%namefile)
    response={
        'message':'El modelo ha sido generado: joblib',
        'archivo':namefile
    }
    return jsonify(response)


def obtencionlistasJS(numnodo):
  #obtiene información de los archivos JSON dado un nodo a explorar
  jnemergency=list()#guarda los strings de tipo de emergencia
  jnumsoc=list()#cuenta las ocurrencias de horas
  jhours=list()#guarda las horas (formato 0.00 a 23.00)
  jndate=list()#guarda fechas
  jrisks=list()#guarda diccionario de riesgos
  varisk=jn1.get(numnodo)[0]['risks'][0]
  jrisks.append(varisk)#guardo el directorio de riesgos para su posterior exploracion
  #para jhours
  #para jnemergency y jndate
  for i in range(len(jn1.get(numnodo)[0]['history'])):
    varauxem=jn1.get(numnodo)[0]['history'][i]['emergency']
    jnemergency.append(varauxem)
    varauxda=jn1.get(numnodo)[0]['history'][i]['date']
    jndate.append(varauxda)
    varauxhr=jn1.get(numnodo)[0]['history'][i]['hour']
    varauxhr=int(varauxhr[0:2])
    jhours.append(varauxhr)
  #preprocesamiento para datos no repetidos en jhours
  jhoursp=[]
  for item in jhours:
      if item not in jhoursp:
          jhoursp.append(item)
  #para jnumsoc
  for item in jhoursp:
    jnumsoc.append(jhours.count(item))#contamos cada hora que se haya presentado por cada 
    #item unico de jhours preprocesado (objetivo: contar emergencias)
  #para jndate yconvertir a datetime
  dtdate=list()
  for y in range(len(jndate)):
    dtdate.append(datetime.datetime.strptime(str(jndate[y][0]), "{'year': '%Y',  'month': '%m', 'day': '%d'}"))
  #para listar tipos de emergencia no repetidos y contarlos
  nremerg=[]#emergencias de cada tipo no repetidas
  jnemergency2=list()#una copia para tener una lista de strs separados SOLO EN ARR DE STRINGS
  countemetype=list()#aqui se depositan los numeros de ocurrencia de cada emergencia segun el orden de nremerg
  for a in jnemergency:
    if len(a)>1:
      jnemergency2.append(str(a[0]))
      jnemergency2.append(str(a[1]))
    else:
      jnemergency2.append(str(a[0]))

  for item in jnemergency2:
      if item not in nremerg:
          nremerg.append(item)#para que las emergencias se muestren como unicas
  #para obtener num de datos
  for item in nremerg:
    countemetype.append(jnemergency2.count(item))
  ##print(countemetype)

  return dtdate,jnemergency2,jhoursp,jnumsoc,nremerg,countemetype,jrisks


#main
app.run(host="0.0.0.0")

#https://www.iartificial.net/regresion-polinomica-en-python-con-scikit-learn/

#https://scikit-learn.org/stable/auto_examples/linear_model/plot_polynomial_interpolation.html#sphx-glr-auto-examples-linear-modeloPR-plot-polynomial-interpolation-py

#https://colab.research.google.com/drive/1rNIUjVRzQnFcvojyzCcGQqc2r0UcGUzD#scrollTo=5xEo0-arRaZz
#https://machinelearningmastery.com/make-predictions-scikit-learn/