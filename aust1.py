#Importamos las librerias que utilizaremos.
import pandas as pd
import os

#Conjuntos con los que se va a trabajar
DatosT="aust1-T.txt"
DatosP="aust1-P.txt"

#Lectura del archivo y prepraracion de los datos
def lectura(file):
    data = []
    with open(file,"r") as f:
        data = f.readlines()  #lectura
    NumElem=int(data[0]) #Numero de Elementos
    NumAttr=int(data[1])#Numero de Atributos
    TypAttr=convNumb(data[2])#Tipo de Atributos
    NClases=int(TypAttr[len(TypAttr)-1])#Numero de Clases
    datos=[[]]*NumElem #Almacena datos
    for i in range(0, len(datos)):
        datos[i]=convNumb(data[i+3])#Convierte cada lineae y la limpia
    head = ["Atributo"]*int(NumAttr+1) #Crea Header
    for i in range(0, len(head)): #Recorremos nuestro header
        head[i]="Atributo %(num)d" % {"num": i+1}#Arreglo para el header para cada atributo
    head[len(head)-1]="Clase"#Nombre a atributo clase
    dframe = pd.DataFrame(datos, columns=head) #Crea un data frame utilizando pandas
    return NumElem, NumAttr, TypAttr, NClases, dframe, head

#Limpiamos los datos
def convNumb(data):
    separador = ","
    separado = data.split(separador)#Separa los datos en una lista
    new_list = [s.replace("\n", "") for s in separado]#Limpia la lista en caso de salto de linea
    return new_list


#Iniciamos el proceso de la creacion del clasificador
class NaiveBayesClassifier:
    def __init__(self, X, y): #X para caracteristicas y Y para las etiquetas a clasificar
        self.X, self.y = X, y 
        self.N = len(self.X) # Tamaño del conjunto de entrenamiento
        self.dim = len(self.X[0]) # Dimensión de la lista de características
        self.attrs = [[] for _ in range(self.dim)] # Aquí almacenaremos las columnas del conjunto de entrenamiento.
        self.salida = {} # Clases de salida con el número de ocurrencias en el conjunto de entrenamiento. En este caso solo tenemos 2 clases
        self.data = [] # Alamacena cada linea [Xi, yi]
        for i in range(len(self.X)):
            for j in range(self.dim):
                if not self.X[i][j] in self.attrs[j]:            # si nunca hemos visto este valor para este atributo antes, 
                    self.attrs[j].append(self.X[i][j])           # luego lo agregamos a la matriz attrs en la posición correspondiente
            if not self.y[i] in self.salida.keys():              # si nunca hemos visto esta clase de salida antes,
                self.salida[self.y[i]] = 1                       # luego lo agregamos a salida y contamos una ocurrencia por ahora
            else:
                self.salida[self.y[i]] += 1 # de lo contrario, incrementamos la ocurrencia de esta salida en el conjunto de entrenamiento en 1
                self.data.append([self.X[i], self.y[i]])              # almacenar la fila
     #La clasificacion de los nuevos elementos la hacemos a partir de este punto 
     # Donde el metodo classufy nos ayuda a realizar la clasificacion     
    def classify(self, entry):
        solucion = None # Resultado final
        max_parcial = -1 # máximo parcial asignamos -1 para que tome en cuenta los siguientes valores
        for y in self.salida.keys():               #Calculamos las probabilidades para las clases 0 y 1
            prob = self.salida[y]/self.N # P(y)     285/621 0.45893719806763283 || 336/621 0.5410628019323671
            for i in range(self.dim):
                cases = [x for x in self.data if x[0][i] == entry[i] and x[1] == y] # Compruba la probabilidad que sea el tipo de clase
                n = len(cases)
                prob *= n/self.N # P *= P(Xi = xi)
            # si tenemos una probabilidad mayor para esta salida que el máximo parcial ...
            if prob > max_parcial:
                max_parcial = prob
                solucion = y            
                #print(solucion)
        return solucion

NumElemT, NumAttrT, TypAttrT, NClasesT, datosT, headT =lectura(DatosT)
print("Entrenamiento")
print("# de elementos: %(num)d" % {"num": NumElemT})  #621
print("# de elementos: %(num)d" % {"num": NumAttrT})  #14
print("# de clases: %(num)d" % {"num": NClasesT})     #2


NumElemP, NumAttrP, TypAttrP, NClasesP, datosP, headP =lectura(DatosP)
print("Conjunto de prueba")
print("# de elementos: %(num)d" % {"num": NumElemP})  #69
print("# de elementos: %(num)d" % {"num": NumAttrP})  #14
print("# de clases: %(num)d" % {"num": NClasesP})     #2    



# valores objetivo como cadena
y = list(map(lambda v: '0' if v == '0' else '1', datosT['Clase'].values))
#'1' if v == '1' else '2', datosT['Clase'].values))
yP = list(map(lambda v: '0' if v == '0' else '1', datosP['Clase'].values)) 
#'1' if v == '1' else '2', datosP['Clase'].values)) 
X = datosT[headT].values # valores de características del conjunto de entrenamiento
XP = datosP[headT].values # valores de características del conjunto de prueba
#Clasifiacdor con los datos de Prueba
nbc = NaiveBayesClassifier(X, y)
CasosTot = len(yP) # tamaño del conjunto de validación
# Ejemplos bien clasificados y ejemplos mal clasificados
acertados = 0
error = 0

for i in range(CasosTot):#Calcula precision del modelo
    predict = nbc.classify(XP[i])
    if yP[i] == predict:
        #print("Acertados ",yP[i], "||" ,nbc.classify(XP[i]))
        acertados += 1
    else:
        #print("No acertados ",yP[i], "||" ,nbc.classify(XP[i]))
        error += 1 
print("Numero de elementos de prueba:", CasosTot)   #69
print("Verdaderos Positivo:", acertados)  #38
print("Falsos Positivos:", error)  #31
print("Precision:", (acertados/CasosTot)*100,"%")  #55.07%

os.system('Rscript aust1/aust.R')   #88.40%