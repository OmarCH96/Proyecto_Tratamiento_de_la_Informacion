#Importamos las librerias que utilizaremos.
import pandas as pd
import os

#Conjuntos con los que se va a trabajar
DatosT="sb1-T.txt"
DatosP="sb1-P.txt"

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
    for i in range(0, len(head)):
        head[i]="Atributo %(num)d" % {"num": i+1}#Arreglo para el header para cada atributo
    head[len(head)-1]="Clase"#Nombre a atributo clase
    dframe = pd.DataFrame(datos, columns=head) #Crea un data frame utilizando pandas
    return NumElem, NumAttr, TypAttr, NClases, dframe, head

#Limpiamos nuestros datos
def convNumb(data):
    separador = ","
    separado = data.split(separador)#Separa los datos en una lista
    new_list = [s.replace("\n", "") for s in separado]#Limpia la lista en caso de salto de linea
    #print(new_list)
    return new_list


#Iniciamos el proceso de la creacion del clasificador
class NaiveBayesClassifier: #Clasificacion de nueva entrada de datos
    def __init__(self, X, y): #X para caracteristicas y Y para las etiquetas a clasificar
                              #Alamacenamos datos en el metodo __init__
        self.X, self.y = X, y 
        self.N = len(self.X) # Tamaño del conjunto de entrenamiento
        self.dim = len(self.X[0]) # Dimensión de la lista de características
        self.attrs = [[] for _ in range(self.dim)] # Aquí almacenaremos las columnas del conjunto de entrenamiento.
        self.salida = {} # Clases de salida con el número de ocurrencias en el conjunto de entrenamiento. En este caso solo tenemos 2 clases
        self.data = [] # Alamacena cada linea [Xi, yi]  contiene todos los datos del conjunto 
        for i in range(len(self.X)):
            for j in range(self.dim):
                # si nunca hemos visto este valor para este atributo antes, 
                # luego lo agregamos a la matriz attrs en la posición correspondiente
                if not self.X[i][j] in self.attrs[j]:
                    self.attrs[j].append(self.X[i][j])
            # si nunca hemos visto esta clase de salida antes,
            # luego lo agregamos a salida y contamos una ocurrencia por ahora
            if not self.y[i] in self.salida.keys(): #self.y[i] contiene la clase.
                self.salida[self.y[i]] = 1
            # de lo contrario, incrementamos la ocurrencia de esta salida en el conjunto de entrenamiento en 1
            else:
                self.salida[self.y[i]] += 1
            # almacenar la fila
            self.data.append([self.X[i], self.y[i]])  #Agregamo a self.X[i] lo que contiene self.y[i]
     #La clasificacion de los nuevos elementos la hacemos a partir de este punto 
     # Donde el metodo classufy nos ayuda a realizar la clasificacion       
    def classify(self, entry):
        solucion = None # Resultado final
        max_parcial = -1 # máximo parcial asignamos -1 para que tome en cuenta los siguientes valores
        for y in self.salida.keys():
            prob = self.salida[y]/self.N # P(y)  Donde self.salida corresponde a la concurrencia sobre el tamaño de nuestro conjunto
            for i in range(self.dim):
                cases = [x for x in self.data if x[0][i] == entry[i] and x[1] == y] # all rows  with Xi = xi
                n = len(cases)
                prob *= n/self.N # P *= P(Xi = xi)
            # si tenemos una probabilidad mayor para esta salida que el máximo parcial ...
            if prob > max_parcial:
                max_parcial = prob
                solucion = y
        return solucion

# Aqui es donde se declara que archivo abrira para la lectura de la linea 10
NumElemT, NumAttrT, TypAttrT, NClasesT, datosT, headT =lectura(DatosT)
print("Entrenamiento")
print("# de elementos: %(num)d" % {"num": NumElemT})        #279
print("# de elementos: %(num)d" % {"num": NumAttrT})        #35
print("# de clases: %(num)d" % {"num": NClasesT})           #19
#print(datosT.head())

NumElemP, NumAttrP, TypAttrP, NClasesP, datosP, headP =lectura(DatosP)
print("Conjunto de prueba")
print("# de elementos: %(num)d" % {"num": NumElemP})        #26
print("# de elementos: %(num)d" % {"num": NumAttrP})        #35
print("# de clases: %(num)d" % {"num": NClasesP})           #19
#print(datosP.head())


# valores objetivo como cadena
y = list(map(lambda v: '0' if v == '0' else 
'1' if v == '1' else
'2' if v == '2' else
'3' if v == '3' else
'4' if v == '4' else 
'5' if v == '5' else
'6' if v == '6' else                    #Agrupamos nuestras clases
'7' if v == '7' else
'8' if v == '8' else
'9' if v == '9' else
'10' if v == '10' else
'11' if v == '11' else
'12' if v == '12' else
'13' if v == '13' else
'14' if v == '14' else
'15' if v == '15' else
'16' if v == '16' else
'17' if v == '17' else '18', datosT['Clase'].values))

yP = list(map(lambda v: '0' if v == '0' else 
'1' if v == '1' else
'2' if v == '2' else
'3' if v == '3' else
'4' if v == '4' else 
'5' if v == '5' else
'6' if v == '6' else
'7' if v == '7' else
'8' if v == '8' else                    #Agrupamos nuestras clases
'9' if v == '9' else
'10' if v == '10' else
'11' if v == '11' else
'12' if v == '12' else
'13' if v == '13' else
'14' if v == '14' else
'15' if v == '15' else
'16' if v == '16' else
'17' if v == '17' else '18', datosP['Clase'].values)) 


X = datosT[headT].values # valores de características del conjunto de entrenamiento
XP = datosP[headT].values # valores de características del conjunto de prueba


nbc = NaiveBayesClassifier(X, y)   #y contiene las clases
CasosTot = len(yP) # tamaño del conjunto de validación

# Ejemplos bien clasificados y ejemplos mal clasificados
acertados = 0
error = 0

for i in range(CasosTot):#Calcula precision del modelo
    predict = nbc.classify(XP[i])
    if yP[i] == predict:
        acertados += 1
    else:
        error += 1

print("Numero de elementos de prueba:", CasosTot)   #26
print("Verdaderos Positivo:", acertados)            #20
print("Falsos Positivos:", error)                   #6
print("Precision:", (acertados/CasosTot)*100,"%")   #76.92%

print("C45")
os.system('Rscript sb1/R.R')  #76.92% de Exactitud