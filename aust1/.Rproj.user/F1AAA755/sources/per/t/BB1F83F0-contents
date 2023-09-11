library(rpart) #  Libreria especifica para la poda de arboles de decision.
library(partykit) # Libreria para el dibujo de Arboles de decision.
library(naniar) # Libreria para facilitar la visualizacion y manipulacion de datos.
library(caret) # Libreria que facilita el preprocesado, entrenamiento, optimizacion y validacion de modelos predictivos.


#Importacion y limpieza de datos.
dataset <- read.delim("aust1-T.txt", header = FALSE, sep = ",", dec = ".")
n_Elementos <- dataset[1,1]
n_Datos <- dataset[2,1]
t_Datos <- dataset[3,]
dataset <- dataset[4:dim(dataset),]
dataset[,(length(dataset))] <- as.factor(dataset[,(length(dataset))])
colnames(dataset)[length(dataset)] <- "Clases"


#Creamos el modelo el paquete con la implementación de árboles de clasificación que utilizaremos.
modelo <- rpart(Clases ~ . , data = dataset,method = "class")
plot(as.party(modelo))
variableArbol <- modelo[["frame"]][["var"]] #Obtenemos los atributosdel arbol
for (i in 1: length(variableArbol)) {   #Asignamos NA a <leaf>
  if(variableArbol[i] == "<leaf>")
    variableArbol[i] <- NA
}
variableArbol <- na.omit(variableArbol)  #Eliminamos NA
variableArbol <- unique(variableArbol)  #Seleccionamos los valores unicos o eliminamos los que se repiten
formula <- "Clases ~ "
for (i in 1:length(variableArbol)) {
  if (i == 1)
    formula <- paste0(formula, variableArbol[i])##Creamos nuestra formula.
  else  
    formula <- paste0(formula, "+", variableArbol[i])
}


#Obtenemos conjunto de prueba
datasetP <- read.delim("aust1-P.txt", header = FALSE, sep = ",", dec = ".")
datasetP <- datasetP[4:dim(datasetP),]
datasetP[,(length(datasetP))] <- as.factor(datasetP[,(length(datasetP))])
colnames(datasetP)[length(datasetP)] <- "Clases"


#Modelo de prediccion
modelo.pred <- predict(modelo, datasetP, type="class")
medida <- confusionMatrix(datasetP$Clases,modelo.pred)


#Creacion del modelo utilizando la formula obtenida con los atributos.
Nuevo_modelo <- rpart(formula, data = dataset,method = "class")##agreamos P
plot(as.party(Nuevo_modelo))
Nuevo_modelo.pred <- predict(Nuevo_modelo, datasetP, type="class")
medida_Nueva <- confusionMatrix(datasetP$Clases,Nuevo_modelo.pred)


#Obtenemos el Rendimiento del modelo
print(paste0("Presicion del modelo: ",medida_Nueva[["overall"]][["Accuracy"]]))
print(paste0("Presicion del modelo utilizando atributos del arbol: ",medida[["overall"]][["Accuracy"]]))
