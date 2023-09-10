# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 23:33:59 2023
Due to Tue Sep 08 17:00:00 2023

@author: Norma Angélica Márquez Sulca
NUA:427278
Materia: Temas Selectos de Física: Machine Learning
"""

#Importamos las librerías
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#Valores
k=5
n_data = 250
centros = 3
desviacion = 0.5
estado = 209

# Generaramos 250 puntos en 3 clases, con desviación estándar de 0.5 y en el estado 209 para reproducibilidad

"""En "data" se almacenan las coordenadas de los puntos mientras en "labels" se guarda un número del 0 al 2,
asignándolo a uno de los 3 centros"""

data, labels = make_blobs(n_samples=n_data, centers=centros, cluster_std=desviacion, random_state=estado)


#Guardamos los primeros 200 puntos en training y los últimos 50 en test
training_data, training_labels = data[:200], labels[:200]
test_data, test_labels = data[200:], labels[200:]

#Verificamos que haya 200 valores en training_data
#len(training_data)

#Verificamos que haya 50 valores en test_data
#len(test_data)

# Graficar los puntos de entrenamiento
colores = ['lightsteelblue', 'pink', 'springgreen']
colores2 = ['darkcyan','hotpink','seagreen']
plt.scatter(training_data[:, 0], training_data[:, 1], c=[colores[label] for label in training_labels])
plt.title("Training data")
plt.show()
print('\n \n \n')
#Comenzamos con KNN

#Definimos una función para calcular las distancias ecludianas del punto de prueba z a los N puntos de X
def distancia(pto1, pto2):
    return np.sqrt(np.sum((pto1 - pto2) ** 2))

# Identificar los k puntos más cercanos (vecinos) a z.
def KNN(training_data, training_labels, test_data, k):
  predicciones = []

  for i in range(len(test_data)):
        distancias = []
        for j in range(len(training_data)):
            dist = distancia(training_data[j], test_data[i])
            distancias.append(dist)

        #Ordenamos de menor a mayor las distancias de cada punto de prueba a los de training data
        #tomamos los primeros k valores en esa lista ordenada
        k_indices_vecinos_cercanos = np.argsort(distancias)[:k]

        #Accedemos a los labels que corresponden a esos índices
        labels_vecinos_cercanos = training_labels[k_indices_vecinos_cercanos]

        #Encontramos las clases únicas que contiene labels_vecinos_cercanos
        #Pedimos un array que cuente cuántas veces está ese valor único en labels_vecinos_cercanos
        labels_solitos, contador = np.unique(labels_vecinos_cercanos, return_counts=True)

        #Seleccionamos el label que aparece un mayor número de veces
        label_videncia = labels_solitos[np.argmax(contador)]

        #Lo añadimos a predicciones
        predicciones.append(label_videncia)

  return predicciones

predicciones_talacha=KNN(training_data, training_labels, test_data, k)

#Graficamos las predicciónes del pseudocódigo
print("Las predicciones siguiendo el pseudo código son:")
plt.scatter(training_data[:, 0], training_data[:, 1], c=[colores[label] for label in training_labels], alpha=0.7, label='puntos de entrenamiento')
plt.scatter(test_data[:, 0], test_data[:, 1], c=[colores2[label] for label in predicciones_talacha], marker='1', label='Predicciones de talacha')
plt.legend()
plt.title('KNN talacha')
plt.show()
print('\n \n \n')

#Reporte de clasificación
reporte_clasificacion_talacha = classification_report(test_labels, predicciones_talacha)
print("Reporte de clasificación del pseudocódigo :\n", reporte_clasificacion_talacha,"\n \n \n")

#Utilizamos KNN con sklearn
KNN_pro = KNeighborsClassifier(n_neighbors=k)
KNN_pro.fit(training_data, training_labels)
predicciones_pro = KNN_pro.predict(test_data)

#Graficamos las predicciones de sklearn
print("Las predicciones con sklearn son:")
plt.scatter(training_data[:, 0], training_data[:, 1], c=training_labels, alpha=0.7, label='puntos de entrenamiento')
plt.scatter(test_data[:, 0], test_data[:, 1], c=predicciones_pro, marker='1', label='Predicciones SKlearn')
plt.legend()
plt.title('KNN SKlearn')
plt.show()
print('\n \n \n')

#Reporte de clasificación
reporte_clasificacion_sk = classification_report(test_labels, predicciones_pro)
print("Reporte de clasificación con sklearn :\n", reporte_clasificacion_sk)
print('\n \n \n')

#Seguimos la documentación de sklearn para fronteras de decisión 
X, y = make_blobs(n_samples=n_data, centers=centros, cluster_std=desviacion, random_state=estado)
training_data, training_labels = X[:200], y[:200]
test_data, test_labels = X[200:], y[200:]

n_neighbors=k
cmap_light = ListedColormap(["lightsteelblue", "pink", "springgreen"])

for weights in ["uniform", "distance"]:
    # we create an instance of Neighbours Classifier and fit the data.
    KNN_pro = KNeighborsClassifier(n_neighbors=k, weights=weights)
    KNN_pro.fit(training_data, training_labels)

    _, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator( KNN_pro, training_data, cmap=cmap_light, ax=ax, response_method="predict", plot_method="pcolormesh", shading="auto" )

    # Plot also the training points
    sns.scatterplot(x=X[:, 0], y=X[:, 1], palette=colores2, hue=y, alpha=1.0, edgecolor="black")
    plt.title("3-Class classification (k = %i, weights = '%s')" % (k, weights))

plt.show()

