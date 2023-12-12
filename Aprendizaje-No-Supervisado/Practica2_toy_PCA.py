# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 18:15:29 2023
Due to Tue Aug 26 17:00:00 2023

@author: Norma Angélica Márquez Sulca
NUA:427278
Materia: Temas Selectos de Física: Machine Learning

Para implementar el algoritmo de Análisis de Componentes Principales (PCA) en su versión de vectores propios de la matriz de covarianza de un dataset dado, se requieren algunos conceptos previos de Álgebra Lineal, Cálculo Vectorial y Probabilidad y Estadística. En esta práctica realizará lo siguiente:

1. Complemente el código (toy-PCA, disponible en archivo adjunto) analizado en clase, para que proyecte los puntos del dataset D sobre el eigenvector principal (vector negro en la gráfica). Esto podrá hacerlo preferentemente con operaciones matriciales entre D y el eigenvector 1. 

2. Grafique los puntos proyectados sobre el eigenvector principal sobre el scatter plot de la primera figura del código. Utilice un color y tamaño de marcador apropiado para identificar facilmente los puntos proyectados.

3. Compare sus proyecciones con las obtenidas mediante la implementación de PCA del paquete sklearn
"""

import numpy as np 
import matplotlib.pyplot as plt

#Dataset (D)
D = np.array([(10,30),(20, 40),(30, 35),(40, 45),(50, 50)])

#Paso 1: Calculamos el vector de medias
mean = np.mean(D, axis =0)

#Paso 2: Calculamos la matriz de covarianza de la matriz D
C = np.cov(D, rowvar=False, bias=True)
#rowvar = Falso, Calcula por columnas y no por fila.
#bias=True, es para dividir entre n y no entre (n-1)

#Paso 3: Cálculo de autovalores y autovectores
eigval, eigvec = np.linalg.eig(C)

#Imprimimos los valores para verificar datos
#print("Eigenvalores:",eigval)
#print("Eigenvectores",eigvec)

#Paso 4: Gráfica del conjunto de datos
plt.figure()
plt.scatter(D[:,0], D[:,1])

#Gráfica de autovalores centrados en la media
plt.arrow(*mean, *eigvec[:, 0]*np.sqrt(eigval[0]), width=0.1, color='k', lw=2,
          overhang=0.1)
plt.arrow(*mean, *eigvec[:, 1]*np.sqrt(eigval[1]), width=0.1, color='r', lw=2,
          overhang=0.1)
plt.grid(True)
plt.axis('equal')
plt.show()
    
#Vamos a ordenas de mayor a menor 
"""ordena los índices que ordenarían el array en orden ascendente, por lo que usamos [::-1]
 para obtener los índices que ordenarían en orden descendente
"""
orden_indice = np.argsort(eigval)[::-1]
eigvec_decre = eigvec[:, orden_indice]
#print("Eigenvectores orden",eigvec_decre)

#Seleccionamos el vector con el autovalor mayor
columna = 0
eigvec_mayor= eigvec_decre[:, columna]

#print("Eigenvalor mayor:", eigvec_mayor)

 
#Proyectamos, definimos una función que lo lleve a cabo, proyecta x en y
""""Además de los vectores añadiremos un parámetro c, cuya única función
es que la proyección esté centrada a la misma altura de lo que lo esta el autovalor (en mean)"""
def proyeccion(x,y, mean):
    cuadrado = np.dot(y, y)
    punto = np.dot(x-mean,y)
    proyecto = punto / cuadrado 
    proyeccion = mean + proyecto * y
    return proyeccion 

#Iniciamos una lista donde guardaremos los valores
puntos_proyec = []

#Ejecutamos la función de proyección por cada punto del dataset
for fila in D:
    proyeccion_final = proyeccion(fila, eigvec_mayor, mean)
    puntos_proyec.append(proyeccion_final)

# Convertir la lista en un array para la gráfica
puntos_proyect_a = np.array(puntos_proyec)

#Verificamos los datos proyectados
#print("Datos proyectados:", puntos_proyect_a)


# Graficamos el autovector y los dos sets de puntos
plt.scatter(D[:, 0], D[:, 1], label='Puntos dataset')
plt.scatter(puntos_proyect_a[:, 0], puntos_proyect_a[:, 1], label='Puntos proyectados', marker='o',  color='crimson')
plt.arrow(*mean, *eigvec[:, 0]*np.sqrt(eigval[0]), width=0.1, color='k', lw=2,
          overhang=0.1)
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()




