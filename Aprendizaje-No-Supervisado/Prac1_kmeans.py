# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 11:42:06 2023

@author: Angie

El algoritmo k-means es empleado para agrupación (clustering) de vectores de características no etiquetados. Al no tener etiquetas para el entrenamiento, se considera un algoritmo de aprendizaje No Supervisado. El siguiente video muestra el proceso iterativo que sigue el método. En general se puede resumir en los siguientes pasos:

1. Definir un número inicial k de centroides de grupo

2. Calcular las distancias de cada uno de los k centroides hacia todos los puntos del dataset. Si el tamaño del dataset es N, el número de distancias calculadas será N*k en caso de que los centroides sean puntos distintos del dataset o (N-k) * k si los centroides son algunos de los mismos puntos del dataset.

3. Asignar cada elemento del dataset al grupo correspondiente a la distancia mínima con su centroide. 

4. Recalcular los centroides y repetir desde el paso 2 hasta cumplir el número de iteraciones deseado.
"""

import random
import numpy as np
import matplotlib.pyplot as plt

# Definimos el número máximo que deseamos de iteraciones
it_max = 55

# Definimos una función para la distancia
def distancia(pto1, pto2):
    return np.sqrt(sum((x - y) ** 2 for x, y in zip(pto1, pto2)))

# Definimos una función para el cálculo de los nuevos centroides que usaremos después
def renovacion_c(AP, gamma, n_j, centroides):
    nuevos_c = np.zeros(centroides.shape)
    for j in range(k):
        if n_j[j] > 0:
            for i in range(n):
                nuevos_c[j] += gamma[i, j] * AP[i]
            nuevos_c[j] /= n_j[j]
    return nuevos_c

# Creamos una función (AP: arreglo puntos)
def k_means(AP, k, it_max):
    n, d = AP.shape
# Seleccionamos el número de centroides k de una lista preexistente
    centroides = AP[np.random.choice(n, k, replace=False)]  

# Ahora debemos iniciar en ceros la matriz de 0 y 1
    for _ in range(it_max):
        gamma = np.zeros((n, k))

# Calculamos la distancia entre cada punto con todos los centroides
        i = 0
        while i < n:
            distancias = [distancia(AP[i], centroide) for centroide in centroides]
            j = np.argmin(distancias)  # Elije el centroide que esté a menor distancia
            gamma[i][j] = 1
            i += 1

        n_j = np.sum(gamma, axis=0)

        c_renovados = renovacion_c(AP, gamma, n_j, centroides)

# Debemos terminar hasta que haya convergencia
        if np.all(np.abs(centroides - c_renovados) < 0.001):
            break

        centroides = c_renovados

    return centroides

  
n = 300
d = 2
k = 3
np.random.seed(15)
AP = np.random.rand(n, d) * 10
centro_grupo = k_means(AP, k, it_max)

# Asignar puntos a grupos
grupoqs = []
for i in range(n):
    distancias = [distancia(AP[i], centroide) for centroide in centro_grupo]
    grupoqs.append(np.argmin(distancias))

# Graficamos
colors = ['hotpink', 'mediumorchid', 'indigo', 'crimson']
for i, grupo in enumerate(grupoqs):
    plt.scatter(AP[i][0], AP[i][1], c=colors[grupo], marker='o', s=7)

# Y para los centroides
for centroid in centro_grupo:
    plt.scatter(centroid[0], centroid[1], c='black', marker='x')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('K-Means')
plt.show()

print("Centroides:\n", centro_grupo)
