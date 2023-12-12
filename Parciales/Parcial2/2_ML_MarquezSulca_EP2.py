# -*- coding: utf-8 -*-
"""
@author: Norma Angélica Márquez Sulca

El dataset_3classes2D.csv adjunto contiene 600 puntos en ℜ2 distribuidos en 3 clases
distintas. A) Utilice validación cruzada de 10 pliegues para evaluar el clasificador
Multilayer Perceptron (MLP) con una arquitectura que tenga 2 neuronas en la capa de
entrada, N neuronas en la capa oculta (donde N es un número de su elección) y 3 neuronas
en la capa de salida. B) Genere una imagen con las fronteras de decisión que obtuvo co n
el clasificador. C) Proporcione una tabla con los accuracies. Guarde en un mismo
directorio su script, la imagen con las fronteras de decisión y una imagen c on la tabla de
accuracies (4 puntos)."""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


#Neuronas en la capa oculta, número de elección propia
N=2
estado = 8
pliegues = 10

# Cargar el dataset
data = pd.read_csv('C:\\Users\\Estudiantes\\Downloads\\dataset_3classes2D.csv')

# Separar las características y las clases
X = data.iloc[:, :2]  
y = data.iloc[:, 2]   

# Clasificador MLP y escalar
mlp = MLPClassifier(hidden_layer_sizes=(N,), max_iter=1000)  
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Definir la validación cruzada de 10 pliegues
cv = KFold(n_splits=pliegues, shuffle=True, random_state=estado)

# Realizar la validación para  accuracies
accuracies = cross_val_score(mlp, X_scaled, y, cv=cv, scoring='accuracy')

# Imprimir 
print("accuracies:", accuracies)

# Entrenar el clasificador 
mlp.fit(X_scaled, y)

# Crear una malla para  fronteras de decisión
h = .02  # tamaño del paso 
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])

# Gráfico de  fronteras de decisión
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, edgecolors='k')
plt.xlabel('1')
plt.ylabel('2')
plt.title('Fronteras')
plt.savefig('fronteras.png')

# Tabla 
# No me dio tiempo de ajustar tamaños de la tabla pero con zoom se puede distinguir y se muestran los valores en la consola
plt.figure(figsize=(8, 6)) 
plt.table(cellText=[[acc] for acc in accuracies],
          rowLabels=[f'Fold {i}' for i in range(1, 11)],
          loc='center',
          cellLoc='center',
          colWidths=[0.1] * 10,
          fontsize=14)
plt.axis('off')
plt.title('Accuracies')
plt.savefig('tabla_accuracies.png')


