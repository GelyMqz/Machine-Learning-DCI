"""
@author: Norma Angélica Márquez Sulca

Utilice el dataset perceptron_train.csv que contiene 500 puntos en ℜ2 distribuidos en 2
clases para entrenar una neurona artificial con el algoritmo Perceptron. Genere una
imagen con la frontera de decisión que obtuvo con el clasificador y guárdela en un
archivo llamado fronteraPerceptron.PDF."""
""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

estado = 5
iteraciones = 500
# Cargamos archivo 
data = np.genfromtxt('C:\\Users\\Estudiantes\\Downloads\\perceptron_train.csv', delimiter=',')

# Dividimos los datos
X = data[:, :2]
y = data[:, 2]

# Se clasifican datos en train y test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=estado)

# Ajustamos
perceptron = Perceptron(max_iter = iteraciones, random_state = estado)
perceptron.fit(X_train, y_train)

# "malla" para los decision boundaary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Predicciones 
prediccion = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
prediccion = prediccion.reshape(xx.shape)

# Gráfica de la frontera y los puntos 
plt.contourf(xx, yy, prediccion, alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=np.where(y==0, 'lightblue', 'orange'), edgecolors='black', marker='o', s=20, linewidth=0.5)

# Guardar figura
plt.savefig('C:\\Users\\Estudiantes\\Downloads\\fronteraPerceptron.pdf')
plt.show()