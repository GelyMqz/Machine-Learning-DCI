# -*- coding: utf-8 -*-

"""
@author: Norma Angélica Márquez Sulca
NUA: 427278

1. Construya un dataset artificial de 40 puntos usando el método make_blobs de sklearn.datasets. El dataset tendrá como centros los 4 puntos de valores posibles a una compuerta lógica AND. Esto es 

centers = [(0, 0), (0, 1), (1, 0), (1,1)]. Use una media  de 0 y desviación estándar de 0.1 en cada cluster. El dataset deberá ser similar al que se muestra en la figura adjunta.

2. Implemente una Neurona de McCulloch-Pitts que reciba dos valores de entrada (sin bias) y utilice una función de activación de tipo escalón (0 o 1). Entrene la neurona usando el algoritmo Perceptron y el dataset de 40 puntos.

3. Genere un scatter plot con el dataset y la recta (hiperplano) que genera la Neurona después de su entrenamiento. Imprima en un PDF su scatter plot con el hiperplano y los puntos pintados de un color asociado a la clase que asignó el clasificador, los pesos finales de la neurona y un reporte de clasificación con matriz de confusión para evaluar el rendimiento de la neurona.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_pdf import PdfPages

# Datos
centros = [(0, 0), (0, 1), (1, 0), (1, 1)]
n_samples = 40
desviacion = 0.1
estado = 42

# Creamos el dataset
X, y = make_blobs(n_samples=n_samples, centers=centros, cluster_std=desviacion, random_state=estado)
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=estado)

# Función de algoritmo de perceptrón
class Perceptron:
    def __init__(self):
        self.w = None

    def step_function(self, z):
        return 1 if z >= 0 else 0

    def weighted_sum_inputs(self, x):
        return np.dot(x, self.w)

    def predict(self, x):
        z = self.weighted_sum_inputs(x)
        return self.step_function(z)

    def fit(self, X, y, epochs=1, step=0.1, verbose=True):
        self.w = np.zeros(X.shape[1])  # Inicializar los pesos a cero
        errors = []

        for epoch in range(epochs):
            error = 0
            for i in range(len(X)):
                x, target = X[i], y[i]
                update = step * (target - self.predict(x))
                self.w += update * x
                error += int(update != 0.0)
            errors.append(error)
            if verbose:
                print(f'Época: {epoch} - Error: {error} - Errores de todas las épocas: {errors}')

        return errors

# Creamos página pdf, figura y ejes
pdf_pages = PdfPages('neuron_plot.pdf')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))


# Se crea neurona y la entrenamos
neuron = Perceptron()
errors = neuron.fit(train_x, train_y, epochs=10, step=0.1)

# Crear un scatter plot de los puntos  
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
axes[0].set_title('Dataset con recta')

# Dibujar la recta
x_vals = np.array(axes[0].get_xlim())
y_vals = -(neuron.w[0] / neuron.w[1]) * x_vals
axes[0].plot(x_vals, y_vals, '--', c="red")

# Guardar la página en el pdf
pdf_pages.savefig()
plt.close()  # Cerrar la figura actual

# Nueva página
fig, ax = plt.subplots(figsize=(7, 7))

# Rendimiento
y_pred = [neuron.predict(xi) for xi in X]
cm = confusion_matrix(y, y_pred)

# Precisión
precision = (cm[0, 0] + cm[1, 1]) / np.sum(cm)

# Agregar texto para pesos, matriz de confusión y precisión en la nueva página
results_text = f'Pesos finales de la neurona: {neuron.w}\n\nMatriz de Confusión:\n{cm}\n\nPrecisión: {precision:.2f}'
ax.text(0.5, 0.5, results_text, fontsize=12, va='center', ha='center')

# Guardar la página con el texto en el pdf
pdf_pages.savefig(fig)

# Cerrar el archivo 
pdf_pages.close()

# Mostrar el archivo 
import webbrowser
webbrowser.open('neuron_plot.pdf')

 
