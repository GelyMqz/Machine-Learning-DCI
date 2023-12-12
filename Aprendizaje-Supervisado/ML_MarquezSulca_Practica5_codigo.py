# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 12:35:28 2023

@author: Norma Angélica Márquez Sulca
NUA: 427278
Materia: Temas Selectos de Física: Machine Learning
"""

#Bibliotecas
import numpy as np
import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from pandas.plotting import table
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.datasets import make_classification, make_blobs

#Valores
K = 5
C = 10
gamma = 0.1
kernel = "rbf"
names = ["KNN", "SVM", "Perceptron", "MLP"]
tasa_apren = 0.3
epocas = 10
estado = 26
centers = [(1, 1), (1, -1), (-1, -1), (-1, 1)]

#Enlistamos los clasificadores
classifiers = [
    KNeighborsClassifier(n_neighbors=K),
    SVC(kernel=kernel, C=C, gamma = gamma),
    Perceptron(eta0 = tasa_apren, max_iter=epocas, random_state= estado),
    MLPClassifier(hidden_layer_sizes=(2, 2, 1), max_iter=epocas, learning_rate_init=tasa_apren, activation='logistic', random_state= estado)
    ]


#Leemos los datasets
dataset1 = pd.read_csv("C:/Users/Angie/Downloads/dataset_classifiers1.csv").values
dataset2 = pd.read_csv("C:/Users/Angie/Downloads/dataset_classifiers2.csv").values
dataset3 = pd.read_csv("C:/Users/Angie/Downloads/dataset_classifiers3.csv").values

#Leemos los datos necesarios

#print(datasets_t[1])


""" Creamos el dataset 4 """

# Generar 1000 puntos alrededor de cada centro
X_xor, y_xor = make_blobs(n_samples=[1000, 1000, 1000, 1000], centers=centers, cluster_std=0.1, random_state=estado)


#Leemos los datos necesarios en la lista de datasets
datasets = [(dataset1[:,1:3], dataset1[:,3]), (dataset2[:,1:3], dataset2[:,3]),(dataset3[:,0:2], dataset3[:,2]),(X_xor, y_xor)]


X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

results_df = pd.DataFrame(columns=["Dataset", "Classifier", "Accuracy"])


"""Con base en: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py"""
# Iterar sobre conjuntos de datos y clasificadores
figure = plt.figure(figsize=(28, 15))
i = 1

for ds_cnt, ds in enumerate(datasets):
    
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=estado)

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    cm = plt.cm.RdBu
    cm_bright = ListedColormap(["red", "blue"])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    
    ax.scatter(        X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k" )
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1
    
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        DecisionBoundaryDisplay.from_estimator(clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5)

        # Graficamos puntos de entrenamiento y de prueba
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors="k",  alpha=0.7)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(x_max - 0.3, y_min + 0.3, ("%.2f" % score).lstrip("0"), size=15, horizontalalignment="right" )
        i += 1
        accuracy = clf.score(X_test, y_test)
        results_df = results_df.append({"Dataset": ds_cnt, "Classifier": name, "Accuracy": accuracy}, ignore_index=True)

results_df.to_csv("accuracies.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 6))

ax.axis("off")

# Creamos la tablita
tab_acc = table(ax, results_df, loc="center", cellLoc="center", colWidths=[0.1, 0.2, 0.2])
tab_acc.auto_set_font_size(False)
tab_acc.set_fontsize(12)
tab_acc.scale(1.5, 1.5)

# Guardarmos la tabla como imagen
plt.savefig("accuracies.png")
plt.tight_layout()
plt.show()
