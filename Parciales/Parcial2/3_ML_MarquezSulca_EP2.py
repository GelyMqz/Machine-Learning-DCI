# -*- coding: utf-8 -*-
"""
@author: Norma Angélica Márquez Sulca

Utilice la imagen frutas.jpg como entrada a una Red Neuronal Convolucional
preentrenada (VGG16, VGG19, ResNet-50, etc). Guarde en una figura de 4x4
subgráficas, 16 imágenes de los filtros que resultan de procesar frutas.jpg por la primera
capa convolucional de su arquitectura
"""
#Primero lo estaba haciendo en colab porque se trababa spider al instalar keras y tensorflow, pero sí
#pude instalarlos al final

#Solo lo probé una vez porque se me trababa spider

#Ref:
#https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/


from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
from tensorflow.keras.utils import load_img 
from tensorflow.keras.utils import img_to_array
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Modelo
model = VGG16()

# Acomodamos para la salida
model = Model(inputs=model.inputs, outputs=model.layers[1].output)
model.summary()

#Carga
path = 'C:\\Users\\Estudiantes\\Downloads\\frutas.jpg' 
img = load_img(path, target_size=(224, 224))

# Hacemos la imagen un array
img = img_to_array(img)

# La expandimos
img = expand_dims(img, axis=0)
print(img.shape)

#Preprocesamiento
img = preprocess_input(img)

#Para 1ra capa oculta
feature_maps = model.predict(img)

# Tamaño cuadricula
square = 4
ix = 1

# Las 16 imágenes
for _ in range(square):
    for _ in range(square):
        ax = pyplot.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
        ix += 1


pyplot.show()