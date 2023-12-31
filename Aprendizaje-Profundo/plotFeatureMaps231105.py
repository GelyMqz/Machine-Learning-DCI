# -*- coding: utf-8 -*-
"""
Created on Tue May 31 21:00:54 2022

@author: TUF-PC8
"""
#El tutorial mostrando estos códigos lo pueden encontrar en la siguiente dirección
# https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/

# plot feature map of first conv layer for given image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims

#En caso de que su equipo les marque el siguiente error:
#OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5 already initialized.
#La solución sería activar la siguiente bandera: (La dejé activa, pero revisen con cuidado)
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#Algunas instrucciones pueden dejar de funcionar por Refactorización (Reubicación o cambio de nombre de paquetes, objetos, etc.)
#from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import load_img 
from tensorflow.keras.utils import img_to_array

# load the model
model = VGG16()

# redefine model to output right after the first hidden layer
model = Model(inputs=model.inputs, outputs=model.layers[1].output)
model.summary()

# load the image with the required shape
# La ruta es relativa, deben colocar su imagen de prueba dentro del mismo directorio que este script
path = 'bird.jpg'
img = load_img(path, target_size=(224, 224))

# convert the image to an array
img = img_to_array(img)

# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
print(img.shape)
## prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model.predict(img)
# plot all 64 maps in an 8x8 squares
square = 8
ix = 1
for _ in range(square):
	for _ in range(square):
		# specify subplot and turn of axis
		ax = pyplot.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		ix += 1
# show the figure
pyplot.show()
