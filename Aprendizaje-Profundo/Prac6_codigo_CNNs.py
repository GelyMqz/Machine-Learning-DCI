# -*- coding: utf-8 -*-
"""

@author: Norma Angélica Márquez Sulca

 Retome el código implementado en clase sobre la CNN - VGG16 (disponibles en archivo adjunto).

2. Usando la imagen bird.jpg, muestre 64 de los filtros que se obtienen con esta red pre-entrenada, al final de cada uno de los 5 bloques convolucionales de la arquitectura. Guarde cada imagen de los filtros resultantes y agréguela a un documento en PDF (cada imagen en una página separada).

3. Repita el paso 2 reemplazando la imagen bird.jpg por una imagen distinta del tema de su elección. Añada al PDF la imagen que seleccionó y el resultado de los filtros en cada capa convolucional.
"""

# -*- coding: utf-8 -*-

#El tutorial mostrando estos códigos lo pueden encontrar en la siguiente dirección
# https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/


# Importamos las bibliotecas
from keras.applications.vgg16 import VGG16
from matplotlib.backends.backend_pdf import PdfPages
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from matplotlib import pyplot
from numpy import expand_dims
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img 
from tensorflow.keras.utils import img_to_array

# Cargamos el modelo
model = VGG16()

# redefine model to output right after the first hidden layer
model = Model(inputs=model.inputs, outputs=model.layers[1].output)
model.summary()

#Hacemos una función para que sea más sencillo usarla con diferentes imágenes
def get_feature_maps(img_path):
    # cargar la imagen con el tamaño necesario
    img = load_img(img_path, target_size=(224, 224))
    
    # Convertir la imagen a un array
    img = img_to_array(img)
    
    # Expandir dimensiones para que represente un solo 'muestra'
    img = expand_dims(img, axis=0)
    
    # Preprocesar la imagen (por ejemplo, escalar los valores de píxeles para el vgg)
    img = preprocess_input(img)
    
    # obtener la malla de características para la primera capa oculta
    feature_maps = model.predict(img)
    
    return feature_maps

# Ruta de la imagen del pájaro
path0 = 'C:\\Users\\Angie\\Downloads\\ML_MarquezSulca_Prac6_CNNs\\birs.jpg'
feature_maps = get_feature_maps(path0)

#Creación pdf
pdf_pages = PdfPages('C:\\Users\\Angie\\Downloads\\ML_MarquezSulca_427278_Prac6_reporteFiltrosCNN.pdf')

#Creamos una función para el guardado de figuras
def save_figs_to_pdf(imagenes, feature_maps):

    for ix in range(imagenes):
        fig, ax = plt.subplots()
        ax.imshow(feature_maps[0, :, :, ix], cmap='gray') 
        ax.set_xticks([])
        ax.set_yticks([])
        
        pdf_pages.savefig(fig)
        pyplot.close(fig)

#LLlamamos a la función que acabamos de crear
imagenes = 64
save_figs_to_pdf(imagenes, feature_maps)

"Con imagen de mariposa"

#Nueva ruta
path1 = 'C:\\Users\\Angie\\Downloads\\ML_MarquezSulca_Prac6_CNNs\\mariposa.jpg'

#Llamamos a la función de feature maps
feature_maps = get_feature_maps(path1)

#Llamamos a la función para continuar el pdf
imagenes = 64
save_figs_to_pdf(imagenes, feature_maps)

#Ahora sí cerramos el pdf
pdf_pages.close()
