# -*- coding: utf-8 -*-
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np
import cv2
import keras
import glob
from cam_models import build_vgg16_GAP
from cam_visualization import generate_cam
import matplotlib.pyplot as plt




#Diretorio dos dados
data_folder = './data_source/val/'

alterado_imgs_path = glob.glob(data_folder + 'alterado/*.jpg')
normal_imgs_path = glob.glob(data_folder + 'normal/*.jpg')
nvalidado_imgs_path = glob.glob(data_folder + 'naovalidado/*.jpg')

#Carregamos o modelo para texte.
#model = keras.models.load_model('small_last4.h5')



#Em casos onde salvamos apenas os pesos
model = build_vgg16_GAP(input_shape=(170, 256,3))
model.load_weights('../checkpoints/normal_alterado/retina_net_6.h5')


#Generator para gerar as imagems de validação
val_data_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

IMAGE_HEIGHT = 170
IMAGE_WIDTH = 256


validation_data_dir = './data_source//normal_alterado/val/'
val_generator = val_data_generator.flow_from_directory(
    validation_data_dir,
    target_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
    batch_size=4,
    interpolation = 'nearest',
    classes = ['normal', 'alterado'])




img_path = './data_source/normal_alterado/val/' + val_generator.filenames[2]

img_path = 'gray_img.jpg'
a = cv2.imread(img_path,0)

cv2.imwrite('gray_img.jpg',a)



img,superimposed_img,pred = generate_cam(img_path,model,img_shape=(170,256))

cv2.imwrite('cam_gray.jpg',superimposed_img)


fig,ax = plt.subplots(2,1)

ax[0].imshow(img)
ax[1].imshow(superimposed_img)




#Aqui salvamos as duas imagens
cv2.imwrite('orginal_0.png',img)
cv2.imwrite('cam_0.png',superimposed_img)














