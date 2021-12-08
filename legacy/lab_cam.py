# -*- coding: utf-8 -*-
import os
import numpy as np
import shutil
import cv2
import tensorflow as tf
from image_utils import cam_visualization
import matplotlib.pyplot as plt
from model_utils.cam_models import build_vgg16_GAP
import glob






#imgs_path = os.listdir('./data_source/normal_alterado/train/alterado/')
#moved_imgs = np.random.choice(imgs_path,167)
#for im in moved_imgs:    
#    try:
#        shutil.move('./data_source/normal_alterado/train/alterado/{}'.format(im),
#                    './data_source/normal_alterado/val/alterado/{}'.format(im))
#    except FileNotFoundError:
#        continue
    

#É interessante testar varios ouputs de layers
model = build_vgg16_GAP(input_shape=(170, 256, 3))
model.load_weights('retina_net_6.h5')

#Agora irei iterar pelos layer e gerar um CAM para cada um
layers_name = ['block1_conv1',
               'block1_conv2',
               'block2_conv1',
               'block2_conv2',
               'block3_conv1',
               'block3_conv2',
               'block3_conv3',
               'block4_conv1',
               'block4_conv2',
               'block4_conv3',
               'block5_conv1',
               'block5_conv2',
               'block5_conv3'
               ]


img_path= '../data_source/covid19/CAM/croped/val/covid//01-04-2020-16-53-00-LPLATFMF.jpg'
for layer_name in layers_name:


    outputs = cam_visualization.generate_cam(img_path,model,(256,256),layer_name)



    cv2.imwrite("./cam_sample/{}_cam.png".format(layer_name),outputs[1])





#Pegar a média de todos os layers para entender as ativações do inicio ao fim
imgs_name = glob.glob("./cam_sample/*")


img = []
i = 1
soma = 0

for img_name in imgs_name:
    
    if len(img) == 0:        
        img = cv2.imread(img_name).astype(float)
        soma += i
        i += 1
    else:
        new_img = cv2.imread(img_name).astype(float)
        img = img + (i*new_img)
        soma += i
        i += 1
    
img = img/soma
img = img.astype(np.uint8)
cv2.imwrite('./cam_sample/media_ponderada.png',img)    
    


#É interessante testar varios ouputs de layers
model = build_vgg16_GAP(input_shape=(256, 256, 3))
model.load_weights('best_covid_vgg16_net_b5c2_b5c3')

img_path= '../data_source/covid19/CAM/croped/val/covid//01-04-2020-16-53-00-LPLATFMF.jpg'


outputs = cam_visualization.generate_cam(img_path,model,(256,256),'block5_conv2')











#from keras.applications.vgg16 import VGG16
#
##Define o Modelo como VGG16
#model = VGG16()
#print(model.summary())
#
##Seleciona os pesos do Modelo
#names = [weight.name for layer in model.layers for weight in layer.weights]
#weights = model.get_weights()[0]
#biases = model.get_weights()[1]
#
#
#
##Imprime os pesos
#for name, weight in zip(names, weights):
#    print(name, weight.shape) #Imprime o nome da camada e o formato dos pesos
#    print(weight) #Imprime os valores dos pesos da camada
#    
##Imprime os pesos
#print('Abaixo seguem os BIAS')
#for name, weight in zip(names, biases):
#    print(name, weight.shape) #Imprime o nome da camada e o formato dos pesos
#    print(weight) #Imprime os valores dos pesos da camada
















