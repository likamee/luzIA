# -*- coding: utf-8 -*-
from tensorflow import keras
from model_utils.cam_models import build_vgg16_GAP
import numpy as np
from image_utils import cam_visualization
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from metrics_redcheck import *
import cv2

#IMPORTANTE
#O CONJUNTO USADA PARA VALIDAÇÃO É ATUALMENTE NOMIADO COMO TESTE NA PASTA
#lEMBRAR DESSA NOMECLATURA SE MAIS DE UMA PESSOA LIDAR COM ISSO


#NESSE PIPELINE SERÁ FEITO A VALIDAÇÃO DOS MODELOS TREINADOS
#AQUI IREMOS INCLUIR AS MÉTRICAS PARA GERAÇÃO DE RELATÓRIO

#O nome do modelo é de extrema importancia.
#Os relatorias terão o mesmo nome que o modelo
MODEL_NAME = "model-vasos.h5"
print('loading the model')

model = build_vgg16_GAP(input_shape=(299, 299, 3))
model.load_weights('{}'.format(MODEL_NAME))



IMAGE_WIDTH = 299
IMAGE_HEIGHT = 299
validation_data_dir = 'data/unifesp_0213_20/encode-vasos/val'

print('loading images')
val_data_generator = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

validation_generator = val_data_generator.flow_from_directory(
    validation_data_dir,
    target_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
    batch_size=1,
    shuffle=False)



preds = model.predict(validation_generator,steps=validation_generator.n)
probs = [x.max() for x in preds]
preds = [x.argmax() for x in preds]
for i in range(len(probs)):
   probs[i] = -probs[i] if preds[i] == 0 else probs[i]

label_map = lambda x: 0 if x.split("/")[0] == "normal" else 1

y_test =  [label_map(x) for x in validation_generator.filenames]

metrics_rc(y_test,preds,probs)

#print(classification_report(y_test,preds))

#geracao das metricas


"""
#Apenas para visualizar o cam

layers = [
    
    "block4_conv2","block4_conv3",
    "block5_conv1","block5_conv2", "block5_conv3"]


img_path = '../data_source/retina_normal_alterado_sem_auditar/test/alterado/{CE56A10C-0B6C-4E75-91BA-DA8886C79482}.jpg'


img,cam, pred = cam_visualization. generate_cam(img_path,model,input_shape=(299,299),layer_name=layers[0])

"""









