# -*- coding: utf-8 -*-
from model_utils.cam_models import build_vgg16_GAP
from image_utils import cam_visualization
import glob
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

#UTILIZAR ESSE SCRIPT PARA A VALIDAÇÃO VISUAL DO RESULTADO DO CAM
#NO CONJUNTO DE DADOS DE VALIDAÇÃO OU IMAGENS NOVAS
#y ==> vai para a próxima imagem
#break ==> para o processo e salva as imagens já conferidas


#Primeiro iremos carregar o modelo que será testado

model = build_vgg16_GAP(input_shape=(170, 256, 3))
model.load_weights('./cam_models_checkpoints/retina_net_6.h5')


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


#Abrimos a pasta contendo as imagens que serão testadas
test_imgs_path = glob.glob('../data_source/train_data/retinografia/retina_normal_alterado/val/alterado/*.jpg')


try:
    conf_imgs = pd.read_csv("./logs/conf_imgs.csv").values
    conf_imgs = list(np.squeeze(conf_imgs))

except FileNotFoundError:
    conf_imgs = []
    
    
for img_name in test_imgs_path:
    
    if img_name.split("\\")[-1] not in conf_imgs:
        
        img = cv2.imread(img_name)
        
        print(img_name)
        plt.imshow(img)
        plt.ion()
        plt.show()
        plt.close()
        
        
        outputs = cam_visualization.generate_cam(img_name,model,(170,256),layers_name[-1])
        
    
        plt.imshow(outputs[1])
        plt.ion()
        plt.show()
        plt.close()
        
        comando = str(input("digite y para proxima imagem:"))
        
        if comando != "y":
            break
        
        
        conf_imgs.append(img_name.split("\\")[-1])
        


audited_df = pd.DataFrame(conf_imgs,columns=["img_names"])
audited_df.to_csv("./logs/conf_imgs.csv",index=False, encoding="utf8")
    
    

    
    
    
    
    
    
    
    
    
    
    