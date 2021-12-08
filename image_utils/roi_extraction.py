# -*- coding: utf-8 -*-
import skimage as skimg 
import numpy as np
import cv2




def get_lung_crop(img_path,input_shape,model):
    """
    Função responsavel por extrair a região
    de interesse das imagens de RX de pulmão.
    """
    
    
    img = skimg.img_as_float(skimg.io.imread(img_path))
    img = skimg.transform.resize(img, input_shape)
    img = skimg.exposure.equalize_hist(img)
    img = np.expand_dims(img, -1)
    
    if img.shape[2] != 1:
        img = img[:,:,0]
    
    
    x = img - img.mean()
    x = x/x.std()
    x = np.expand_dims(x, axis=0)
    
    
    
    
    pred = model.predict(x)
    
    #Pegamos apenas a região que se encontra o pulmão
    mask = np.where(pred[0][:,:,0] > 0.77,1,0)
    mask = np.expand_dims(mask,axis=3)
    
    orig_img = cv2.imread(img_path)
    orig_img = cv2.resize(orig_img,(256,256))
    
    crop = orig_img * mask
    
    return crop