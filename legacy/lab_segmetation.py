# -*- coding: utf-8 -*-
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from image_utils import roi_extraction
import glob
import cv2



input_shape = (256,256)
model = load_model('./segmentation_models_checkpoints/trained_model.hdf5')


imgs_paths = glob.glob('../data_source/covid19/CAM/normal/val/normal/*')


croped_imgs_dir = '../data_source/covid19/CAM/croped/val/normal/'
for img_path in imgs_paths:

    
    crop = roi_extraction.get_lung_crop(img_path,input_shape,model)
    cv2.imwrite(croped_imgs_dir + img_path.split('\\')[-1],crop)













