import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn import preprocessing
import random
import tensorflow as tf


folder = "data/refuge_data"
images_dest = "data/refuge_data/algo"
image_size=299

def readjson(folder):
    df = pd.read_json(folder+"/index.json", orient='index')
    return df

def create_dataset(df,folder):
    count = 0
    for idx, row in df.iterrows():
        cond = 'normais' if row['Label'] == 0 else 'escavacao'
        image_path = os.path.join(folder+'/images',row['ImgName'])
        image_path_dest = os.path.join(images_dest+'/'+cond+'/',row['ImgName'])
        try:
            image = cv2.imread(image_path)
            image = cv2.resize(image,(image_size,image_size))
            if cv2.imwrite(image_path_dest, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100]):
                print(count)
            else:
                print("Error")
        except:
            continue
        count = count + 1
        
df = readjson(folder+'/val')
create_dataset(df,folder+'/val')

x=0