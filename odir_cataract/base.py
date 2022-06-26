import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn import preprocessing
import random
import tensorflow as tf

df = pd.read_csv("data/odir/full_df.csv")

def has_cataract(text):
    if "cataract" in text:
        return 1
    else:
        return 0


df["left_cataract"] = df["Left-Diagnostic Keywords"].apply(lambda x: has_cataract(x))
df["right_cataract"] = df["Right-Diagnostic Keywords"].apply(lambda x: has_cataract(x))

left_cataract_imgs = []
right_cataract_imgs = []
for i in range(len(df)):
    if df["left_cataract"][i] == 1:
        left_cataract_imgs.append(df['Left-Fundus'][i])
    if df["right_cataract"][i] == 1:
        right_cataract_imgs.append(df['Right-Fundus'][i])


def is_normal(text):
    if "normal fundus" in text:
        return 1
    else:
        return 0


df["left_normal"] = df["Left-Diagnostic Keywords"].apply(lambda x: is_normal(x))
df["right_normal"] = df["Right-Diagnostic Keywords"].apply(lambda x: is_normal(x))


left_normal_imgs = []
right_normal_imgs = []
for i in range(len(df)):
    if df["left_normal"][i] == 1:
        left_normal_imgs.append(df['Left-Fundus'][i])
    if df["right_normal"][i] == 1:
        right_normal_imgs.append(df['Right-Fundus'][i])



cataract = np.concatenate((left_cataract_imgs,right_cataract_imgs),axis=0)
normal = np.concatenate((left_normal_imgs,right_normal_imgs),axis=0)


images_dir = "data/odir/preprocessed_images/"
images_dest = "data/odir/algo/"
image_size=299
labels = []
dataset = []

def create_dataset(image_category,label):
    count = 0
    for img in image_category:
        cond = 'normais' if label == 0 else 'catarata'
        image_path = os.path.join(images_dir,img)
        image_path_dest = os.path.join(images_dest+'/'+cond+'/',img)
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
        dataset.append([np.array(image),np.array(label)])
    random.shuffle(dataset)
    print(count)
    return dataset

dataset = create_dataset(cataract,1)
dataset = create_dataset(normal,0)