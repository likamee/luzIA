import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn import preprocessing
import random
import tensorflow as tf
import pickle

len(os.listdir("data/odir/preprocessed_images"))
df = pd.read_csv("data/odir/full_df.csv")
df.head()
count = 1
f = plt.figure(figsize=(50,20))
for Class in df['labels'].unique():
    seg = df[df['labels']==Class]
    address = seg.sample().iloc[0]['filename']
    dataset_dir = "data/odir/preprocessed_images/"
    img = cv2.imread(dataset_dir+ address)
    #print(img)
    ax = f.add_subplot(2, 4,count)
    ax = plt.imshow(img)
    ax = plt.title(Class,fontsize= 30)
    count = count + 1
plt.show()
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
print(len(left_cataract_imgs))
print(len(right_cataract_imgs))
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
print(len(left_normal_imgs))
print(len(right_normal_imgs))
import random
print(len(left_normal_imgs))
print(len(right_normal_imgs))
cataract = np.concatenate((left_cataract_imgs,right_cataract_imgs),axis=0)
normal = np.concatenate((left_normal_imgs,right_normal_imgs),axis=0)
print(len(cataract),len(normal))
images_dir = "data/odir/preprocessed_images/"
image_size=299
labels = []
dataset = []
def create_dataset(image_category,label):
    for img in image_category:
        image_path = os.path.join(images_dir,img)
        try:
            image = cv2.imread(image_path)
            image = cv2.resize(image,(image_size,image_size))
        except:
            continue
        
        dataset.append([np.array(image),np.array(label)])
    random.shuffle(dataset)
    return dataset

dataset = create_dataset(cataract,1)
dataset = create_dataset(normal,0)
len(dataset)
X = np.array([i[0] for i in dataset]).reshape(-1,image_size,image_size,3)
Y = np.array([i[1] for i in dataset])
X.shape, Y.shape
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.1, random_state=4)
X_val = X_train[-50:]
Y_val = Y_train[-50:]
X_train = X_train[:-50]
Y_train = Y_train[:-50]

with open('xtrain.txt', 'wb') as fp:
    pickle.dump(X_train, fp)

with open('ytrain.txt', 'wb') as fp:
    pickle.dump(Y_train, fp)

with open('xval.txt', 'wb') as fp:
    pickle.dump(X_val, fp)

with open('yval.txt', 'wb') as fp:
    pickle.dump(Y_val, fp)


""" with open ('xtrain.txt', 'rb') as fp:
    X_train = pickle.load(fp)

with open ('ytrain.txt', 'rb') as fp:
    Y_train = pickle.load(fp)

with open ('xval.txt', 'rb') as fp:
    X_val = pickle.load(fp)

with open ('yval.txt', 'rb') as fp:
    Y_val = pickle.load(fp) """

        
print(f"X_train Shape: {X_train.shape}, Y_train Shape: {Y_train.shape}")
print(f"X_val Shape: {X_val.shape}, Y_val Shape: {Y_val.shape}")
print(f"X_test Shape: {X_test.shape}, Y_test Shape: {Y_test.shape}")
from tensorflow.keras.applications import ResNet50, VGG16
resnet = VGG16(weights="imagenet", include_top = False, input_shape=(image_size,image_size,3))



for layer in resnet.layers:
    layer.trainable = False

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten,Dense

model = Sequential()
model.add(resnet)
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor="val_loss",patience=7, verbose=1)
callbacks = [early_stopping]

with open('xtrain.txt', 'wb') as fp:
    pickle.dump(X_train, fp)

with open('xval.txt', 'wb') as fp:
    pickle.dump(X_val, fp)

with open('ytrain.txt', 'wb') as fp:
    pickle.dump(Y_train, fp)

with open('yval.txt', 'wb') as fp:
    pickle.dump(Y_val, fp)

hist = model.fit(X_train, Y_train, batch_size=8, epochs=100, validation_data=(X_val, Y_val),
                    verbose=1,callbacks=callbacks)

import matplotlib.pyplot as plt
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()
loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.figure()

plt.show()
model.evaluate(X_test, Y_test, batch_size=8)
y_pred = model.predict(X_test, batch_size=8)
y_pred[y_pred <= 0.5] = 0.
y_pred[y_pred > 0.5] = 1.
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(Y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
from sklearn.metrics import classification_report
print(classification_report(Y_test, y_pred))