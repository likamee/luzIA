# -*- coding: utf-8 -*-
from tensorflow import keras
from model_utils import cam_models
from tensorflow.keras.callbacks import ModelCheckpoint

#PIPELINE RESPONSAVEL POR LEVAR OS PARAMETROS PARA TREINO DO MODELO




#Configurações do treino
IMAGE_WIDTH = 299
IMAGE_HEIGHT  = 299

BATCH_SIZE =  10 #com 6 chegamos no melhor tempo a melhor acc
EPOCHS = 30 #30 epochs se provou o sufuciente para treino

#Aqui os diretórios de treino, test e validação
training_data_dir = 'data/unifesp_classificadas/encode-escavacao/train'
validation_data_dir = 'data/unifesp_classificadas/encode-escavacao/val'

     

#Aqui é compilado o modelo que será treinado
model = cam_models.build_vgg16_GAP(input_shape=(299, 299, 3), trainable_layers=['block5_conv1','block5_conv2','block5_conv3','block5_conv1','block5_conv2','block5_conv3'])




#Gerador do conjunto de treino
#TODO: lembrar de implementar um treino em batch
training_data_generator = keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255,
    fill_mode='nearest', 
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=50
    )


#Um gerador para os dados de validação da rede
val_data_generator = keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
                                                                


train_generator = training_data_generator.flow_from_directory(
    training_data_dir,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=['normal','alterado'],
    interpolation = 'nearest', #Testar outros tipos de resize para alimentar a rede
    target_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
    shuffle=True)


validation_generator = val_data_generator.flow_from_directory(
    validation_data_dir,
    target_size=(IMAGE_HEIGHT,IMAGE_WIDTH),
    batch_size=32,
    interpolation = 'nearest',
    classes = ['normal','alterado'])




class_weight = {'normal':2.2 , 'alterado':1}
mc = ModelCheckpoint('model-excavation.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)

# Train the model
model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.samples/train_generator.batch_size,
      epochs=EPOCHS,
      validation_data=validation_generator,
      validation_steps=validation_generator.samples/validation_generator.batch_size,
      verbose=1,
      callbacks=[mc])
     # class_weight=class_weight)
 
# Save the model
model.save('model-excavation.h5')




















