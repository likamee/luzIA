# -*- coding: utf-8 -*-
from tensorflow.keras.applications import VGG16, InceptionResNetV2, InceptionV3, VGG19, ResNet50
from tensorflow.keras import models, Sequential
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, Conv2D, MaxPooling2D, BatchNormalization, Lambda, Activation
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives, BinaryAccuracy, Precision, Recall, AUC
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.regularizers import l2


METRICS = [      
      BinaryAccuracy(name='accuracy'),
      AUC(name='auc')
    ]



def global_average_pooling(x):
    return K.mean(x, axis = (2, 3))

def global_average_pooling_shape(input_shape):
    return input_shape[0:2]



#best 12 layers eyepacs and 6 messidor2
def build_vgg16_GAP(n_layers = 12, type_train = 'n', name_test = '', input_shape=(299, 299, 3), trainable_layers = None):
    """
    Atualmente o modelo mais eficiente na deteccao das retinopatias.
    Consiste de layers de uma VGG16 treinados no conjunto de
    dados imagenet
    """
    
    #Se o tipo de treino for normal, iremos fazer a transferencia de conhecimento apenas da ImageNet
    #Caso for FT, precisamos todas as layers do modelo com LR baixo
    #Caso for uma transferencia de conhecimento, frizar algumas layers e treinar em cima das demais com dados novos

    # ----------------------------- #

    #Congelo os layer que não irei treinar
    #Textar deixar os ultimos layers treinaveis
    
    #Os indexes dos layers que serão treinados
    #Se a lista de layers para treino for None então
    #não sera treinada nem um layer
    if type_train == 'n':
        print('Generating model with ImageNet weights')        
        name_test = name_test+'_base'
        vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        
        x = vgg_conv.output
        x = Lambda(global_average_pooling, output_shape=global_average_pooling_shape)(x)
        predictions = Dense(2, activation = 'softmax', kernel_initializer='uniform')(x)

        model = Model(inputs=vgg_conv.input, outputs=predictions)
    else:
        print('Loading the base model to '+type_train)
        model = load_model(name_test+'.hdf5')
        name_test = name_test+'_'+type_train
    count = 0
    if type_train == 'ft':
        for layer in model.layers[::-1]:            
            layer.trainable = True
        lr = 0.0001
    else:
        for layer in model.layers[::-1]:

            if not isinstance(layer,Conv2D) or count > n_layers:
                layer.trainable = False
            else:
                layer.trainable = True
                count = count + 1
        lr = 0.01
        
    opt = SGD(learning_rate=lr, decay=1e-6, momentum=0.9, nesterov=True)
    #opt = Adam(learning_rate=lr, decay=1e-6)

    model.summary()
    #opt = RMSprop(lr=0.001, decay=4e-5, momentum=0.0)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt, metrics=[METRICS])
                           
    return model, name_test

def build_vgg19_GAP(input_shape=(299, 299, 3), trainable_layers = None, n_layers = 3):
    """
    Atualmente o modelo mais eficiente na deteccao das retinopatias.
    Consiste de layers de uma VGG16 treinados no conjunto de
    dados imagenet
    """
    
    
    vgg_conv = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)
    #Congelo os layer que não irei treinar
    #Textar deixar os ultimos layers treinaveis
    
    #Os indexes dos layers que serão treinados
    #Se a lista de layers para treino for None então
    #não sera treinada nem um layer
    x = vgg_conv.output
    x = Lambda(global_average_pooling, output_shape=global_average_pooling_shape)(x)
    predictions = Dense(2, activation = 'softmax', kernel_initializer='uniform')(x)

    model = Model(inputs=vgg_conv.input, outputs=predictions)

    count = 0
    for layer in vgg_conv.layers[::-1]:

        if not isinstance(layer,Conv2D) or count > n_layers:
            layer.trainable = False
        else:
            layer.trainable = True
            count = count + 1
	
    opt = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #opt = RMSprop(lr=0.001, decay=4e-5, momentum=0.0)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt, metrics=[METRICS])
                           
    return model


def build_alexNET(input_shape):
    MODEL = models.Sequential()

    # 1st Convolutional Layer
    MODEL.add(Conv2D(filters=96, input_shape=input_shape, kernel_size=(11,11), strides=(4,4), padding='same', kernel_constraint=max_norm(3), bias_constraint=max_norm(3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    MODEL.add(Activation('relu'))
    # Pooling
    MODEL.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
    # Batch Normalisation before passing it to the next layer
    MODEL.add(BatchNormalization())

    # 2nd Convolutional Layer
    MODEL.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='same',  kernel_constraint=max_norm(3), bias_constraint=max_norm(3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    MODEL.add(Activation('relu'))
    # Pooling
    MODEL.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
    # Batch Normalisation
    MODEL.add(BatchNormalization())

    # 3rd Convolutional Layer
    MODEL.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same',  kernel_constraint=max_norm(3), bias_constraint=max_norm(3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    MODEL.add(Activation('relu'))
    # Batch Normalisation
    MODEL.add(BatchNormalization())

    # 4th Convolutional Layer
    MODEL.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same',  kernel_constraint=max_norm(3), bias_constraint=max_norm(3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    MODEL.add(Activation('relu'))
    # Batch Normalisation
    MODEL.add(BatchNormalization())

    # 5th Convolutional Layer
    MODEL.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',  kernel_constraint=max_norm(3), bias_constraint=max_norm(3), kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
    MODEL.add(Activation('relu'))
    # Pooling
    MODEL.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='same'))
    # Batch Normalisation
    MODEL.add(BatchNormalization())

    # Passing it to a dense layer
    MODEL.add(Flatten())
    # 1st Dense Layer
    MODEL.add(Dense(4096, input_shape=(384*384*3,)))
    MODEL.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    MODEL.add(Dropout(0.3))
    # Batch Normalisation
    MODEL.add(BatchNormalization())

    # 2nd Dense Layer
    MODEL.add(Dense(4096))
    MODEL.add(Activation('relu'))
    # Add Dropout
    MODEL.add(Dropout(0.3))
    # Batch Normalisation
    MODEL.add(BatchNormalization())

    # 3rd Dense Layer
    MODEL.add(Dense(1000))
    MODEL.add(Activation('relu'))
    # Add Dropout
    MODEL.add(Dropout(0.3))
    # Batch Normalisation
    MODEL.add(BatchNormalization())

    # Output Layer
    MODEL.add(Dense(2))
    MODEL.add(Activation('softmax'))

    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #MODEL.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
    MODEL.compile(loss='binary_crossentropy', optimizer='adadelta',metrics=[METRICS])

    return MODEL

def build_InceptionV3_GAP(input_shape=(299, 299, 3), trainable_layers = None, n_layers = 62):
    """
    Modelo benchmark (mas e bosta)
    """

    incep3 = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

    x = incep3.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation = 'softmax')(x)

    model = Model(inputs=incep3.input, outputs=predictions)

    count = 0
    for layer in incep3.layers[::-1]:

        if count > n_layers:
            layer.trainable = False
        else:
            layer.trainable = True
            count = count + 1
            	
    opt = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #opt = RMSprop(lr=0.001, decay=4e-5, momentum=0.0)
    model.compile(loss='binary_crossentropy',optimizer=opt, metrics=[METRICS])
                           
    return model

def build_inception_resnetv2(input_shape, n_layers = 50):
    net = InceptionResNetV2(weights='imagenet', input_tensor=None, include_top=False, input_shape=input_shape)

    x = net.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)

    model = Model(inputs=net.input, outputs=predictions)

    count = 0
    for layer in net.layers[::-1]:

        if count > n_layers:
            layer.trainable = False
        else:
            layer.trainable = True
            count = count + 1
            	
    opt = SGD(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    #opt = RMSprop(lr=0.001, decay=4e-5, momentum=0.0)
    model.compile(loss='binary_crossentropy',optimizer=opt, metrics=[METRICS])

    return model


def build_resnet50(n_layers = 12, type_train = 'n', name_test = '', input_shape=(299, 299, 3), trainable_layers = None):
    resnet = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in resnet.layers:
        layer.trainable = False

    model = Sequential()
    model.add(resnet)
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1,activation="sigmoid"))
    model.summary()
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

    return model, name_test