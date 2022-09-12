# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 12:51:55 2022

@author: yxliao
"""

import os
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import Input, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def load_data(array_p, array_pl):
    x = [] 
    y = []    
    for ar in array_p:
        min_max_scaler = preprocessing.MinMaxScaler()
        arr = min_max_scaler.fit_transform(ar)
        x.append(arr)
    for al in array_pl:
        min_max_scaler = preprocessing.MinMaxScaler()
        arl = min_max_scaler.fit_transform(al)
        y.append(arl)
    x = np.expand_dims(np.array(x), axis=3)
    y = np.expand_dims(np.array(y), axis=3)
    np.random.seed(116)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=420)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25, random_state=420)
    return x_train, x_valid, x_test, y_train, y_valid, y_test


def conv2d_block(input_tensor, n_filters=64, kernel_size=3, batchnorm=True, padding='same'):
    # the first layer
    x = Conv2D(n_filters, kernel_size, padding=padding, kernel_initializer='he_normal')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # the second layer
    x = Conv2D(n_filters, kernel_size, padding=padding, kernel_initializer='he_normal')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x 

def get_unet(input_data, n_filters=64, dropout=0.1, batchnorm=True, padding='same'):
    # contracting path
    c1 = conv2d_block(input_data, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm, padding=padding)
    p1 = MaxPooling2D((2, 2))(c1)
  
    c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm, padding=padding)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm, padding=padding)
    p3 = MaxPooling2D((2, 2))(c3)
        
    c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm, padding=padding)
    #p4 = Dropout(dropout)(c4)
    p4 = MaxPooling2D((2, 2))(c4)
        
    c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm, padding=padding)
    p5 = Dropout(dropout)(c5) 

    # extending path
    u6 = Conv2D(n_filters * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(p5))
    u6 = concatenate([c4, u6], axis=3)
    #u6 = concatenate([Conv2DTranspose(n_filters * 8, (2, 2), strides=(2, 2), padding='same')(p5), c4], axis=3)
    c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm, padding=padding)

    u7 = Conv2D(n_filters * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c6))
    u7 = concatenate([c3, u7], axis=3)
    #u7 = concatenate([Conv2DTranspose(n_filters * 4, (2, 2), strides=(2, 2), padding='same')(c6), c3], axis=3)
    c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm, padding=padding)

    u8 = Conv2D(n_filters * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c7))
    u8 = concatenate([c2, u8], axis=3)
    #u8 = concatenate([Conv2DTranspose(n_filters * 2, (2, 2), strides=(2, 2), padding='same')(c7), c2], axis=3)
    c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm, padding=padding)
        
    u9 = Conv2D(n_filters * 1, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(c8))
    u9 = concatenate([c1, u9], axis=3)
    #u9 = concatenate([Conv2DTranspose(n_filters * 1, (2, 2), strides=(2, 2), padding='same')(c8), c1], axis=3)
    c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm, padding=padding)
    c9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(c9)
    outputs = Conv2D(1, 1, activation='sigmoid')(c9)
    model = Model(inputs=[input_data], outputs=[outputs])
    return model

def PI(path_pi):
    files= os.listdir(path_pi)
    rt_p = []
    array_p= []
    ms_p = []
    for file in files:
        position = path_pi +'\\'+ file
        choose_spec = np.loadtxt(position)
        B = np.zeros([256, 256],dtype=float)
        N = np.unique(choose_spec[:,0])
        if ((256-len(N)) % 2) == 0:
            a = np.array(0).repeat(int((256-len(N))/2))
            b = np.array(0).repeat(int((256-len(N))/2))
        else:
            a = np.array(0).repeat(int((256-len(N))/2))
            b = np.array(0).repeat(int((256-len(N))/2)+1)
        left = np.hstack((a,N,b))
        C = np.around(choose_spec[np.argmax(choose_spec[:,2]),1],3)
        k1 = np.around(np.arange(C-0.01,C+0.01,0.005),3)
        k2 = np.around(np.arange(C-1.28,C+1.28,0.01),3)
        k3 = np.delete(np.unique(np.hstack((k1,k2))),np.searchsorted(np.unique(np.hstack((k1,k2))), C))
        rows = dict(zip(list(range(0, len(B))), list(left)))
        cols = dict(zip(list(range(0, len(B)+1)), list(k3)))
        for row in range(256):
            for i in range(len(choose_spec)):
                if rows[row] == choose_spec[i,0]:
                    for col in range(256):
                        if cols[col]<choose_spec[i,1]<=cols[col+1]:
                            B[col][row] = choose_spec[i,2]
        rt_p.append(left)
        array_p.append(B)
        ms_p.append(k3)
    return array_p, rt_p, ms_p
    
    
def PL(path_pl, path_pi):
    files= os.listdir(path_pl)
    choose_label = []
    for file in files:
        position = path_pl +'\\'+ file
        choose_label.append(np.loadtxt(position))
    array_pl = []
    for i in range(len(PI(path_pi)[1])):
        B = np.zeros([256, 256],dtype=float)
        rows = dict(zip(list(range(0, len(B))), list(PI(path_pi)[1][i])))
        cols = dict(zip(list(range(0, len(B)+1)), list(PI(path_pi)[2][i])))   
        for row in range(256):
            for label in choose_label:
                for i in range(len(label)):
                    if rows[row] == label[i,0]:
                        for col in range(256):
                            if cols[col]<label[i,1]<=cols[col+1]:
                                B[col][row] = label[i,2]
        array_pl.append(B)
    return array_pl
    
def NI(path_ni):
    array_zs = []
    files= os.listdir(path_ni)
    for file in files:
        position = path_ni+'\\'+ file
        array_zs.append(np.loadtxt(position))
    return array_zs

def NL():
    array_zsl = []
    for i in range(100):
        f = np.zeros([256, 256],dtype=float)
        array_zsl.append(f)
    return array_zsl

def DeepPIC_train(x_train, y_train, batch_size, epochs, x_valid, y_valid, saved_model_path):
    input_data = Input(shape=(256, 256, 1))
    model = get_unet(input_data, n_filters=64, dropout=0.5, batchnorm=True, padding='same')
    model.compile(optimizer=Adam(lr = 0.001), loss="binary_crossentropy", metrics=["accuracy"])
    #model.summary()
    callbacks = [
        EarlyStopping(patience=60, verbose=1),
        ReduceLROnPlateau(factor=0.8, patience=3, min_lr=0.00005, verbose=1),
        ModelCheckpoint(saved_model_path, 
                        monitor='loss', verbose=1, save_best_only=True, save_weights_only=True)]
    #train the model
    results = model.fit(x_train, y_train, batch_size, epochs,
                        callbacks=callbacks,validation_data=(x_valid, y_valid))
    #model.save_weights('new_unet.h5', overwrite=True)     
    return results


if __name__=="__main__":
    path_pi = './dataset/positive samples/inputs'
    path_pl = './dataset/positive samples/labels'
    path_ni = './dataset/negative samples/inputs'
    
    array_p = PI(path_pi)[0]
    array_pl = PL(path_pl, path_pi)
    array_zs = NI(path_ni)
    array_zsl = NL()
    array_p.extend(array_zs)
    array_pl.extend(array_zsl)
    
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_data(array_p, array_pl)
    results = DeepPIC_train(x_train, y_train, 2, 40, x_valid, y_valid, '../new.unet.h5')  #batch_size=2, epochs=40
    
    
    