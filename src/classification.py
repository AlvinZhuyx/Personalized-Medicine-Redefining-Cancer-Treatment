# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 22:50:07 2017

@author: zhuya
"""
#import word_embedding as we  
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers 

'''
define a neural network model as the classifier
@param: 
    Text_INPUT_DIM : dimension of the input document embedding
    Gene_INPUT_DIM: dimension of the encoded gene 
    Variation_INPUT_DIM: dimension of the encoded gene variation
@return:
    a nerual network model
'''
def nn_baseline_model(Text_INPUT_DIM = 200, Gene_INPUT_DIM = 25, Variation_INPUT_DIM = 25):
    model = Sequential()
    model.add(Dense(256, input_dim=Text_INPUT_DIM+ Gene_INPUT_DIM + Variation_INPUT_DIM, init='normal', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, init='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(80, init='normal', activation='relu'))
    model.add(Dense(9, init='normal', activation="softmax"))
    
    #optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) original type
    optimizer = optimizers.Adadelta(lr=0.5, rho=0.95, epsilon=1e-08, decay=0.0) 
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

'''
train the nerual_network model:
@param:
    model: pre-defined nerual network model
    train_set: training set 
    encoded_y: ground truth data(one-hot encoding)
    filename: filename of the pretrained network
@return:
    model: trained nn model
'''
def train_nn_model(model, train_set, encoded_y, filename = 'best_weight_predict_all.h5'):
    if os.path.isfile(filename):
        model.load_weights(filename)
        print('successful load\n')
    else:
        print('begin training\n')
        best_acc = 0
        acc = []
        val_acc = []
        loss = []
        val_loss = []
        for i in range(30):
            estimator=model.fit(train_set, encoded_y, validation_split=0.2, epochs=2, batch_size=64)
            if (best_acc < estimator.history['val_acc'][-1] * 100):
                best_acc = estimator.history['val_acc'][-1] * 100
                model.save_weights(filename)
            print("Training accuracy: %.2f%% / Best validation accuracy: %.2f%%" % (100*estimator.history['acc'][-1], best_acc))
            acc += estimator.history['acc']
            val_acc += estimator.history['val_acc']
            loss += estimator.history['loss']
            val_loss += estimator.history['val_loss']
            
        #plot the history for loss and accuracy
        plt.plot(acc)
        plt.plot(val_acc)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.show()
        plt.plot(loss)
        plt.plot(val_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'valid'], loc='upper left')
        plt.show()
    return model

'''
#predict for the new data


#enha
n = 50

a = 0.2
y_predict_final = np.zeros((test_size, 9))
y_predict_final += y_pred * (1 - a);

for i in range(test_size):
    for j in range(n):
        y_predict_final[i] += a/n * encoded_y[int(dic_test_id[i][j])]


train_totaldata = np.zeros((train_size, 9 * (n+1)))
test_totaldata = np.zeros((test_size, 9 * (n+1)))
for i in range (train_size):
    train_totaldata[i][:9] = encoded_y[i]
    for j in range (n):
        train_totaldata[i][9 + j * 9 : 9 + (j + 1) * 9] = encoded_y[int(dic_train_id[i][j])]

for i in range (test_size):
    test_totaldata[i][:9] = y_pred[i]
    for j in range (n):
        test_totaldata[i][9 + j * 9 : 9 + (j + 1) * 9] = encoded_y[int(dic_test_id[i][j])]

def weight_model():
    model = Sequential()
    model.add(Dense(256, input_dim = 9 * (n+1), init='normal', activation='relu')) 
    model.add(Dense(9, init='normal', activation="softmax"))
    optimizer = optimizers.Adadelta(lr=0.5, rho=0.95, epsilon=1e-08, decay=0.0) 
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

weightmodel = weight_model()
weightmodel.summary()
best_acc = 0
for i in range(100):
    estimator = weightmodel.fit(train_totaldata, encoded_y, validation_split=0.2, epochs=1, batch_size=64)
    if (best_acc < estimator.history['val_acc'][-1] * 100):
        best_acc = estimator.history['val_acc'][-1] * 100
        #weightmodel.save_weights('best_weight_final.h5')
        
print("Training accuracy: %.2f%% / Best validation accuracy: %.2f%%" % (100*estimator.history['acc'][-1], best_acc))
#weightmodel.load_weights('best_weight_final.h5')
y_predict_final = weightmodel.predict_proba(test_totaldata)
'''