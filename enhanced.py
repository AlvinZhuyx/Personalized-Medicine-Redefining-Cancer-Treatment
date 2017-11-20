# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 22:21:28 2017

@author: zhuya
"""
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers 

train_code = np.hstack((truncated_one_hot_gene[:train_size], truncated_one_hot_variation[:train_size]))
test_code = np.hstack((truncated_one_hot_gene[train_size:], truncated_one_hot_variation[train_size:]))
total_code =  np.concatenate((train_code, test_code), axis=0)
total_text_arrays = np.concatenate((text_train_arrays,text_test_arrays),axis = 0)

def dictionary_model():
    model = Sequential()
    model.add(Dense(256, input_dim= Gene_INPUT_DIM*2, init='normal', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, init='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(200, init='normal', activation='linear'))
    
    #optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) original type
    optimizer = optimizers.Adadelta(lr=3, rho=0.95, epsilon=1e-08, decay=0.0) 
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model

dicmodel = dictionary_model()
dicmodel.summary()
best_loss = 1e300
for i in range(20):
    estimator = dicmodel.fit(total_code, total_text_arrays, epochs=5, batch_size=64)
    dic_loss = dicmodel.evaluate(x = test_code, y=text_test_arrays)
    print("")
    print("test loss: %.4f" % dic_loss)
    if (dic_loss < best_loss):
        best_loss = dic_loss
        dicmodel.save_weights('best_weight.h5')

print("best loss: %.4f" % best_loss)


dicmodel.load_weights('best_weight.h5')
train_text_predict = dicmodel.predict(train_code)
test_text_predict = dicmodel.predict(test_code)

tmp_distance = np.zeros((train_size))
dic_train_id = np.zeros((train_size, 50))
dic_test_id = np.zeros((test_size, 50))

for i in range(train_size):
    for j in range(train_size):
        tmp_distance[j] = np.sqrt(np.sum(np.square(train_text_predict[i] - text_train_arrays[j])))
    tmp_sort = np.argsort(tmp_distance)
    dic_train_id[i] = tmp_sort[:50]


for i in range(test_size):
    for j in range(train_size):
        tmp_distance[j] = np.sqrt(np.sum(np.square(test_text_predict[i] - text_train_arrays[j])))
    tmp_sort = np.argsort(tmp_distance)
    dic_test_id[i] = tmp_sort[:50]  
