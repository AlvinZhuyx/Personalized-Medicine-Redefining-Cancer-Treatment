# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 22:21:28 2017

@author: zhuya
"""
#this code use the embedded gene and validation directly to find the nearest neighbour
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers 

tmp_distance = np.zeros((train_size))
dic_train_id = np.zeros((train_size, 50))
dic_test_id = np.zeros((test_size, 50))

train_code = np.hstack((truncated_one_hot_gene[:train_size], truncated_one_hot_variation[:train_size]))
test_code = np.hstack((truncated_one_hot_gene[train_size:], truncated_one_hot_variation[train_size:]))

for i in range(train_size):
    for j in range(train_size):
        tmp_distance[j] = np.sqrt(np.sum(np.square(train_code[i] - train_code[j])))
    tmp_sort = np.argsort(tmp_distance)
    dic_train_id[i] = tmp_sort[1:51]


for i in range(test_size):
    for j in range(train_size):
        tmp_distance[j] = np.sqrt(np.sum(np.square(test_code[i] - train_code[j])))
    tmp_sort = np.argsort(tmp_distance)
    dic_test_id[i] = tmp_sort[:50]  
'''
#here use the text itself to find the nearest

for i in range(train_size):
    for j in range(train_size):
        tmp_distance[j] = np.sqrt(np.sum(np.square(text_train_arrays[i] - text_train_arrays[j])))
    tmp_sort = np.argsort(tmp_distance)
    dic_train_id[i] = tmp_sort[1:51]


for i in range(test_size):
    for j in range(train_size):
        tmp_distance[j] = np.sqrt(np.sum(np.square(text_test_arrays[i] - text_train_arrays[j])))
    tmp_sort = np.argsort(tmp_distance)
    dic_test_id[i] = tmp_sort[:50]  
'''