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
import numpy as np
import util
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

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

def train_nn_model_simple(model, train_set, encoded_y, filename = 'best_weight_predict_all.h5'):
    print('begin training\n')
    best_acc = 0
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    for i in range(30):
        estimator=model.fit(train_set, encoded_y, validation_split=0.01, epochs=2, batch_size=64)
        if (best_acc < estimator.history['val_acc'][-1] * 100):
            best_acc = estimator.history['val_acc'][-1] * 100
            model.save_weights(filename)
    return model

def nn_cross_validation(X, y, skf = StratifiedKFold(n_splits=10, random_state = 66, shuffle = True)):
    ret = []
    mat = 0
    LOSS = []
    ACC = []
    NMI = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = nn_baseline_model(25, 25, 200)
        train_nn_model_simple(model, X_train, y_train, 'temp.h5')

        y_pred_proba = model.predict_proba(X_test)
        y_pred = np.argmax(y_predict, axis = 1)

        acc = accuracy_score(y_test, y_pred)
        nmi = normalized_mutual_info_score(y_pred, y_test)
        loss = log_loss(y_test, y_pred_proba)
        ACC.append(acc)
        LOSS.append(loss)
        NMI.append(nmi)
        cnf_matrix = confusion_matrix(y_test, y_pred)
        mat = mat + cnf_matrix
    print("Accuracy: %.4f ± %.4f" % (np.mean(ACC), np.std(ACC)))
    print("NMI: %.4f ± %.4f" % (np.mean(NMI), np.std(NMI)))
    print("Log_loss: %.4f ± %.4f" % (np.mean(LOSS), np.std(LOSS)))
    mat = mat / skf.get_n_splits()
    mat = np.array(mat)
    mat = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
    util.plot_confusion_matrix(mat, classes='', normalize=True,
                      title='Confusion matrix')
    return mat


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