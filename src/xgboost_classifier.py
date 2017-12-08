# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 10:43:19 2017

@author: zhuya
"""
# this script using xgboost to do the prediction

import xgboost as xgb
import sklearn
import pandas as pd
import numpy as np
import util   
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics.cluster import normalized_mutual_info_score
# use xgboost as classifier
#@param: 
#   train_set: training data input
#   train_y: training label
#   test_set: test data input
#   fold: validation fold
#   max_depth: max depth of each boost tree
#   n_estimator: number of boost tree used
#@return:
#   y-final: value predicted by xgboost classifier        
    
def xgbclassifier(train_set, train_y, test_set, fold = 10, max_depth = 6, n_estimators=1000):

    y_foldpred = []
    y_foldlabel = []
    ACC = []
    LOSS = []
    NMI = []
    
    denom  = 0
    for i in range(fold):
        params = {
                'eta': 0.03333,
                'max_depth': max_depth,
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'num_class': 9,
                'seed': i,
                'silent': True
                }
        xtrain,xval,ytrain,yval = sklearn.model_selection.train_test_split(train_set, train_y, test_size = 0.18, random_state = i)
        watchlist = [(xgb.DMatrix(xtrain,ytrain), 'train'), (xgb.DMatrix(xval, yval), 'valid')]
        model = xgb.train(params, xgb.DMatrix(xtrain, ytrain), n_estimators,  watchlist, verbose_eval=25, early_stopping_rounds=100)
        yvalpredict = model.predict(xgb.DMatrix(xval), ntree_limit=model.best_ntree_limit)
        yvaltmp = np.argmax(yvalpredict, axis = 1)
        y_foldlabel.extend(list(yval))
        y_foldpred.extend(list(yvaltmp))
        nmi = normalized_mutual_info_score(yvaltmp, yval)
        NMI.append(nmi)
        loss = sklearn.metrics.log_loss(yval, yvalpredict, labels = list(range(9)))
        acc = sklearn.metrics.accuracy_score(yval, yvaltmp)
        LOSS.append(loss)
        ACC.append(acc)
        print("final validation loss:")
        print(loss)
        print("final validation accuracy:")
        print(acc)
        pred = model.predict(xgb.DMatrix(test_set), ntree_limit=model.best_ntree_limit + 80)
        y_final = pred.copy()
        if denom != 0:
            pred = model.predict(xgb.DMatrix(test_set), ntree_limit=model.best_ntree_limit + 80)
            y_final += pred
            denom += 1
        else:
            denom = 1
    
    print("Accuracy: %.4f ± %.4f" % (np.mean(ACC), np.std(ACC)))
    print("NMI: %.4f ± %.4f" % (np.mean(NMI), np.std(NMI)))
    print("Log_loss: %.4f ± %.4f" % (np.mean(LOSS), np.std(LOSS)))
    cnf_matrix = confusion_matrix(y_foldlabel, y_foldpred)
    mat = np.array(cnf_matrix/len(y_foldlabel))
    util.plot_confusion_matrix(mat, classes='', normalize=True, title='Confusion matrix')
    y_final /= denom
    return y_final




