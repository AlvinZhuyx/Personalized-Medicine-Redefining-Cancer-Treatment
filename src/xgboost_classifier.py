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
    
def xgbclassifier(train_set, train_y, test_set, fold = 5, max_depth = 6, n_estimators=1000):

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
        x1,x2,y1,y2 = sklearn.model_selection.train_test_split(train_set, train_y, test_size = 0.18, random_state = i)
        watchlist = [(xgb.DMatrix(x1,y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]
        model = xgb.train(params, xgb.DMatrix(x1, y1), n_estimators,  watchlist, verbose_eval=50, early_stopping_rounds=100)
        score1 = sklearn.metrics.log_loss(y2, model.predict(xgb.DMatrix(x2), ntree_limit=model.best_ntree_limit), labels = list(range(9)))
        print(score1)
        pred = model.predict(xgb.DMatrix(test_set), ntree_limit=model.best_ntree_limit+80)
        y_final = pred.copy()
        #if score < 0.9:
        if denom != 0:
            pred = model.predict(xgb.DMatrix(test_set), ntree_limit=model.best_ntree_limit+80)
            y_final += pred
            denom += 1
        else:
            denom = 1
    print(denom)
    y_final /= denom
    return y_final
    
'''
#enhanced
n = 5

a = 0.25
y_predict_final = np.zeros((test_size, 9))
y_predict_final += y_final * (1 - a);

for i in range(test_size):
    for j in range(n):
        y_predict_final[i] += a/n * encoded_y[int(dic_test_id[i][j])]
'''



