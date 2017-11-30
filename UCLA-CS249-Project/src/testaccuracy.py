# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 23:13:01 2017

@author: zhuya
"""

import pandas as pd
import numpy as np
#import classification as cl

test_solution = pd.read_csv("../data/stage_2_private_solution.csv", sep = ",")
test_id = test_solution['ID'].values
test_result = np.array(test_solution.drop('ID', axis = 1))
actualsize = len(test_result)

pred = np.array(y_predict_final)

#mysum = 0
myloss = 0
for i in range(actualsize):
    truth = np.argmax(test_result[i])
    #predict = np.argmax(pred[test_id[i]-1])
    #mysum += (truth == predict)
    myloss += -np.log(pred[test_id[i] - 1][truth])
    
#accuracy = 100 * mysum / actualsize
averageloss = myloss / actualsize

print("Test loss: %.2f " % averageloss)
submission = pd.DataFrame(pred)
submission.insert(0, 'ID', test_x['ID'])
submission.columns = ['ID','class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9']
submission.to_csv("submission_all.csv",index=False)
submission.head()





