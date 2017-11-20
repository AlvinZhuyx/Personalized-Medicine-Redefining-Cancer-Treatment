# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 11:02:20 2017

@author: zhuya
"""

import os
import re #处理正则表达式
import tqdm #显示进度条 
import string
import pandas as pd
import numpy as np
import keras

#data loading
train_variant = pd.read_csv("../data/training_variants")
test_variant = pd.read_csv("../data/stage2_test_variants.csv")
#因为此时选用||分割数据，而head的地方是用，分割故会出错，因此要特别指定name
train_text = pd.read_csv("../data/training_text", sep = "\|\|", engine = 'python', header=None, skiprows=1, names=["ID","Text"])
test_text = pd.read_csv("../data/stage2_test_text.csv", sep = "\|\|", engine = 'python', header=None, skiprows=1, names=["ID","Text"])
train = pd.merge(train_variant, train_text, how = 'inner', on = 'ID')

train_y = train['Class'].values
train_x = train.drop('Class', axis = 1) #axis代表前面的label 'Class'是包含在第一维里的
train_size = len(train_x)

test_x = pd.merge(test_variant, test_text, how = 'inner', on = 'ID')
test_size = len(test_x)
print('Number of training variants: %d' % (train_size))
print('Number of test variants: %d' % (test_size))
test_index = test_x['ID'].values

all_data = np.concatenate((train_x, test_x), axis=0)
all_data = pd.DataFrame(all_data)
all_data.columns = ["ID", "Gene", "Variation", "Text"]
print(all_data.head())

