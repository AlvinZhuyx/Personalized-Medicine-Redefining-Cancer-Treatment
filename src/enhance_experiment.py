'''
This code is for our experiment running the enhanced model mentioned in section 8
'''
import os
import re
import tqdm
import string
import pandas as pd
import numpy as np
import util
from sklearn.decomposition import TruncatedSVD
from data_preprocessing import *
import word_embedding_load as wel
from classification import *
from xgboost_classifier import *
from testaccuracy import *
from enhanced import *

[all_data, train_size, test_size, train_x, train_y, test_x] = util.loadData()
sentences = data_preprocess(all_data)


Text_INPUT_DIM=200
GENE_INPUT_DIM=25
TEXT_INPUT_DIM=200
PATH = '../model/doc2vec/'
modelName='docEmbeddings_win2_load_all.d2v'

param = util.Doc2VecParam(1, 2, 200, 1e-4, 5, 4, 30, 1)


svd = TruncatedSVD(n_components=GENE_INPUT_DIM, n_iter=25, random_state=12)

text_model = wel.loadTextModel(PATH + modelName)


truncated_one_hot_gene = wel.getGeneVec(all_data, svd)
truncated_one_hot_variation = wel.getVariationVec(all_data, svd)
text_train_arrays, text_test_arrays = wel.getTextVec(text_model, train_size, test_size, TEXT_INPUT_DIM)

print(text_train_arrays.shape)
print(text_test_arrays.shape)
text_train_arrays[0]

train_dic, test_dic = getdictionary(train_size, test_size, GENE_INPUT_DIM, text_train_arrays, text_test_arrays, truncated_one_hot_gene, truncated_one_hot_variation)

encoded_y = pd.get_dummies(train_y)
encoded_y = np.array(encoded_y)
print(encoded_y.shape)

n = 10
train_set = np.zeros((train_size, GENE_INPUT_DIM * 2 + TEXT_INPUT_DIM + n*9))
test_set = np.zeros((test_size, GENE_INPUT_DIM * 2 + TEXT_INPUT_DIM + n*9))

for i in range(train_size):
    train_set[i][:GENE_INPUT_DIM] = truncated_one_hot_gene[i]
    train_set[i][GENE_INPUT_DIM:2*GENE_INPUT_DIM] = truncated_one_hot_variation[i]
    train_set[i][2*GENE_INPUT_DIM: 2*GENE_INPUT_DIM + TEXT_INPUT_DIM] = text_train_arrays[i]
    for j in range(n):
        train_set[i][2*GENE_INPUT_DIM + TEXT_INPUT_DIM + j*9: 2*GENE_INPUT_DIM + TEXT_INPUT_DIM + (j + 1)*9] = encoded_y[int(train_dic[i][j+1])]

for i in range(test_size):
    test_set[i][:GENE_INPUT_DIM] = truncated_one_hot_gene[i + train_size]
    test_set[i][GENE_INPUT_DIM:2*GENE_INPUT_DIM] = truncated_one_hot_variation[i + train_size]
    test_set[i][2*GENE_INPUT_DIM: 2*GENE_INPUT_DIM + TEXT_INPUT_DIM] = text_train_arrays[i]
    for j in range(n):
        test_set[i][2*GENE_INPUT_DIM + TEXT_INPUT_DIM + j*9: 2*GENE_INPUT_DIM + TEXT_INPUT_DIM + (j + 1)*9] = encoded_y[int(test_dic[i][j])]

print(train_set.shape)
print(test_set.shape)
train_set[0, 25:50]




#the xgboost classfication model
#first deal with the input label, transfrom it from 1-9 to 0-8(required by the xgboost)

for i in range(len(train_y)):
    train_y[i] -=1  
y_predict = xgbclassifier(train_set, train_y, test_set, 1, 8, 1000)
savesubmisstion(y_predict, test_x, filename = "submission_allwin2loadenhance.csv")
