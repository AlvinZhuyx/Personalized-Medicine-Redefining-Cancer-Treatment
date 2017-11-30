# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 14:56:55 2017

@author: zhuya
"""
import library as lib
#import data_preprocessing as dp
from gensim.models import Doc2Vec
import pandas as pd
import numpy as np

Text_INPUT_DIM=300


text_model=None
filename='docEmbeddings_30_clean.d2v'
if lib.os.path.isfile(filename):
    text_model = Doc2Vec.load(filename)
else:
    #text_model = Doc2Vec(min_count=1, window=5, size=Text_INPUT_DIM, sample=1e-4, negative=5, workers=4, iter=5,seed=1)
    param = Doc2VecWrapper()
    text_model = Doc2VecWrapper(param)
    text_model.build_vocab(sentences)
    text_model.train(sentences, total_examples=text_model.corpus_count, epochs=text_model.iter)
    text_model.save(filename)
    
text_train_arrays = np.zeros((train_size, Text_INPUT_DIM))
text_test_arrays = np.zeros((test_size, Text_INPUT_DIM))

for i in range(train_size):
    text_train_arrays[i] = text_model.docvecs['Text_'+str(i)]

j = 0
for i in range(train_size, test_size + train_size):
    text_test_arrays[j] = text_model.docvecs['Text_'+str(i)]
    j += 1

from sklearn.decomposition import TruncatedSVD
Gene_INPUT_DIM=25

svd = TruncatedSVD(n_components=25, n_iter=Gene_INPUT_DIM, random_state=12)

one_hot_gene = pd.get_dummies(all_data['Gene'])
truncated_one_hot_gene = svd.fit_transform(one_hot_gene.values)

one_hot_variation = pd.get_dummies(all_data['Variation'])
truncated_one_hot_variation = svd.fit_transform(one_hot_variation.values)

train_set = np.hstack((truncated_one_hot_gene[:train_size], truncated_one_hot_variation[:train_size], text_train_arrays))
test_set = np.hstack((truncated_one_hot_gene[train_size:], truncated_one_hot_variation[train_size:], text_test_arrays))

encoded_y = pd.get_dummies(train_y)
encoded_y = np.array(encoded_y)


