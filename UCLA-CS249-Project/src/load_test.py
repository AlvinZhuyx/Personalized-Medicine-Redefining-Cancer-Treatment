# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 23:23:39 2017

@author: zhuya
"""

#this file is used to load the data of 2nd stage

import os
import re #处理正则表达式
import tqdm #显示进度条 
import string
import pandas as pd
import numpy as np
import keras

test_variant = pd.read_csv("../data/stage2_test_variants.csv")
test_text = pd.read_csv("../data/stage2_test_text.csv", sep = "\|\|", engine = 'python', header=None, skiprows=1, names=["ID","Text"])
test_solution = pd.read_csv("../data/stage_2_private_solution.csv", sep = ",")
test_x = pd.merge(test_variant, test_text, how = 'inner', on = 'ID')
print(test_x.head())


#data preprocessing
from nltk.corpus import stopwords
from gensim.models.doc2vec import LabeledSentence
from gensim import utils 

def constructLabeledSentences(data):
    sentences=[]
    for index, row in data.iteritems():
        sentences.append(LabeledSentence(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
    return sentences

def textClean(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]    
    text = " ".join(text)
    return(text)
    
def cleanup(text):
    text = textClean(text)
    text= text.translate(str.maketrans("","", string.punctuation))
    return text

testsentences = test_x['Text'].apply(cleanup)
size = len(testsentences)
print(testsentences.head())

#word embedding
from gensim.models import Doc2Vec
from sklearn.decomposition import TruncatedSVD
Gene_INPUT_DIM=25
Text_INPUT_DIM = 200
filename='docEmbeddings_5_loadw30.d2v'
if os.path.isfile(filename):
    text_model = Doc2Vec.load(filename)
    
text_model.infer_vector(testsentences, alpha=0.1, min_alpha=0.0001, steps=50)

text_test_arrays = np.zeros((size, Text_INPUT_DIM))
for i in range(size):
    text_test_arrays[i] = text_model.docvecs['Text_'+str(i)]

svd = TruncatedSVD(n_components=25, n_iter=Gene_INPUT_DIM, random_state=12)
one_hot_gene = pd.get_dummies(test_x['Gene'])
truncated_one_hot_gene = svd.fit_transform(one_hot_gene.values)

one_hot_variation = pd.get_dummies(test_x['Variation'])
truncated_one_hot_variation = svd.fit_transform(one_hot_variation.values)
test_set = np.hstack((truncated_one_hot_gene, truncated_one_hot_variation, text_test_arrays))

#you should train a deep model first
predict_test =  model.predict_proba(test_set)

test_id = test_solution['ID'].values
test_result = np.array(test_solution.drop('ID', axis = 1))
actualsize = len(test_result)

pred = np.array(predict_test)
mysum = 0
for i in range(actualsize):
    truth = np.argmax(test_result[i])
    predict = np.argmax(pred[test_id[i] - 1])
    mysum += (truth == predict)
    
accuracy = 100 * mysum / actualsize

print("Test accuracy: %.2f %%" % accuracy)
submission = pd.DataFrame(predict_test)
submission.insert(0, 'ID', test_x['ID'])
submission.columns = ['ID','class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9']
submission.to_csv("submission_all.csv",index=False)
submission.head()


