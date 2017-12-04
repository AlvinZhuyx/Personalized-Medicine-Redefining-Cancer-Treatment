# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:39:10 2017

@author: zhuya
"""
#import data_load as dl

from nltk.corpus import stopwords #nlp库
from gensim.models.doc2vec import LabeledSentence
#Gensim is a Python library for *topic modelling*, *document indexing* and *similarity retrieval* with large corpora.
from gensim import utils as gutils
from util import *

'''
## text data preprocessing ##
3 utilites
1 driver function: data_preprocessing()

Clean up the data, and convert it to a list of words

'''
def constructLabeledSentences(data):
    sentences=[]
    for index, row in data.iteritems():
        sentences.append(LabeledSentence(gutils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
    return sentences

def textClean(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)#[^]表示match所有不在集合里的元素
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]    
    text = " ".join(text)
    return(text)
    
def cleanup(text):
    text = textClean(text)
    #maketrans(x,y,z)表示建立映射表，使得x被替换为y并且删除z，translate接受映射表为参数并且完成映射
    text= text.translate(str.maketrans("","", string.punctuation))#去除标点符号
    return text

'''
@param all_data, the raw data loaded from loadData() function, containing 4 fields: [ID, Gene, Variation, Text]
@return 
'''
def data_preprocess(all_data):
    allText = all_data['Text'].apply(cleanup) # a list of str
    sentences = constructLabeledSentences(allText)
    return sentences

if __name__ == '__main__':
    sentences = data_preprocess(all_data)