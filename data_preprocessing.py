# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:39:10 2017

@author: zhuya
"""
#import data_load as dl

from nltk.corpus import stopwords #nlp库
from gensim.models.doc2vec import LabeledSentence
#Gensim is a Python library for *topic modelling*, *document indexing* and *similarity retrieval* with large corpora.
from gensim import utils 

def constructLabeledSentences(data):
    sentences=[]
    for index, row in data.iteritems():
        sentences.append(LabeledSentence(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
    return sentences

#去掉特殊字符以及停用词
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

allText = all_data['Text'].apply(cleanup)
sentences = constructLabeledSentences(allText)
print("\n")
print(allText.head())