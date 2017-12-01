#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import tqdm
import string
import pandas as pd
import numpy as np
import util
from sklearn.decomposition import TruncatedSVD
#from data_preprocessing import *
import word_embedding_load as wel
import baseline_classification as bc
import matplotlib.pyplot as plt

GENE_INPUT_DIM = 25


def runTextModelEval(textModelName = [], PATH = '../model/doc2vec/'):
	'''
	Given a list of existed Text Model name, load them and get the baseline results one by one.
	Baseline evaluation please see baseline_classification.py

	@param:
		textModelName, a list of TextModel name
		PATH, the path to the model TextModel folder, default set to be ../model/doc2vec/
	@return: null

	'''

	[all_data, train_size, test_size, train_x, train_y, test_x] = util.loadData()
	sentences = util.data_preprocess(all_data)
	svd = TruncatedSVD(n_components=GENE_INPUT_DIM, random_state=12)
	for textModel in textModelName:

		try:
			model = wel.loadTextModel(PATH + textModel)
		except:
			print('Failed on ' + textModel)
			continue
		if model == None:
			print('Failed on ' + textModel)
			continue
		text_train_arrays, text_test_arrays = wel.getTextVec(model, train_size, test_size, 200)
		truncated_one_hot_gene = wel.getGeneVec(all_data, svd)
		truncated_one_hot_variation = wel.getVariationVec(all_data, svd)
		train_set = np.hstack((truncated_one_hot_gene[:train_size], truncated_one_hot_variation[:train_size], text_train_arrays))
		test_set = np.hstack((truncated_one_hot_gene[train_size:], truncated_one_hot_variation[train_size:], text_test_arrays))
		encoded_y = pd.get_dummies(train_y)
		encoded_y = np.array(encoded_y)

		X = np.array(train_set)
		y = np.array(bc.getLabels(encoded_y))
		print('Results for TextModel: ' + textModel)
		cm = bc.baseline(X, y)

#TODO!
def runFeatLenEval(textModel, featLen = []):
	'''
	Given textModel and a list of feature length, conduct truncated SVD to reduce the length of feature vector.
	Then running baseline evalution, please also see baseline_classification.py

	@param:


	'''
	
	return
		
