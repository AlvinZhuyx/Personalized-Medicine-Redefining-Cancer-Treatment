#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import util
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold


'''
Create some baseline result using document vector representation
Try different length of vector.
The vector representation consists of 3 parts:
	Gene, 
	Variation, 
	Text
First two are sparse onehot-encoding while the third is dense representation comptued from text model

'''



def computeSim(v1, v2, DIM, weight=[1,1,1]):
	'''
	Compute the similarity score between two vectors, from 0 to 1, weighted on three parts

	@param:
		v1,
		v2, two vectors, [(GENE_INPUT_DIM + VAR_INPUT_DIM + TEXT_INPUT_DIM), 1]
		DIM, the DIM of each vector representation, [GENE_INPUT_DIM, VAR_INPUT_DIM, TEXT_INPUT_DIM]
		weight, [3, 1] linear weight to be applied to similarity score
	'''
	weight = weight / (np.sum(weight))
	GENE_INPUT_DIM = DIM[0]
	VAR_INPUT_DIM = DIM[1]
	TEXT_INPUT_DIM = DIM[2]
	s_gene = getCos(v1[0 : GENE_INPUT_DIM], v2[0 : GENE_INPUT_DIM])
	s_var = getCos(v1[GENE_INPUT_DIM : (GENE_INPUT_DIM+VAR_INPUT_DIM)], v2[GENE_INPUT_DIM : (GENE_INPUT_DIM + VAR_INPUT_DIM)])

	t1 = v1[(GENE_INPUT_DIM+VAR_INPUT_DIM):]
	t2 = v2[(GENE_INPUT_DIM+VAR_INPUT_DIM):]
	s_text = getCos(t1, t2)
	score = [s_gene, s_var, s_text]
	s = np.dot(score, weight)
	return s

def getCos(v1, v2):
	s = np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2))
	return s


def getLabels(encoding):
	'''
	Input the one-hot encoding for the dataset, return the labels corresponding to it.

	@param: encoding, one-hot encoding, supposed to be in shape [N, 9] (we expect 9 classes for N records)
	@return: labels, ranging 0 to 8
	'''
	return [ np.where(r==1)[0][0] for r in encoding ]



def baseline(X, y, skf = StratifiedKFold(n_splits=10, random_state = 66, shuffle = True)):
	'''
	Giving X and y vector representation and index splits, conduct N-fold validation
	Results evaluated by the metrics: ACC, NMI, Log loss, Confusion_matrix.

	@param: 
		X, feature vector of records, [N, FEATURE_LENGTH]
		y, label of records, [N, ]
		skf, default set to be StratifiedKFold(n_splits=10, random_state = 66, shuffle = True)

	@return: confusion_matrix averaged over N folds

	@output:
		Mean value and standard deviation of each metric,
		confusion matrix,
		Confusion matrix plot
	'''
	ret = []
	mat = 0
	LOSS = []
	ACC = []
	NMI = []
	for train_index, test_index in skf.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		neigh = KNeighborsClassifier(n_neighbors=3)
		neigh.fit(X_train, y_train)
		y_pred = neigh.predict(X_test)
		y_pred_proba = neigh.predict_proba(X_test)
		acc = accuracy_score(y_test, y_pred)
		nmi = normalized_mutual_info_score(y_pred, y_test)
		loss = log_loss(y_test, y_pred_proba)
		ACC.append(acc)
		LOSS.append(loss)
		NMI.append(nmi)

		cnf_matrix = confusion_matrix(y_test, y_pred)
		mat = mat + cnf_matrix
	print("Accuracy: %.4f ± %.4f" % (np.mean(ACC), np.std(ACC)))
	print("NMI: %.4f ± %.4f" % (np.mean(NMI), np.std(NMI)))
	print("Log_loss: %.4f ± %.4f" % (np.mean(LOSS), np.std(LOSS)))
	mat = mat / skf.get_n_splits()
	mat = np.array(mat)
	mat = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
	util.plot_confusion_matrix(mat, classes='', normalize=True,
                      title='Confusion matrix')
	return mat


def getBaseline(X, y, clf, skf = StratifiedKFold(n_splits=10, random_state = 66, shuffle = True)):
	ret = []
	mat = 0
	LOSS = []
	ACC = []
	NMI = []
	for train_index, test_index in skf.split(X, y):
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		clf.fit(X_train, y_train)
		y_pred = clf.predict(X_test)
		y_pred_proba = clf.predict_proba(X_test)
		acc = accuracy_score(y_test, y_pred)
		nmi = normalized_mutual_info_score(y_pred, y_test)
		loss = log_loss(y_test, y_pred_proba)
		ACC.append(acc)
		LOSS.append(loss)
		NMI.append(nmi)

		cnf_matrix = confusion_matrix(y_test, y_pred)
		mat = mat + cnf_matrix
	print("Accuracy: %.4f ± %.4f" % (np.mean(ACC), np.std(ACC)))
	print("NMI: %.4f ± %.4f" % (np.mean(NMI), np.std(NMI)))
	print("Log_loss: %.4f ± %.4f" % (np.mean(LOSS), np.std(LOSS)))
	mat = mat / skf.get_n_splits()
	mat = np.array(mat)
	mat = mat.astype('float') / mat.sum(axis=1)[:, np.newaxis]
	util.plot_confusion_matrix(mat, classes='', normalize=True,
                      title='Confusion matrix')
	return mat


