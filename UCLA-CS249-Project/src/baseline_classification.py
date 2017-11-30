#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from util import *

'''
Create some baseline result using document vector representation
Try different length of vector.
The vector representation consists of 3 parts:
	Gene, 
	Variation, 
	Text
First two are sparse onehot-encoding while the third is dense representation comptued from text model

'''


'''
Compute the similarity score between two vectors, from 0 to 1, weighted on three parts

@param:
	v1,
	v2, two vectors, [(GENE_INPUT_DIM + VAR_INPUT_DIM + TEXT_INPUT_DIM), 1]
	DIM, the DIM of each vector representation, [GENE_INPUT_DIM, VAR_INPUT_DIM, TEXT_INPUT_DIM]
	weight, [3, 1] linear weight to be applied to similarity score
'''
def computeSim(v1, v2, DIM, weight=[1,1,1]):
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



