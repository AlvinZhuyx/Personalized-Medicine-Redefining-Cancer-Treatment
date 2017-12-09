# Personalized-Medicine-Redefining-Cancer-Treatment
UCLA-CS249-Project  
Team: SQLZW  
[Kaggle Contest](https://www.kaggle.com/c/msk-redefining-cancer-treatment)  

## Dependencies:
Python 3.5+ with Anaconda  
[Tensorflow](https://www.tensorflow.org), [keras](https://keras.io), [xgboost](http://xgboost.readthedocs.io/en/latest/), [gensim](https://radimrehurek.com/gensim/models/word2vec.html), [nltk](http://www.nltk.org), sklearn, pandas

## Codes:
All codes and demos in [src](./src/)

### * [demo_NN.ipynb](./src/demo_NN.ipynb)  
Complete system pipeline, demo using Neural Network  
### * [demo_xgboost.ipynb](./src/demo_xgboost.ipynb)  
Complete syste pipeline, demo training xgboost  
### * [demo_feature_baseline.ipynb](./src/demo_feature_baseline.ipynb)  
Demo using KNN to create baseline results for text model  
### [util.py](./src/util.py)  
This file contains most utility function used in this project.  
### [baseline_classification.py](./src/baseline_classification.py)  
Functions concerning baseline classifier using KNN.  
### [word_embedding_load.py](./src/word_embedding_load.py)  
Fucntions on training word embedding model using PubMed-based vectors introduced by Chiu et al. (2016).  
### [feature_selection_evaluation.py](./src/feature_selection_evaluation.py)  
About feature and model evaluation  
### [nn_classification.py](./src/nn_classification.py)  
NN model initialization and classification  
### [enhance_experiment.py](./src/enhance_experiment.py) & [enhanced.py](./src/enhanced.py)  
More attempt on enhancement, discussed in section 8  

## Data:
[Kaggle](https://www.kaggle.com/c/msk-redefining-cancer-treatment/data) Official dataset from Kaggle  
[bio_nlp_vec](https://github.com/cambridgeltl/BioNLP-2016) Word vectors from BioNLP-2016  
[Pre-trained Model](https://drive.google.com/drive/folders/1703i996nsfiDldvK8_aTT1G2nX4i1Qnu?usp=sharing) Here is the url for pre-trained word embeddings

## Report:
[Final Report](./CS249_final_report_SQLZW.pdf)  
[5 min Presentation](./term_project_presentation.pptx)

## Contributors:
Team SQLZW  
Yaxuan Zhu, Qi(Quincy) Qu, Don Lee, Yiming Wang, Jiyuan Shen
