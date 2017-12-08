！！！！！ThiS version is little outdated ！！！！Please refer to the release branch for our latest version ！！！！！！！！
！！！！！ThiS version is little outdated ！！！！Please refer to the release branch for our latest version ！！！！！！！！
！！！！！ThiS version is little outdated ！！！！Please refer to the release branch for our latest version ！！！！！！！！
！！！！！ThiS version is little outdated ！！！！Please refer to the release branch for our latest version ！！！！！！！！
！！！！！ThiS version is little outdated ！！！！Please refer to the release branch for our latest version ！！！！！！！！
！！！！！ThiS version is little outdated ！！！！Please refer to the release branch for our latest version ！！！！！！！！

This code is for Kaggle competition: Personalized Medicine: Redefining Cancer Treatment

The codes are python based and it's organized as follows:
The data_load.py load csv data
The data_preprocessing.py do data cleaning
The word_embedding_load.py usede to get doc embedding or load the existed word embedding trained on PubMed by Chiu.(2016) and trained doc vector based on these vectors.
The NN classification.py trained a classification model on word embedding;
The xgboost_classifier.py using the xgboost tree to do the classfication;(using word embedding as input)
The testaccuracy.py calculation the test accuracy by the classification model we just trained;
The load_test.py load the stage-2 data and test the accuracy;
The enhanced.py and enhanced_baseline.py is some method I used to try to enhance the result.

PS: Some of our code refer to the kaggle kernel "Doc2Vec with Keras(0.77)".

For the data:
docEmbeddings_win2_loadall is using all the labeled data(both training, 1st stage, 2nd stage test data) to train embedding and load the word embedding of PubMed with windows length = 2
docEmbeddings_win30_loadall is using all the labeled data(both training, 1st stage, 2nd stage test data) to train embedding and load the word embedding of PubMed with windows length = 30

###################################
#below added by Quincy @2017/11/28
###################################
Dependencies:
p7zip
wget
Re
gensim
Pandas
nltk
numpy
keras
tensorflow


here is the url for pre-trained word embeddings:
https://drive.google.com/drive/folders/1703i996nsfiDldvK8_aTT1G2nX4i1Qnu?usp=sharing
