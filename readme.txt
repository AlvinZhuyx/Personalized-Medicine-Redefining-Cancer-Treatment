This code is for Kaggle competition: Personalized Medicine: Redefining Cancer Treatment
The codes are based on Mr. Aly Osama's open source code "Doc2Vec with Keras(0.77)" and we try to improve the results by make several changes to it.

The codes are python based and it's organized as follows:
The data_load.py load csv data
The data_preprocessing.py do data cleaning
The word_embedding.py do doc embedding by Doc2Vec;
The word_embedding_load.py is another way to get doc embedding: we load the existed word embedding trained on PubMed by Chiu.(2016) and trained doc vector based on these vectors.
The classification.py trained a classification model on word embedding;(now we use NN)
The xgboost_classifier.py using the xgboost tree to do the classfication;(using word embedding as input)
The xgb_dataprecessing.py is the code of https://www.kaggle.com/the1owl/redefining-treatment-0-57456, which use transform and fit_transform to do data preprocessing (we use it as the baseline of our xgboost method)
The testaccuracy.py calculation the test accuracy by the classification model we just trained;
The load_test.py load the stage-2 data and test the accuracy;
The enhanced.py and enhanced_baseline.py is some method I used to try to enhance the result.


For the data:
docEmbeddings_5_clean is the doc embedding get from doc2vec and trained for 5 epoch;
docEmbeddings_30_clean is the doc embedding get from doc2vec and trained for 30 epoch;
docEmbeddings_5_load is the doc embedding get from doc2vec with loaded word embedding and trained for 5 epoch(windows = 2);
docEmbeddings_30_load is the doc embedding get from doc2vec with loaded word embedding and trained for 30 epoch(windows = 2);
docEmbeddings_30_loadwin30 is the doc embedding get from doc2vec with loaded word embedding and trained for 30 epoch(windows = 30);
docEmbeddings_30_loads2 is the doc emdedding for the seconde tage data
docEmbeddings_30_loadall is using all the labeled data(both training, 1st stage, 2nd stage test data) to train embedding

The pre-trained word embedding: 
https://drive.google.com/drive/folders/1h0vK_ZibfgCFH_XCHqKA0aZs3jlj-37m?usp=sharing


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

