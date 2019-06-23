# Shallow techniques
# TFIDF, Count Vectorization, Naive Bayes, XGBoost
# Support Vector Machines, Logistic Regression


# Dependencies

import warnings
warnings.filterwarnings("ignore")

# Data

import pandas as pd
dataset = pd.read_csv('bt_data_train_set_1_5.csv').fillna('0')

# Transforms target variable into 0s and 1s for classification

from sklearn import preprocessing
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(dataset.Name.values)

# Returns every row as a string inside of a list 

data_str = ''
for i in dataset.itertuples():
    data_str = data_str + str(i.Message)
    
# Tokenizes text

import nltk
from nltk import word_tokenize
tokenized_text = word_tokenize(data_str)

# Appends list as a function to retrieve 
# Nltk part of speech tags

from nltk.tag import pos_tag_sents
list_of_tagged_words = nltk.pos_tag(tokenized_text) 

''''Based on hash-tables, which are continuous vectors 
    similar to python dictionaries, set_pos transforms 
    list_of_tagged_words into a highly optimized,
    iterable method that will make sure pos_tags is 
    contained within the object its called, which is 
    important because we only want the features in 
    pos_tags included in the final version of  
    list_of_tagged_words before we split the train 
    and test sets.'''

pos_set = (set(list_of_tagged_words))

'''Specifies the parts of speech
   we want to capture and groups 
   them together.''' 
pos_tags = ['PRP','PRP$', 'WP', 
            'WP$','JJ','JJR','VB', 
            'VBD','VBG', 'VBN','VBP', 
            'VBZ','JJS','EX','IN','CD',
            'CC','NN','NNS','NNP','NNPS']

# Removes the 1st index of set object

list_of_words = set(map(lambda tuple_2: tuple_2[0], filter(lambda tuple_2: tuple_2[1] in pos_tags, pos_set)))

# Transforms bt_1 & bt_5 Message vectors 
# Based on functions from list_of_words

dataset['pos_features'] = dataset['Message'].apply(lambda x: str([w for w in str(x).split() if w in list_of_words]))

# Split data into xtrain/ytrain xval/yval sets

from sklearn.model_selection import train_test_split
xtrain, xval, ytrain, yval = train_test_split(dataset.Message.values,y, 
                                                  stratify=y, 
                                                  random_state=42, 
                                                  test_size=0.1, shuffle=True)

# Import glove embeddings

from tqdm import tqdm
import numpy as np

glove_vectors = {}
e = open('glove.840B.300d.txt') # Need the full representation which includes stopwords
for p in tqdm(e):
    real_num = p.split(' ')
    word = real_num[0]
    coefs = np.asarray(real_num[1:], dtype='float32')
    glove_vectors[word] = coefs
e.close()
print('Found %s word vectors.' % len(glove_vectors))

'''Multinomial/bernoulli naive bayes, logistic regression 
   xgboost and a support vector classifier
   all with countvec and tfidf weighted features.'''

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.naive_bayes import MultinomialNB
multi_nb = Pipeline([("count_vectorizer",
                      CountVectorizer(analyzer=lambda x: x,token_pattern=r'\w{1,}',ngram_range=(1,3),
                                      stop_words='english')),("multinomial nb",MultinomialNB())])

multi_nb_tfidf = Pipeline([("tfidf_vectorizer",
                            TfidfVectorizer(analyzer=lambda x: x,min_df=3,max_features=None,
                                            strip_accents='unicode',token_pattern=r'\w{1,}',
                                            ngram_range=(1,3),use_idf=1,smooth_idf=1,sublinear_tf=1,
                                            stop_words='english')),("multinomial nb",MultinomialNB())])

from sklearn.naive_bayes import BernoulliNB
bern_nb = Pipeline([("count_vectorizer",
                     CountVectorizer(analyzer=lambda x: x,token_pattern=r'\w{1,}',ngram_range=(1,3),
                                     stop_words='english')),("bernoulli nb",BernoulliNB())])

bern_nb_tfidf = Pipeline([("tfidf_vectorizer",
                           TfidfVectorizer(analyzer=lambda x: x,min_df=3,max_features=None,
                                           strip_accents='unicode',token_pattern=r'\w{1,}',
                                           ngram_range=(1,3), use_idf=1,smooth_idf=1,sublinear_tf=1,
                                           stop_words='english')),("bernoulli nb",BernoulliNB())])

from sklearn.linear_model import LogisticRegression
log_reg = Pipeline([("count_vectorizer",
                     CountVectorizer(analyzer=lambda x: x,token_pattern=r'\w{1,}',
                                     ngram_range=(1,3),stop_words='english')),
                    ("logistic regression", LogisticRegression(C=1.0))])

log_reg_tfidf = Pipeline([("tfidf_vectorizer",
                           TfidfVectorizer(analyzer=lambda x: x,min_df=3,max_features=None,
                                           strip_accents='unicode',token_pattern=r'\w{1,}',
                                           ngram_range=(1,3),use_idf=1,smooth_idf=1,sublinear_tf=1,
                                           stop_words='english')),("logistic regression",LogisticRegression(C=1.0))])

import xgboost as xgb
from xgboost.sklearn import XGBClassifier 
xgb = Pipeline([("count_vectorizer", 
                 CountVectorizer(analyzer=lambda x: x,token_pattern=r'\w{1,}',
                                 ngram_range=(1,3),stop_words='english')),("xg boost",
                                                                           XGBClassifier(max_depth=7,
                                                                                         n_estimators=200,
                                                                                         colsample_bytree=0.8,
                                                                                         subsample=0.8,nthread=10,
                                                                                         learning_rate=0.1))])

xgb_tfidf = Pipeline([("tfidf_vectorizer",
                       TfidfVectorizer(analyzer=lambda x: x,min_df=3,  max_features=None,
                                       strip_accents='unicode',token_pattern=r'\w{1,}',
                                       ngram_range=(1,3),use_idf=1,smooth_idf=1,sublinear_tf=1,
                                       stop_words='english')),("xg boost",
                                                               XGBClassifier(max_depth=7,
                                                                             n_estimators=200,
                                                                             colsample_bytree=0.8,
                                                                             subsample=0.8,nthread=10,
                                                                             learning_rate=0.1))])

from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
svc = Pipeline([("count_vectorizer", 
                 CountVectorizer(analyzer=lambda x: x,token_pattern=r'\w{1,}',
                                 ngram_range=(1, 3), stop_words='english')),("linear svc",
                                                                             SVC(kernel="linear"))])

svc_tfidf = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x,min_df=3,
                                                           max_features=None,strip_accents='unicode',
                                                           token_pattern=r'\w{1,}',ngram_range=(1, 3),
                                                           use_idf=1,smooth_idf=1,sublinear_tf=1,
                                                           stop_words='english')), ("linear svc", 
                                                                                      SVC(kernel="linear"))])


'''Both classes vectorizes text by taking the mean 
   of all the  vectors corresponding to individual 
   words in a given vector mapping.''' 

class CountVectorizerEmbeddings(object):
    def __init__(self, glove):
        self.glove = glove
        if len(glove)>0:
            self.dim=len(glove[next(iter(glove_vectors))])
        else:
            self.dim=0
            
    def fit(self, X, y):
        return self 

    def transform(self, X):
        return np.array([
            np.mean([self.glove[w] for w in words if w in self.glove] 
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

from collections import defaultdict

class TfidfVectorizerEmbeddings(object):
    def __init__(self, glove):
        self.glove = glove
        self.word2weight = None
        if len(glove)>0:
            self.dim=len(glove[next(iter(glove_vectors))])
        else:
            self.dim=0
        
    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        '''If a word was never seen - it must be at least as infrequent
           as any of the known words - so the default idf is the max of 
           known idf's.'''
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf, 
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()]) 
    
        return self
    
    def transform(self, X):
        return np.array([
                np.mean([self.glove[w] * self.word2weight[w]
                         for w in words if w in self.glove] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])


'''Glove vectors passing through a stack of random decision trees
   that will be trained using tf-idf weighted + glove weighted vectors'''

from sklearn.ensemble import ExtraTreesClassifier
stacked_r_tree_glove_vectors = Pipeline([(
    "glove vectorizer", CountVectorizerEmbeddings(glove_vectors)),
    ("stacked trees", ExtraTreesClassifier(n_estimators=1000))])

stacked_r_tree_glove_vectors_tfidf = Pipeline([(
    "glove vectorizer", TfidfVectorizerEmbeddings(glove_vectors)),
    ("stacked trees", ExtraTreesClassifier(n_estimators=1000))])


# Places algorithm variables in a neat tabulated format

from tabulate import tabulate
# all 6 models
all_models = [('multi_nb', multi_nb),
              ('multi_nb_tfidf', multi_nb_tfidf),
              ('bern_nb', bern_nb),
              ('bern_nb_tfidf', bern_nb_tfidf),
              ('log_reg',log_reg),
              ('log_reg_tfidf',log_reg_tfidf),
              ('xgb',xgb),
              ('xgb_tfidf',xgb_tfidf),
              ('svc', svc),
              ('svc_tfidf', svc_tfidf),
              ('glove_vectors', stacked_r_tree_glove_vectors),
              ('glove_vectors_tfidf', stacked_r_tree_glove_vectors_tfidf)]


# Takes average of each algorithms output via the weighted f1 scoring metric

from sklearn.model_selection import cross_val_score
disordered_scores = [(name,cross_val_score(model,xtrain,ytrain,
                                         scoring= 'f1_weighted',
                                         cv=2).mean()) for name,model in all_models]

# Sorts and prints the score of each algorithm

scores = sorted(disordered_scores, key=lambda x: -x[1])
print (tabulate(scores, floatfmt=".4f", headers=("model", 'score')))
