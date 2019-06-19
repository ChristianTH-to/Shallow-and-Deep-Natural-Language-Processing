# Exploring the data with word2vec and GloVe


# Dependencies

import warnings
warnings.filterwarnings("ignore")

# Data

import pandas as pd
dataset = pd.read_csv('bt_10.csv').fillna('')

extra_stopwords = []

# Dependencies

import re
import nltk
import gensim
from nltk.corpus import stopwords
from stop_words import get_stop_words
from gensim.parsing.preprocessing import STOPWORDS

# Removes stopwords, lowercases and splits words into tokens

corpus = []
for i in range(0, 8461):
    clean_text = re.sub('[^a-zA-Z]', ' ', str(dataset['Message'][i]))
    clean_text = clean_text.lower()
    clean_text = clean_text.split()
    clean_text = [word for word in clean_text if not word in set(stopwords.words('english'))]
    clean_text = [word for word in clean_text if not word in set(get_stop_words('english'))]
    clean_text = [word for word in clean_text if word not in STOPWORDS]
    #clean_text = [word for word in clean_text if word not in extra_stopwords]
    corpus.append(clean_text)  


# Word2vec model

from gensim.models import Word2Vec
bt2Vec = Word2Vec(sentences = corpus, size = 100, window = 5, min_count = 4, workers = 8,
                  sg = 0, iter = 30, alpha = 0.020)
bt2Vec = bt2Vec.wv

# Prints most similar word to the query

word_vectors = bt2Vec.wv
bt2Vec.wv.most_similar('cool')
result = word_vectors.similar_by_word('gimme')
print("{}: {:.4f}".format(*result[0]))

# Detects n-grams in data

from gensim.models import Phrases
bigram_transformer = Phrases(corpus)
model = Word2Vec(bigram_transformer[corpus], min_count=4)


# Dependencies

from gensim import models
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile

# Loads GloVe vectors

glove_file = datapath('glove.840B.300d.txt')
# word2vec_glove_file = 'glove.840B.300d.word2vec.txt'#get_tmpfile('word2vec.txt')


# Converts GloVe vectors to Word2vec format

glove2word2vec(glove_file,word2vec_glove_file)
g2w2v = 'glove.840B.300d.word2vec.txt'
model = KeyedVectors.load_word2vec_format(g2w2v, binary=False)
model.most_similar('friendship')


# GloVe vector visualization of co-occurrence probabilities 
# Using a principal component analysis scatterplot 

# Reference: https://nlp.stanford.edu/pubs/glove.pdf

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
def display_pca_scatterplot(model, words=None, sample=0):
    if words == None:
        if sample > 0:
            words = np.random.choice(list(model.vocab.keys()), sample)
        else:
            words = [word for word in model.vocab]
            
    word_vectors = np.array([model[w] for w in words])
    
    twodim = PCA().fit_transform(word_vectors)[:,:2] # PCA reduces dimensionality of embedding matrix
    
    plt.figure(figsize=(10,8))
    #figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.scatter(twodim[:,0], twodim[:,1], edgecolors='k', c='r')
    for word, (x,y) in zip(words, twodim):
        plt.text(x+0.05,y+0.05,word)

display_pca_scatterplot(model, ['ice','steam','solid','gas','water','fashion'])

display_pca_scatterplot(model, ['territory','territories','pricing','price','analysis','analyses',
                               'crisis','crises','fish','fishes','ox','oxen','leaf','leaves',
                               'help','helping','expect','expected','fellowship','ownership',
                               'kinship','internship','will','could','possible','possibility',
                               'endless','ageless','lossless'])