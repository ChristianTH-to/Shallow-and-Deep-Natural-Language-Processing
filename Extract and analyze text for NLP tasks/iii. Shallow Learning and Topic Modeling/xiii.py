# Compare the distance between users probability distributions


# Dependencies

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# Comparing the distances between each user will highlight
# how similar or dissimilar each user is to each other

# Load data bt_1

import pandas as pd

bt_1_dataset = pd.read_csv('/floyd/input/bt_2000_data/bt_1.csv').fillna('0')

extra_stopwords = []

# bt_1 data preprocessing: transform the vector 9 times
# Apply regex to remove numerals

import re
bt_1_dataset['Message'] = bt_1_dataset.Message.map(lambda x: re.sub(r'\d+', '', x))

# Convert words to lowercase
bt_1_dataset['Message'] = bt_1_dataset.Message.map(lambda x: x.lower())

print(bt_1_dataset['Message'][0][:500])

# Tokenize the vector

from nltk.tokenize import RegexpTokenizer
bt_1_dataset['Message'] = bt_1_dataset.Message.map(lambda x: RegexpTokenizer(r'\w+').tokenize(x))

print(bt_1_dataset['Message'][0][:25])

# Lemmatize all words in vector

import nltk
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
bt_1_dataset['Message'] = bt_1_dataset.Message.map(lambda x: [lemmatizer.lemmatize(token) for token in x])

print(bt_1_dataset['Message'][0][:25])

# Remove stopwords from vector

from nltk.corpus import stopwords
nltk.download('stopwords')
from gensim.parsing.preprocessing import STOPWORDS
from stop_words import get_stop_words
stop_en = stopwords.words('english')

bt_1_dataset['Message'] = bt_1_dataset.Message.map(lambda x: [t for t in x if t not in stop_en])
bt_1_dataset['Message'] = bt_1_dataset.Message.map(lambda x: [t for t in x if not t in set(get_stop_words('english'))])
bt_1_dataset['Message'] = bt_1_dataset.Message.map(lambda x: [t for t in x if t not in STOPWORDS])
bt_1_dataset['Message'] = bt_1_dataset.Message.map(lambda x: [t for t in x if t not in extra_stopwords])

print(bt_1_dataset['Message'][0][:25])

# Remove words with less than 2 characters

bt_1_dataset['Message'] = bt_1_dataset.Message.map(lambda x: [t for t in x if len(t) > 1])

print(bt_1_dataset['Message'][0][:25])

# bt_1 n-gram preprocessing

import numpy as np
# Puts the preprocessed text into an array
bt_1_docs = np.array(bt_1_dataset['Message']) 


# Add bigrams and trigrams that appear 
# more than 10 times to docs 

from gensim.models import Phrases
bigram = Phrases(bt_1_docs, min_count=10)
trigram = Phrases(bigram[bt_1_docs])
quadgram = Phrases(trigram[bt_1_docs])
quingram = Phrases(quadgram[bt_1_docs])
sexgram = Phrases(quingram[bt_1_docs])

for idx in range(len(bt_1_docs)):
    for token in bigram[bt_1_docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document
            bt_1_docs[idx].append(token)
    for token in trigram[bt_1_docs[idx]]:
        if '_' in token:
            # Token is a trigram, add to document
            bt_1_docs[idx].append(token)
    for token in quadgram[bt_1_docs[idx]]:
        if '_' in token:
            # Token is a quadgram, add to document
            bt_1_docs[idx].append(token)    
    for token in quingram[bt_1_docs[idx]]:
        if '_' in token:
            # Token is a quingram, add to document
            bt_1_docs[idx].append(token)
    for token in sexgram[bt_1_docs[idx]]:
        if '_' in token:
            # Token is a sexgram, add to document
            bt_1_docs[idx].append(token)    
            

# Put bt_1 into a dictionary

from gensim import corpora, models
import numpy as np
np.random.seed(2017)

bt_1_texts = bt_1_dataset['Message'].values
print('Total Number of documents: %d' % len(bt_1_texts))

# Makes an index to word dictionary

bt_1_dictionary = corpora.Dictionary(bt_1_texts)
print('Number of unique tokens in initital documents:', len(bt_1_dictionary))

# Filter words that occur in less than 20% of documents

bt_1_dictionary.filter_extremes(no_below=8, no_above=0.2) # Originally 10 & 0.2  but anywhere between 5 & 20 works fine
print('Number of unique tokens after removing rare and common tokens:', len(bt_1_dictionary))

# Converts the dictionary into a bag-of-words

bt_1_corpus = [bt_1_dictionary.doc2bow(text) for text in bt_1_texts]


# bt_1 lda model
from gensim.models import ldamodel
# LDA model extracts topics from the corpus
bt_1_ldamodel = models.ldamodel.LdaModel(bt_1_corpus, id2word=bt_1_dictionary,alpha='auto',eta='auto',  
                                    num_topics=32, passes=100, iterations=400, minimum_probability=0)



# Load data bt_2
bt_2_dataset = pd.read_csv('/floyd/input/bt_2000_data/bt_2.csv').fillna('0')

# bt_2 data preprocessing: transform the vector 9 times
# Apply regex to remove numerals
import re
bt_2_dataset['Message'] = bt_2_dataset.Message.map(lambda x: re.sub(r'\d+', '', x))

# Convert words to lowercase
bt_2_dataset['Message'] = bt_2_dataset.Message.map(lambda x: x.lower())
print(bt_2_dataset['Message'][0][:500])

# Tokenize the vector
from nltk.tokenize import RegexpTokenizer
bt_2_dataset['Message'] = bt_2_dataset.Message.map(lambda x: RegexpTokenizer(r'\w+').tokenize(x))
print(bt_2_dataset['Message'][0][:25])

# Lemmatize all words in vector
import nltk
#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
bt_2_dataset['Message'] = bt_2_dataset.Message.map(lambda x: [lemmatizer.lemmatize(token) for token in x])
print(bt_2_dataset['Message'][0][:25])

# Remove stopwords from vector
from nltk.corpus import stopwords
#nltk.download('stopwords')
from gensim.parsing.preprocessing import STOPWORDS
from stop_words import get_stop_words
stop_en = stopwords.words('english')

bt_2_dataset['Message'] = bt_2_dataset.Message.map(lambda x: [t for t in x if t not in stop_en])
bt_2_dataset['Message'] = bt_2_dataset.Message.map(lambda x: [t for t in x if not t in set(get_stop_words('english'))])
bt_2_dataset['Message'] = bt_2_dataset.Message.map(lambda x: [t for t in x if t not in STOPWORDS])
bt_2_dataset['Message'] = bt_2_dataset.Message.map(lambda x: [t for t in x if t not in extra_stopwords])
print(bt_2_dataset['Message'][0][:25])

# Remove words with less than 2 characters
bt_2_dataset['Message'] = bt_2_dataset.Message.map(lambda x: [t for t in x if len(t) > 1])
print(bt_2_dataset['Message'][0][:25])


# bt_2 n-gram preprocessing
import numpy as np
# puts the preprocessed text into an array
bt_2_docs = np.array(bt_2_dataset['Message']) 

# Add bigrams and trigrams that appear 
# more than 10 times to docs 

from gensim.models import Phrases
bigram = Phrases(bt_2_docs, min_count=10)
trigram = Phrases(bigram[bt_2_docs])
quadgram = Phrases(trigram[bt_2_docs])
quingram = Phrases(quadgram[bt_2_docs])
sexgram = Phrases(quingram[bt_2_docs])

for idx in range(len(bt_2_docs)):
    for token in bigram[bt_2_docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document
            bt_2_docs[idx].append(token)
    for token in trigram[bt_2_docs[idx]]:
        if '_' in token:
            # Token is a trigram, add to document
            bt_2_docs[idx].append(token)
    for token in quadgram[bt_2_docs[idx]]:
        if '_' in token:
            # Token is a quadgram, add to document
            bt_2_docs[idx].append(token)    
    for token in quingram[bt_2_docs[idx]]:
        if '_' in token:
            # Token is a quingram, add to document
            bt_2_docs[idx].append(token)
    for token in sexgram[bt_1_docs[idx]]:
        if '_' in token:
            # Token is a sexgram, add to document
            bt_1_docs[idx].append(token)          
            

# Put bt_2 in a dictionary
from gensim import corpora, models
from gensim.corpora import Dictionary
import numpy as np
np.random.seed(2017)

bt_2_texts = bt_2_dataset['Message'].values
print('Total Number of documents: %d' % len(bt_2_texts))

# Makes an index to word dictionary
bt_2_dictionary = corpora.Dictionary(bt_2_texts)
print('Number of unique tokens in initital documents:', len(bt_2_dictionary))

# Filter words that occur in less than 20% of documents
bt_2_dictionary.filter_extremes(no_below=7, no_above=0.2) # originally 10 & 0.2  but anywhere between 5 & 20 works ok
print('Number of unique tokens after removing rare and common tokens:', len(bt_2_dictionary))

# Converts the dictionary into a bag-of-words
bt_2_corpus = [bt_2_dictionary.doc2bow(text) for text in bt_2_texts]


# bt_2 lda model
# LDA model extracts topics from the corpus
bt_2_ldamodel = models.ldamodel.LdaModel(bt_2_corpus, id2word=bt_2_dictionary,alpha='auto',eta='auto',  
                                    num_topics=32, passes=100, iterations=400, minimum_probability=0)


# bt_1 topics
bt_1_topics = ['black pipe','cleveland','rib','gay',
             'fucking','love','feel', 'free', 'party',
             'worth', 'hoke','happy','foot','tv','weed',
             'hard paint','good luck','olga','hair','gas',
             'sex','especially','pretty','hope','basically',
             'dream','hit','bit','ben krenke','weird','saying',
             'okay','doesnt','understand','fuck','job','hard',
             'night','weekend','fucked','sorry','school','cheap',
             'literally','crazy','mom','year_old','home','year',
             'old','bitch','song']
# bt_2 topics
bt_2_topics = ['black pipe','bike','hard','suck','dick','fart noise',
              'life','buy','ride','sorry','nah','working','easy',
              'worst','comment','win','pissed','interview','bad',
              'high','game','rib','drink','fast','dog','smoke weed',
              'kit','happy birthday','apparently','lol','cleveland','sweet',
              'hang','summer','get paper','good','jesus christ','idea',
              'gay','dumb','jesus','sound','god damn','house','health insurance',
              'stock','set nickname']


#  Transforms user topics to bag of words 
bt_1_bow = bt_1_ldamodel.id2word.doc2bow(bt_1_topics)
bt_1 = bt_1_ldamodel[bt_1_bow]

bt_2_bow = bt_2_ldamodel.id2word.doc2bow(bt_2_topics)
bt_2 = bt_2_ldamodel[bt_2_bow]


# Computes the similarity of bt_1 & bt_2
from gensim.matutils import hellinger
print('Similarity between bt_1 & bt_2:', hellinger(bt_1,bt_2))
