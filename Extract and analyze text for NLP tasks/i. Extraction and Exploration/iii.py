# Violin plots of feature distributions by user

# Stdlib
import warnings
warnings.filterwarnings('ignore')
import string
import re

# Third party
import pandas as pd
import numpy as np

import gensim
from gensim.parsing.preprocessing import STOPWORDS
import nltk
from nltk.tag import pos_tag
from nltk.stem.porter import PorterStemmer 
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
from stop_words import get_stop_words

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
color = sns.color_palette()


# Import data
dataset = pd.read_csv('bt_data_train_set_1_5.csv').fillna('nan')

# Number of words in the text 
dataset["num_words"] = dataset["Message"].apply(lambda x: len(str(x).split()))

# Number of unique words in the text 
dataset["num_unique_words"] = dataset["Message"].apply(lambda x: len(set(str(x).split())))

# Number of characters in the text 
dataset["num_chars"] = dataset["Message"].apply(lambda x: len(str(x)))

# Number of stopwords in the text 
dataset["num_stopwords"] = dataset["Message"].apply(lambda x: len([w for w in str(x).lower().split() if w in stopwords]))

# Number of punctuations in the text 
dataset["num_punctuations"] =dataset['Message'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))

# Number of upper case words in the text 
dataset["num_words_upper"] = dataset["Message"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

# Number of title case words in the text 
dataset["num_words_title"] = dataset["Message"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

# Average length of the words in the text 
dataset["mean_word_len"] = dataset["Message"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

# Number of adjectives
# import nltk
#pos_list = nltk.pos_tag

#dataset["fraction_adj"] = dataset["Message"].apply(lambda x: len([w for w in str(x).lower().split() in pos_list if w[1] in ('JJ','JJR','JJS')]))

#dataset['fraction_adj'] = dataset.apply(lambda row: fraction_adj(row),axis=1) 

# Plot feature counts
dataset['num_words'].loc[dataset['num_words']>80] = 80 #truncated for better visuals
plt.figure(figsize=(12,8))
sns.pointplot(x='Name', y='num_words', data=dataset)
plt.xlabel('User', fontsize=20)
plt.ylabel('Number of words in text', fontsize=15)
plt.title("Number of words by User", fontsize=20)
plt.show()

dataset['num_unique_words'].loc[dataset['num_unique_words']>80] = 80 
plt.figure(figsize=(12,8))
sns.pointplot(x='Name', y='num_unique_words', data=dataset)
plt.xlabel('User', fontsize=12)
plt.ylabel('Number of unique words in text', fontsize=12)
plt.title("Number of unique words by User", fontsize=15)
plt.show()

dataset['num_chars'].loc[dataset['num_chars']>80] = 80 
plt.figure(figsize=(12,8))
sns.pointplot(x='Name', y='num_chars', data=dataset)
plt.xlabel('User', fontsize=12)
plt.ylabel('Number of characters in text', fontsize=12)
plt.title("Number of characters by User", fontsize=15)
plt.show()

dataset['num_stopwords'].loc[dataset['num_stopwords']>80] = 80 
plt.figure(figsize=(12,8))
sns.pointplot(x='Name', y='num_stopwords', data=dataset)
plt.xlabel('User', fontsize=12)
plt.ylabel('Number of stop words in text', fontsize=12)
plt.title("Number of stop words by User", fontsize=15)
plt.show()

dataset['num_punctuations'].loc[dataset['num_punctuations']>80] = 80 
plt.figure(figsize=(12,8))
sns.pointplot(x='Name', y='num_punctuations', data=dataset)
plt.xlabel('User', fontsize=12)
plt.ylabel('Number of punctuations in text', fontsize=12)
plt.title("Number of punctuations by User", fontsize=15)
plt.show()

dataset['num_words_upper'].loc[dataset['num_words_upper']>80] = 80 
plt.figure(figsize=(12,8))
sns.pointplot(x='Name', y='num_words_upper', data=dataset)
plt.xlabel('User', fontsize=12)
plt.ylabel('Number of upper case words in text', fontsize=12)
plt.title("Number of upper case words by User", fontsize=15)
plt.show()

dataset['num_stopwords'].loc[dataset['num_stopwords']>80] = 80 
figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')
user1 = ['bt_5']
user2 = ['bt_1']
# Iterate through the both users
for user in user1:
        subset = dataset[dataset['Name'] == user]
sns.distplot(subset["num_stopwords"], hist = False, kde = True,color='green',
                 kde_kws = {'shade': True, 'linewidth': 3},label = 'bt_5')
plt.show()

dataset['mean_word_len'].loc[dataset['mean_word_len']>80] = 80 
plt.figure(figsize=(12,8))
sns.violinplot(x='Name', y='mean_word_len', data=dataset)
plt.xlabel('User', fontsize=12)
plt.ylabel('Average length of words in text', fontsize=12)
plt.title("Average length of words by User", fontsize=15)
plt.show()

# Part of speech violin plots
extra_stopwords = []
corpus = []
for i in range(0, 105185):
    clean_text = re.sub('[^a-zA-Z]', ' ', str(dataset['Message'][i]))
    clean_text = clean_text.lower()
    clean_text = clean_text.split()
    #clean_text = [word for word in clean_text if not word in set(stopwords.words('english'))]
    clean_text = [word for word in clean_text if not word in set(get_stop_words('english'))]
    clean_text = [word for word in clean_text if word not in STOPWORDS]
    clean_text = [word for word in clean_text if word not in extra_stopwords]
    corpus.append(clean_text)  

all_text_without_sw = ''
for i in dataset.itertuples():
    all_text_without_sw = all_text_without_sw + str(i.Message)

tokenized_text = word_tokenize(all_text_without_sw)
list_of_tagged_words = nltk.pos_tag(tokenized_text)
set_pos = (set(list_of_tagged_words))
nouns = ['PRP', 'PRP$', 'WP', 'WP$','VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
        'JJ', 'JJR', 'JJS','EX','IN','NN','NNS','NNP','NNPS','CD','CC']
list_of_words = set(map(lambda tuple_2: tuple_2[0], filter(lambda tuple_2: tuple_2[1] in nouns, set_pos)))
dataset['pos_pppn'] = dataset['Message'].apply(lambda x: len([w for w in str(x).lower().split() if w in list_of_words]))

dataset['pos_pppn'].loc[dataset['pos_pppn']>80] = 80 # truncated for better visuals
plt.figure(figsize=(12,8))
sns.violinplot(x='Name', y='pos_pppn', data=dataset,scale='count') # pairplot x='Name',
#sns.pairplot(dataset, hue ='Name', palette='Set2', diag_kind='kde')
#dataset['pos_nouns'].hist(figsize = (12, 12));
 
plt.xlabel('User', fontsize=12)
plt.ylabel('Number of Prepositional Phrases in Text', fontsize=15)
plt.title("Number of Prepositional Phrases by User", fontsize=20)
plt.show()

# Cleaning/refining the sample
dataset = pd.read_csv('bt_4.csv').fillna('')
extra_stopwords = []
bt_4_additional_stopwords = extra_stopwords

# List initialization 
corpus = []
for i in range(0, 38954):
    clean_text = re.sub('[^a-zA-Z]', ' ', str(dataset['Message'][i]))
    clean_text = clean_text.lower()
    clean_text = clean_text.split()
    # text stemming & stop word removal
    ps = PorterStemmer() 
    clean_text = [ps.stem(word) for word in clean_text if not word in set(stopwords.words('english'))]
    clean_text = [word for word in clean_text if not word in set(get_stop_words('english'))]
    clean_text = [word for word in clean_text if word not in STOPWORDS]
    clean_text = [word for word in clean_text if word not in bt_4_additional_stopwords]
    corpus.append(clean_text) 
