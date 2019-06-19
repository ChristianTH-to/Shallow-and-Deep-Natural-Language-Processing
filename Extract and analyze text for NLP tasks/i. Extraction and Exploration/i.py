# Data extraction


# Opens a path to the raw data

import urllib
from urllib.request import Request
bp_message_data = urllib.request.urlopen('file:///Users/christianth/Downloads/facebook-christianhardy1023%20(1)/messages/bluestraveler2000_6dec36cd06/message.html',timeout=1)

# Loads & parses data using pandas & beautifulsoup

import pandas as pd
bp = bp_message_data.read()
import bs4
from bs4 import BeautifulSoup
soup = bs4.BeautifulSoup(bp, 'html.parser') 

print(soup)

# Extract names of users 

import csv
d = csv.writer(open('bt_name_data_R.csv', 'w'))

d.writerow('Name')

data_name = soup.find_all('div', class_ = '_3-96 _2pio _2lek _2lel')

for data_name in data_name:
    names = data_name.contents
    d.writerow(names)

print(names)

# Extract users dates & times

d = csv.writer(open('bt_date_data_R.csv', 'w'))

d.writerow(['Date & Time'])

data_date_time = soup.find_all('div', class_ = '_3-94 _2lem')

for data_date_time in data_date_time:
    dates_times = data_date_time.contents
    d.writerow(dates_times)  

print(dates_times) 

# Extract users messages

m_data = soup.div(class_ = '_3-96 _2let')
data_m = soup.find_all(class_ = '_3-96 _2let')

d = csv.writer(open('bt_message_data.csv', 'w'))

d.writerow(['Message'])
data_message = soup.find_all('div', class_ = '_3-96 _2let')

for data_message in data_message:
    messages = data_message.get('_3-96 _2let')
    d.writerow([messages])
    print([messages])

print([messages])


# Initial data exploration & feature extraction

# Data

import pandas as pd
dataset = pd.read_csv('bt_fb_messenger_data.csv').fillna('')
# shape of data
print('Training Data Shape : ', dataset.shape)

# Number of occurances for each user

import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()

names = dataset['Name'].value_counts()
plt.figure(figsize=(36,16)) # 36,16 # 12,4
sns.barplot(names.index, names.values, alpha=0.8, color=color[1])
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('User Name', fontsize=12)
plt.show()

# Number of characters

dataset['char_count'] = dataset['Message'].str.len() 
dataset[['Message','char_count']].head()

# sample: bt_4

bt_4 = pd.read_csv('bt_4.csv').fillna('')
# shape of data
print('bt_4 data shape: ', bt_4.shape)

# Custom stopwords list

extra_stopwords = []


# Define a function to clean up text by removing personal pronouns, stopwords and punctuation

from nltk.corpus import stopwords
stopwords = stopwords.words('english')
import string
import spacy

# Loads pre-trained English spacy model (CNN)

nlp = spacy.load('en_core_web_sm')
punctuations = string.punctuation

# Function that visualizes the most common words across the entire dataset

def cleanup_text(docs, logging=False):
    texts = []
    counter = 1
    for doc in docs:
        if counter % 1000 == 0 and logging:
            print("Processed %d out of %d documents." % (counter, len(docs)))
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
        tokens = [word for word in tokens if word not in extra_stopwords]
        tokens = ' '.join(tokens)
        tokens = re.sub("(^|\W)\d+($|\W)", " ", tokens)
        tokens = re.sub('[^A-Za-z0-9]+', '', tokens)
        texts.append(tokens)
    return pd.Series(texts)
    

# Collect all text associated to bt_4

from collections import Counter
bt_4_text = [text for text in dataset[dataset['Name'] == 'bt_4']['Message']]

# Clean bt_4 text

bt_4_clean = cleanup_text(bt_4_text)
bt_4_clean = ' '.join(bt_4_clean).split()

# Remove words with 's

bt_4_clean = [word for word in bt_4_clean if word != '\'s']

# Count all unique words

bt_4_counts = Counter(bt_4_clean)
bt_4_common_words = [word[0] for word in bt_4_counts.most_common(30)]
bt_4_common_counts = [word[1] for word in bt_4_counts.most_common(30)]

# Plot 30 most commonly occuring words

plt.figure(figsize=(20, 12))
sns.barplot(x=bt_4_common_words, y=bt_4_common_counts)
plt.title('Most Common Words used by bt_4')
plt.show() 
