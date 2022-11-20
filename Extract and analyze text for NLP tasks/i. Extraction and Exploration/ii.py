# Zipf's Law

# Stdlib
import re

# Third party
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline


# Import data
df = pd.read_csv('bt_train_data.csv').fillna(' ')
df = df[['Message']]

# Lower cases all words in df
df['clean_text'] = df.Message.apply(lambda x: re.sub('[^A-Za-z\']', ' ', x.lower()))

# Counts all words from data and scales them logarithmically
word_list = ' '.join(df.clean_text.values).split(' ')
words = pd.DataFrame(word_list, columns=['word'])
word_counts = words.word.value_counts().reset_index()
word_counts.columns = ['word', 'log']
word_counts['bt_2000_word_rank'] = word_counts.log.rank(ascending=False)

# Plot Zipf's Law
f, ax = plt.subplots(figsize=(12, 12))
ax.set(xscale="log", yscale="log")
sns.regplot("log", "bt_2000_word_rank", word_counts, ax=ax, scatter_kws={"s": 100})
