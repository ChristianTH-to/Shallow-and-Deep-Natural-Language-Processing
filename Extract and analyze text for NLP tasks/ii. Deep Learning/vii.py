# Deciding if the classification problem should be modeled as binary or multi-class
# with Evolutionary Data Measures


# Dependencies

import os 
from edm import report

# Data

import pandas as pd
dataset = pd.read_csv('bt_train_data_evaluation_set.csv').fillna('')
df = dataset.sample(frac=0.30)
messages = df['Message'].fillna('_##_').values
labels = df['Name'].values


# Returns a set of metrics and statistics that says 
# how likely it is that any algorithm we train can learn 
# from the dataset

print(report.get_difficulty_report(messages, labels))
