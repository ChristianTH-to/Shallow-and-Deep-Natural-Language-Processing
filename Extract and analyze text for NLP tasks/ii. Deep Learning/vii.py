# Deciding if the classification problem should be modeled as binary or multi-class
# with Evolutionary Data Measures

# Stdlib
import os 

# Third party
import pandas as pd
from edm import report

# Import data
dataset = pd.read_csv('bt_train_data_evaluation_set.csv').fillna('')
df = dataset.sample(frac=0.30)
messages = df['Message'].fillna('_##_').values
labels = df['Name'].values

# Returns a set of metrics and statistics that says 
# how likely it is that any algorithm we train can learn 
# from the dataset
print(report.get_difficulty_report(messages, labels))
