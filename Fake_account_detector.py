# -*- coding: utf-8 -*-
"""
Fake Account Detector
"""

import os 
os.chdir(r'D:\Dataset\Side_project_Fake_account_detector')

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt 
%matplotlib inline 
import seaborn as sns

### load the dataset and get some overview
#fk_df = pd.read_csv('fake_account.csv') the format of the dataset is not perfect
# I would liek to treat it like a txt file 

# for fake account 
fk = open('fake_account.csv', encoding='utf-8')
fk_list = []

text = fk.readlines()
for i in text:
    text_split = i.split('\t', 1)
    fk_list.append([text_split[0], text_split[1].strip('\n'), 1])

df = pd.DataFrame(fk_list, columns=['id', 'post', 'fake'])

# for legitimate account 
le = open('legitimate_account.csv', encoding='utf-8')
le_list = []

text = le.readlines()
for i in text:
    text_split = i.split('\t')
    le_list.append([text_split[0], text_split[5].strip('\n'), 0])
    
df_le = pd.DataFrame(le_list, columns=['id', 'post', 'fake'])

df = pd.concat([df, df_le], axis=0, ignore_index=True)

del df_le

print(df.shape)


### EDA and engineering for new features 

## post length 

df['post_len'] = df['post'].apply(lambda x: len(x))
df['post_len'].describe()

# visualization for the distribution of post length by account type 
fig, axes = plt.subplots(figsize=[10, 10])
for fake_type in np.unique(df['fake']):
    sns.distplot(df['post_len'][df['fake'] == fake_type], bins=50,
                 kde=True, label=str(fake_type), hist_kws={'alpha':0.5,
                    'edgecolor':'none'})
axes.legend()
axes.set_title('Post Length by Fake or Not')
axes.set_xlabel('Post Length')

## cut the post into words and count the frequent words for each fake type 
import time
start_time = time.time()
import jieba 
df['post_cut'] = df['post'].apply(lambda x: ';'.join(list(jieba.cut(x))))
print(time.time() - start_time)

# remove the stopwords and non-word symbol, and then count the words 
# using mapping function 

import os 
os.chdir(r'D:\Code\Python\Add_in')
stop_words = open('Stop_words_Simplified_Chinese_English.txt', 'r', encoding = 'utf-8') # read the file with chinese words 
stop_words = stop_words.readlines() 
stop_words = [x.strip('\n') for x in stop_words]  
import re              
              
def remove_stop_punc(text):
    text = re.sub(r'\W', ' ', text) # remove punctuations
    word_list = list(jieba.cut(text))
    word_list = ';'.join([word for word in word_list if not word in (stop_words+[' '])])
    
    return word_list

start_time = time.time()    
df['post_cut_filtered'] = df['post'].apply(lambda x: remove_stop_punc(x))
print(time.time() - start_time)

### output the dataframe for saving time 
#os.chdir(r'D:\Dataset\Side_project_Fake_account_detector')
#df.to_csv('Account.csv', encoding='utf-8', index_label=False)

# print the words with high frequencies in each fake type 

df['post_cut_filtered'] = df['post_cut_filtered'].astype(str)
df['post_cut_filtered'] = df['post_cut_filtered'].apply(lambda x: x.split(';'))

from collections import Counter
fake_counter = {}
for fake_type in np.unique(df['fake']):
    fake_counter[fake_type] = Counter([word for text in df['post_cut_filtered'][df['fake'] == fake_type]
                                              for word in text])
print(fake_counter[0].most_common(1000))  
print(fake_counter[1].most_common(1000))  
"""
# loading the dataset

import os 
os.chdir(r'D:\Dataset\Side_project_Fake_account_detector')
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt 
%matplotlib inline 
import seaborn as sns
df = pd.read_csv('Account.csv', encoding='utf-8')

"""