import numpy as np
import pandas as pd
from textblob import TextBlob
import nltk
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from wordcloud import WordCloud
from collections import Counter
import collections
import seaborn as sns
import re, string
import sys
import time
import matplotlib.pyplot as plt

#import csv with reviews (star, text and date)
yelp = pd.read_csv('pocodeets.csv') 


yelp.info()

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 202 entries, 0 to 201
Data columns (total 4 columns):
stars         202 non-null int64
date          202 non-null object
text          202 non-null object
word_count    202 non-null int64
dtypes: int64(2), object(2)
memory usage: 6.4+ KB


yelp['review length'] = yelp['text'].apply(len)
sns.set_style('white')
%matplotlib inline
g = sns.FacetGrid(yelp,col='stars')
g.map(plt.hist,'review length')

#divide negative and postive sentiment by star ratings
yelpbadreviews = yelp[(yelp.stars <=  2)]
yelpgoodreviews = yelp[(yelp.stars >= 4)]

badreviews = yelpbadreviews.text
goodreviews = yelpgoodreviews.text

stopwords = nltk.corpus.stopwords.words('english')
new_words=['really']
for i in new_words:
    stopwords.append(i)
print(stopwords)


def tokenize(s):
    word_list = re.findall(r'\w+', s.lower())
    filtered_words = [word for word in word_list if word not in stopwords]
    return filtered_words
def count_ngrams(lines, min_length=2, max_length=4):
    lengths = range(min_length, max_length + 1)
    ngrams = {length: collections.Counter() for length in lengths}
    queue = collections.deque(maxlen=max_length)
# Helper function to add n-grams at start of current queue to dict
    def add_queue():
        current = tuple(queue)
        for length in lengths:
            if len(current) >= length:
                ngrams[length][current[:length]] += 1
# Loop through all lines and words and add n-grams to dict
    for line in lines:
        for word in tokenize(line):
            queue.append(word)
            if len(queue) >= max_length:
                add_queue()
# Make sure we get the n-grams at the tail end of the queue
    while len(queue) > min_length:
        queue.popleft()
        add_queue()
        return ngrams
def print_most_frequent(ngrams, num=10):
    for n in sorted(ngrams):
        print('----- {} most common {}-word phrase -----'.format(num, n))
        for gram, count in ngrams[n].most_common(num):
            print('{0}: {1}'.format(' '.join(gram), count))
        print('')
def print_word_cloud(ngrams, num=5):
    words = []
    for n in sorted(ngrams):
        for gram, count in ngrams[n].most_common(num):
            s = ' '.join(gram)
            words.append(s)
            
    cloud = WordCloud(width=1440, height= 1080,max_words= 200).generate(' '.join(words))
    plt.figure(figsize=(20, 15))
    plt.imshow(cloud)
    plt.axis('off');
    plt.show()
    print('')
    
most_frequent_badreviews = count_ngrams(badreviews,max_length=4)
print_word_cloud(most_frequent_badreviews, 10)
print_most_frequent(most_frequent_badreviews, num= 10)


most_frequent_goodreviews = count_ngrams(goodreviews,max_length=4)
print_word_cloud(most_frequent_goodreviews, 10)
print_most_frequent(most_frequent_goodreviews, num= 10)
