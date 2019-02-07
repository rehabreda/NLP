# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 17:55:54 2019

@author: rehab
"""

import nltk
import re
import urllib.request
import bs4 as bs
from gensim.models  import Word2Vec
from nltk.corpus import stopwords


## getting data
source=urllib.request.urlopen("https://en.wikipedia.org/wiki/Global_warming").read()

soup=bs.BeautifulSoup(source,'lxml')

text = ""
for paragraph in soup.find_all('p'):
    text += paragraph.text

# Preprocessing the data
text = re.sub(r'\[[0-9]*\]',' ',text)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d',' ',text)
text = re.sub(r'\s+',' ',text)

# Preparing the dataset
sentences = nltk.sent_tokenize(text)

sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

for i in range(len(sentences)):
    sentences[i] = [word for word in sentences[i] if word not in stopwords.words('english')]



model= Word2Vec(sentences,min_count=1)
word=model.wv.vocab

vector=model.wv['warming']

similar=model.wv.most_similar('global')
    