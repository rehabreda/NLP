# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 23:12:32 2019

@author: rehab
"""

import bs4 as bs
import urllib.request
import re
import nltk
import heapq


# getting data 
source=urllib.request.urlopen('https://en.wikipedia.org/wiki/Global_warming').read()

soup=bs.BeautifulSoup(source,'lxml')

text=" "
for paragraph in soup.find_all('p'):
    text+=paragraph.text
    
    
# preprocessing text 
text=re.sub(r"\[[0-9]*\]"," ",text)   
text=re.sub(r"\s+"," ",text)
clean_text=text.lower()
clean_text=re.sub(r"\W"," ",clean_text)
clean_text=re.sub(r"\d"," ",clean_text)
clean_text=re.sub(r"\s+"," ",clean_text)

sentences=nltk.sent_tokenize(text)
stop_words=nltk.corpus.stopwords.words('english')


# building histogram 
word2count={}
for word in nltk.word_tokenize(clean_text):
    if word not in stop_words:
        if word not in word2count.keys():
            word2count[word]=1
        else:
             word2count[word]+=1
             
# building weighted histogram 
for key in word2count.keys():
    word2count[key]= word2count[key]/max(word2count.values())
    
    
sent2score={}

for sent in sentences:
    for word in nltk.word_tokenize(sent.lower()):
        if word in word2count.keys():
            if len(sent.split(' '))<30:
                if sent not in sent2score.keys():
                    sent2score[sent]=word2count[word]
                else:
                    sent2score[sent]+=word2count[word]
    
    
             
            
            
            
best_sentences=heapq.nlargest(5,sent2score,key=sent2score.get)  

print('---------------------------------------------------------')
for sent in best_sentences:
    print(sent)
          