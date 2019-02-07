# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 21:20:45 2019

@author: rehab
"""

# import libraries
import pickle
import re
import tweepy
from tweepy import OAuthHandler
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# initialize  keys
consumer_key = 'yoIwFkjZGYDa49aO16XqSNqcN'
consumer_secret = 'gl4LQOItV7Z1aFwNrlvaiKJ3t8o8h99blMIAmnmdHxYjzjRAxO' 
access_token = '624310916-E7fDF2IE8P6bfY1oVFglASf6F8RnxMd3vgSXFqnZ'
access_secret ='ID9JcoXHsDcKtvNcnmBGcCQhUlO0wmwAxBJ6LCesiUAas'


auth=OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_token,access_secret)
args=['facebook']
api=tweepy.API(auth,timeout=50)


list_tweets=[]
query=args[0]

if len(args)==1:
    for status in tweepy.Cursor(api.search,q=query+"-filter:retweets",lang='en',result_type='recent').items(100):
        list_tweets.append(status.text)
        
        
## load  classifier and tfidf 
with open('classifier.pickle','rb') as f:
    classifier=pickle.load(f)
        
        
with open('vectorizer.pickle','rb') as f:
    vectorizer=pickle.load(f)        
    

#print(classifier.predict(vectorizer.transform(['you are  good person ,have a nice life'])))    

# processing tweets
cleaned_tweets=[]
for tweet in list_tweets:
    tweet = re.sub(r"^https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*\s", " ", tweet)
    tweet = re.sub(r"\s+https://t.co/[a-zA-Z0-9]*$", " ", tweet)
    tweet = tweet.lower()
    tweet = re.sub(r"that's","that is",tweet)
    tweet = re.sub(r"there's","there is",tweet)
    tweet = re.sub(r"what's","what is",tweet)
    tweet = re.sub(r"where's","where is",tweet)
    tweet = re.sub(r"it's","it is",tweet)
    tweet = re.sub(r"who's","who is",tweet)
    tweet = re.sub(r"i'm","i am",tweet)
    tweet = re.sub(r"she's","she is",tweet)
    tweet = re.sub(r"he's","he is",tweet)
    tweet = re.sub(r"they're","they are",tweet)
    tweet = re.sub(r"who're","who are",tweet)
    tweet = re.sub(r"ain't","am not",tweet)
    tweet = re.sub(r"wouldn't","would not",tweet)
    tweet = re.sub(r"shouldn't","should not",tweet)
    tweet = re.sub(r"can't","can not",tweet)
    tweet = re.sub(r"couldn't","could not",tweet)
    tweet = re.sub(r"won't","will not",tweet)
    tweet=re.sub(r"\W"," ",tweet)
    tweet=re.sub(r"\d"," ",tweet)
    tweet=re.sub(r"^[a-z]\s+"," ",tweet)
    tweet=re.sub(r"\s+[a-z]$"," ",tweet)
    tweet=re.sub(r"\s+[a-z]\s+"," ",tweet)
    tweet=re.sub(r"\s+"," ",tweet)
    cleaned_tweets.append(tweet)
    
    
vectorized=vectorizer.transform(cleaned_tweets).toarray() 

predicted=classifier.predict(vectorized)   

c=Counter(predicted)
pos=c[1]
neg=c[0]


objects=['positive','negative']
y_pos=np.arange(len(objects))
plt.bar(y_pos,[pos,neg],alpha=.5)
plt.xticks(y_pos,objects)
plt.title('number of positive and negative tweets')
plt.show()    