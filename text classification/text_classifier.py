# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 00:40:45 2019

@author: rehab
"""

# import libraries
import numpy as np 
import re 
import nltk
import pickle
from nltk.corpus import stopwords
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer , TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# importing data 
reviews=load_files('txt_sentoken/')
X,y=reviews.data ,reviews.target

# storing datainto pickle file 
with open('X.pickle','wb') as f:
    pickle.dump(X,f)
    
with open('y.pickle','wb') as f:
    pickle.dump(y,f)
    
# loading data 
with open('X.pickle','rb') as f:
    X=pickle.load(f)
    
with open('y.pickle','rb') as f:
    y=pickle.load(f)
        
    

# preprocessing 
corpus=[]
for i in range(0,len(X)):
    review=re.sub(r'\W',' ',str(X[i]))
    review=review.lower()
    review=re.sub(r'\s+[a-z]\s+',' ',review)
    review=re.sub(r'^[a-z]\s+',' ',review)
    review=re.sub(r'\s+',' ',review)
    corpus.append(review)
    
# create bag of words model
vectorizer=CountVectorizer(max_features=2000,min_df=3,max_df=.6,stop_words=stopwords.words('english')) 
X=vectorizer.fit_transform(corpus).toarray()  


# convert from bag of words model to tfidf model
transformer=TfidfTransformer()
X=transformer.fit_transform(X).toarray()


# create tfidf model 
# create bag of words model
vectorizer=TfidfVectorizer(max_features=2000,min_df=3,max_df=.6,stop_words=stopwords.words('english')) 
X=vectorizer.fit_transform(corpus).toarray()  


# spliting data into train   and test
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.2)

# building logistic regression classifier
classifier=LogisticRegression()
classifier.fit(x_train,y_train)

# test model
predicted=classifier.predict(x_test)
cm=confusion_matrix(y_test,predicted)

# save model
with open('classifier.pickle','wb') as f:
    pickle.dump(classifier,f)
    
with open('vectorizer.pickle','wb') as f:
    pickle.dump(vectorizer,f)   
    
    
    
 # load classifier 
with open('classifier.pickle','rb') as f:
    clf=pickle.load(f)
    
with open('vectorizer.pickle','rb') as f:
    tfidf=pickle.load(f)
    
sample=['you are nice person , have good life']   
sample=tfidf.transform(sample).toarray()
print(clf.predict(sample)) 
        
    
    

