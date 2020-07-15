# -*- coding: utf-8 -*-


import pickle 
from keras.models import load_model
from django.core.files.storage import default_storage
import numpy as np 
import pandas as pd 
import os 
import pickle 
import re 
import string
from keras.applications.vgg16 import VGG16 , preprocess_input
from keras.preprocessing.image import load_img ,img_to_array 
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.utils import to_categorical
from keras.layers import LSTM ,Input ,Embedding ,Dropout ,Dense
from keras.layers.merge import  add
from keras.utils.vis_utils import plot_model
import os
import requests











class ImageCaption:
    
    ### remove startseq and endseq from sequence
    @staticmethod
    def cleanup_summary(summary):
      index=summary.find('startseq ')
      if index >-1:
        summary=summary[len('startseq '):]
      index=summary.find(' endseq')
      if index >-1:
        summary=summary[:index]  
      return  summary 


    @staticmethod
    def word_for_id(integer,tokenizier):
      for word,id in tokenizier.word_index.items():
        if id==integer:
          return word
      return None   
    
    
    # extract features from each photo in the directory
    @staticmethod
    def extract_features(filename):
    	# load the model
    	model = VGG16()
    	# re-structure the model
    	model.layers.pop()
    	model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    	# load the photo
    	image = load_img(filename, target_size=(224, 224))
    	# convert the image pixels to a numpy array
    	image = img_to_array(image)
    	# reshape data for the model
    	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    	# prepare the image for the VGG model
    	image = preprocess_input(image)
    	# get features
    	feature = model.predict(image, verbose=0)
    	return feature
    
    # generate a description for an image
    @staticmethod
    def generate_desc(model, tokenizer, photo, max_length):
    	# seed the generation process
    	in_text = 'startseq'
    	# iterate over the whole length of the sequence
    	for _ in range(max_length):
    		# integer encode input sequence
    		sequence = tokenizer.texts_to_sequences([in_text])[0]
    		# pad input
    		sequence = pad_sequences([sequence], maxlen=max_length)
    		# predict next word
    		yhat = model.predict([photo,sequence], verbose=0)
    		# convert probability to integer
    		yhat = np.argmax(yhat)
    		# map integer to word
    		word = ImageCaption.word_for_id(yhat, tokenizer)
    		# stop if we cannot map the word
    		if word is None:
    			break
    		# append as input for generating the next word
    		in_text += ' ' + word
    		# stop if we predict the end of the sequence
    		if word == 'endseq':
    			break
    	return in_text
    
    @staticmethod
    def image_caption_lstm_cnn(image):
        max_length=34   ## from training
        file_name =default_storage.save(image.name, image)
        path=default_storage.path(file_name)
        model=load_model('generate_caption\model.h5')
        with open(default_storage.path('generate_caption\\tokenizer.pkl'),'rb') as f:
            tokenizer=pickle.load(f)
        photo = ImageCaption.extract_features(path)
        description = ImageCaption.generate_desc(model, tokenizer, photo, max_length)
        description = ImageCaption.cleanup_summary(description)
        os.remove(path)
        return description
    
    
    
    def neural_talk2(image):
        file_name =default_storage.save(image.name, image)
        path=default_storage.path(file_name)
        r = requests.post(
            "https://api.deepai.org/api/neuraltalk",
            files={
                'image': open(path, 'rb'),
            },
            headers={'api-key': 'quickstart-QUdJIGlzIGNvbWluZy4uLi4K'}
        )
        return(r.json())
        
        
        
        
        

