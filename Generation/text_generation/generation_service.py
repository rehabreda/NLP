# -*- coding: utf-8 -*-

from keras.models import load_model
from random import randint
import pickle 
from keras.preprocessing.sequence import pad_sequences








class TextGeneration:
    
    @staticmethod
    def generate_text(model,tokenizer,seq_length,seed_text,n_words):
        
          result=[]
          in_text=seed_text
          for _ in range(n_words):
            encoded_text=tokenizer.texts_to_sequences([in_text])[0]
            padded_text=pad_sequences([encoded_text],maxlen=seq_length, padding='pre', truncating='pre')
            prediction_word=model.predict_classes(padded_text,verbose=0)
            output_word=''
            for w,i in tokenizer.word_index.items():
              if prediction_word[0]== i:
                output_word=w
                break
            in_text +=' '+output_word
            result.append(output_word)
          return ' '.join(result)
    
    
    @staticmethod
    def generate_text_with_lstm(seed_text,num_words):
        model = load_model('text_generation/model.h5')
        tokenizer = pickle.load(open('text_generation/tokenizer.pkl', 'rb'))
        generated_data=TextGeneration.generate_text(model,tokenizer,50,seed_text,num_words)
        return generated_data
        
    

