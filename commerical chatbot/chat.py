# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:42:44 2020

@author: rehab
"""

import random 
import json 
import torch 
from nltk_utls import Tokenize,Stem,bag_of_words
from model import  NeuralNet


device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json','r') as f :
    intents=json.load(f)
    
FILE='data.pth'
data=torch.load(FILE)
input_size =data['input_size']
hidden_size=data['hidden_size']
num_classes=data['output_size']
all_words=data['all_words']
model_state=data['model_state']
tags=data['tags']
   


model=NeuralNet(input_size,hidden_size,num_classes).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name='Sam'
print("Let's chat! (type 'quit' to exit)")
while True:
    sentence=input('You: ')
    if sentence== 'quit':
        break
    x=bag_of_words(Tokenize(sentence),all_words)
    x=x.reshape(1,x.shape[0])
    x=torch.from_numpy(x).to(device)
    
    output=model(x)
    _,predicted=torch.max(output,dim=1)
    tag=tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.85:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")
            