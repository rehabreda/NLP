import nltk 
import numpy as np 
#nltk.download('punkt')

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

stemmer=PorterStemmer()
lemmatizier=WordNetLemmatizer()


def Tokenize(sentence):
    return nltk.word_tokenize(sentence)



def Stem(word):
    return stemmer.stem(word.lower())





def bag_of_words(tokenized_sent,all_words):
    
    tokenized_sent=[Stem(w) for w in tokenized_sent]
    bag=np.zeros(len(all_words),dtype=np.float32)
    for indx,w in enumerate(all_words):
        if w in tokenized_sent:
            bag[indx]=1.0
    return bag         



#unit test
# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]

# bag=bag_of_words(sentence,words)
# print(bag)









