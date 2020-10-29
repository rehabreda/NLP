# Arabic Sentiment Analysis Using Bert

training arabert model on arabic data reviews.

## Datasets

The dataset combines reviews from hotels, books, movies, products and a few airlines. It has three classes (Mixed, Negative and Positive). 
Most were mapped from reviewers' ratings with 3 being mixed, above 3 positive and below 3 negative. Each row has a label and text separated by a tab (tsv). 
Text (reviews) were cleaned by removing Arabic diacritics and non-Arabic characters. The dataset has no duplicate reviews.


* [dataset](https://www.kaggle.com/abedkhooli/arabic-100k-reviews) - the dataset link

## Model 

'aubmindlab/bert-base-arabertv01'
* [model ](https://github.com/huggingface/transformers/blob/master/model_cards/aubmindlab/bert-base-arabertv01) - the model link





### Prerequisites

What things you need to install the software and how to install them


```
pip install "tensorflow_gpu>=2.0.0"
pip install transformers==2.5.1
pip install ktrain
```



## Running the tests

Deploy using Flask API and test through postmam.


![Positive](https://github.com/rehabreda/NLP/blob/master/Arabic%20Sentiment%20Analysis%20Using%20Bert/example1.PNG)


![Negative](https://github.com/rehabreda/NLP/blob/master/Arabic%20Sentiment%20Analysis%20Using%20Bert/example2.PNG)