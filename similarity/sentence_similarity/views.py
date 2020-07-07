from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
import json
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import tensorflow_hub as hub
# Create your views here.


module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"

@swagger_auto_schema(method='post',operation_description="GET Similarity Between Sentences", responses={200:'Sucess',404: 'Not Found'} ,request_body=openapi.Schema(
    type=openapi.TYPE_OBJECT, 
    properties={
        'sentence1': openapi.Schema(type=openapi.TYPE_STRING),
        'sentence2': openapi.Schema(type=openapi.TYPE_STRING),
    }
))


@api_view(['POST'])
def sentence_similarity(request):
    
    sent1=request.data['sentence1']
    sent2=request.data['sentence2']
    embed = hub.Module(module_url)
    messages = [sent1,sent2]

    with tf.Session() as session:
      session.run([tf.global_variables_initializer(), tf.tables_initializer()])
      message_embeddings = session.run(embed(messages))
      
    cos_sim = cosine_similarity(message_embeddings[0].reshape(1,-1),message_embeddings[1].reshape(1,-1))  
    return JsonResponse({'similarity':str(cos_sim[0][0])}, safe=False)
    