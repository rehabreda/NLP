
from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
import json
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import tensorflow_hub as hub
from text_generation.generation_service import TextGeneration
# Create your views here.




@swagger_auto_schema(method='post',operation_description="Generate text from seed text", responses={200:'Sucess',404: 'Not Found'} ,request_body=openapi.Schema(
    type=openapi.TYPE_OBJECT, 
    properties={
        'seed_text': openapi.Schema(type=openapi.TYPE_STRING),
        'num_words': openapi.Schema(type=openapi.TYPE_INTEGER),
        
    }
))


@api_view(['POST'])
def text_generation(request):
    
    seed_text=request.data['seed_text']
    num_words=request.data['num_words']
    data=TextGeneration.generate_text_with_lstm(seed_text,num_words)
    
    return JsonResponse({'generated text': data}, safe=False)
    
