from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
import json
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import tensorflow_hub as hub
from generate_caption.image_caption_service import ImageCaption
from rest_framework.parsers import MultiPartParser
from rest_framework.decorators import parser_classes
# Create your views here.



file = openapi.Parameter('file', openapi.IN_FORM, type=openapi.TYPE_FILE)

@swagger_auto_schema(method='post',manual_parameters=[file],operation_description="Generate caption using CNN+LSTM", responses={200:'Sucess',404: 'Not Found'})


@api_view(['POST'])
@parser_classes([MultiPartParser])
def image_caption_using_lstm_cnn(request):
    
    image=request.FILES['file']
    
    description=ImageCaption.image_caption_lstm_cnn(image)
    
    return JsonResponse({'generated text': description}, safe=False)




file = openapi.Parameter('file', openapi.IN_FORM, type=openapi.TYPE_FILE)

@swagger_auto_schema(method='post',manual_parameters=[file],operation_description="Generate caption using NeuralTalk2", responses={200:'Sucess',404: 'Not Found'})


@api_view(['POST'])
@parser_classes([MultiPartParser])
def image_caption_using_neuraltalk2(request):
    
    image=request.FILES['file']
    
    description=ImageCaption.neural_talk2(image)
    
    return JsonResponse({'generated text': description}, safe=False)


