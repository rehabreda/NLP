
from django.urls import path ,include
from . import views




urlpatterns = [
    path('sentence/similarity',views.sentence_similarity,name='sentence_similarity')
   
   
]
# -*- coding: utf-8 -*-

