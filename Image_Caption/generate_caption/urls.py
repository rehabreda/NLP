from django.urls import path ,include
from . import views




urlpatterns = [
    path('generate/caption',views.image_caption_using_lstm_cnn,name='generate_caption'),
    path('neural/talk',views.image_caption_using_neuraltalk2,name='neural_talk')
   
   
]