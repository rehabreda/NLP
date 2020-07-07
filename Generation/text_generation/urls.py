
from django.urls import path ,include
from . import views




urlpatterns = [
    path('text/generation',views.text_generation,name='text_generation')
   
   
]