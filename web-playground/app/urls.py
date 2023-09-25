from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('similar_words', views.similar_words, name='similar_words')
#    path('recommendation', views.recommendation, name='recommendation')
    
]
