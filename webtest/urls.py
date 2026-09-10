from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/recommend/', views.recommend_api, name='recommend_api'),
    path('data/', views.data_browser, name='data_browser'),
    path('api/data/', views.data_api, name='data_api'),
]