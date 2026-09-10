from django.urls import path
from . import views

app_name = 'webtest'
urlpatterns = [
    path('', views.index, name='index'),
    path('api/recommend/', views.recommend_api, name='recommend_api'),
    path('data/', views.data_browser, name='data_browser'),
    path('api/data/', views.data_api, name='data_api'),
    #path('api/reason/', views.reason_api, name='reason_api'),
    #path('api/feedback/', views.feedback_api, name='feedback_api'),
]