from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/recommend/', views.recommend_api, name='recommend_api'),
    path('data/', views.data_browser, name='data_browser'),
    path('api/data/', views.data_api, name='data_api'),
    path('api/reason/', views.reason_api, name='reason_api'),
    path('api/feedback/', views.feedback_api, name='feedback_api'),
    path('api/feedback/list/', views.feedback_list_api, name='feedback_list_api'),
    path('feedback/', views.feedback_page, name='feedback_page'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('api/seed_candidates/', views.seed_candidates_api, name='seed_candidates_api'),
]