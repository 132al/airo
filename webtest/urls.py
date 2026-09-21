from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('api/recommend/', views.recommend_api, name='recommend_api'),
    path('data/', views.data_browser, name='data_browser'),
    path('api/data/', views.data_api, name='data_api'),
    path('api/reason/', views.reason_api, name='reason_api'),
    path('api/netease_link/', views.netease_link_api, name='netease_link_api'),
    path('api/feedback/', views.feedback_api, name='feedback_api'),
    path('api/feedback/list/', views.feedback_list_api, name='feedback_list_api'),
    path('api/playlist/import/', views.import_playlist_api, name='import_playlist_api'),
    path('api/profile/summary/', views.profile_summary_api, name='profile_summary_api'),
    path('feedback/', views.feedback_page, name='feedback_page'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('api/seed_candidates/', views.seed_candidates_api, name='seed_candidates_api'),
    path('api/recommend_by_intent/', views.recommend_by_intent_api, name='recommend_by_intent_api'),
    path('api/recommend_for_me/', views.recommend_for_me_api, name='recommend_for_me_api'),
]