from django.contrib import admin
from .models import Album, Comment

@admin.register(Album)
class AlbumAdmin(admin.ModelAdmin):
    list_display = ['title', 'artist', 'genre', 'rating', 'rating_count']
    search_fields = ['title', 'artist']
    list_filter = ['genre']

@admin.register(Comment)
class CommentAdmin(admin.ModelAdmin):
    list_display = ['user', 'album', 'rating', 'comment_time', 'useful']
    search_fields = ['user', 'content']