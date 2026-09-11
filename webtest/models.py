# webtest/models.py

from django.db import models
from django.contrib.auth.models import User


class MusicKnowledge(models.Model):
    """音乐知识库（映射到 music.db 的 music_knowledge 表）"""
    entity = models.CharField(max_length=500)
    text = models.TextField()
    url = models.CharField(max_length=500, blank=True)
    entity_type = models.CharField(max_length=50, blank=True)

    class Meta:
        managed = False
        db_table = 'music_knowledge'
        app_label = 'webtest'


class UserProfile(models.Model):
    """用户档案"""
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="profile",
    )
    profile_json = models.TextField(default="{}")
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        managed = True
        db_table = "user_profile"
        app_label = 'webtest'


class UserFeedback(models.Model):
    """用户对单首歌的反馈（一个用户一首歌只有一条）"""
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name="feedbacks",
    )
    track_id = models.CharField(max_length=100)
    track_name = models.CharField(max_length=200)
    artist_name = models.CharField(max_length=200)
    artist_genres = models.CharField(max_length=500, default="")
    feedback = models.CharField(max_length=20)  # like / dislike
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        managed = True
        db_table = "user_feedback"
        app_label = 'webtest'
        unique_together = (("user", "track_id"),)