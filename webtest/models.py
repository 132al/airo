# webtest/models.py

from django.db import models
from django.contrib.auth.models import User


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
    # Embeat 归类索引：Qdrant 里对这两个字段建了 integer 索引，可直接用于过滤/扩散召回。
    # 反馈时由前端带上（Qdrant payload 本就返回这两个字段）。
    artist_idx = models.IntegerField(default=0)
    artist_genre_idx = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        managed = True
        db_table = "user_feedback"
        app_label = 'webtest'
        unique_together = (("user", "track_id"),)