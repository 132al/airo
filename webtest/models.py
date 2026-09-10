# webtest/models.py

from django.db import models


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