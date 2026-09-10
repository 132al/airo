from django.db import models

class Album(models.Model):
    """专辑主表"""
    music_id = models.CharField(max_length=20, unique=True, verbose_name="豆瓣音乐ID")
    title = models.CharField(max_length=200, verbose_name="专辑名称")
    artist = models.CharField(max_length=100, verbose_name="歌手")
    genre = models.CharField(max_length=50, blank=True, null=True, verbose_name="流派")
    release_date = models.CharField(max_length=20, blank=True, null=True, verbose_name="发行日期")
    rating = models.FloatField(blank=True, null=True, verbose_name="评分")
    rating_count = models.IntegerField(default=0, verbose_name="评分人数")
    comment_count = models.IntegerField(default=0, verbose_name="评论总数")
    
    class Meta:
        db_table = 'album'
        verbose_name = "专辑信息"
        verbose_name_plural = "专辑信息"
    
    def __str__(self):
        return f"{self.title} - {self.artist}"


class Comment(models.Model):
    """评论表，与专辑一对多"""
    album = models.ForeignKey(Album, on_delete=models.CASCADE, related_name='comments', verbose_name="所属专辑")
    comment_id = models.CharField(max_length=20, unique=True, verbose_name="评论ID")
    user = models.CharField(max_length=100, verbose_name="用户昵称")
    user_url = models.URLField(blank=True, null=True, verbose_name="用户主页")
    rating = models.CharField(max_length=10, blank=True, null=True, verbose_name="用户评分（星数）")
    comment_time = models.DateTimeField(blank=True, null=True, verbose_name="评论时间")
    location = models.CharField(max_length=50, blank=True, null=True, verbose_name="用户所在地")
    content = models.TextField(verbose_name="评论内容")
    useful = models.IntegerField(default=0, verbose_name="有用数")
    is_truncated = models.BooleanField(default=False, verbose_name="是否被截断")
    
    class Meta:
        db_table = 'comment'
        verbose_name = "评论信息"
        verbose_name_plural = "评论信息"
    
    def __str__(self):
        return f"{self.user} 评论 {self.album.title}"