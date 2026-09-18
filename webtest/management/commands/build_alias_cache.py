# webtest/management/commands/build_alias_cache.py
"""
生成画像所需的精简流派别名缓存。

背景
----
profile_service 需要一张 "流派名 → genre_idx" 的别名表，用于把用户反馈里的
artist_genres 文本、以及意图推荐的标签，映射到 Qdrant 的 artist_genre_idx。

原本这张表在每次进程启动时都要解析 2.8MB 的 tags_full.json，耗时约 900ms。
本命令把它预生成成一个约 100KB 的扁平 JSON，加载时间降到约 15ms。

用法
----
    python manage.py build_alias_cache
"""

from django.core.management.base import BaseCommand

from webtest import profile_service


class Command(BaseCommand):
    help = "生成精简的流派别名缓存（data/genre_alias.json）"

    def handle(self, *args, **options):
        self.stdout.write("正在构建别名缓存...")
        n = profile_service.build_alias_cache(verbose=False)
        self.stdout.write(self.style.SUCCESS(
            f"完成：{n} 条别名 -> {profile_service.ALIAS_CACHE_FILE}"
        ))
