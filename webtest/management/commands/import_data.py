import json
import os
from datetime import datetime
from django.core.management.base import BaseCommand
from webtest.models import Album, Comment

class Command(BaseCommand):
    help = '从JSON文件导入豆瓣音乐数据'

    def add_arguments(self, parser):
        parser.add_argument('json_file', type=str, help='JSON数据文件路径')

    def handle(self, *args, **options):
        file_path = options['json_file']
        
        if not os.path.exists(file_path):
            self.stdout.write(self.style.ERROR(f'文件不存在: {file_path}'))
            return
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        total = len(data)
        self.stdout.write(f'开始导入 {total} 张专辑...')
        
        for idx, (music_id, album_data) in enumerate(data.items(), 1):
            # 检查专辑是否已存在，避免重复导入
            album, created = Album.objects.get_or_create(
                music_id=music_id,
                defaults={
                    'title': album_data.get('title', ''),
                    'artist': album_data.get('artist', ''),
                    'genre': album_data.get('genre', ''),
                    'release_date': album_data.get('release_date', ''),
                    'rating': float(album_data.get('rating', 0)) if album_data.get('rating') else None,
                    'rating_count': int(album_data.get('rating_count', 0)),
                    'comment_count': int(album_data.get('comment_count', 0)),
                }
            )
            
            if created:
                self.stdout.write(f'  [{idx}/{total}] 导入专辑: {album.title}')
            else:
                self.stdout.write(f'  [{idx}/{total}] 专辑已存在，跳过: {album.title}')
                continue
            
            # 导入评论
            comments = album_data.get('preview_comments', [])
            for comment_data in comments:
                comment_id = comment_data.get('comment_id')
                if not comment_id:
                    continue
                    
                # 解析时间
                time_str = comment_data.get('comment_time', '')
                try:
                    comment_time = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                except:
                    comment_time = None
                
                Comment.objects.get_or_create(
                    comment_id=comment_id,
                    defaults={
                        'album': album,
                        'user': comment_data.get('user', ''),
                        'user_url': comment_data.get('user_url', ''),
                        'rating': comment_data.get('rating', ''),
                        'comment_time': comment_time,
                        'location': comment_data.get('location', ''),
                        'content': comment_data.get('content', ''),
                        'useful': int(comment_data.get('useful', 0)),
                        'is_truncated': comment_data.get('is_truncated', False),
                    }
                )
            
            if idx % 5 == 0:
                self.stdout.write(f'  已完成 {idx}/{total} 张专辑')
        
        self.stdout.write(self.style.SUCCESS(f'导入完成！共导入 {Album.objects.count()} 张专辑，{Comment.objects.count()} 条评论'))