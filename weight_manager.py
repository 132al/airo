# music_query.py
"""
音乐知识库交互式查询工具
支持按实体名称搜索，显示完整的词条内容
"""

import sqlite3
import os
from typing import List, Dict, Optional

class MusicQueryTool:
    """音乐知识库查询工具"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.conn.text_factory = str  # 确保文本正确显示
        
    def search(self, keyword: str, max_results: int = 10) -> List[Dict]:
        """搜索包含关键词的实体"""
        results = []
        
        # 1. 搜索实体名称（精确匹配优先）
        self.cursor.execute("""
            SELECT id, entity, text, url, entity_type
            FROM music_knowledge
            WHERE entity LIKE ?
            ORDER BY 
                CASE 
                    WHEN entity = ? THEN 1
                    WHEN entity LIKE ? THEN 2
                    ELSE 3
                END,
                LENGTH(text) DESC
            LIMIT ?
        """, (f'%{keyword}%', keyword, f'{keyword}%', max_results))
        
        for row in self.cursor.fetchall():
            results.append({
                'id': row[0],
                'entity': row[1],
                'text': row[2],
                'url': row[3],
                'entity_type': row[4],
                'match_type': 'entity'
            })
        
        # 如果实体匹配不够，补充文本匹配
        if len(results) < max_results:
            remaining = max_results - len(results)
            # 获取已查到的ID
            existing_ids = [r['id'] for r in results]
            
            if existing_ids:
                placeholders = ','.join(['?' for _ in existing_ids])
                self.cursor.execute(f"""
                    SELECT id, entity, text, url, entity_type
                    FROM music_knowledge
                    WHERE text LIKE ?
                    AND id NOT IN ({placeholders})
                    ORDER BY LENGTH(text) DESC
                    LIMIT ?
                """, (f'%{keyword}%', *existing_ids, remaining))
            else:
                self.cursor.execute("""
                    SELECT id, entity, text, url, entity_type
                    FROM music_knowledge
                    WHERE text LIKE ?
                    ORDER BY LENGTH(text) DESC
                    LIMIT ?
                """, (f'%{keyword}%', remaining))
            
            for row in self.cursor.fetchall():
                results.append({
                    'id': row[0],
                    'entity': row[1],
                    'text': row[2],
                    'url': row[3],
                    'entity_type': row[4],
                    'match_type': 'text'
                })
        
        return results
    
    def get_by_id(self, entity_id: int) -> Optional[Dict]:
        """根据ID获取完整实体"""
        self.cursor.execute("""
            SELECT id, entity, text, url, entity_type
            FROM music_knowledge
            WHERE id = ?
        """, (entity_id,))
        
        row = self.cursor.fetchone()
        if row:
            return {
                'id': row[0],
                'entity': row[1],
                'text': row[2],
                'url': row[3],
                'entity_type': row[4]
            }
        return None
    
    def display_results(self, results: List[Dict], show_full: bool = False):
        """显示搜索结果"""
        if not results:
            print("\n❌ 未找到相关结果")
            return
        
        print(f"\n✅ 找到 {len(results)} 条结果")
        print("=" * 70)
        
        for i, r in enumerate(results, 1):
            match_tag = "🎯" if r['match_type'] == 'entity' else "📄"
            text_len = len(r['text']) if r['text'] else 0
            
            print(f"\n【{i}】{match_tag} {r['entity']}")
            print(f"   ID: {r['id']}")
            print(f"   URL: {r['url']}")
            print(f"   文本长度: {text_len:,} 字符")
            print(f"   匹配方式: {r['match_type']}")
            
            if show_full:
                print(f"\n   📝 完整内容:")
                print("-" * 50)
                print(r['text'] if r['text'] else "（无文本内容）")
                print("-" * 50)
            else:
                # 显示预览
                text_preview = r['text'][:500] + "..." if r['text'] and len(r['text']) > 500 else r['text']
                print(f"\n   📝 内容预览:")
                print(f"   {text_preview}")
        
        print("\n" + "=" * 70)
        print("💡 提示: 输入 'show <编号>' 查看完整内容")
        print("   例如: show 1")
    
    def display_full_content(self, results: List[Dict], index: int):
        """显示指定条目的完整内容"""
        if index < 1 or index > len(results):
            print(f"❌ 无效的编号，请输入 1-{len(results)}")
            return
        
        r = results[index - 1]
        print("\n" + "=" * 70)
        print(f"📖 {r['entity']}")
        print("=" * 70)
        print(f"URL: {r['url']}")
        print(f"ID: {r['id']}")
        print(f"文本长度: {len(r['text']):,} 字符")
        print("\n" + "-" * 70)
        print(r['text'] if r['text'] else "（无文本内容）")
        print("-" * 70)
    
    def interactive(self):
        """交互式查询"""
        print("\n" + "=" * 70)
        print("🎵 音乐知识库查询工具")
        print("=" * 70)
        print("📖 输入关键词搜索音乐实体（艺术家、专辑、歌曲等）")
        print("📖 输入 'show <编号>' 查看完整内容")
        print("📖 输入 'stats' 查看数据库统计")
        print("📖 输入 'quit' 或 'exit' 退出")
        print("=" * 70)
        
        last_results = []
        
        while True:
            try:
                user_input = input("\n🔍 请输入查询: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 再见！")
                    break
                
                if user_input.lower() == 'stats':
                    self.show_stats()
                    continue
                
                if user_input.lower().startswith('show '):
                    try:
                        idx = int(user_input.split(' ')[1])
                        if last_results:
                            self.display_full_content(last_results, idx)
                        else:
                            print("⚠️ 请先进行搜索")
                    except ValueError:
                        print("❌ 请输入有效的编号，例如: show 1")
                    continue
                
                # 执行搜索
                results = self.search(user_input, max_results=10)
                last_results = results
                self.display_results(results, show_full=False)
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 错误: {e}")
    
    def show_stats(self):
        """显示数据库统计信息"""
        self.cursor.execute("SELECT COUNT(*) FROM music_knowledge")
        total = self.cursor.fetchone()[0]
        
        self.cursor.execute("""
            SELECT entity_type, COUNT(*) 
            FROM music_knowledge 
            GROUP BY entity_type
        """)
        types = self.cursor.fetchall()
        
        print("\n📊 数据库统计")
        print("=" * 70)
        print(f"总记录数: {total:,}")
        print(f"实体类型分布:")
        for etype, count in types:
            print(f"  {etype}: {count:,} ({count/total*100:.1f}%)")
        print("=" * 70)
    
    def close(self):
        """关闭数据库连接"""
        self.conn.close()


def quick_search(db_path: str, keyword: str, max_results: int = 5):
    """快速搜索（非交互式）"""
    tool = MusicQueryTool(db_path)
    try:
        results = tool.search(keyword, max_results)
        tool.display_results(results, show_full=True)
    finally:
        tool.close()


if __name__ == "__main__":
    db_path = r'F:\airo\aipro\data\music.db'
    
    # 检查文件是否存在
    if not os.path.exists(db_path):
        print(f"❌ 数据库文件不存在: {db_path}")
        exit(1)
    
    # 交互式模式
    tool = MusicQueryTool(db_path)
    try:
        tool.interactive()
    finally:
        tool.close()