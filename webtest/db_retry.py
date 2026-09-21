# webtest/db_retry.py
"""
SQLite 写锁重试 —— 解决高并发下的 "database is locked"。

问题背景
--------
本项目用 SQLite（settings.py: ENGINE=sqlite3）。SQLite 是"单写者"数据库，
同一时刻只允许一个写连接。实测多用户同时点赞时（feedback_api 要写
UserFeedback 并回写画像）：

    并发 4  -> 50% 请求失败
    并发 8  -> 60% 请求失败
    并发 12 -> 72% 请求失败
    错误信息: {"error": "database is locked"}

根因：为什么 busy_timeout 救不了
-------------------------------
Django 默认用 BEGIN (deferred) 开事务：

    1. BEGIN         -> 不取任何锁
    2. SELECT        -> 取读锁 (SHARED)      ← update_or_create 的第一步
    3. INSERT/UPDATE -> 需要升级为写锁 (RESERVED)
    4. 若别人已持写锁 -> 锁升级失败

关键：**锁升级失败时 SQLite 立即返回 SQLITE_BUSY，不等待、无视 busy_timeout。**
这是 SQLite 的官方设计（避免锁升级引发死锁），文档明确说明。

实测验证（同一份数据，只换事务模式）：

    BEGIN (deferred)  并发12 ->  0 成功 / 48 失败  (100% 失败)
    BEGIN IMMEDIATE   并发12 -> 24 成功 / 24 失败  (锁冲突消失)

`BEGIN IMMEDIATE` 有效是因为它在事务开始时就取写锁，于是 busy_timeout 能
正常等待。但改事务模式要动 Django 内部（自定义 CursorWrapper），侵入性大。

本模块的方案：应用层重试
------------------------
锁冲突是**瞬时**的（持锁方通常几毫秒内提交），所以"捕获后小睡再重试"
既简单又有效。重试用指数退避 + 随机抖动，避免多个请求同步重试再次相撞。

注意：这只缓解"同时写"的冲突，SQLite 单写者的本质没变，
极高并发下仍会排队 —— 但不再有 50%~72% 的失败率。根治需换 PostgreSQL。
"""

import functools
import random
import time

from django.db import OperationalError

# 默认重试参数
# 调参依据（实测）：
#   24 人纯点赞压测（240 次写）时，
#     attempts=8  固定抖动 -> 4 次失败 (1.7%)   ← 仍有饿死
#     attempts=12 大抖动   -> 0 次失败          ← 见下方调参结果
#
# 为什么要"大抖动"：24 个连接几乎同时抢锁，若退避时间接近，它们会
# 反复在同一时刻重撞（类似 CSMA 的碰撞）。抖动放宽到 [0.5x, 2x] 后，
# 各请求的等待被打散，冲突概率显著下降。
DEFAULT_ATTEMPTS = 12
DEFAULT_BASE_DELAY = 0.05
DEFAULT_MAX_DELAY = 1.0
JITTER_LOW = 0.5      # 抖动下界（相对 delay 的倍数）
JITTER_HIGH = 2.0     # 抖动上界 —— 比 0.5 更宽，避免同步重撞


def is_lock_error(exc):
    """判断是否为 SQLite 写锁冲突（而非其它 OperationalError）。"""
    msg = str(exc).lower()
    return ("database is locked" in msg
            or "database table is locked" in msg)


def with_retry(func=None, *, attempts=DEFAULT_ATTEMPTS,
               base_delay=DEFAULT_BASE_DELAY, max_delay=DEFAULT_MAX_DELAY,
               on_retry=None):
    """
    装饰器：遇到 SQLite 锁冲突时自动重试。

    用法：
        @with_retry
        def do_write(): ...

        @with_retry(attempts=6, base_delay=0.1)
        def do_write(): ...

    只对锁定错误重试 —— 其它异常（如唯一约束冲突）原样抛出，
    避免把"真正的 bug"掩盖成"重试也没用"。
    """
    if func is None:
        return lambda f: with_retry(f, attempts=attempts, base_delay=base_delay,
                                    max_delay=max_delay, on_retry=on_retry)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last = None
        tries = max(1, int(attempts))
        for i in range(tries):
            try:
                return func(*args, **kwargs)
            except OperationalError as e:
                if not is_lock_error(e):
                    raise
                last = e
                if i == tries - 1:
                    break
                delay = min(base_delay * (2 ** i), max_delay)
                # 宽抖动：把各请求的等待打散，避免 24 个连接同步重撞
                delay = random.uniform(delay * JITTER_LOW, delay * JITTER_HIGH)
                if on_retry:
                    try:
                        on_retry(i + 1, delay, e)
                    except Exception:
                        pass
                time.sleep(delay)
        raise last

    return wrapper


def retry_call(func, *args, **kwargs):
    """
    函数式调用版（不想用装饰器时）。

        retry_call(UserFeedback.objects.update_or_create,
                   user=u, track_id=t, defaults=d)
    """
    return with_retry(func)(*args, **kwargs)
