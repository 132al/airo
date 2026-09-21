# ============================================================================
# SQLite 写锁重试 · 回归测试
#
# 背景（本测试守护的核心事实）：
#   SQLite 单写者 + Django 的 BEGIN(deferred) 模式下 busy_timeout 不生效，
#   多用户同时点赞会撞 "database is locked"。实测修复前：
#       并发 4 -> 58% 失败   并发 8 -> 75% 失败   并发 12 -> 79% 失败
#   修复（应用层重试，见 webtest/db_retry.py）后：0% 失败。
#
# 用法：python test_db_retry.py
# ============================================================================

import os
import sys
import threading
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runtime_env  # noqa: F401
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "web_project.settings")
django.setup()

from django.db import OperationalError, connection
from django.contrib.auth.models import User
from webtest.db_retry import (with_retry, retry_call, is_lock_error,
                              DEFAULT_ATTEMPTS)
from webtest.models import UserFeedback, UserProfile
from webtest import profile_service

U = "__retry__"
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}"
          + (f"  {detail}" if detail else ""))


def cleanup():
    UserFeedback.objects.filter(user__username__startswith=U).delete()
    UserProfile.objects.filter(user__username__startswith=U).delete()
    User.objects.filter(username__startswith=U).delete()


def main():
    print("#" * 74)
    print("#  SQLite 写锁重试 · 回归测试")
    print("#" * 74)

    # ---------- 1. 锁错误识别 ----------
    print()
    print("1 · 锁错误识别")
    check("识别 'database is locked'",
          is_lock_error(OperationalError("database is locked")))
    check("识别 'database table is locked'",
          is_lock_error(OperationalError("database table is locked")))
    check("大小写不敏感",
          is_lock_error(OperationalError("Database Is Locked")))
    check("不误判 'no such table'",
          not is_lock_error(OperationalError("no such table: x")))
    check("不误判唯一约束",
          not is_lock_error(OperationalError("UNIQUE constraint failed")))

    # ---------- 2. 重试行为 ----------
    print()
    print("2 · 重试行为")
    cnt = {"n": 0}

    @with_retry(attempts=5, base_delay=0.005)
    def flaky():
        cnt["n"] += 1
        if cnt["n"] < 3:
            raise OperationalError("database is locked")
        return "ok"

    try:
        check("锁冲突后能重试成功", flaky() == "ok", f"第 {cnt['n']} 次成功")
    except Exception as e:
        check("锁冲突后能重试成功", False, f"异常 {e}")

    cnt2 = {"n": 0}

    @with_retry(attempts=3, base_delay=0.005)
    def always():
        cnt2["n"] += 1
        raise OperationalError("database is locked")

    raised = False
    try:
        always()
    except OperationalError:
        raised = True
    check("重试耗尽后仍抛出原异常", raised)
    check("重试次数符合 attempts=3", cnt2["n"] == 3, f"实际 {cnt2['n']} 次")

    cnt3 = {"n": 0}

    @with_retry(attempts=5, base_delay=0.005)
    def other():
        cnt3["n"] += 1
        raise OperationalError("no such table: nope")

    try:
        other()
    except OperationalError:
        pass
    check("非锁错误不重试（立即抛出）", cnt3["n"] == 1, f"实际 {cnt3['n']} 次")

    cnt4 = {"n": 0}

    def fn(a, b=0):
        cnt4["n"] += 1
        if cnt4["n"] < 2:
            raise OperationalError("database is locked")
        return a + b

    check("retry_call 可用且传参正确", retry_call(fn, 1, b=2) == 3)
    check("默认重试次数 >= 5（需覆盖并发争抢）",
          DEFAULT_ATTEMPTS >= 5, f"DEFAULT_ATTEMPTS={DEFAULT_ATTEMPTS}")

    # ---------- 3. 并发写：核心断言 ----------
    print()
    print("3 · 并发写不再失败（核心）")
    cleanup()
    names = [f"{U}{i}" for i in range(24)]
    for n in names:
        User.objects.create_user(username=n, password="x")

    stats = {"ok": 0, "lock": 0, "other": 0}
    lk = threading.Lock()

    def writer(name):
        for k in range(5):
            try:
                u = User.objects.get(username=name)

                # 与 feedback_api 相同的组合：写反馈 + 回写画像
                @with_retry
                def _w():
                    UserFeedback.objects.update_or_create(
                        user=u, track_id=f"{name}_{k}",
                        defaults={"track_name": f"T{k}", "artist_name": "A",
                                  "artist_genres": "pop", "feedback": "like",
                                  "artist_idx": 7000 + k, "artist_genre_idx": 1})
                    return profile_service.rebuild_profile(u)

                _w()
                with lk:
                    stats["ok"] += 1
            except OperationalError as e:
                with lk:
                    if is_lock_error(e):
                        stats["lock"] += 1
                    else:
                        stats["other"] += 1
            except Exception:
                with lk:
                    stats["other"] += 1
            finally:
                connection.close()

    for conc in (4, 8, 24):
        stats["ok"] = stats["lock"] = stats["other"] = 0
        ths = [threading.Thread(target=writer, args=(names[i % len(names)],))
               for i in range(conc)]
        t0 = time.time()
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        el = time.time() - t0
        check(f"并发 {conc} 写操作零锁失败",
              stats["lock"] == 0 and stats["other"] == 0,
              f"ok={stats['ok']} lock={stats['lock']} "
              f"other={stats['other']} ({el:.2f}s)")

    # ---------- 4. 数据完整性 ----------
    print()
    print("4 · 数据完整性")
    db_cnt = UserFeedback.objects.filter(
        user__username__startswith=U, feedback="like").count()
    check("反馈记录已成功落库", db_cnt > 0, f"n={db_cnt}")
    profs = list(UserProfile.objects.filter(user__username__startswith=U))
    check("画像已成功落库", len(profs) > 0, f"n={len(profs)}")
    check("画像内容非空",
          all(p.profile_json and p.profile_json != "{}" for p in profs))

    cleanup()

    print()
    print("=" * 74)
    print(f"  通过 {len(PASS)} / {len(PASS) + len(FAIL)}")
    if FAIL:
        print("  失败项:")
        for f in FAIL:
            print(f"    - {f}")
    print("=" * 74)
    return 0 if not FAIL else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    finally:
        cleanup()


