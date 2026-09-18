#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

# 必须在任何数值/ML 库被导入之前设置运行时环境变量
# （BLAS 线程限制 + HuggingFace 离线模式），详见 runtime_env.py
import runtime_env  # noqa: F401


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'web_project.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()
