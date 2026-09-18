"""
ASGI config for web_project project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/6.1/howto/deployment/asgi/
"""

import os

# 必须在 django / 数值库之前设置运行时环境（BLAS 线程 + HF 离线），详见 runtime_env.py
import runtime_env  # noqa: F401

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'web_project.settings')

application = get_asgi_application()
