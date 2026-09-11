# webtest/middleware.py

from django.shortcuts import redirect


class LoginRequiredMiddleware:
    """全站登录中间件：未登录一律跳 /login/"""

    EXEMPT_PATHS = (
        "/login/",
        "/register/",
        "/logout/",
        "/static/",
        "/admin/",
    )

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        path = request.path_info
        if not request.user.is_authenticated:
            if not any(path.startswith(p) for p in self.EXEMPT_PATHS):
                return redirect(f"/login/?next={path}")
        return self.get_response(request)