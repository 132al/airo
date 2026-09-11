# webtest/forms.py

import re
from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import AuthenticationForm


class RegisterForm(forms.Form):
    username = forms.CharField(
        max_length=150,
        label="用户名",
        widget=forms.TextInput(attrs={"placeholder": "只能包含字母、数字和下划线"}),
    )
    password1 = forms.CharField(
        label="密码",
        widget=forms.PasswordInput(attrs={"placeholder": "输入密码"}),
    )
    password2 = forms.CharField(
        label="确认密码",
        widget=forms.PasswordInput(attrs={"placeholder": "再输一遍"}),
    )

    def clean_username(self):
        username = self.cleaned_data["username"]
        if not re.match(r"^[A-Za-z0-9_]+$", username):
            raise forms.ValidationError("用户名只能包含字母、数字和下划线")
        if User.objects.filter(username=username).exists():
            raise forms.ValidationError("该用户名已被使用")
        return username

    def clean(self):
        cleaned = super().clean()
        p1 = cleaned.get("password1")
        p2 = cleaned.get("password2")
        if p1 and p2 and p1 != p2:
            raise forms.ValidationError("两次输入的密码不一致")
        return cleaned


class LoginForm(AuthenticationForm):
    username = forms.CharField(
        label="用户名",
        widget=forms.TextInput(attrs={"placeholder": "用户名"}),
    )
    password = forms.CharField(
        label="密码",
        widget=forms.PasswordInput(attrs={"placeholder": "密码"}),
    )