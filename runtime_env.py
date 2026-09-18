# runtime_env.py
"""
运行时环境变量引导 —— 必须在 import torch / numpy / sklearn / transformers 之前执行。

为什么需要这个文件
------------------
本机内存紧张（16GB 物理内存，Qdrant 常驻 ~5GB，Ollama 加载模型 ~2.6GB，
可用内存经常低于 1GB），会导致两类"看起来像 bug"的故障：

1. OpenBLAS 内存分配失败
   `sentence_transformers` 会连带 import sklearn/scipy，触发 OpenBLAS
   按 CPU 核数创建线程池。每个线程都要预留栈与缓冲区，内存不足时报：
       OpenBLAS error: Memory allocation still failed after 10 retries, giving up.
   进程直接崩溃（或被 shell 显示为无输出卡死）。

   解决：把 BLAS/OMP 线程数限制为 1，线程池内存占用降到最低。
   对本项目完全够用 —— 我们只编码单条 query 字符串，不需要并行矩阵运算。

2. HuggingFace 联网校验超时
   即使模型已缓存在本地，transformers 仍会对每个配置文件发起 HEAD 请求
   校验版本；网络不通时会重试 5 次（每个文件约 1 分钟），表现为"加载极慢"。

   解决：开启 HF 离线模式，强制只读本地缓存。
   （模型已缓存在 %USERPROFILE%\\.cache\\huggingface\\hub\\models--BAAI--bge-small-zh-v1.5）

用法：在 manage.py / wsgi.py / asgi.py 的**最顶部** `import runtime_env`。
"""

import os

# ---------------------------------------------------------------
# 1. 限制数值库线程数（避免 OpenBLAS 内存分配失败）
#    必须在 import numpy / torch / sklearn 之前设置才生效
# ---------------------------------------------------------------
for _var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    # 不覆盖用户显式设置的值
    os.environ.setdefault(_var, "1")

# ---------------------------------------------------------------
# 2. HuggingFace 离线模式（跳过联网校验，直接用本地缓存）
#    如需重新下载模型，临时设置 HF_HUB_OFFLINE=0 即可
# ---------------------------------------------------------------
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ---------------------------------------------------------------
# 3. 控制台编码（Windows 默认 GBK，打印中文/特殊字符可能抛 UnicodeEncodeError）
# ---------------------------------------------------------------
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
