@echo off
REM ============================================================
REM  Ollama 健康检查 / 自动修复
REM
REM  背景：Ollama 在 Windows 上存在"空壳托盘"状态不一致问题 ——
REM  服务进程 ollama.exe 被系统回收后，托盘 ollama app.exe 仍在，
REM  并占据"已启动"标志位，导致：
REM    1) 11434 端口无人监听，前端报 WinError 10061
REM    2) 手动启动会被判定"existing instance found"直接退出
REM  本脚本检测到该状态后，会杀掉残留并干净重启。
REM
REM  用法：ollama_doctor.bat
REM ============================================================
setlocal

set OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe
set OLLAMA_APP=%LOCALAPPDATA%\Programs\Ollama\ollama app.exe
set ENDPOINT=http://127.0.0.1:11434/api/version

echo ============================================================
echo  Ollama 健康检查
echo ============================================================
echo.

REM ---- 1. 端口探测 ----
echo [1/4] 探测 %ENDPOINT%
curl -s --max-time 5 %ENDPOINT% > "%TEMP%\_ollama_probe.txt" 2>nul
if %ERRORLEVEL% EQU 0 (
    echo       服务正常
    type "%TEMP%\_ollama_probe.txt"
    echo.
    goto :show_models
)
echo       服务无响应（连接被拒绝）
echo.

REM ---- 2. 清理残留进程 ----
echo [2/4] 清理残留进程
taskkill /F /IM "ollama.exe" >nul 2>&1
taskkill /F /IM "ollama app.exe" >nul 2>&1
timeout /t 3 /nobreak >nul
echo       已清理
echo.

REM ---- 3. 重启服务 ----
echo [3/4] 启动 Ollama
if not exist "%OLLAMA_APP%" (
    echo       错误：找不到 %OLLAMA_APP%
    echo       请确认 Ollama 已安装
    goto :fail
)
start "" "%OLLAMA_APP%"

REM 轮询等待最多 60 秒
set /a tries=0
:wait
set /a tries+=1
timeout /t 3 /nobreak >nul
curl -s --max-time 5 %ENDPOINT% > "%TEMP%\_ollama_probe.txt" 2>nul
if %ERRORLEVEL% EQU 0 goto :started
if %tries% GEQ 20 goto :timeout_msg
echo       等待中... (%tries%/20)
goto :wait

:started
echo       启动成功（等待 %tries% 次）
type "%TEMP%\_ollama_probe.txt"
echo.
echo       提示：首次推理需要把模型加载到 GPU，约 5-15 秒
echo.

REM ---- 4. 列出模型 ----
echo [4/4] 已安装模型
"%OLLAMA_EXE%" list
echo.
echo ============================================================
echo  修复完成
echo ============================================================
goto :eof

:show_models
echo [2/4] 无需修复
echo.
echo [3/4] 已安装模型
"%OLLAMA_EXE%" list
echo.
echo ============================================================
echo  一切正常
echo ============================================================
goto :eof

:timeout_msg
echo.
echo       错误：等待 60 秒后服务仍未就绪
echo       请手动检查：%LOCALAPPDATA%\Ollama\server.log
goto :fail

:fail
echo.
echo ============================================================
echo  修复失败
echo ============================================================
exit /b 1
