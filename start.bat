@echo off
setlocal
title 天地熔炉 Tiandi Furnace
cd /d "%~dp0"

echo ================================================
echo   天地熔炉 Tiandi Furnace · 一键点火
echo ================================================
echo.

REM ---------- 1. 定位 tiandi.exe ----------
set "EXE=%~dp0target\release\tiandi.exe"
if not exist "%EXE%" set "EXE=%~dp0target\debug\tiandi.exe"
if not exist "%EXE%" (
    echo [错误] 未找到 tiandi.exe
    echo   请先在项目目录执行构建：
    echo     cargo build --release
    echo.
    pause
    exit /b 1
)

REM ---------- 2. 检查工作区 ----------
set "WS=%~dp0.kernel-ws"
if not exist "%WS%\tiandi.db" (
    echo [错误] 工作区数据缺失（.kernel-ws\tiandi.db）
    echo   首次使用请先运行 tiandi init
    echo.
    pause
    exit /b 1
)
if not exist "%WS%\.kernel\kernel.json" (
    echo [提示] 未检测到训练内核，仅演示可用
    echo   真实训练请先运行 tiandi kernel install
    echo.
)

REM ---------- 3. 端口占用检查 ----------
set "PORT=18765"
netstat -an | findstr /c:":%PORT% " | findstr /c:"LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [提示] 端口 %PORT% 已被占用，服务将尝试后续端口
    echo   UI 会自动探测实际端口。
    echo.
)

REM ---------- 4. 启动服务（新窗口，--web 自动打开浏览器） ----------
echo 正在点火：http://127.0.0.1:%PORT%
echo 服务日志在「天地熔炉·服务」窗口，请勿关闭。
echo.
start "天地熔炉 · 服务" "%~dp0server_window.bat" %PORT%

REM ---------- 5. 等待健康检查（最多 20 秒） ----------
set "OK="
for /L %%i in (1,1,20) do (
    curl -s -o nul --max-time 1 "http://127.0.0.1:%PORT%/api/health" >nul 2>&1
    if not errorlevel 1 (
        set "OK=1"
        goto :ready
    )
    timeout /t 1 /nobreak >nul
)

:ready
if defined OK (
    echo.
    echo [OK] 天地熔炉已点火：http://127.0.0.1:%PORT%
    echo   浏览器应已自动打开；若未打开请手动访问。
) else (
    echo.
    echo [警告] 服务未在 %PORT% 端口响应
    echo   请查看服务窗口日志；UI 会自动探测 18765-18774。
)
echo.
echo 停止方式：关闭「天地熔炉·服务」窗口即停止服务。
echo.
timeout /t 15 /nobreak >nul
endlocal