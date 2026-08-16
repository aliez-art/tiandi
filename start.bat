@echo off
chcp 65001 >nul
setlocal
title 天地熔炉 Tiandi Furnace
cd /d "%~dp0"

echo ================================================
echo   天地熔炉 Tiandi Furnace - 一键点火
echo ================================================
echo.

REM ---------- 1. 定位 tiandi.exe ----------
set "EXE=%~dp0target\release\tiandi.exe"
if not exist "%EXE%" set "EXE=%~dp0target\debug\tiandi.exe"
if not exist "%EXE%" (
    echo [错误] 未找到 tiandi.exe
    echo   请在项目目录执行构建：
    echo     cargo build --release
    echo.
    pause
    exit /b 1
)

REM ---------- 1.5 发布版 UI 目录（覆盖编译路径推导，发布包必需） ----------
if exist "%~dp0ui\dist" set "TIANDI_UI_DIR=%~dp0ui\dist"

REM ---------- 2. 检查工作区（不存在则首次自动初始化） ----------
set "WS=%~dp0.kernel-ws"
if not exist "%WS%\tiandi.db" (
    if not exist "%WS%" (
        echo [提示] 未找到工作区，正在自动初始化 .kernel-ws ...
        "%EXE%" init "%WS%"
        if errorlevel 1 (
            echo [错误] 工作区初始化失败
            echo.
            pause
            exit /b 1
        )
    ) else (
        echo [错误] 工作区目录 .kernel-ws 存在但没有数据库 tiandi.db
        echo   请检查后重试，或删除 .kernel-ws 让本脚本重新初始化
        echo.
        pause
        exit /b 1
    )
)

REM ---------- 3. 端口占用提示 ----------
netstat -an | findstr /c:":18765 " | findstr /c:"LISTENING" >nul 2>&1
if not errorlevel 1 (
    echo [提示] 端口 18765 已被占用，服务将自动回退到 18766-18774
    echo.
)

REM ---------- 4. 启动服务（前台运行，Ctrl-C 停止） ----------
echo 正在点火：http://127.0.0.1:18765
echo 浏览器将自动打开；关闭本窗口或按 Ctrl-C 即停止服务。
echo.
"%EXE%" server --dir "%WS%" --web
echo.
echo [天地熔炉] 服务已停止（退出码 %errorlevel%）
pause
endlocal