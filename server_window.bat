@echo off
chcp 65001 >nul
setlocal
set "EXE=%~dp0target\release\tiandi.exe"
if not exist "%EXE%" set "EXE=%~dp0target\debug\tiandi.exe"
echo [天地熔炉] 服务启动中... Ctrl-C 或关闭本窗口即停止。
"%EXE%" server --dir "%~dp0.kernel-ws" --port %1 --web
echo.
echo [天地熔炉] 服务已停止（退出码 %errorlevel%）
pause