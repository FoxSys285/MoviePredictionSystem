@echo off
setlocal
title Movie Rating Prediction

set "LAUNCH_SCRIPT=%~dp0project\scripts\start.ps1"
if not exist "%LAUNCH_SCRIPT%" (
    echo Cannot find project\scripts\start.ps1 next to this file.
    pause
    exit /b 1
)

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%LAUNCH_SCRIPT%" %*
set "LAUNCH_EXIT_CODE=%ERRORLEVEL%"
if not "%LAUNCH_EXIT_CODE%"=="0" (
    echo.
    echo Startup failed. The browser was not opened. Read the error above.
    pause
)
exit /b %LAUNCH_EXIT_CODE%
