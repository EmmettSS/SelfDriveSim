@echo off
chcp 65001 >nul
title SelfDriveSim Server

set "BASE_DIR=%~dp0"
if exist "%BASE_DIR%selfdriving_sim\manage.py" (
    cd /d "%BASE_DIR%selfdriving_sim"
) else if exist "%BASE_DIR%manage.py" (
    cd /d "%BASE_DIR%"
) else (
    echo [ERROR] Could not find manage.py!
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment .venv not found in %CD%!
    pause
    exit /b 1
)

echo ===================================================
echo       SelfDriveSim - Autonomous Driving Simulator
echo ===================================================
echo Starting Daphne secure server on 0.0.0.0:8000 ...
echo.
echo Desktop Simulation:  http://localhost:8000/
echo Mobile Dashboard:    https://[YOUR_IP]:8000/dashboard/
echo ===================================================
echo.

:: Automatically open browser after 2 seconds
start "" cmd /c "timeout /t 2 /nobreak >nul & start http://localhost:8000/"

:: Launch Daphne secure server
.venv\Scripts\python.exe manage.py runsecure 0.0.0.0:8000

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Server stopped with error code %ERRORLEVEL%.
    pause
)
