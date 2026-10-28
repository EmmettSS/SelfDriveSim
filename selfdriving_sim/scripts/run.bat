@echo off
rem ============================================================
rem  SelfDriveSim launcher for Windows.
rem  Checks Redis, applies migrations, collects static files,
rem  prints the URLs, then serves ASGI on 0.0.0.0:8000.
rem
rem  Usage:  scripts\run.bat          (default port 8000)
rem          set PORT=9000 && scripts\run.bat
rem ============================================================
setlocal enabledelayedexpansion

cd /d "%~dp0.."

rem --- configuration ------------------------------------------
if "%PORT%"=="" set PORT=8000
if "%TLS_PORT%"=="" set TLS_PORT=8443
if "%REDIS_HOST%"=="" set REDIS_HOST=127.0.0.1
if "%REDIS_PORT%"=="" set REDIS_PORT=6379
set "SIMULATION_TLS_PORT=%TLS_PORT%"

rem --- interpreter: prefer the local virtualenv ----------------
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

echo ==^> Using interpreter: %PY%
%PY% --version

rem --- 1. Redis must answer ------------------------------------
echo ==^> Checking Redis at %REDIS_HOST%:%REDIS_PORT% ...
where redis-cli >nul 2>nul
if %errorlevel%==0 (
    redis-cli -h %REDIS_HOST% -p %REDIS_PORT% ping 2>nul | findstr /C:PONG >nul
    if errorlevel 1 (
        echo.
        echo ERROR: Redis did not answer at %REDIS_HOST%:%REDIS_PORT%.
        echo.
        echo Start it with one of:
        echo   - WSL:        sudo systemctl enable --now redis-server
        echo   - Docker:     docker run -d -p %REDIS_PORT%:6379 redis:7
        echo   - Memurai:    https://www.memurai.com/   (native Windows Redis)
        echo Or point REDIS_HOST / REDIS_PORT at a running instance.
        echo.
        pause
        exit /b 1
    )
    echo     Redis is up.
) else (
    echo     redis-cli not found; skipping the ping check.
)

rem --- 2. database ---------------------------------------------
echo ==^> Applying migrations ...
%PY% manage.py migrate --noinput
if errorlevel 1 goto :failed

rem --- 3. static files -----------------------------------------
echo ==^> Collecting static files ...
%PY% manage.py collectstatic --noinput
if errorlevel 1 goto :failed

rem --- 4. URLs -------------------------------------------------
rem First IPv4 address ipconfig reports: the one the phone reaches on
rem the same Wi-Fi.
set "LAN_IP="
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /C:"IPv4"') do (
    if not defined LAN_IP set "LAN_IP=%%a"
)
if defined LAN_IP set "LAN_IP=%LAN_IP: =%"
if not defined LAN_IP set "LAN_IP=<LAN-IP>"

echo.
echo     PC renderer (fullscreen):   http://localhost:%PORT%/
echo     Phone dashboard:            https://%LAN_IP%:%TLS_PORT%/dashboard/
echo.
echo     Phone setup - scan the QR code shown on the monitor page, same Wi-Fi,
echo       then accept the one-time self-signed certificate warning:
echo         Android Chrome:  Advanced -^> Proceed to <IP>
echo         iOS Safari:      Show Details -^> visit this website
echo     HTTPS on port %TLS_PORT% is what makes the camera work on Android and iOS.
echo     Guide: docs\guide.fa.md (fa) - docs\guide.en.md (en)
echo.
echo     Keyboard fallback drives the car when no phone command arrives.
echo.

rem --- 5. serve ------------------------------------------------
echo ==^> Serving on 0.0.0.0:%PORT% (http) and 0.0.0.0:%TLS_PORT% (https) - Ctrl+C to stop ...
%PY% manage.py runsecure %PORT%
goto :eof

:failed
echo.
echo ERROR: a step failed - see the output above.
pause
exit /b 1
