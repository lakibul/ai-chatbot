@echo off
REM Put the assistant on the public internet with a Cloudflare quick tunnel.
REM
REM   start-tunnel.bat          -> port 8000
REM   start-tunnel.bat 8010     -> port 8010
REM
REM Needs cloudflared:  winget install --id Cloudflare.cloudflared -e
REM
REM WARNING: the URL this prints is reachable by ANYONE on the internet who
REM has it. There is no login. Close this window to take it offline.

cd /d "%~dp0"

set PORT=%1
if "%PORT%"=="" set PORT=8000

set CFD=%ProgramFiles(x86)%\cloudflared\cloudflared.exe
if not exist "%CFD%" set CFD=%ProgramFiles%\cloudflared\cloudflared.exe
if not exist "%CFD%" (
  echo.
  echo   ERROR: cloudflared not found.
  echo   Install it once with:
  echo       winget install --id Cloudflare.cloudflared -e
  echo.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo   ERROR: .venv not found in %CD%
  echo   Run the setup steps in README.md first.
  echo.
  pause
  exit /b 1
)

REM Only one process may hold the Qdrant index. Refuse to start a second.
netstat -ano | findstr /r /c:"LISTENING" | findstr /c:":%PORT% " >nul
if not errorlevel 1 (
  echo.
  echo   ERROR: something is already listening on port %PORT%.
  echo   Stop the other server first, or pass a different port:
  echo       start-tunnel.bat 8010
  echo.
  pause
  exit /b 1
)

REM No ROOT_PATH here: the app serves at / so the tunnel URL needs no suffix.
set PYTHONIOENCODING=utf-8

echo.
echo   Starting the assistant on port %PORT% in a second window...
start "Nodi Tech Assistant (port %PORT%)" ".venv\Scripts\python.exe" -m uvicorn src.api.app:app --host 127.0.0.1 --port %PORT%

echo   Waiting for it to be ready...
powershell -NoProfile -Command "1..90 | ForEach-Object { try { if ((Invoke-WebRequest -UseBasicParsing -TimeoutSec 3 'http://127.0.0.1:%PORT%/health').StatusCode -eq 200) { exit 0 } } catch {}; Start-Sleep 2 }; exit 1"
if errorlevel 1 (
  echo.
  echo   ERROR: the server did not come up. Check the other window.
  echo.
  pause
  exit /b 1
)

echo   Ready. Opening the tunnel - your public link appears below.
echo.
"%CFD%" tunnel --no-autoupdate --url http://127.0.0.1:%PORT%

echo.
echo   Tunnel closed. The link no longer works.
echo   Close the server window too.
pause
