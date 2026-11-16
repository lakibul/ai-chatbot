@echo off
REM Start the assistant so other devices on this network can reach it.
REM
REM   start-lan.bat          -> port 8000
REM   start-lan.bat 8080     -> port 8080
REM
REM Binds to 0.0.0.0 (every network interface) rather than 127.0.0.1
REM (this machine only). That is the whole difference between the LAN
REM address working and not working.

cd /d "%~dp0"

set PORT=%1
if "%PORT%"=="" set PORT=8000

if not exist ".venv\Scripts\python.exe" (
  echo.
  echo   ERROR: .venv not found in %CD%
  echo   Run the setup steps in README.md first.
  echo.
  pause
  exit /b 1
)

set ROOT_PATH=ai-chatbot
set PYTHONIOENCODING=utf-8

echo.
echo   Nodi Tech Assistant
echo   ===================
echo.
echo   This machine:  http://127.0.0.1:%PORT%/ai-chatbot
echo.
echo   Other devices on this network:
REM Parsing ipconfig by token position is fragile; ask Windows directly.
for /f "delims=" %%a in ('powershell -NoProfile -Command "(Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' }).IPAddress"') do (
  echo                  http://%%a:%PORT%/ai-chatbot
)
echo.
echo   If other devices cannot connect, open PowerShell as Administrator
echo   and run this once:
echo.
echo     New-NetFirewallRule -DisplayName 'Nodi %PORT%' -Direction Inbound -LocalPort %PORT% -Protocol TCP -Action Allow -Profile Any
echo.
echo   Press Ctrl+C to stop the server.
echo.

".venv\Scripts\python.exe" -m uvicorn src.api.app:app --host 0.0.0.0 --port %PORT%

echo.
echo   Server stopped.
pause
