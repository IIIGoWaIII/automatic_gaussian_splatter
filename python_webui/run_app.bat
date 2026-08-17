@echo off
cd /d "%~dp0"

set PYTHON=python
where python >nul 2>nul
if errorlevel 1 set PYTHON=py -3.13

echo Checking for virtual environment...
if not exist ".venv" (
    echo Creating virtual environment...
    %PYTHON% -m venv .venv
    echo Installing dependencies...
    call .venv\Scripts\activate
    pip install -r requirements.txt
) else (
    echo Virtual environment found.
    call .venv\Scripts\activate
)

echo Allowing inbound connections on port 8000 (best effort)...
netsh advfirewall firewall delete rule name="Gaussian Splatter WebUI" >nul 2>nul
netsh advfirewall firewall add rule name="Gaussian Splatter WebUI" dir=in action=allow protocol=TCP localport=8000 >nul 2>nul
if errorlevel 1 (
    echo   Could not add firewall rule. Run this file once as Administrator,
    echo   or allow python.exe when Windows Firewall prompts.
)

echo Starting WebUI...
echo The console shows LAN/Tailnet URLs once the server is up.
echo Open http://127.0.0.1:8000 in your browser.
python server.py

pause
