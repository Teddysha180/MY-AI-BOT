@echo off
chcp 65001 >nul
cd /d %~dp0
:loop
echo [%date% %time%] Starting Artovix >> artovix.log
python -u bot.py >> artovix.log 2>&1
echo [%date% %time%] Bot exited with code %ERRORLEVEL% - restarting in 5s >> artovix.log
timeout /t 5 /nobreak >nul
goto loop
