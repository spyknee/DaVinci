@echo off
cd /d F:\DaVinci
call %~dp0..\ai-agent\.venv\Scripts\activate

:loop
echo.
set /p MSG="You: "
if /i "%MSG%"=="exit" goto end
python __main__.py chat "%MSG%"
goto loop

:end
echo Goodbye.
pause
