@echo off
call F:\ai-agent\.venv\Scripts\activate

:loop
echo.
set /p MSG="You: "
if /i "%MSG%"=="exit" goto end
python -m davinci chat "%MSG%"
goto loop

:end
echo Goodbye.
pause