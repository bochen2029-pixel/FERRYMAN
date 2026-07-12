@echo off
rem FERRYMAN scheduled batch wrapper — timestamped append-log per day.
rem Resolves its own location so the tree runs from ANY path (%~dp0 = this bin\ dir).
set FERRYMAN_HOME=%~dp0..
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist "%FERRYMAN_HOME%\logs" mkdir "%FERRYMAN_HOME%\logs"
set LOG=%FERRYMAN_HOME%\logs\batch_%date:~-4%%date:~4,2%%date:~7,2%.log
echo ================ batch fire %date% %time% ================>> "%LOG%" 2>&1
python "%FERRYMAN_HOME%\src\ferryman.py" batch >> "%LOG%" 2>&1
echo ---------------- exit %errorlevel% %time% ---------------->> "%LOG%" 2>&1
