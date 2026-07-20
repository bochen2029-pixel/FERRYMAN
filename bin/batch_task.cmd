@echo off
rem FERRYMAN scheduled batch wrapper — timestamped append-log per day.
rem QC B-03: portable via %~dp0 (bin\); creates logs\ if missing so the redirect
rem never fails before Python even starts.
rem REM-3.3 (R-11): python resolved with py-launcher fallback (FERRYMAN_PYTHON overrides).
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set HERE=%~dp0
if not exist "%HERE%..\logs" mkdir "%HERE%..\logs"
set LOG=%HERE%..\logs\batch_%date:~-4%%date:~4,2%%date:~7,2%.log
set "FM_PY=python"
if defined FERRYMAN_PYTHON set "FM_PY=%FERRYMAN_PYTHON%"
where %FM_PY% >nul 2>nul || set "FM_PY=py -3"
echo ================ batch fire %date% %time% ================>> "%LOG%" 2>&1
%FM_PY% "%HERE%..\src\ferryman.py" batch >> "%LOG%" 2>&1
echo ---------------- exit %errorlevel% %time% ---------------->> "%LOG%" 2>&1
