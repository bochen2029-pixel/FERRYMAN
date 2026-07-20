@echo off
rem FERRYMAN CLI launcher — runs the orchestrator on system Python (venv-runtime role).
rem QC B-03: resolves the tree from this file's own location (%~dp0 = bin\) — portable
rem to any unpack path; FERRYMAN_HOME (if set) still wins inside ferryman.py itself.
rem REM-3.3 (R-11): a stock box may have only the py-launcher — resolve python, fall
rem back to `py -3`, and let FERRYMAN_PYTHON override both.
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if defined FERRYMAN_PYTHON (
  "%FERRYMAN_PYTHON%" "%~dp0..\src\ferryman.py" %*
  goto :eof
)
where python >nul 2>nul
if %errorlevel%==0 (
  python "%~dp0..\src\ferryman.py" %*
) else (
  py -3 "%~dp0..\src\ferryman.py" %*
)
