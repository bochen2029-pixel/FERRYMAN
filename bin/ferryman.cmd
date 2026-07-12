@echo off
rem FERRYMAN CLI launcher — runs the orchestrator on system Python (venv-runtime role).
rem Resolves its own location so the tree runs from ANY path (%~dp0 = this bin\ dir).
set FERRYMAN_HOME=%~dp0..
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python "%FERRYMAN_HOME%\src\ferryman.py" %*
