@echo off
REM ── FERRYMAN image head — start ComfyUI HEADLESS on :8188 ─────────────────────────────
REM The ComfyUI *Desktop* Electron app (ComfyUI.exe) does NOT reliably serve the API when
REM launched non-interactively, and it binds :8000 not :8188. So drive the backend directly:
REM   its uv venv python  +  the bundled ComfyUI code  +  --base-directory (the models/output root).
REM Comes up in ~25-30s; `comfy_client.py` / `image_from_doc.py` then talk to http://127.0.0.1:8188.
REM QC C-03: paths resolve per-user (%USERPROFILE% / %LOCALAPPDATA%) instead of a hardcoded
REM username; override COMFYUI_BASE / COMFYUI_PY / COMFYUI_MAIN if ComfyUI lives elsewhere.
if not defined COMFYUI_BASE set "COMFYUI_BASE=%USERPROFILE%\Documents\ComfyUI"
if not defined COMFYUI_PY   set "COMFYUI_PY=%COMFYUI_BASE%\.venv\Scripts\python.exe"
if not defined COMFYUI_MAIN set "COMFYUI_MAIN=%LOCALAPPDATA%\Programs\ComfyUI\resources\ComfyUI\main.py"
if not exist "%COMFYUI_PY%"   echo [start_comfyui] python not found: %COMFYUI_PY%  (install ComfyUI Desktop or set COMFYUI_PY) && exit /b 2
if not exist "%COMFYUI_MAIN%" echo [start_comfyui] ComfyUI main.py not found: %COMFYUI_MAIN%  (set COMFYUI_MAIN) && exit /b 2
"%COMFYUI_PY%" "%COMFYUI_MAIN%" --base-directory "%COMFYUI_BASE%" --port 8188 --listen 127.0.0.1
