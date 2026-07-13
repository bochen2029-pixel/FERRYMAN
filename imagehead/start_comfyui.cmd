@echo off
REM ── FERRYMAN image head — start ComfyUI HEADLESS on :8188 ─────────────────────────────
REM The ComfyUI *Desktop* Electron app (ComfyUI.exe) does NOT reliably serve the API when
REM launched non-interactively, and it binds :8000 not :8188. So drive the backend directly:
REM   its uv venv python  +  the bundled ComfyUI code  +  --base-directory (the models/output root).
REM Comes up in ~25-30s; `comfy_client.py` / `image_from_doc.py` then talk to http://127.0.0.1:8188.
REM (Machine-specific paths = this dev box; a future `ferryman doctor` should resolve them.)
"C:\Users\user\Documents\ComfyUI\.venv\Scripts\python.exe" "C:\Users\user\AppData\Local\Programs\ComfyUI\resources\ComfyUI\main.py" --base-directory "C:\Users\user\Documents\ComfyUI" --port 8188 --listen 127.0.0.1
