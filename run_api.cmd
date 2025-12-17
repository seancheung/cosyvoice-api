@echo off
setlocal enabledelayedexpansion

cd /D "%~dp0"

set HF_HOME=%~dp0huggingface
set HF_ENDPOINT=https://hf-mirror.com
set MODELSCOPE_CACHE=%~dp0modelscope

call install.cmd

@REM run
call uv run api.py --allow-cors --model-dir "CosyVoice/pretrained_models/CosyVoice2-0.5B" || ( echo. && echo failed to start app && goto :error )

:end
exit /b 0

:error
pause