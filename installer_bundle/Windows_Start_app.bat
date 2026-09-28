@echo off
setlocal

pushd "%~dp0Whisper-WebUI" || exit /b 1

call .\venv\Scripts\activate.bat || goto launch_failed

REM SET CUDA_VISIBLE_DEVICES=0  - this is used to set certain CUDA device visible only used
set PYTHONWARNINGS=ignore
set HF_HUB_ENABLE_HF_TRANSFER=1
set HF_HOME=models
REM SET CUDA_VISIBLE_DEVICES=1

python app.py %*
set "APP_EXIT_CODE=%ERRORLEVEL%"
goto launch_done

:launch_failed
echo Virtual environment not found. Run Windows_Install_Update.bat first.
set "APP_EXIT_CODE=1"
:launch_done
popd
pause
exit /b %APP_EXIT_CODE%
