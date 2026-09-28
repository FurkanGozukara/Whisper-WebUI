@echo off
setlocal
pushd "%~dp0" || exit /b 1

if not exist "..\requirements_whisper.txt" goto missing_distribution
if not exist "..\uv_build_constraints.txt" goto missing_distribution

if not exist "venv\Scripts\python.exe" (
    echo Creating Python 3.12 venv...
    py -3.12 -m venv venv || goto install_failed
)
"venv\Scripts\python.exe" -c "import sys; raise SystemExit(sys.version_info[:2] != (3, 12))" || goto wrong_python
call "venv\Scripts\activate.bat" || goto install_failed

set UV_SKIP_WHEEL_FILENAME_CHECK=1
set UV_LINK_MODE=copy
set UV_COMPILE_BYTECODE=1
python -m pip install -U pip uv || goto install_failed
uv pip install -r "..\requirements_whisper.txt" --index-strategy unsafe-best-match --build-constraints "..\uv_build_constraints.txt" || goto install_failed
uv pip install git+https://github.com/NVIDIA/NeMo.git --index-strategy unsafe-best-match --build-constraints "..\uv_build_constraints.txt" || goto install_failed
python "..\DownloadModels.py" || goto install_failed

echo Requirements installed successfully.
popd
pause
exit /b 0

:missing_distribution
echo The distribution requirements are missing. Extract the complete installer archive first.
goto install_failed
:wrong_python
echo Python 3.12 is required. Run Windows_Install_Update.bat to rebuild the environment.
:install_failed
echo Installation failed. Please save the full console log for debugging.
popd
pause
exit /b 1
