@echo off
setlocal
pushd "%~dp0" || exit /b 1

echo WARNING. For this auto installer to work you need to have installed Python 3.12.10+, Git, FFmpeg, cuDNN 9.17+, CUDA 13.0, Visual Studio Community Edition with All c++ options
echo This tutorial shows all requirements step by step : https://youtu.be/DrhUHnYfwC0
echo This requirements will help you run all AI apps

set UV_SKIP_WHEEL_FILENAME_CHECK=1
set UV_LINK_MODE=copy
REM Compile the libraries to bytecode during install; otherwise the first app start does it and looks frozen
set UV_COMPILE_BYTECODE=1

pip install requests

if not exist "Whisper-WebUI\.git" (
    git clone https://github.com/FurkanGozukara/Whisper-WebUI || goto install_failed
)

cd Whisper-WebUI || goto install_failed

git reset --hard || goto install_failed

git pull || goto install_failed

REM A venv made with another Python version (3.11 in older releases) cannot be reused
if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" -c "import sys; raise SystemExit(not sys.version_info[:2] == (3, 12))" >nul 2>&1
    if errorlevel 1 (
        echo Existing venv is not Python 3.12. Removing it so it can be recreated...
        rmdir /s /q venv
        if exist venv (
            echo Could not remove the old venv. Close the app and any window using it, then run this installer again.
            goto install_failed
        )
    )
)

py --version >nul 2>&1
if "%ERRORLEVEL%" == "0" (
    echo Python launcher is available. Generating Python 3.12 VENV
    py -3.12 -m venv venv
) else (
    echo Python launcher is not available, generating VENV with default Python. Make sure that it is 3.12
    python -m venv venv
)

"venv\Scripts\python.exe" -c "import sys; raise SystemExit(sys.version_info[:2] != (3, 12))" || goto wrong_python
call .\venv\Scripts\activate.bat || goto install_failed

echo installing requirements

python -m pip install --upgrade pip || goto install_failed

pip install uv || goto install_failed

uv pip install wheel || goto install_failed

cd ..

uv pip install -r requirements_whisper.txt --index-strategy unsafe-best-match --build-constraints uv_build_constraints.txt || goto install_failed

uv pip install git+https://github.com/NVIDIA/NeMo.git --index-strategy unsafe-best-match --build-constraints uv_build_constraints.txt || goto install_failed

python DownloadModels.py || goto install_failed


REM Show completion message
echo Virtual environment made and installed properly - save logs to send me if any errors occurs

REM Pause to keep the command prompt open
popd
pause
exit /b 0

:wrong_python
echo Python 3.12 is required. Install Python 3.12 and its launcher, then run this installer again.
:install_failed
echo Installation failed. Please save the full console log and send it for debugging.
popd
pause
exit /b 1
