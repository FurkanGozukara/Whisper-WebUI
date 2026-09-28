@echo off
setlocal
pushd "%~dp0" || exit /b 1

call venv\Scripts\activate.bat || goto launch_failed
python app.py %*
set "APP_EXIT_CODE=%ERRORLEVEL%"
goto launch_done

:launch_failed
echo Virtual environment not found. Run the installer first.
set "APP_EXIT_CODE=1"
:launch_done
popd
pause
exit /b %APP_EXIT_CODE%
