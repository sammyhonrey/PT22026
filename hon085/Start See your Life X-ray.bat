@echo off
title See your Life X-ray
cd /d "%~dp0"
set PY=
where py >nul 2>nul && set PY=py -3
if not defined PY (where python >nul 2>nul && set PY=python)
if not defined PY goto :nopython
%PY% -c "import fastapi, uvicorn, sklearn, mediapipe" >nul 2>nul
if errorlevel 1 (
  echo Installing the Python packages. This is needed only once and takes a few minutes...
  %PY% -m pip install -r requirements.txt
  if errorlevel 1 goto :failed
)
%PY% scripts\fetch_assets.py
if errorlevel 1 goto :failed
%PY% run.py
goto :end
:nopython
echo.
echo  Python was not found on this computer.
echo  1. Open https://www.python.org/downloads/ and install Python 3.12.
echo  2. On the first install screen, tick "Add python.exe to PATH".
echo  3. Double-click this file again.
goto :end
:failed
echo.
echo  Something went wrong. Check your internet connection and try again.
:end
pause
