@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
pushd "%ROOT%" >nul

set "PYTHON_EXE="
set "CONDA_ENV_DIR=%ROOT%.conda-env"

if defined TADA_PYTHON (
  call "%TADA_PYTHON%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)" >nul 2>&1
  if not errorlevel 1 set "PYTHON_EXE=%TADA_PYTHON%"
)

if defined TADA_CONDA_ENV_DIR set "CONDA_ENV_DIR=%TADA_CONDA_ENV_DIR%"
if not defined PYTHON_EXE if exist "%CONDA_ENV_DIR%\python.exe" set "PYTHON_EXE=%CONDA_ENV_DIR%\python.exe"

if not defined PYTHON_EXE (
  echo [ERROR] No managed Python environment was found. Run install.bat first.
  popd >nul
  exit /b 1
)

call "%PYTHON_EXE%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python 3.10 or newer is required. Run install.bat with a supported interpreter.
  popd >nul
  exit /b 1
)

if not exist "%ROOT%frontend\dist\index.html" (
  echo [ERROR] frontend\dist\index.html is missing. Run install.bat first.
  popd >nul
  exit /b 1
)

if not exist "%ROOT%backend\vendor\tada\modules\tada.py" (
  echo [ERROR] backend\vendor is missing. Run install.bat first.
  popd >nul
  exit /b 1
)

set "CUDA_VISIBLE_DEVICES=0"
if "%TADA_DEVICE%"=="" set "TADA_DEVICE=cuda:0"
if "%TADA_ENCODER_DEVICE%"=="" set "TADA_ENCODER_DEVICE=cpu"
if "%TADA_DISABLE_TORCH_COMPILE%"=="" set "TADA_DISABLE_TORCH_COMPILE=1"
if "%TADA_ENABLE_CPU_OFFLOAD%"=="" set "TADA_ENABLE_CPU_OFFLOAD=0"
if "%TADA_DEFAULT_STEPS%"=="" set "TADA_DEFAULT_STEPS=10"
if "%TADA_MODEL_NAME%"=="" set "TADA_MODEL_NAME=HumeAI/tada-3b-ml"
set "HF_HOME=%ROOT%.hf_cache"
set "HF_HUB_CACHE=%ROOT%.hf_cache\hub"
set "HF_XET_CACHE=%ROOT%.hf_cache\xet"
set "HF_ASSETS_CACHE=%ROOT%.hf_cache\assets"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "RUNTIME_PACKAGE_MODE="
if exist "%ROOT%.runtime_package_mode" set /p RUNTIME_PACKAGE_MODE=<"%ROOT%.runtime_package_mode"
set "PYTHONPATH_BASE=%ROOT%"
if /I "%RUNTIME_PACKAGE_MODE%"=="target" if exist "%ROOT%.python_packages" set "PYTHONPATH_BASE=%ROOT%.python_packages;%PYTHONPATH_BASE%"
if defined PYTHONPATH (
  set "PYTHONPATH=%PYTHONPATH_BASE%;%PYTHONPATH%"
) else (
  set "PYTHONPATH=%PYTHONPATH_BASE%"
)
set "PYTHONUTF8=1"
if not defined TADA_SERVER_HOST set "TADA_SERVER_HOST="
if not defined TADA_SERVER_PORT set "TADA_SERVER_PORT="
if not defined TADA_ALLOW_LAN_ACCESS set "TADA_ALLOW_LAN_ACCESS="
set "LAUNCH_SETTINGS_FILE=%ROOT%.tmp\launch_settings.env"
if not exist "%ROOT%.tmp" mkdir "%ROOT%.tmp" >nul 2>&1
del /q "%LAUNCH_SETTINGS_FILE%" >nul 2>&1
call "%PYTHON_EXE%" "%ROOT%tools\print_launch_settings.py" > "%LAUNCH_SETTINGS_FILE%" 2>nul
if exist "%LAUNCH_SETTINGS_FILE%" (
  for /f "usebackq tokens=1,* delims==" %%A in ("%LAUNCH_SETTINGS_FILE%") do set "%%A=%%B"
)
if "%TADA_SERVER_HOST%"=="" set "TADA_SERVER_HOST=127.0.0.1"
if "%TADA_SERVER_PORT%"=="" set "TADA_SERVER_PORT=7878"
if "%TADA_ALLOW_LAN_ACCESS%"=="" set "TADA_ALLOW_LAN_ACCESS=false"
if not defined TADA_STARTUP_ADMIN_KEY_TTL_SECONDS set "TADA_STARTUP_ADMIN_KEY_TTL_SECONDS=300"
if not defined TADA_STARTUP_ADMIN_KEY_DISPLAY_SECONDS set "TADA_STARTUP_ADMIN_KEY_DISPLAY_SECONDS=15"
set "TADA_STARTUP_ADMIN_KEY="
set "STARTUP_ADMIN_KEY_FILE=%ROOT%.tmp\startup_admin_key.txt"
del /q "%STARTUP_ADMIN_KEY_FILE%" >nul 2>&1
call "%PYTHON_EXE%" "%ROOT%tools\generate_startup_admin_key.py" > "%STARTUP_ADMIN_KEY_FILE%" 2>nul
if exist "%STARTUP_ADMIN_KEY_FILE%" (
  set /p TADA_STARTUP_ADMIN_KEY=<"%STARTUP_ADMIN_KEY_FILE%"
  del /q "%STARTUP_ADMIN_KEY_FILE%" >nul 2>&1
)

if defined TADA_STARTUP_ADMIN_KEY (
  echo.
  echo ============================================================
  echo Temporary startup admin key ^(valid for %TADA_STARTUP_ADMIN_KEY_TTL_SECONDS% seconds after server start^):
  echo %TADA_STARTUP_ADMIN_KEY%
  echo Copy it now if you need emergency admin access in the browser.
  echo This screen clears automatically in %TADA_STARTUP_ADMIN_KEY_DISPLAY_SECONDS% seconds...
  echo ============================================================
  timeout /t %TADA_STARTUP_ADMIN_KEY_DISPLAY_SECONDS% /nobreak >nul
  cls
) else (
  echo [WARN] Temporary startup admin key could not be generated.
)

echo Using Python: "%PYTHON_EXE%"
if /I "%TADA_ALLOW_LAN_ACCESS%"=="true" (
  echo Starting TADA server on http://%TADA_SERVER_HOST%:%TADA_SERVER_PORT% ^(LAN enabled; use this machine's IP address from other devices^)
) else (
  echo Starting TADA server on http://%TADA_SERVER_HOST%:%TADA_SERVER_PORT%
)
call "%PYTHON_EXE%" -m uvicorn backend.server_app:app --host "%TADA_SERVER_HOST%" --port "%TADA_SERVER_PORT%"
set "EXIT_CODE=%ERRORLEVEL%"
popd >nul
pause
exit /b %EXIT_CODE%
