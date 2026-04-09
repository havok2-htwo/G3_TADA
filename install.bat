@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
pushd "%ROOT%" >nul

set "CONDA_CMD="
set "CONDA_ENV_DIR=%ROOT%.conda-env"
set "LOCAL_CONDA_HOME=%ROOT%.conda"
set "CONDA_PYTHON_VERSION=3.12"
set "PYTHON_EXE="

if defined TADA_CONDA_EXE set "CONDA_CMD=%TADA_CONDA_EXE%"
if not defined CONDA_CMD if defined CONDA_EXE set "CONDA_CMD=%CONDA_EXE%"
if defined TADA_CONDA_ENV_DIR set "CONDA_ENV_DIR=%TADA_CONDA_ENV_DIR%"
if defined TADA_LOCAL_CONDA_HOME set "LOCAL_CONDA_HOME=%TADA_LOCAL_CONDA_HOME%"
if defined TADA_CONDA_PYTHON_VERSION set "CONDA_PYTHON_VERSION=%TADA_CONDA_PYTHON_VERSION%"

if not exist "%LOCAL_CONDA_HOME%" mkdir "%LOCAL_CONDA_HOME%" >nul 2>&1
if not exist "%LOCAL_CONDA_HOME%\pkgs" mkdir "%LOCAL_CONDA_HOME%\pkgs" >nul 2>&1
if not exist "%LOCAL_CONDA_HOME%\envs" mkdir "%LOCAL_CONDA_HOME%\envs" >nul 2>&1
if not exist "%LOCAL_CONDA_HOME%\bld" mkdir "%LOCAL_CONDA_HOME%\bld" >nul 2>&1
if not exist "%LOCAL_CONDA_HOME%\localappdata" mkdir "%LOCAL_CONDA_HOME%\localappdata" >nul 2>&1
if not exist "%LOCAL_CONDA_HOME%\appdata" mkdir "%LOCAL_CONDA_HOME%\appdata" >nul 2>&1
if not exist "%ROOT%.tmp" mkdir "%ROOT%.tmp" >nul 2>&1

set "LOCALAPPDATA=%LOCAL_CONDA_HOME%\localappdata"
set "APPDATA=%LOCAL_CONDA_HOME%\appdata"
set "CONDA_PKGS_DIRS=%LOCAL_CONDA_HOME%\pkgs"
set "CONDA_ENVS_PATH=%LOCAL_CONDA_HOME%\envs"
set "CONDA_BLD_PATH=%LOCAL_CONDA_HOME%\bld"
set "CONDA_NUMBER_CHANNEL_NOTICES=0"
set "CONDA_REPORT_ERRORS=false"
set "TMP=%ROOT%.tmp"
set "TEMP=%ROOT%.tmp"
set "TMPDIR=%ROOT%.tmp"

if /I not "%TADA_USE_SYSTEM_PROXY%"=="1" (
  set "HTTP_PROXY="
  set "HTTPS_PROXY="
  set "ALL_PROXY="
  set "http_proxy="
  set "https_proxy="
  set "all_proxy="
  set "GIT_HTTP_PROXY="
  set "GIT_HTTPS_PROXY="
)

set "CUDA_DIR_NAME="
if not defined CONDA_OVERRIDE_CUDA if defined CUDA_PATH for %%I in ("%CUDA_PATH%") do set "CUDA_DIR_NAME=%%~nI"
if not defined CONDA_OVERRIDE_CUDA if /I "%CUDA_DIR_NAME:~0,1%"=="v" set "CONDA_OVERRIDE_CUDA=%CUDA_DIR_NAME:~1%"

if defined TADA_PYTHON (
  call "%TADA_PYTHON%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)" >nul 2>&1
  if not errorlevel 1 set "PYTHON_EXE=%TADA_PYTHON%"
)

if not defined PYTHON_EXE (
  if not defined TADA_CONDA_EXE (
    for /f "delims=" %%I in ('where conda 2^>nul') do (
      if not defined CONDA_CMD set "CONDA_CMD=%%I"
    )
  )
  call "%CONDA_CMD%" --version >nul 2>&1
  if errorlevel 1 (
    echo [ERROR] conda was not found. Install Miniconda/Anaconda or set TADA_PYTHON to a ready Python 3.12 environment.
    goto fail
  )

  if not exist "%CONDA_ENV_DIR%\python.exe" (
    echo [0/4] Creating local conda environment in "%CONDA_ENV_DIR%" ...
    call "%CONDA_CMD%" create -p "%CONDA_ENV_DIR%" python=%CONDA_PYTHON_VERSION% pip -y
    if errorlevel 1 goto fail
  ) else (
    echo [0/4] Reusing existing conda environment in "%CONDA_ENV_DIR%" ...
  )

  if not exist "%CONDA_ENV_DIR%\python.exe" (
    echo [ERROR] The conda environment was created, but python.exe was not found at "%CONDA_ENV_DIR%\python.exe".
    goto fail
  )
  set "PYTHON_EXE=%CONDA_ENV_DIR%\python.exe"
)

set "TADA_CONDA_ENV_DIR=%CONDA_ENV_DIR%"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "PYTHONUTF8=1"

echo [1/4] Using Python: "%PYTHON_EXE%"
call "%PYTHON_EXE%" -c "import sys; print(sys.version)"
if errorlevel 1 goto fail

echo [2/4] Torch target index: %TADA_TORCH_INDEX_URL%
if "%TADA_TORCH_INDEX_URL%"=="" echo [2/4] Torch target index: default ^(Windows NVIDIA preferred^)

echo [3/4] Running shared runtime installer...
call "%PYTHON_EXE%" "%ROOT%tools\install_runtime.py"
if errorlevel 1 goto fail

echo [4/4] Install finished successfully.
echo Run start.bat to launch the server.
popd >nul
exit /b 0

:fail
echo.
echo [ERROR] Install failed with errorlevel %ERRORLEVEL%.
popd >nul
exit /b 1
