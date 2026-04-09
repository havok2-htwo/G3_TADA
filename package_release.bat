@echo off
setlocal EnableExtensions

set "ROOT=%~dp0"
pushd "%ROOT%" >nul

set "PYTHON_EXE="
set "CONDA_ENV_DIR=%ROOT%.conda-env"

if defined TADA_PYTHON (
  call "%TADA_PYTHON%" -c "import sys" >nul 2>&1
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
  echo [ERROR] Python 3.10 or newer is required to build the Windows release bundle.
  popd >nul
  exit /b 1
)

set "PYTHONUTF8=1"
call "%PYTHON_EXE%" "%ROOT%tools\build_release_bundle.py"
set "EXIT_CODE=%ERRORLEVEL%"
popd >nul
exit /b %EXIT_CODE%
