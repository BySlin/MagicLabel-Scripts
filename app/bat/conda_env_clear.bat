@echo off
setlocal enabledelayedexpansion

if not defined CONDA_SHLVL (
    goto :eof
)

set "oldPath=%PATH%"
call set "newPath=%%oldPath:*;=%%"

if "%newPath%"=="%oldPath%" (
    set "PATH="
) else (
    set "PATH=%newPath%"
)

endlocal & (
    set "PATH=%PATH%"
    set "CONDA_SHLVL="
    set "CONDA_BAT="
    set "CONDA_EXE="
)
