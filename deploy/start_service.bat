@echo off
REM 看门循环:run.py 退出后 15 秒自动拉起。
REM BRAIN_HOME  = 仓库根目录(默认取本 bat 的上一级)
REM BRAIN_PYTHON = 要用的解释器(默认交给 PATH)

if "%BRAIN_HOME%"=="" set BRAIN_HOME=%~dp0..
if "%BRAIN_PYTHON%"=="" (
    if exist "C:\miniconda3\envs\chatservice\python.exe" (
        set BRAIN_PYTHON=C:\miniconda3\envs\chatservice\python.exe
    ) else if exist "D:\miniconda\envs\chatservice\python.exe" (
        set BRAIN_PYTHON=D:\miniconda\envs\chatservice\python.exe
    ) else (
        set BRAIN_PYTHON=python
    )
)

cd /d "%BRAIN_HOME%"
if not exist logs mkdir logs

:loop
"%BRAIN_PYTHON%" run.py --reset >> logs\launcher.log 2>&1
timeout /t 15 /nobreak > nul
goto loop
