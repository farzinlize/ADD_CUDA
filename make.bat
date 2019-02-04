@echo off
SETLOCAL
SET source=app.cu kernels.cu helper_functions.c fuzzy_timing.c
SET __defines=
if "%~1"=="" goto :command
if "%1" == "clean" goto :clean

:loop_args
if "%1" == "debug" set __defines=%__defines% -DDEBUG
if "%1" == "overlap" set __defines=%__defines% -DOVERLAP
if "%1" == "test" set __defines=%__defines% -DTEST
shift
if NOT "%~1"=="" goto :loop_args

:command
echo %source% %__defines%
nvcc -o add_cuda.exe %source% %__defines%
ENDLOCAL
goto :eof

:clean
echo clean
del add_cuda.exe add_cuda.exp add_cuda.lib add_cuda_debug.exe add_cuda_debug.exp add_cuda_debug.lib
ENDLOCAL