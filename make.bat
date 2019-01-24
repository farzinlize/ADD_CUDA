@echo off
if "%1" == "clean" goto :clean
if "%1" == "debug" goto :debug

nvcc -o add_cuda.exe kernel.cu helper_functions.c fuzzy_timing.c
goto :eof

:debug
nvcc -o add_cuda_debug.exe kernel.cu helper_functions.c fuzzy_timing.c -D DEBUG
goto :eof

:clean
del add_cuda.exe add_cuda.exp add_cuda.lib add_cuda_debug.exe add_cuda_debug.exp add_cuda_debug.lib