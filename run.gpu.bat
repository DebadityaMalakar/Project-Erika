@echo off
echo Compiling with CUDA using cl.exe...

:: Set cl.exe path for nvcc
set "CL_PATH=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64"

:: Run nvcc using cl.exe
nvcc -o erika_gpu main.cpp neurons.cpp loader.cpp model.cpp -I. --std=c++17 -arch=sm_86 -ccbin "%CL_PATH%" --allow-unsupported-compiler
if %errorlevel% neq 0 (
    echo CUDA Compilation failed!
    exit /b %errorlevel%
)
echo CUDA Compilation Successful!

echo Running with CUDA...
erika_gpu
pause
