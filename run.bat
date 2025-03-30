@echo off
echo Compiling without CUDA...
g++ -std=c++17 -o erika main.cpp neurons.cpp loader.cpp
if %errorlevel% neq 0 (
    echo Compilation failed!
    exit /b %errorlevel%
)
echo Compilation Successful!

echo Running...
erika
pause
