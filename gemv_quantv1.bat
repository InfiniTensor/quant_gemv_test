@echo off
set CXX=g++
set OPENCL_HEADERS="D:\code\opencl_headers\OpenCL-Headers"
set OPENCL_LIB="C:\Program Files (x86)\Common Files\Intel\Shared Libraries\lib"

%CXX% -o gemv_quantv1.exe gemv_quantv1.cpp -I%OPENCL_HEADERS% -L%OPENCL_LIB% -lOpenCL -std=c++17
echo Build finished!
pause