// opencl_gemv_q8.cpp
/*
    课程作业：学习实现并优化Q8_0量化算子，尽可能多的应用课程讲解的优化方案，以实现更好的性能。
    1. 权重读取之后反量化到高位计算（已提供基础实现）；
    2. 激活值量化到低位做计算；
    3. 应用课程讲解的通用优化：应用共享内存，使用子组，使用dot等优化；
    4. 优化内存访问，权重拆分
    5. 使用image图像内存存储权重；
    6. ...
*/
/*
    量化作业示例-权重A：BlockQ8_0，激活值B：half
    内核测试：
    将实现的kernel放到 kernelSource 中（可以根据优化方案调整使用的量化函数，内核参数等）,根据实际路径修改环境变量，执行脚本即可编译。
*/

#define CL_TARGET_OPENCL_VERSION 200
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <algorithm>
#include <cstdint>
#include <random>
#include <CL/cl.h>
#include "half.hpp"
#include <fstream>
#include <sstream>

using namespace std;
using half_float::half;

constexpr int NUM_WARMUP = 100;
constexpr int NUM_ITER = 1000;
constexpr int MAX_INT = 99999;
constexpr int Block_size = 32;
constexpr float SCALE = 100000.0f;
constexpr float QMAX = 127.f;
constexpr float EPS = 1e-6f;

// Q8_0量化块
struct BlockQ8_0
{
    half d;
    int8_t q[Block_size];
};

// 工具函数
void checkErr(cl_int err, const char *name)
{
    if (err != CL_SUCCESS)
    {
        fprintf(stderr, "ERROR: %s (%d)\n", name, err);
        exit(EXIT_FAILURE);
    }
}

vector<float> generateMatrix_F32(int rows, int cols)
{
    static mt19937 gen(66); // 固定种子可复现
    uniform_real_distribution<float> dist(-0.5f, 0.5f);

    vector<float> matrix(rows * cols);
    for (auto &val : matrix)
        val = dist(gen);
    return matrix;
}
vector<half> generateMatrix_F16(int rows, int cols)
{
    static mt19937 gen(66);
    uniform_real_distribution<float> dist(-0.5f, 0.5f);

    vector<half> matrix(rows * cols);
    for (auto &val : matrix)
        val = half(dist(gen));
    return matrix;
}

void gemv(vector<half> &C, const vector<half> &A, const vector<half> &B, int m, int n, int k, float alpha, float beta)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            float sum = 0.0f;
            for (int l = 0; l < k; l++)
            {
                sum += (float)A[j * k + l] * (float)B[i * k + l];
            }
            if (beta != 0.0f)
                C[i * m + j] = (half)(alpha * sum + beta * (float)C[i * m + j]);
            else
                C[i * m + j] = (half)(alpha * sum);
        }
    }
}

// void quant_gemv(vector<half> &C, const vector<BlockQ8_0> &A, const vector<half> &B, int m, int n, int k, float alpha, float beta)
// {
//     for (int i = 0; i < n; i++)
//     {
//         for (int j = 0; j < m; j++)
//         {
//             float sum = 0;
//             for (int l = 0; l < k / 32; l++)
//             {
//                 for (int o = 0; o < 32; o++)
//                 {
//                     sum += (float)(int)A[j * k / 32 + l].q[o] * (float)B[i * k + l * 32 + o] * (float)A[j * k / 32 + l].d;
//                 }
//             }
//             if (beta != 0)
//                 C[i * m + j] = (half)(alpha * sum + beta * (float)C[i * m + j]);
//             else
//                 C[i * m + j] = (half)(alpha * sum);
//         }
//     }
// }

half computeAbsoluteError(const vector<half> &C_cpu, const vector<half> &C_gpu)
{
    float max_abs = 0.0f;
    for (size_t i = 0; i < C_cpu.size(); ++i)
    {
        float a = (float)C_cpu[i];
        float b = (float)C_gpu[i];
        float err = fabsf(a - b);
        if (err > max_abs)
            max_abs = err;
    }
    return half(max_abs);
}

inline int8_t clamp_int8(int v)
{
    return static_cast<int8_t>(std::max(-128, std::min(127, v)));
}

// 激活 B 的对称 per-32 量化：输出 qB(int8) 与 sB(half per-block)
void quantB_q8_per32(const vector<half> &B, vector<int8_t> &qB, vector<half> &sB, int k)
{
    int blocks = k / Block_size;
    qB.resize(k);
    sB.resize(blocks);
    for (int j = 0, jj = 0; j < k; j += Block_size, ++jj)
    {
        float max_abs = 0.f;
        for (int l = j; l < j + Block_size; ++l)
            max_abs = max(max_abs, fabsf((float)B[l]));
        half sd = (half)((max_abs > 0.f) ? (max_abs / QMAX) : EPS);
        sB[jj] = sd;
        for (int l = j; l < j + Block_size; ++l)
        {
            float v = (float)B[l] / (float)sd;
            int iv = (int)roundf(v);
            qB[l] = clamp_int8(iv);
        }
    }
}
// 量化最后一维 量化块存储
// M: m x k ; produce BlockQ8_0: m x (k / 32)
void quantv1(const vector<half> &M, vector<BlockQ8_0> &q_M, int m, int k, int blocks)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0, jj = 0; j < k; j += Block_size, jj += 1)
        {
            float max_abs = -1.0f;
            for (int l = j; l < j + Block_size; l++)
            {
                max_abs = max(max_abs, fabsf((float)M[i * k + l]));
            }
            half qd = (half)(max_abs / QMAX);
            if ((float)qd == 0.0f)
                qd = half(EPS);
            q_M[i * blocks + jj].d = qd;
            for (int l = j; l < j + Block_size; l++)
            {
                float v = (float)M[i * k + l] / (float)qd;
                int iv = (int)roundf(v);
                q_M[i * blocks + jj].q[l - j] = clamp_int8(iv);
            }
        }
    }
}

// 量化最后一维 量化分离存储
// M: m x k ; produce q_M (m*k chars) and q_d_M (m * (k/32) halves)
void quantv2(const vector<half> &M, vector<char> &q_M, vector<half> &q_d_M, int m, int k)
{
    int blocks = k / Block_size;
    q_M.assign(m * k, 0);
    q_d_M.assign(m * blocks, half(0.0f));
    for (int i = 0; i < m; i++)
    {
        for (int j = 0, jj = 0; j < k; j += Block_size, jj += 1)
        {
            float max_abs = 0.f;
            for (int l = j; l < j + Block_size; l++)
            {
                max_abs = max(max_abs, fabsf((float)M[i * k + l]));
            }
            half qd = (half)(max_abs / QMAX);
            if ((float)qd == 0.0f)
                qd = half(EPS);
            q_d_M[i * blocks + jj] = qd;
            for (int l = j; l < j + Block_size; l++)
            {
                float v = (float)M[i * k + l] / (float)qd;
                int iv = (int)roundf(v);
                q_M[i * k + l] = clamp_int8(iv);
            }
        }
    }
}

static std::string loadTextFile(const std::string &path)
{
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs)
    {
        throw std::runtime_error("Failed to open file: " + path);
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    return oss.str();
}

const char *kernelPath = "./kernel.cl";
// kernel 示例

// ---------- Host 辅助：编译、运行、衡量函数 ----------
void printDeviceInfo(cl_device_id device)
{
    char name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL);
    cout << "Device: " << name << endl;
    cl_uint subs;
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(subs), &subs, NULL);
    cout << "Compute units: " << subs << endl;
    cl_bool imageSupport = CL_FALSE;
    clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(imageSupport), &imageSupport, NULL);
    cout << "Image support: " << (imageSupport ? "yes" : "no") << endl;
    // subgroup support query (OpenCL 2.0/2.1 may differ across vendors)
    // 这里仅输出提示，具体是否可用以 clGetDeviceInfo(CL_DEVICE_EXTENSIONS) 判断
    size_t ext_size = 0;
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 0, NULL, &ext_size);
    string exts(ext_size, '\0');
    clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, ext_size, &exts[0], NULL);
    cout << "Extensions: " << exts << endl;
}

double KernelTest(const string &kernelName,
                  //   const char *kernelSrc,
                  cl_context context,
                  cl_device_id device,
                  cl_command_queue queue,
                  const vector<BlockQ8_0> &BlockA, // m x (k / 32)
                  const vector<int8_t> &qB,        // n * k (int8)
                  const vector<half> &sB,          // k/32 per-block scale for B (n==1)
                  vector<half> &C_gpu,
                  const vector<half> &C_ref,
                  int m, int n, int k,
                  float alpha, float beta)
{
    auto src = loadTextFile(kernelPath);
    const char *kernelSource = src.c_str();
    const size_t src_len = src.size();
    cl_int err;
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, &src_len, &err);
    checkErr(err, "clCreateProgramWithSource");
    const char *buildOptions = "-cl-std=CL2.0";
    err = clBuildProgram(program, 1, &device, buildOptions, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        // print build log
        size_t log_size = 0;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        string log(log_size, '\0');
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], NULL);
        cerr << "Build failed:\n"
             << log << endl;
        exit(1);
    }
    cl_kernel kernel = clCreateKernel(program, kernelName.c_str(), &err);
    checkErr(err, "clCreateKernel");

    int blocks = k / Block_size;
    size_t sizeBlockA = (size_t)(m * blocks) * sizeof(BlockQ8_0);
    size_t sizeQB = (size_t)n * (size_t)k * sizeof(int8_t);
    size_t sizeSB = (size_t)(k / Block_size) * sizeof(half);
    size_t sizeC = (size_t)n * (size_t)m * sizeof(half);

    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeBlockA, (void *)BlockA.data(), &err);
    checkErr(err, "clCreateBuffer A");
    cl_mem bufQB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeQB, (void *)qB.data(), &err);
    checkErr(err, "clCreateBuffer qB");
    cl_mem bufSB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeSB, (void *)sB.data(), &err);
    checkErr(err, "clCreateBuffer sB");
    cl_mem bufC = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeC, (void *)C_gpu.data(), &err);
    checkErr(err, "clCreateBuffer C");

    // set kernel args
    int argIdx = 0;
    clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &bufQB);
    clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &bufSB);
    clSetKernelArg(kernel, argIdx++, sizeof(cl_mem), &bufC);
    // strides
    // A:m*k B:n*k C:n*m
    // A * B^T = C^T
    int as = m * k / Block_size, ars = k / Block_size, acs = 1;
    // qB layout matches B: n*k contiguous; for n==1, row-stride 1, col-stride k
    int bs = n * k, brs = 1, bcs = k;
    int cs = n * m, crs = 1, ccs = n;
    clSetKernelArg(kernel, argIdx++, sizeof(int), &as);
    clSetKernelArg(kernel, argIdx++, sizeof(int), &ars);
    clSetKernelArg(kernel, argIdx++, sizeof(int), &acs);
    clSetKernelArg(kernel, argIdx++, sizeof(int), &bs);
    clSetKernelArg(kernel, argIdx++, sizeof(int), &brs);
    clSetKernelArg(kernel, argIdx++, sizeof(int), &bcs);
    clSetKernelArg(kernel, argIdx++, sizeof(int), &cs);
    clSetKernelArg(kernel, argIdx++, sizeof(int), &crs);
    clSetKernelArg(kernel, argIdx++, sizeof(int), &ccs);
    clSetKernelArg(kernel, argIdx++, sizeof(int), &m);
    clSetKernelArg(kernel, argIdx++, sizeof(int), &n);
    clSetKernelArg(kernel, argIdx++, sizeof(int), &k);
    clSetKernelArg(kernel, argIdx++, sizeof(float), &alpha);
    clSetKernelArg(kernel, argIdx++, sizeof(float), &beta);

    // NDRange: 1D: (m) — 选择安全的 LWS，并将 GWS 向上取整到 LWS 的倍数
    size_t devMaxWG = 0;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(devMaxWG), &devMaxWG, NULL);
    size_t devMaxItems[3] = {0, 0, 0};
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(devMaxItems), &devMaxItems, NULL);
    size_t kernelMaxWG = 0;
    clGetKernelWorkGroupInfo(kernel, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(kernelMaxWG), &kernelMaxWG, NULL);

    size_t lwsCandidate = 256; // 目标 LWS
    size_t lws0 = lwsCandidate;
    lws0 = std::min(lws0, devMaxWG);
    lws0 = std::min(lws0, devMaxItems[0]);
    lws0 = std::min(lws0, kernelMaxWG);
    if (lws0 == 0)
        lws0 = 1; // 兜底

    size_t lws[1] = {lws0};
    size_t gws[1] = {((size_t)m + lws[0] - 1) / lws[0] * lws[0]};

    // warmup
    for (int i = 0; i < NUM_WARMUP; ++i)
    {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, gws, lws, 0, NULL, NULL);
        checkErr(err, "clEnqueueNDRangeKernel warmup");
        clFinish(queue);
    }

    vector<cl_event> evs(NUM_ITER);
    for (int i = 0; i < NUM_ITER; ++i)
    {
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, gws, lws, 0, NULL, &evs[i]);
        checkErr(err, "clEnqueueNDRangeKernel timed");
    }
    err = clWaitForEvents(NUM_ITER, evs.data());
    checkErr(err, "clWaitForEvents");

    double total_ms = 0;
    for (int i = 0; i < NUM_ITER; ++i)
    {
        cl_ulong t0, t1;
        clGetEventProfilingInfo(evs[i], CL_PROFILING_COMMAND_START, sizeof(t0), &t0, NULL);
        clGetEventProfilingInfo(evs[i], CL_PROFILING_COMMAND_END, sizeof(t1), &t1, NULL);
        double ms = (double)(t1 - t0) / 1e6;
        total_ms += ms;
        clReleaseEvent(evs[i]);
    }
    double avg_ms = total_ms / NUM_ITER;

    err = clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeC, C_gpu.data(), 0, NULL, NULL);
    checkErr(err, "clEnqueueReadBuffer");

    half max_abs = computeAbsoluteError(C_ref, C_gpu);

    printf("Kernel: %s | avg_time = %.3f ms | max_abs_err = %f",
           kernelName.c_str(), avg_ms, (float)max_abs);

    // cleanup
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufQB);
    clReleaseMemObject(bufSB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);

    return avg_ms;
}

int main()
{
    // A:m*k B:n*k C:n*m
    // A * B^T = C^T
    int m = 122753;
    int n = 1; // gemv, n值固定为1
    int k = 2304;
    float alpha = 0.5f;
    float beta = 0.0f;

    if (n != 1)
    {
        cerr << "Unsupported configuration: this GEMV variant expects n == 1" << endl;
        return 1;
    }
    if (k % 32 != 0)
    {
        cerr << "k must be multiple of 32 for Q8_0 block size 32" << endl;
        return 1;
    }

    // 生成随机数据
    vector<half> A = generateMatrix_F16(m, k);
    vector<half> B = generateMatrix_F16(n, k);
    vector<half> C_gpu(n * m);
    vector<half> C_ref(n * m);

    // 量化 A -> qchars + scales
    int blocks = k / Block_size;
    vector<BlockQ8_0> BlockA(m * blocks);
    quantv1(A, BlockA, m, k, blocks);

    // 量化激活 B -> qB(int8) + sB(half per-32)
    vector<int8_t> qB(k);
    vector<half> sB(blocks);
    quantB_q8_per32(B, qB, sB, k);

    // CPU 计算
    gemv(C_ref, A, B, m, n, k, alpha, beta);
    // gemv1(C_ref, BlockA, B, m, n, k, alpha, beta);

    // OpenCL 初始化
    cl_int err;
    cl_platform_id platform;
    cl_device_id device;

    err = clGetPlatformIDs(1, &platform, NULL);
    checkErr(err, "clGetPlatformIDs");
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    checkErr(err, "clGetDeviceIDs");

    // printDeviceInfo(device);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkErr(err, "clCreateContext");

    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &err);
    checkErr(err, "clCreateCommandQueueWithProperties");

    // Kernel test
    KernelTest("gemv_q8_base", context, device, queue,
               BlockA, qB, sB, C_gpu, C_ref, m, n, k, alpha, beta);
    // cleanup
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
