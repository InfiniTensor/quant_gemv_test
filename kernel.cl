#define CL_TARGET_OPENCL_VERSION 200
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef struct
{
    half d;
    char qs[32];
} BlockQ8_0;
// 激活值量化到低位：B 拆分为 qB(int8) + sB(half, 每32元素一块)
__kernel void gemv_q8_base(__global const BlockQ8_0 *A,
                           __global const char *qB,
                           __global const half *sB,
                           __global half *C,
                           int as, int ars, int acs, int bs, int brs, int bcs,
                           int cs, int crs, int ccs,
                           int M, int N, int K, float alpha, float beta)
{

    int row_id = get_global_id(0);
    // 允许 GWS 向上取整：越界线程直接返回
    if (row_id >= M)
        return;

    // 本地缓存当前块的 qB[32]，供同一工作组复用
    __local char l_qB[32];

    BlockQ8_0 valueA;
    float sum = 0.0f;

    for (int i = 0; i < K / 32; i++)
    {
        // 组内协作：加载 qB 的第 i 个 32 元素块到本地内存
        int lid = get_local_id(0);
        if (lid < 32)
        {
            l_qB[lid] = qB[(i * 32 + lid) * brs];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // 读取 A 的对应块
        valueA = *(A + row_id * ars + i * acs);

        // 使用 char4 向量分段做乘加，减少循环开销
        int acci = 0;
        #pragma unroll
        for (int t = 0; t < 8; ++t)
        {
            char4 aa = vload4(t, valueA.qs);
            char4 bb = vload4(t, l_qB);
            acci += (int)aa.s0 * (int)bb.s0;
            acci += (int)aa.s1 * (int)bb.s1;
            acci += (int)aa.s2 * (int)bb.s2;
            acci += (int)aa.s3 * (int)bb.s3;
        }

        // 缩放：权重块 d 与激活块 sB[i]
        half sb = sB[i];
    sum += (float)acci * (float)valueA.d * (float)sb;

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    __global half *p = C + row_id * crs;
    if (beta != 0)
        *p = (half)(beta * (*p) + alpha * sum);
    else
        *p = (half)(alpha * sum);
}

// 使用 image2D 存储权重的 SoA 版：A 的 int8 权重放入 RGBA 像素（每像素4个有符号int8）
__kernel void gemv_q8_soa_img(read_only image2d_t AqImg,
                              __global const half *Ad,
                              __global const char *qB,
                              __global const half *sB,
                              __global half *C,
                              int M, int K, float alpha, float beta)
{
    int row = get_global_id(0);
    if (row >= M) return;

    int blocks = K >> 5;   // K/32
    int baseX  = 0;        // 每个块占 8 个像素（8*4 = 32）
    __local char l_qB[32];
    float sum = 0.0f;

    for (int bi = 0; bi < blocks; ++bi)
    {
        int lid = get_local_id(0);
        if (lid < 32)
            l_qB[lid] = qB[bi * 32 + lid];
        barrier(CLK_LOCAL_MEM_FENCE);

        int x0 = baseX + bi * 8;
        int acci = 0;
        #pragma unroll
        for (int t = 0; t < 8; ++t)
        {
            int2 coord = (int2)(x0 + t, row);
            int4 px = read_imagei(AqImg, coord);
            char qb0 = l_qB[4 * t + 0];
            char qb1 = l_qB[4 * t + 1];
            char qb2 = l_qB[4 * t + 2];
            char qb3 = l_qB[4 * t + 3];
            acci += px.x * (int)qb0;
            acci += px.y * (int)qb1;
            acci += px.z * (int)qb2;
            acci += px.w * (int)qb3;
        }

        float d  = (float)Ad[row * blocks + bi];
        float sb = (float)sB[bi];
        sum += (float)acci * d * sb;

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    half oldv = C[row];
    C[row] = (beta != 0.0f) ? (half)(beta * (float)oldv + alpha * sum)
                            : (half)(alpha * sum);
}
