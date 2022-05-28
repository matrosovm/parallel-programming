#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define WORK_GROUP_SIZE 16

__kernel void matrix_multiplication(__global const float* a,
                                   __global        float* b,
                                   __global        float* c,
                                   unsigned int M,
                                   unsigned int K,
                                   unsigned int N)
{
    const unsigned int global_x = get_global_id(0);
    const unsigned int global_y = get_global_id(1);
    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);
    if (global_x >= M || global_y >= K)
        return;

    __local float buffer_a[WORK_GROUP_SIZE * WORK_GROUP_SIZE];
    __local float buffer_b[WORK_GROUP_SIZE * WORK_GROUP_SIZE];

    float sum = 0.0f;
    for (unsigned int k = 0; k * WORK_GROUP_SIZE < K; ++k) {
        buffer_a[local_y * WORK_GROUP_SIZE + local_x] = k * WORK_GROUP_SIZE + local_x < K ?
            a[global_y * K + (k * WORK_GROUP_SIZE + local_x)] : 0.0f;

        buffer_b[local_y * WORK_GROUP_SIZE + local_x] = k * WORK_GROUP_SIZE + local_y < K ?
            b[(k * WORK_GROUP_SIZE + local_y) * N + global_x] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);
        for (unsigned int l = 0; l < WORK_GROUP_SIZE; ++l) {
            sum += buffer_a[local_y * WORK_GROUP_SIZE + l] * buffer_b[l * WORK_GROUP_SIZE + local_x];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[global_y * N + global_x] = sum;
}