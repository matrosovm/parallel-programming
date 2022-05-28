#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 128

__kernel void sum(__global const unsigned int* as, __global unsigned int* res, unsigned int n)
{
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    __local unsigned int local_as[WORK_GROUP_SIZE];
    local_as[local_id] = global_id < n ? as[global_id] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (unsigned int i = WORK_GROUP_SIZE; i > 1; i /= 2) {
        if (local_id < i / 2) {
            local_as[local_id] += local_as[local_id + i / 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (local_id == 0) {
        atomic_add(res, local_as[local_id]);
    }
}