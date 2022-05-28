#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void merge(__global const float* as, __global float* out, 
                    const uint size, const uint n) {
    const uint id = get_global_id(0);
    if (id >= size) {
        return;
    }
    uint left_b = id - id % (2 * n);
    uint left_e = left_b + n;
    uint right_b = left_e;
    uint right_e = right_b + n;

    uint l = id >= right_b ? left_b : right_b;
    uint r = l == left_b ? left_e : right_e;

    const float el = as[id];
    while (l < r) {
        uint mid = (l + r) / 2;
        if (as[mid] < el) {
            l = mid + 1;
        } else if (as[mid] == el && id >= right_b) {
            l = mid + 1;
        } else {
            r = mid;
        } 
    }
    out[l + id - right_b] = el;
}