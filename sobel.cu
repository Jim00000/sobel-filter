#include <iostream>

__global__ void _sobel_process_kernel_(unsigned char* d_src, unsigned char* d_dst, int row, int col)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int src_center = idy * col + idx;
    if(idy >= row || idx >= col)
        return;
    int lidx = (idx <= 0)? 0 : idx - 1;
    int ridx = (idx >= col - 1)? col - 1 : idx + 1;
    int uidy = (idy <= 0)? 0 : idy - 1;
    int didy = (idy >= row - 1)? row - 1 : idy + 1;
    int src_left = idy * col + lidx;
    int src_right = idy * col + ridx;
    int src_up = uidy * col + idx;
    int src_up_left = uidy * col + lidx; 
    int src_up_right = uidy * col + ridx;
    int src_down = didy * col + idx;
    int src_down_left = didy * col + lidx;
    int src_down_right = didy * col + ridx;
    float src_left_r = d_src[src_left];
    float src_right_r = d_src[src_right];
    float src_up_r = d_src[src_up];
    float src_up_left_r = d_src[src_up_left];
    float src_up_right_r = d_src[src_up_right];
    float src_down_r = d_src[src_down];
    float src_down_left_r = d_src[src_down_left];
    float src_down_right_r = d_src[src_down_right];
    float GX = 1 * src_up_right_r + 2 * src_right_r + 1 * src_down_right_r - 1 * src_up_left_r - 2 * src_left_r - 1 * src_down_left_r;
    float GY = 1 * src_up_left_r + 2 * src_up_r + 1 * src_up_right_r - 1 * src_down_left_r - 2 * src_down_r - 1 * src_down_right_r;
    float G = sqrt(pow(GX, 2) + pow(GY, 2));
    unsigned char gray = static_cast<unsigned char>(G);
    d_dst[src_center] = gray;
}

__global__ void _split_channel_kernel_(unsigned char* d_src, unsigned char* d_r, unsigned char* d_g, unsigned char* d_b, int row, int col)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int id = idy * col + idx;
    if(id >= row * col)
        return;
    d_r[id] = d_src[id * 3 + 0];
    d_g[id] = d_src[id * 3 + 1];
    d_b[id] = d_src[id * 3 + 2];
}

extern void _sobel_process_(unsigned char* src, unsigned char* dst, int row, int col)
{
    unsigned char* d_src = nullptr;
    unsigned char* d_dst = nullptr;
    const size_t ARRAY_BYTES = row * col * sizeof(unsigned char);
    cudaMalloc((void**) &d_src, ARRAY_BYTES);
    cudaMalloc((void**) &d_dst, ARRAY_BYTES);
    cudaMemcpy(d_src, src, ARRAY_BYTES, cudaMemcpyHostToDevice);
    dim3 threads(32, 32);
    dim3 blocks(col / threads.x + 1, row / threads.y + 1);
    _sobel_process_kernel_<<<blocks, threads>>>(d_src, d_dst, row, col);
    cudaDeviceSynchronize();
    cudaMemcpy(dst, d_dst, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_dst);
}

extern void _split_channel_(unsigned char* src, unsigned char* r, unsigned char* g, unsigned char* b, int row, int col)
{
    unsigned char* d_src = nullptr;
    unsigned char* d_r = nullptr;
    unsigned char* d_g = nullptr;
    unsigned char* d_b = nullptr;
    const size_t ARRAY_BYTES = row * col * sizeof(unsigned char);
    cudaMalloc((void**) &d_src, ARRAY_BYTES * 3);
    cudaMalloc((void**) &d_r, ARRAY_BYTES);
    cudaMalloc((void**) &d_g, ARRAY_BYTES);
    cudaMalloc((void**) &d_b, ARRAY_BYTES);
    cudaMemcpy(d_src, src, ARRAY_BYTES * 3, cudaMemcpyHostToDevice);
    dim3 threads(1, 1);
    dim3 blocks(col / threads.x + 1, row / threads.y + 1);
    _split_channel_kernel_<<<blocks, threads>>>(d_src, d_r, d_g, d_b, row, col);
    cudaDeviceSynchronize();
    cudaMemcpy(r, d_r, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(g, d_g, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, d_b, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    cudaFree(d_src);
    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
}