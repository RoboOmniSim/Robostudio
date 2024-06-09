#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Structure to hold a 3D point
struct Point3D {
    float x, y, z;
};

// CUDA kernel to compute the barycentric coordinates
__global__ void computeBarycentricCoordinates(Point3D point, Point3D* tetrahedron, float* barycentricCoords) {
    __shared__ Point3D shared_tetrahedron[4];

    // Load tetrahedron vertices into shared memory
    if (threadIdx.x < 4) {
        shared_tetrahedron[threadIdx.x] = tetrahedron[threadIdx.x];
    }
    __syncthreads();

    // Tetrahedron vertices
    Point3D p0 = shared_tetrahedron[0];
    Point3D p1 = shared_tetrahedron[1];
    Point3D p2 = shared_tetrahedron[2];
    Point3D p3 = shared_tetrahedron[3];

    // Compute vectors
    float v0x = p1.x - p0.x, v0y = p1.y - p0.y, v0z = p1.z - p0.z;
    float v1x = p2.x - p0.x, v1y = p2.y - p0.y, v1z = p2.z - p0.z;
    float v2x = p3.x - p0.x, v2y = p3.y - p0.y, v2z = p3.z - p0.z;
    float vpx = point.x - p0.x, vpy = point.y - p0.y, vpz = point.z - p0.z;

    // Compute dot products
    float d00 = v0x * v0x + v0y * v0y + v0z * v0z;
    float d01 = v0x * v1x + v0y * v1y + v0z * v1z;
    float d02 = v0x * v2x + v0y * v2y + v0z * v2z;
    float d11 = v1x * v1x + v1y * v1y + v1z * v1z;
    float d12 = v1x * v2x + v1y * v2y + v1z * v2z;
    float dp0 = vpx * v0x + vpy * v0y + vpz * v0z;
    float dp1 = vpx * v1x + vpy * v1y + vpz * v1z;
    float dp2 = vpx * v2x + vpy * v2y + vpz * v2z;

    // Compute the denominator of the barycentric coordinates
    float denom = d00 * (d11 * d12 - d01 * d12) - d01 * (d01 * d12 - d02 * d11) + d02 * (d01 * d11 - d02 * d11);

    // Compute barycentric coordinates
    float u = (dp0 * (d11 * d12 - d01 * d12) - dp1 * (d01 * d12 - d02 * d11) + dp2 * (d01 * d11 - d02 * d11)) / denom;
    float v = (d00 * (dp1 * d12 - dp2 * d11) - d01 * (dp0 * d12 - dp2 * d01) + d02 * (dp0 * d11 - dp1 * d01)) / denom;
    float w = (d00 * (d11 * dp2 - dp1 * d12) - d01 * (d01 * dp2 - dp0 * d12) + d02 * (d01 * dp1 - dp0 * d11)) / denom;
    float t = 1.0f - u - v - w;

    // Store the result in the output array
    barycentricCoords[0] = u;
    barycentricCoords[1] = v;
    barycentricCoords[2] = w;
    barycentricCoords[3] = t;
}

extern "C" void computeBarycentric(Point3D point, Point3D* tetrahedron, float* barycentricCoords) {
    // Allocate device memory
    Point3D* d_tetrahedron;
    float* d_barycentricCoords;
    cudaMalloc((void**)&d_tetrahedron, 4 * sizeof(Point3D));
    cudaMalloc((void**)&d_barycentricCoords, 4 * sizeof(float));

    // Copy data to the device
    cudaMemcpy(d_tetrahedron, tetrahedron, 4 * sizeof(Point3D), cudaMemcpyHostToDevice);

    // Launch the kernel
    computeBarycentricCoordinates<<<1, 1>>>(point, d_tetrahedron, d_barycentricCoords);

    // Copy the result back to the host
    cudaMemcpy(barycentricCoords, d_barycentricCoords, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_tetrahedron);
    cudaFree(d_barycentricCoords);
}