#include <stdio.h>
#include <math.h>

#define N 16

__global__ void add(int* a, int* b, int* c) {
    int localIdx = blockIdx.x*blockDim.x + threadIdx.x;
    
    if( localIdx < N ) {
        c[localIdx] = a[localIdx] + b[localIdx];
    }
}

int main( int argc, char** argv ) {
    int *a, *b, *c; 
    
    cudaMallocManaged( (void**)&a, N * sizeof(int) );
    cudaMallocManaged( (void**)&b, N * sizeof(int) );
    cudaMallocManaged( (void**)&c, N * sizeof(int) );
    
    // Initialize arrays a and b with data
    for (int i=0; i < N; i++) {
        a[i] = 2*i;
        b[i] = -i;
    }
    
    // Compute the number of block necessary based on a constant number of threads per block
    // Be careful - this can launch more threads than we need, we need to handle this in the kernel!
    int threadsPerBlock = 1024;
    int blocks = (int)ceil((float)N/threadsPerBlock);

    // Launch the kernel
    add<<<blocks,threadsPerBlock>>>(a, b, c);
    
    cudaDeviceSynchronize();
    
    for (int i=0; i < N; i++) {
        printf( "%d + %d = %d\n", a[i], b[i], c[i] );
    }
 
    cudaFree( a );
    cudaFree( b );
    cudaFree( c );
    
    return 0;
}

