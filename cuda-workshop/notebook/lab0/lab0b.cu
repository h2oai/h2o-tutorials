#include <stdio.h>

__global__ void hello() {
    printf("Hello, CUDA! Thread [%d] in block [%d]\n", threadIdx.x, blockIdx.x);
}

int main( int argc, char** argv ) {
    hello<<<1,1>>>(); // asynchronous call!
    cudaDeviceSynchronize(); // wait for all operations on the GPU to finish
    return 0;
}
