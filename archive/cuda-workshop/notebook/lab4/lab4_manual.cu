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
    int a[N], b[N], c[N]; 
    int *dev_a, *dev_b, *dev_c;

    // Initialize arrays a and b with data
    for (int i=0; i < N; i++) {
        a[i] = 2*i;
        b[i] = -i;
    }
    
    // TODO: Allocate memory on the GPU for dev_a, dev_b, dev_c
    // TODO: they all should be able store N elements of size int
    
    // TODO: Copy the data from "a" to dev_a and from "b" to dev_b
    // TODO: remember about the direction

    // Compute the number of block necessary based on a constant number of threads per block
    // Be careful - this can launch more threads than we need, we need to handle this in the kernel!
    int threadsPerBlock = 1024;
    int blocks = (int)ceil((float)N/threadsPerBlock);

    // Launch the kernel
    add<<<blocks,threadsPerBlock>>>(dev_a, dev_b, dev_c);
    
    // TODO: Move the result back from dev_c to "c"
    
    for (int i=0; i < N; i++) {
        printf( "%d + %d = %d\n", a[i], b[i], c[i] );
    }
 
    // TODO: remember to free all the memory you allocated.
    
    return 0;
}

