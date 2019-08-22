#include <stdio.h>

#define N 64

__global__ void matmul( int *A, int *B, int *C ) {
    int val = 0;

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        for ( int k = 0; k < N; ++k ) {
            val += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = val;
    }
}

int main() {
    int *A, *B, *C;

    int size = N * N * sizeof (int);
    
    // Allocate memory
    cudaMallocManaged (&A, size);
    cudaMallocManaged (&B, size);
    cudaMallocManaged (&C, size);

    // Initialize memory
    for( int row = 0; row < N; ++row ) {
        for( int col = 0; col < N; ++col ) {
            A[row*N + col] = row;
            B[row*N + col] = col+2;
            C[row*N + col] = 0;
        }
    }

    dim3 threads_per_block (16, 16, 1); // A 16 x 16 x 1 block threads
    dim3 number_of_blocks ((N / threads_per_block.x) + 1, (N / threads_per_block.y) + 1, 1);

    matmul <<< number_of_blocks, threads_per_block >>> ( A, B, C );
    cudaDeviceSynchronize(); 
    
    // Check if we got it all correct
    bool error = false;
    for( int row = 0; row < N; ++row ) {
        for( int col = 0; col < N; ++col ) {
            int val = 0;
            for( int k = 0; k < N; ++k ) {
                val += A[row * N + k] * B[k * N + col];
            }
            if(C[row * N + col] != val) {
                error = true;
            }
        }
    }
    
    if(error) {
        printf("Incorrect result!");
    } else {
        printf("Success!");
    }
    
    // Free all our allocated memory
    cudaFree(A); cudaFree(B); cudaFree(C); 
}
