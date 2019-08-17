#include <stdio.h>

#define N 64

#define TILE_DIM 16

__global__ void matmul(int *A, int *B, int *C) {
    __shared__ int Asub[TILE_DIM][TILE_DIM];
    __shared__ int Bsub[TILE_DIM][TILE_DIM];
    
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
          
    int sum = 0;
    
    for (int k = 0; k < (TILE_DIM + N - 1)/TILE_DIM; k++) {
        if (k*TILE_DIM + threadIdx.x < N && row < N) {
            Asub[threadIdx.y][threadIdx.x] = A[row*N + k*TILE_DIM + threadIdx.x];
        } else {
            Asub[threadIdx.y][threadIdx.x] = 0.0;
        }

        if (k*TILE_DIM + threadIdx.y < N && col < N) {
            Bsub[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*N + col];
        } else {
            Bsub[threadIdx.y][threadIdx.x] = 0.0;
        }

        __syncthreads();

        for (int n = 0; n < TILE_DIM; ++n) {
            sum += Asub[threadIdx.y][n] * Bsub[n][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[((blockIdx.y * blockDim.y + threadIdx.y)*N)+(blockIdx.x*blockDim.x)+threadIdx.x] = sum;
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
