#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <iostream>

#define N 16

int main( int argc, char** argv ) {
    // Allocation of 3 vectors on the device
    thrust::device_vector<int> a(N);
    thrust::device_vector<int> b(N);
    thrust::device_vector<int> c(N);
    
    // Thrust allows us to assign values easily from host code
    for (int i=0; i < N; i++) {
        a[i] = 2*i;
        b[i] = -i;
    }
    
    thrust::transform(a.begin(), a.end(),
                      b.begin(),
                      c.begin(),
                      thrust::plus<int>());

    // Moving data from the device memory to the host memory is as simple as assigning device_vector to host_vector
    thrust::host_vector<int> a_host = a;
    thrust::host_vector<int> b_host = b;
    thrust::host_vector<int> c_host = c;
    
    for (int i=0; i < N; i++) {
        printf( "%d + %d = %d\n", a_host[i], b_host[i], c_host[i] );
    }
    
    return 0;
}