#include <iostream>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#define USECPSEC 1000000ULL

//generic timer function from https://stackoverflow.com/questions/69136940/timing-kernel-execution-with-cpu-timers
unsigned long long myCPUTimer(void)
{
	timeval tv;
	gettimeofday(&tv,0);
	return ((tv.tv_sec*USECPSEC) + tv.tv_usec);
}

//tutorial from https://developer.nvidia.com/blog/even-easier-introduction-cuda/
__global__ //need this to run as GPU kernel
void add(int n, float *x, float *y)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < n; i+=stride)
		y[i] = x[i] + y[i];
}

int main(void)
{
	unsigned long long t1 = myCPUTimer();
	int N = 1<<20; //total number of elements
	int blockSize =1024;
	int numBlocks = (N + blockSize - 1)/(blockSize); //number of blocks 
	std::cout << "total elements:" << N << std::endl;
	std::cout << "num blocks:" << numBlocks << std::endl;
	std::cout << "blocksize:" << blockSize << std::endl;
	//CPU allocation
	//float *x = new float[N];
	//float *y = new float[N];

	//GPU allocation
	float *x, *y;
	cudaMallocManaged(&x, N*sizeof(float));
	cudaMallocManaged(&y, N*sizeof(float));

	//initialize
	for (int i = 0; i<N; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	//run [CPU]
	//add(N,x,y);

	//pre-fetch arrays: ensures all data is transferred to GPU at once instead of being transfered on-demand when needed by kernel; minimizes host-to-device transfers
	cudaMemPrefetchAsync(x, N*sizeof(float), 0, 0);
	cudaMemPrefetchAsync(y, N*sizeof(float), 0, 0);

	//run [GPU]
	add<<<numBlocks, blockSize>>>(N,x,y);
	cudaDeviceSynchronize();

	float maxError = 0.0f;
	for (int i=0; i<N; i++)
	{
		maxError = fmax(maxError, fabs(y[i]-3.0f));
	}
	std::cout << "Max error: " << maxError << std::endl;

	//CPU free
	//delete [] x;
	//delete [] y;

	//GPU free
	cudaFree(x);
	cudaFree(y);
	unsigned long long t2 = myCPUTimer();
	unsigned long long tottime = (t2 - t1);
	float membw = (N*sizeof(float) +N*sizeof(float) +N*sizeof(float))/(tottime*1e3);
	float thruput = N/(tottime*1e3); 

	std::cout << "Total execution time (microseconds):" << tottime << std::endl;
	std::cout << "Effective memory bandwidth (GB/s):" << membw << std::endl;
	std::cout << "Effective throughput (GFLOPS):" << thruput << std::endl;
	return 0;
}
