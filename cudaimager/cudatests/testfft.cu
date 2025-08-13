#include <iostream>
#include <complex>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <cufftw.h>
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

//__global__ 
void myfft(int n, int nout, double *input, double *output, int index, int stride)
{
	//int index = blockIdx.x * blockDim.x + threadIdx.x;
	int outindex = (index/n)*nout;
	//int stride = blockDim.x * gridDim.x;
	double pi = 22.0/7.0;
	//printf("before loop %d %d %d \n",n,index,stride);
	for (int i = 0; i < nout; i += stride)//index/2; i < (index/2)+nout; i+=stride)
	{
		//printf(">>%d %d \n",i,nout);
		for (int j = 0; j < n; j += stride)//index; j < index+n; j +=stride)
		{
			output[2*(outindex + i)] += input[index + j]*cos(2*pi*i*j/n);
			output[2*(outindex + i) + 1] -= input[index + j]*sin(2*pi*i*j/n);

			//printf("%d %d %f\n",i,j,input[j]*cos(2*pi*(i-index)*(j-index)/n));
			//output[2*i] += input[j]*cos(2*pi*((i/stride)-index/2 )*((j/stride)-index)/n);
			//output[2*i + 1] -= input[j]*sin(2*pi*((i/stride)-index/2)*((j/stride)-index)/n);
		}
	}
	//printf("after loop");
}

int main(void)
{
	unsigned long long t1 = myCPUTimer();
	
	//GPU allocation -- 175x175x25x16, want 1D FFT along time axis
	int gridsize = 175;
	int nsamps = 25;
	int nsamps_out = nsamps/2;
	nsamps_out += 1;
	int nchans = 16;

	int Ntotal = gridsize*gridsize*nsamps*nchans;
	int Nout = gridsize*gridsize*nsamps_out*nchans;
	int Nout_ri = Nout*2;
	cufftDoubleReal *in_image;
	cufftDoubleComplex *out_image;
	double *comp_image;
	double *incomp_image;

	cudaMallocManaged(&in_image, Ntotal*sizeof(cufftDoubleReal));
	cudaMallocManaged(&out_image, Nout*sizeof(cufftDoubleComplex));
	cudaMallocManaged(&comp_image, Nout_ri*sizeof(double));
	cudaMallocManaged(&incomp_image, Ntotal*sizeof(double));

	std::cout << "gridsize: " << gridsize << std::endl;
	std::cout << "nsamps: " << nsamps << std::endl;
	std::cout << "nchans: " << nchans << std::endl;
	std::cout << "Ntotal: " << Ntotal << std::endl;
	std::cout << "Nout: " << Nout << std::endl;

	//initialize
	for (int i = 0; i<Ntotal; i++) {
		in_image[i] = 1.0f;
		incomp_image[i] = 1.0f;
	}
	for (int i = 0; i<Nout; i++) {
		out_image[i].x =0.0f;
		out_image[i].y = 0.0f;
		comp_image[2*i] = 0.0f;
		comp_image[2*i] = 0.0f;
	}
	cudaMemPrefetchAsync(in_image,Ntotal*sizeof(cufftDoubleReal),0,0);
       	cudaMemPrefetchAsync(out_image,Nout*sizeof(cufftDoubleComplex),0,0);
        //cudaMemPrefetchAsync(incomp_image,Ntotal*sizeof(cufftDoubleReal),0,0);
        //cudaMemPrefetchAsync(comp_image,Nout*sizeof(cufftDoubleComplex),0,0);

	//create simple plan for 1D FFT    
	cufftHandle plan;
	int batchsize = gridsize*gridsize*nchans;

	//data layout options
	/*
	int *inembed;
	int *oembed;
	cudaMallocManaged(&inembed, sizeof(int));
	cudaMallocManaged(&oembed, sizeof(int));
	inembed[0] = Ntotal;
	oembed[0] = Nout;
	*/
	int inembed = Ntotal;
	int oembed = Nout;
	int istride = 1;
	int idist = nsamps;
	int ostride = 1;
	int odist = nsamps_out;

	cufftPlanMany(&plan, 1, &nsamps, &inembed, istride, idist,
        	&oembed, ostride, odist, CUFFT_D2Z, batchsize);
	

	//we want to use the double-precision real-to-complex version
	cufftResult res = cufftExecD2Z(plan,in_image,out_image);
	cudaDeviceSynchronize();
	cufftDestroy(plan);

	printf("RESULT:%d\n",res);
	//for comparison
	//std::cout << "num threads: " << gridsize*gridsize << std::endl;
	//myfft<<<1,gridsize*gridsize>>>(nsamps,incomp_image,comp_image);
	for (int i = 0; i < gridsize*gridsize*nchans; i += 1)
	{
		//printf(">%d\n",i);
		myfft(nsamps,nsamps_out,incomp_image,comp_image,i*nsamps,1);
	}

	//release plan
	//cudaDeviceSynchronize();

	
	
	float maxError_r = 0.0f;
	float maxError_i = 0.0f;
	double pi = 22.0/7.0;
	std::cout << "test " << cos(2*pi/Ntotal) << std::endl;
	int starti = nsamps_out*nchans;
	for (int i=starti; i<starti+nsamps_out*nchans; i += 1)
	{
		std::cout << "real,myfft: " << comp_image[2*i] << std::endl;
		std::cout << "real,cufft: " << out_image[i].x << std::endl;
		maxError_r = fmax(maxError_r,fabs(out_image[i].x - comp_image[2*i]));
		maxError_i = fmax(maxError_i,fabs(out_image[i].y - comp_image[2*i + 1]));
	}
	/*
	for (int i=0; i<nsamps_out*nchans; i += 1)
	{
		std::cout << "real,myfft: " << comp_image[2*i*gridsize*gridsize] << std::endl;
		std::cout << "real,cufft: " << out_image[i*gridsize*gridsize].x << std::endl;
		//std::cout << "imag: " << comp_image[2*i + 1] << std::endl;
		maxError_r = fmax(maxError_r, fabs(out_image[i*gridsize*gridsize].x-comp_image[2*i*gridsize*gridsize]));
		maxError_i = fmax(maxError_i, fabs(out_image[i*gridsize*gridsize].y-comp_image[2*i*gridsize*gridsize + 1]));
	}
	*/
	std::cout << "Max error (real): " << maxError_r << std::endl;
	std::cout << "Max error (imag): " << maxError_i << std::endl;

	

	//GPU free
	cudaFree(in_image);
	cudaFree(incomp_image);
	cudaFree(out_image);
	cudaFree(comp_image);
	unsigned long long t2 = myCPUTimer();
	unsigned long long tottime = (t2 - t1);
	float membw = (Ntotal*sizeof(cufftDoubleReal) +Nout*sizeof(cufftDoubleComplex))/(tottime*1e3);
	float thruput = Ntotal/(tottime*1e3); 

	std::cout << "Total execution time (microseconds):" << tottime << std::endl;
	std::cout << "Effective memory bandwidth (GB/s):" << membw << std::endl;
	std::cout << "Effective throughput (GFLOPS):" << thruput << std::endl;
	return 0;
}
