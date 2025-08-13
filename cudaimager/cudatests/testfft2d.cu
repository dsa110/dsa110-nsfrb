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

void my2dfft(int n, int nout, double *input, double *output, int index, int stride)
{
	double pi = 22.0/7.0;
	int outindex = (index/n)*nout;
	for (int i =0; i<nout; i += stride)
	{
		for (int ii =0; ii<nout; ii += stride)
		{
			for (int j = 0; j<n; j += stride)
			{
				for (int jj = 0; jj<n; jj+= stride)
				{
					output[outindex+i*n + ii] += input[2*(index + j*n + jj)]*cos(2*pi*((i-nout/2)*(j-n/2) + (ii-nout/2)*(jj-n/2))/n) + input[2*(index + j*n + jj) + 1]*sin(2*pi*((i-nout/2)*(j-n/2) + (ii-nout/2)*(jj-n/2))/n);
				}

			}
		}
	}
}

__global__
void my2difftshift(int n, int batchsize, cufftDoubleReal *input, cufftDoubleReal *output, int trim)
{
	int shiftby = n/2;
	int jnew = 0;
	int knew = 0;
	for (int i = 0; i<batchsize; i+=1) 
	{
		for (int j = trim; j<trim+n; j +=1)
		{
			for (int k = trim; k<trim+n; k+=1)
			{
				jnew = (j+shiftby-trim)%n;
				knew = (k+shiftby-trim)%n;
				output[i*n*n + jnew*n + knew] = input[i*n*n + j*n + k];
			}
		}
	}
}

__global__
void my2difftshift_complex(int n, int batchsize, cufftDoubleComplex *input, cufftDoubleComplex *output, int trim)
{
        int shiftby = n/2;
        int jnew = 0;
        int knew = 0;
        for (int i = 0; i<batchsize; i+=1)
        {
                for (int j = trim; j<trim+n; j +=1)
                {
                        for (int k = trim; k<trim+n; k+=1)
                        {
                                jnew = (j+shiftby-trim)%n;
                                knew = (k+shiftby-trim)%n;
                                output[i*n*n + jnew*n + knew] = input[i*n*n + j*n + k];
                        }
                }
        }
}


int main(void)
{
	unsigned long long t1 = myCPUTimer();
	

	//GPU allocation -- complex 25x16x175x175, want 2D FFT along spatial axes

	int gridsize = 175;
	int gridsize_out = (gridsize - 1)*2; 
	int nsamps = 25;
	int nchans = 16;

	int Ntotal = gridsize*gridsize*nsamps*nchans;
	int Nout = gridsize_out*gridsize_out*nsamps*nchans;
	cufftDoubleComplex *in_image;
	cufftDoubleComplex *in_image_shift;
	cufftDoubleReal *out_image;
	cufftDoubleReal *out_image_shift;
	double *comp_image;
	double *incomp_image;

	cudaMallocManaged(&in_image, Ntotal*sizeof(cufftDoubleComplex));
	cudaMallocManaged(&in_image_shift, Ntotal*sizeof(cufftDoubleComplex));
	cudaMallocManaged(&out_image, Nout*sizeof(cufftDoubleReal));
	cudaMallocManaged(&out_image_shift, Ntotal*sizeof(cufftDoubleReal));
	cudaMallocManaged(&comp_image, Nout*sizeof(double));
	cudaMallocManaged(&incomp_image, Ntotal*2*sizeof(double));

	std::cout << "gridsize: " << gridsize << std::endl;
	std::cout << "nsamps: " << nsamps << std::endl;
	std::cout << "nchans: " << nchans << std::endl;
	std::cout << "Ntotal: " << Ntotal << std::endl;
	std::cout << "Nout: " << Nout << std::endl;

	//initialize
	for (int i = 0; i<Ntotal; i++) {
		in_image[i].x = 1.0f;
		in_image[i].y = 1.0f;
		in_image_shift[i].x = 1.0f;
                in_image_shift[i].y = 1.0f;
		incomp_image[2*i] = 1.0f;
		incomp_image[2*i + 1] = 1.0f;
		out_image_shift[i] = 0.0f;
	}
	for (int i = 0; i<Nout; i++) {
		out_image[i] =0.0f;
		comp_image[i] = 0.0f;
	}
	cudaMemPrefetchAsync(in_image,Ntotal*sizeof(cufftDoubleComplex),0,0);
       	cudaMemPrefetchAsync(out_image,Nout*sizeof(cufftDoubleReal),0,0);
	cudaMemPrefetchAsync(in_image_shift,Ntotal*sizeof(cufftDoubleComplex),0,0);
        cudaMemPrefetchAsync(out_image_shift,Ntotal*sizeof(cufftDoubleReal),0,0);


        //cudaMemPrefetchAsync(incomp_image,Ntotal*sizeof(cufftDoubleReal),0,0);
        //cudaMemPrefetchAsync(comp_image,Nout*sizeof(cufftDoubleComplex),0,0);

	//create simple plan for 1D FFT    
	cufftHandle plan;
	int batchsize = nchans*nsamps;

	//data layout options
	/*
	int *inembed;
	int *oembed;
	cudaMallocManaged(&inembed, sizeof(int));
	cudaMallocManaged(&oembed, sizeof(int));
	inembed[0] = Ntotal;
	oembed[0] = Nout;
	*/
	int inembed[2]; 
	inembed[0] = gridsize*gridsize;//Ntotal;
	inembed[1] = gridsize;
	int oembed[2];
       	oembed[0] = gridsize_out*gridsize_out; //Nout;
	oembed[1] = gridsize_out;
	int istride = 1;
	int idist = gridsize*gridsize;
	int ostride = 1;
	int odist = gridsize_out*gridsize_out;
	int narr[2];
	narr[0] = gridsize;
	narr[1] = gridsize;

	cufftPlanMany(&plan, 2, narr, inembed, istride, idist,
        	oembed, ostride, odist, CUFFT_Z2D, batchsize);
	

	//we want to use the double-precision real-to-complex version
	my2difftshift_complex<<<1, batchsize>>>(gridsize,batchsize,in_image,in_image_shift,0);
	cufftResult res = cufftExecZ2D(plan,in_image_shift,out_image);
	my2difftshift<<<1, batchsize>>>(gridsize,batchsize,out_image,out_image_shift,0);
	cudaDeviceSynchronize();
	cufftDestroy(plan);

	printf("RESULT:%d\n",res);
	//for comparison
	/*for (int i = 0; i < nsamps*nchans; i += 1)
	{
		//printf(">%d\n",i);
		my2dfft(gridsize,gridsize_out,incomp_image,comp_image,i*gridsize*gridsize,1);
	}*/

	//release plan
	//cudaDeviceSynchronize();

	
	
	float maxError_r = 0.0f;
	float maxError_i = 0.0f;
	int starti = 0;
	for (int i=starti; i<starti+gridsize*gridsize; i += 1)
	{
		//std::cout << "real,myfft: " << comp_image[i] << std::endl;
		std::cout << "INDEX: " << i <<std::endl;
		std::cout << "real,cufft: " << out_image_shift[i] << std::endl;
		maxError_r = fmax(maxError_r,fabs(out_image_shift[i] - comp_image[i]));
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
	cudaFree(in_image_shift);
	cudaFree(incomp_image);
	cudaFree(out_image);
	cudaFree(out_image_shift);
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
