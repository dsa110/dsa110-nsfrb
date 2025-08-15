#include <iostream>
#include "cuda_imager.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <fstream>
#include <stdlib.h>
#include <cstring>
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
	int fidx = ((blockIdx.x * blockDim.x) + threadIdx.x) * batchsize;

	int shiftby = n/2;
	int jnew = 0;
	int knew = 0;
	for (int i = fidx; i<fidx+batchsize; i+=1) 
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
	int fidx = ((blockIdx.x * blockDim.x) + threadIdx.x) * batchsize;
	
	int shiftby = n/2;
        int jnew = 0;
        int knew = 0;
        for (int i = fidx; i<fidx+batchsize; i+=1)
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

void my2difftshift_quad(int gridsize, int gridsize_out, int num_grids, cufftDoubleReal *image, cufftDoubleReal *image_shift)
{
	/*
	   Implements inverse fftshift on padded image from cufft in place
	*/
	unsigned long id1 = 0;
	unsigned long id2 = 0;
	printf("GRIDSIZE %d %d\n",gridsize,gridsize_out);
	int shiftby = gridsize/2;
	for (int g =0; g<num_grids; g++)
	{
		//first move from first to fourth quadrant
		for (int i=0; i<gridsize;i++)
		{
			for (int j=0; j<gridsize;j++)
			{
				id1 = (g*gridsize*gridsize) + (((i+shiftby)%gridsize)*gridsize + (j+shiftby)%gridsize);
				id2 = (g*gridsize_out*gridsize_out) + (i*gridsize_out + j);

				image_shift[id1] = image[id2];
			}
		}
	
		
	}





}
//Briggs weighted gridding of visibilities
__global__
void uniform_grid_singlefreq(double *U, double *V, double *W, long nbase, long nbase_thread,double *fobs_GHz, int num_chans, int num_chans_per_node, int gridsize, double uv_max, double grid_res, float robust, unsigned int *i_indices, unsigned int *j_indices, unsigned int *i_conj_indices, unsigned int *j_conj_indices)
{
	/*
	   This function uniformly grids visibilities for the given observing frequency
	*/
	int fidx = blockIdx.x * nbase;
	int bidx = threadIdx.x * nbase_thread;
	double lambda_m = C_GHZ_M/fobs_GHz[blockIdx.x];
	int shiftby = gridsize/2;

	for (int i = bidx; i < bidx+nbase_thread; i+=1)
	{
		i_indices[fidx + i] = ((U[i]/lambda_m) + uv_max)/grid_res;
		j_indices[fidx + i] = ((V[i]/lambda_m) + uv_max)/grid_res;
		i_conj_indices[fidx + i] = gridsize - i_indices[fidx + i] - 1;
		j_conj_indices[fidx + i] = gridsize - j_indices[fidx + i] - 1;

		i_indices[fidx + i] = (i_indices[fidx + i] + shiftby)%gridsize;
		j_indices[fidx + i] = (j_indices[fidx + i] + shiftby)%gridsize;
		i_conj_indices[fidx + i] = (i_conj_indices[fidx + i] + shiftby)%gridsize;
                j_conj_indices[fidx + i] = (j_conj_indices[fidx + i] + shiftby)%gridsize;
	}

}

__global__
void briggs_weight_singlefreq(long nbase, int gridsize, unsigned int *i_indices, unsigned int *j_indices, double *bweights, float robust, unsigned int *Wk, double *vis_weights, int *flagbase, int *flagchans, int *flagcorrs, int *flagants, int nflagbase, int nflagchans, int nflagcorrs, int nflagants, uint8_t *ANT1, uint8_t *ANT2)
{
	/*
	   This function computes grid weights using Briggs robustness parameter
	*/
	int fidx = threadIdx.x * nbase;
	int Widx = threadIdx.x * (gridsize*gridsize);
	unsigned int min_index_i = 0;
	unsigned int min_index_j = 0;
	unsigned int min_index = 0;
	int gridpointidx = 0;
	double vis_weight_sum = 0;

	// get the minimum grid index value
	bool flagflag =0;
	for (int i = 0; i < nbase; i+=1)
	{
		for (int k_f=0; k_f<nflagbase; k_f++)
		{
			if (i==flagbase[k_f])
			{
				flagflag=1;
			}
		}
		if (flagflag==1)
		{
			flagflag=0;
			continue;
		}
		for (int k_f=0; k_f<nflagants; k_f++)
		{
			if (ANT1[i]==flagants[k_f] || ANT2[i]==flagants[k_f])
			{
				flagflag=1;
			}
		}
                if (flagflag==1)
                {
                        flagflag=0;
                        continue;
                }

				
		if (i_indices[i] < min_index_i) {min_index_i = i_indices[i];}
		if (j_indices[i] < min_index_j) {min_index_j = j_indices[i];}
	}
	min_index = min_index_i*gridsize + min_index_j;

	// count the number of occurences of each index; get sums for counts and vis weights
	for (int i = 0; i < nbase; i+=1)
	{
		for (int k_f=0; k_f<nflagbase; k_f++)
                {
                        if (i==flagbase[k_f])
                        {
                                flagflag=1;
                        }
                }
                if (flagflag==1)
                {
                        flagflag=0;
                        continue;
                }
                for (int k_f=0; k_f<nflagants; k_f++)
                {
                        if (ANT1[i]==flagants[k_f] || ANT2[i]==flagants[k_f])
                        {
                                flagflag=1;
                        }
                }
                if (flagflag==1)
                {
                        flagflag=0;
                        continue;
                }

		gridpointidx = i_indices[fidx + i]*gridsize + j_indices[fidx + i] - min_index;
		Wk[Widx + gridpointidx] += 1;
		vis_weight_sum += vis_weights[fidx + i];
	}
	double Wk2_sum = 0;
	for (int j = 0; j < gridsize*gridsize; j+=1)
	{
		Wk2_sum += Wk[Widx + j]*Wk[Widx + j];
	}

	// compute weighting factor
	double f2 = pow((5 * pow(10,-robust)),2)/(Wk2_sum / vis_weight_sum);

	// compute weights
	double bweights_sum =0;
	for (int i = 0; i < nbase; i+=1)
	{
		for (int k_f=0; k_f<nflagbase; k_f++)
                {
                        if (i==flagbase[k_f])
                        {
                                flagflag=1;
                        }
                }
                if (flagflag==1)
                {
                        flagflag=0;
                        continue;
                }
                for (int k_f=0; k_f<nflagants; k_f++)
                {
                        if (ANT1[i]==flagants[k_f] || ANT2[i]==flagants[k_f])
                        {
                                flagflag=1;
                        }
                }
                if (flagflag==1)
                {
                        flagflag=0;
                        continue;
                }

		gridpointidx = i_indices[fidx + i]*gridsize + j_indices[fidx + i] - min_index;
		bweights[fidx + i] = vis_weights[fidx + i] / (1 + Wk[Widx + gridpointidx]*f2);
		bweights_sum += bweights[fidx + i];
	}

	// normalize weights
	for (int i = 0; i < nbase; i+=1)
	{
		for (int k_f=0; k_f<nflagbase; k_f++)
                {
                        if (i==flagbase[k_f])
                        {
                                flagflag=1;
                        }
                }
                if (flagflag==1)
                {
                        flagflag=0;
                        continue;
                }
                for (int k_f=0; k_f<nflagants; k_f++)
                {
                        if (ANT1[i]==flagants[k_f] || ANT2[i]==flagants[k_f])
                        {
                                flagflag=1;
                        }
                }
                if (flagflag==1)
                {
                        flagflag=0;
                        continue;
                }

		bweights[fidx + i] /= bweights_sum;
	}


}


//__global__
void grid_data_CPU(int num_chans, int num_time_samples, int num_chans_per_node, int nbase, int nbase_thread,
		cuComplex *data, unsigned int *i_indices, unsigned int *j_indices, 
		unsigned int *i_conj_indices, unsigned int *j_conj_indices,
	       	double *bweights, int gridsize, cuDoubleComplex *vis_grid, int tmpblockIdx, int tmpthreadIdx)
{	
	/*
	   This function applies pre-computed weights to and grids visibilities.
	   Note the grid should be arranged time x channel x U x V to properly
	   interface with cufft.
		
	   Input data is arranged subband x time x baseline x channel x polarization
	   Input weights are arranged channel x baseline
	 */
	//indices for weights
	int fidx = tmpblockIdx * nbase; //blockIdx.x * nbase;
        int bidx = tmpthreadIdx * nbase_thread; //threadIdx.x * nbase_thread;

	//indices for data
	int sidx_data = tmpblockIdx/num_chans_per_node; //blockIdx.x/num_chans_per_node;
	sidx_data *= (num_time_samples * nbase * num_chans_per_node * 2);
	int fidx_data = (tmpblockIdx % num_chans_per_node)*2; //(blockIdx.x % num_chans_per_node)*2; //channels
	int bidx_data = tmpthreadIdx * nbase_thread * num_chans_per_node * 2; //threadIdx.x * nbase_thread * num_chans_per_node * 2; //baseline

	//indices for grid
	int fidx_grid = tmpblockIdx * gridsize * gridsize; //blockIdx.x * gridsize * gridsize;

	int data_index = 0;
	int weight_index = 0;
	int grid_index = 0;
	int grid_conj_index =0 ;
	for (int k=0; k<nbase_thread; k++)
	{
		//weight/i/j index
		weight_index = fidx + bidx + k;
		
		printf(">weight index: %d\n",weight_index);
		for (int kt=0; kt<num_time_samples; kt++)
		{
			//grid index
                        grid_index = (kt*num_chans_per_node*gridsize*gridsize) + fidx_grid + (i_indices[weight_index]*gridsize + j_indices[weight_index]);
                        grid_conj_index = (kt*num_chans_per_node*gridsize*gridsize) + fidx_grid + (i_conj_indices[weight_index]*gridsize + j_conj_indices[weight_index]);
			printf(">>grid index: %d %d \n",grid_index,grid_conj_index);
			for (int kp=0; kp<2; kp++)
			{
				//data index
				data_index = sidx_data + (kt*nbase*num_chans_per_node*2) + bidx_data + (k*num_chans_per_node*2) + fidx_data + kp;
				printf(">>>data index: %d\n",data_index);
				printf(">>>>operation: %f + i%f x %f\n",data[data_index].x,data[data_index].y,bweights[weight_index]);

				//weight and add to grid
				if (bweights!=nullptr)
				{
					vis_grid[grid_index].x += data[data_index].x*bweights[weight_index];
					vis_grid[grid_index].y += data[data_index].y*bweights[weight_index];			
					vis_grid[grid_conj_index].x += data[data_index].x*bweights[weight_index];
					vis_grid[grid_conj_index].y -= data[data_index].y*bweights[weight_index];
				}
				else
				{
					vis_grid[grid_index].x += data[data_index].x;
                                	vis_grid[grid_index].y += data[data_index].y;
                                	vis_grid[grid_conj_index].x += data[data_index].x;
                                	vis_grid[grid_conj_index].y -= data[data_index].y;
				}
                        }
			printf("----------------------\n\n");
	
		}
	}
}


__global__
void grid_data(int num_chans, int num_time_samples, int num_chans_per_node, int nbase, int nbase_thread,
                cuComplex *data, unsigned int *i_indices, unsigned int *j_indices,
                unsigned int *i_conj_indices, unsigned int *j_conj_indices,
                double *bweights, int gridsize, cuDoubleComplex *vis_grid,
		int *flagbase, int *flagchans, int *flagcorrs, int *flagants, 
		int nflagbase, int nflagchans, int nflagcorrs, int nflagants,
		uint8_t *ANT1, uint8_t *ANT2)
{
        /*
           This function applies pre-computed weights to and grids visibilities.
           Note the grid should be arranged time x channel x U x V to properly
           interface with cufft.

           Input data is arranged subband x time x baseline x channel x polarization
           Input weights are arranged channel x baseline
         */
        //indices for weights
        int fidx = blockIdx.x * nbase;
        
	//check if channel is flagged
	for (int k_f=0; k_f < nflagcorrs; k_f++)
	{
		if (k_f*num_chans_per_node <= blockIdx.x && blockIdx.x <= (k_f+1)*num_chans_per_node)
		{
			return;
		}
	}
	for (int k_f=0; k_f < nflagchans; k_f++)
	{
		if (blockIdx.x == k_f)
		{
			return;
		}
	}

	
	
	int bidx = threadIdx.x * nbase_thread;

        //indices for data
        int sidx_data = blockIdx.x/num_chans_per_node;
        sidx_data *= (num_time_samples * nbase * num_chans_per_node * 2);
        int fidx_data = (blockIdx.x % num_chans_per_node)*2; //channels
        int bidx_data = threadIdx.x * nbase_thread * num_chans_per_node * 2; //baseline

        //indices for grid
        int fidx_grid = blockIdx.x * gridsize * gridsize;

        int data_index = 0;
        int weight_index = 0;
        int grid_index = 0;
        int grid_conj_index =0 ;
	bool flagflag =0;
        for (int k=0; k<nbase_thread; k++)
        {
		//check if baseline is flagged
		for (int k_f=0; k_f<nflagbase; k_f++)
		{
			if (bidx==flagbase[k_f])
			{
				flagflag = 1;
			}
		}
		if (flagflag==1) 
		{
			flagflag =0;
			continue;
		}
		
		for (int k_f=0; k_f<nflagants; k_f++)
		{
			if (ANT1[bidx]==flagants[k_f] || ANT2[bidx]==flagants[k_f])
			{
				flagflag = 1;
			}
		}
		if (flagflag==1)
                {
                        flagflag =0;
                        continue;
                }

                //weight/i/j index
                weight_index = fidx + bidx + k;

                for (int kt=0; kt<num_time_samples; kt++)
                {
                        //grid index
                        grid_index = (kt*num_chans_per_node*gridsize*gridsize) + fidx_grid + (i_indices[weight_index]*gridsize + j_indices[weight_index]);
                        grid_conj_index = (kt*num_chans_per_node*gridsize*gridsize) + fidx_grid + (i_conj_indices[weight_index]*gridsize + j_conj_indices[weight_index]);
                        for (int kp=0; kp<2; kp++)
                        {
                                //data index
                                data_index = sidx_data + (kt*nbase*num_chans_per_node*2) + bidx_data + (k*num_chans_per_node*2) + fidx_data + kp;
				
				//skip if nan
				if (isnan((double)(data[data_index].x)) || isnan((double)(data[data_index].y)) 
						|| isnan(bweights[weight_index]))
				{
					continue;
				}

                                //weight and add to grid
                                if (bweights!=nullptr)
                                {
                                        vis_grid[grid_index].x += data[data_index].x*bweights[weight_index];
                                        vis_grid[grid_index].y += data[data_index].y*bweights[weight_index];
                                        vis_grid[grid_conj_index].x += data[data_index].x*bweights[weight_index];
                                        vis_grid[grid_conj_index].y -= data[data_index].y*bweights[weight_index];
                                }
                                else
                                {
                                        vis_grid[grid_index].x += data[data_index].x;
                                        vis_grid[grid_index].y += data[data_index].y;
                                        vis_grid[grid_conj_index].x += data[data_index].x;
                                        vis_grid[grid_conj_index].y -= data[data_index].y;
                                }
                        }

                }
        }
}

 
void *cufftJITCallbackStoreD(void *dataOut,
                                        unsigned long long offset,
                                        cufftDoubleReal element,
                                        void *callerInfo,
                                        void *sharedPointer)
{
	/*
	   callback routine essentially FFT-shifts the output data
	   */
	int gridsize_out = *((int *)sharedPointer);
	int gridsize = (gridsize_out/2) + 1;
	int gidx = offset/(gridsize_out*gridsize_out);
	
	
	int i_idx = (offset - gidx)/gridsize_out;
	int j_idx = (offset - gidx - (i_idx*gridsize_out));
	int i_quad = (i_idx/gridsize);
	int j_quad = (j_idx/gridsize);
	int shiftby  = gridsize/2;

	i_idx = (i_quad*gridsize) + (((i_idx - i_quad*gridsize) + shiftby)%gridsize);
	j_idx = (j_quad*gridsize) + (((j_idx - j_quad*gridsize) + shiftby)%gridsize);
	
	(&offset)[0] = (gidx*gridsize_out*gridsize_out) + (i_idx*gridsize_out + j_idx);

}




// Data structure to hold command line arguments
struct cmdargs {
	//input data
	char *filelabel;
	char *timestamp;
	char *filedir;
	int num_gulps = -1;
	int gulp_offset = 0;
	int num_time_samples = 25;
	char *path;
	char *outpath;
	bool verbose = 0;
	bool search = 0;
	bool save = 0;
	long max_base = 4656;
	
	//injection parameters
	bool inject = 0;
	bool slowinject = 0;
	double snr_inject =-1;
	double snr_min_inject = 1e7;
	double snr_max_inject = 1e8;
	double dm_inject = -1;
	int width_inject = -1;
	int offsetRA_inject = 0;
	int offsetDEC_inject = 0;
	bool offline = 0;
	bool inject_noiseonly = 0;
	bool inject_noiseless =0;
	int num_inject = 0;
	bool flat_field = 0;
	bool gauss_field = 0;
	bool point_field = 0;

	//imaging parameters
	bool sbname = 0;
	int num_chans = 16;
	int num_chans_per_node = 8;
	bool briggs = 0;
	float robust = 0;
	double sleeptime = 0;
	double bmin = 20;
	double bmax = 20000;
	bool wstack = 0;
	bool wstack_parallel = 0;
	int Nlayers = 18;
	int gridsize = 301;
	int pixperFWHM = 3;

	
	//flagging parameters
	bool flagSWAVE = 0;
	bool flagBPASS = 0;
	bool flagFRCBAND = 0;
	bool flagBPASSBURST = 0;
	int *flagcorrs;
	int *flagants;
	int *flagchans;
	int *flagbase;
	int nflagcorrs = 0;
	int nflagants = 0;
	int nflagchans = 0;
	int nflagbase = 0;

	//processing parameters
	int maxProcesses = 16;
	bool multiimage = 0;
	bool multisend = 0;
	float stagger_multisend = 0;
	int port = 8080;
	int *multiport;	
};

cmdargs *parseargs(cmdargs *args, int argc, char *argv[])
{
	/*
	   Parses command line arguments and stores results in args struct
	*/
	int i =0;
	while (i < argc)
	{
		printf("%s\n",argv[i]);
		if (argv[i][0] == 45 && argv[i][1] == 45)
		{
			if (strcmp(argv[i] + 2,"filelabel") == 0) {args->filelabel = argv[i+1];}
			else if (strcmp(argv[i] + 2,"timestamp") == 0) {args->timestamp = argv[i+1];}
			else if (strcmp(argv[i] + 2,"filedir") == 0) {args->filedir = argv[i+1];}
			else if (strcmp(argv[i] + 2,"num_gulps") == 0) {args->num_gulps = atoi(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"gulp_offset") == 0) {args->gulp_offset = atoi(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"num_time_samples") == 0) {args->num_time_samples = atoi(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"path") == 0) {args->path = argv[i+1];}
			else if (strcmp(argv[i] + 2,"verbose") == 0) {args->verbose = 1;}
			else if (strcmp(argv[i] + 2,"search") == 0) {args->search = 1;}
			else if (strcmp(argv[i] + 2,"save") == 0) {args->save = 1;}
			else if (strcmp(argv[i] + 2,"max_base") == 0) {args->max_base = strtol(argv[i+1],NULL,0);}
			else if (strcmp(argv[i] + 2,"inject") == 0) {args->inject = 1;}
			else if (strcmp(argv[i] + 2,"slowinject") == 0) {args->slowinject = 1;}
			else if (strcmp(argv[i] + 2,"snr_inject") == 0) {args->snr_inject = atof(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"snr_min_inject") == 0) {args->snr_min_inject = atof(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"snr_max_inject") == 0) {args->snr_max_inject = atof(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"dm_inject") == 0) {args->dm_inject = atof(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"width_inject") == 0) {args->width_inject = atoi(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"offsetRA_inject") == 0) {args->offsetRA_inject = atoi(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"offsetDEC_inject") == 0) {args->offsetDEC_inject = atoi(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"offline") == 0) {args->offline = 1;}
			else if (strcmp(argv[i] + 2,"inject_noiseonly") == 0) {args->inject_noiseonly = 1;}
			else if (strcmp(argv[i] + 2,"inject_noiseless") == 0) {args->inject_noiseless = 1;}
			else if (strcmp(argv[i] + 2,"num_inject") == 0) {args->num_inject = atoi(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"flat_field") == 0) {args->flat_field = 1;}
			else if (strcmp(argv[i] + 2,"gauss_field") == 0) {args->gauss_field = 1;}
			else if (strcmp(argv[i] + 2,"point_field") == 0) {args->point_field = 1;}
			else if (strcmp(argv[i] + 2,"sbname") == 0) {args->sbname = 1;}
			else if (strcmp(argv[i] + 2,"num_chans") == 0) {args->num_chans = atoi(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"num_chans_per_node") == 0) {args->num_chans_per_node = atoi(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"briggs") == 0) {args->briggs = 1;}
			else if (strcmp(argv[i] + 2,"robust") == 0) {args->robust = atof(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"sleeptime") == 0) {args->sleeptime = atof(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"bmin") == 0) {args->bmin = atof(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"bmax") == 0) {args->bmax = atof(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"wstack") == 0) {args->wstack = 1;}
			else if (strcmp(argv[i] + 2,"wstack_parallel") == 0) {args->wstack_parallel = 1;}
			else if (strcmp(argv[i] + 2,"Nlayers") == 0) {args->Nlayers = atoi(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"gridsize") == 0) {args->gridsize = atoi(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"pixperFWHM") == 0) {args->pixperFWHM = atoi(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"flagSWAVE") == 0) {args->flagSWAVE = 1;}
			else if (strcmp(argv[i] + 2,"flagBPASS") == 0) {args->flagBPASS = 1;}
			else if (strcmp(argv[i] + 2,"flagBPASSBURST") == 0) {args->flagBPASSBURST = 1;}
			else if (strcmp(argv[i] + 2,"flagcorrs") == 0) {
				int tmp_i = 1;
				while ((i+tmp_i < argc-1) && (argv[i+tmp_i][0] != 45 || argv[i+tmp_i][1] != 45))
				{
					tmp_i += 1;
				}
				args->nflagcorrs = tmp_i;
				
				cudaMallocManaged(&(args->flagcorrs), tmp_i*sizeof(int));
				for (int j = 0; j < tmp_i; j+= 1)
				{
					(args->flagcorrs)[j] = atoi(argv[i+1+j]);

				}
			}
			else if (strcmp(argv[i] + 2,"flagants") == 0) {
				int tmp_i = 1;
                                while ((i+tmp_i < argc-1) && (argv[i+tmp_i][0] != 45 || argv[i+tmp_i][1] != 45))
                                {
                                        tmp_i += 1;
                                }
				args->nflagants = tmp_i;

				cudaMallocManaged(&(args->flagants), tmp_i*sizeof(int));
                                for (int j = 0; j < tmp_i; j+= 1)
                                {
                                        (args->flagants)[j] = atoi(argv[i+1+j]);
                                }
			}
			else if (strcmp(argv[i] + 2,"flagchans") == 0) {
                                int tmp_i = 1;
                                while ((i+tmp_i < argc-1) && (argv[i+tmp_i][0] != 45 || argv[i+tmp_i][1] != 45))
				{
                                        tmp_i += 1;
                                }
				args->nflagchans = tmp_i;

				cudaMallocManaged(&(args->flagchans), tmp_i*sizeof(int));
                                for (int j = 0; j < tmp_i; j+= 1)
                                {       
                                        (args->flagchans)[j] = atoi(argv[i+1+j]);
                                }
			}
			else if (strcmp(argv[i] + 2,"flagbase") == 0) {
                                int tmp_i = 1;
                                while ((i+tmp_i < argc-1) && (argv[i+tmp_i][0] != 45 || argv[i+tmp_i][1] != 45))
				{
                                        tmp_i += 1;
                                }
				args->nflagbase = tmp_i;


				cudaMallocManaged(&(args->flagbase), tmp_i*sizeof(int));
                                for (int j = 0; j < tmp_i; j+= 1)
                                {
                                        (args->flagbase)[j] = atoi(argv[i+1+j]);
                                }
			}
			else if (strcmp(argv[i] + 2,"maxProcesses") == 0) {args->maxProcesses = atoi(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"multiimage") == 0) {args->multiimage = 1;}
			else if (strcmp(argv[i] + 2,"multisend") == 0) {args->multisend = 1;}
			else if (strcmp(argv[i] + 2,"stagger_multisend") == 0) {args->stagger_multisend = atoi(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"port") == 0) {args->port = atoi(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"multiport") == 0) {
                                int tmp_i = 1;
                                while ((i+tmp_i < argc-1) && (argv[i+tmp_i][0] != 45 || argv[i+tmp_i][1] != 45))
				{
                                        tmp_i += 1;
                                }

				cudaMallocManaged(&(args->multiport), tmp_i*sizeof(int));
                                for (int j = 0; j < tmp_i; j+= 1)
                                {       
                                        (args->multiport)[j] = atoi(argv[i+1+j]);
                                }
			}
			else {printf("Invalid argument \'%s\'",argv[i]);}
		}
		i++;
	}

	return args;
}


void setup_frequencies(double *freq_axis_fullres,double *freq_axis_fullres_GHz, double *freq_axis, double *freq_axis_GHz) {
        for (int i =0; i < maxchans; i+=1)
        {
                freq_axis_fullres[i] = 1000*(1.53-(i*0.25/8192)); //MHz
                freq_axis_fullres_GHz[i] = (1.53-(i*0.25/8192)); //GHz
        }

        for (int j=0; j < nchans; j+=1)
        {
                freq_axis[j] = 0;
                freq_axis_GHz[j] = 0;
                for (int jj=j*(maxchans/nchans); jj < (j+1)*maxchans/nchans; jj+=1)
                {
                        freq_axis[j] += freq_axis_fullres[jj];//MHz
                        freq_axis_GHz[j] += freq_axis_fullres_GHz[jj];//GHz
                }
                freq_axis[j] /= (maxchans/nchans);
                freq_axis_GHz[j] /= (maxchans/nchans);
        }
}

int main(int argc, char *argv[])
{
	double pi = 22.0/7.0;
	unsigned long long t1 = myCPUTimer();
	cmdargs args_obj;
	cmdargs *args = &args_obj;
	//cudaMallocManaged(&args,sizeof(cmdargs));
	parseargs(args,argc,argv);
	printf("Done Parsing Data\n");


	//create freq axis
	setup_frequencies(freq_axis_fullres,freq_axis_fullres_GHz,freq_axis,freq_axis_GHz);
	chanbw = freq_axis[1]-freq_axis[0]; //MHz
	fmin_=freq_axis[nchans-1]; //MHz
	fmax_=freq_axis[0]; //MHz
	fc_ = (fmin_+fmax_)/2; //MHz
	lambdamin_ = C_MHZ_M/fmax_; //m
	lambdamax_ = C_MHZ_M/fmin_; //m
	lambdac_ = C_MHZ_M/fc_; //m
	lambdaref_ = C_MHZ_M/freq_axis_fullres[0]; //m
	int num_tot_chans = args->num_chans*args->num_chans_per_node;
	double *fobs;//[num_tot_chans];
	double *fobs_GHz;//[num_tot_chans];
	cudaMallocManaged(&fobs,num_tot_chans*sizeof(double));
	cudaMallocManaged(&fobs_GHz,num_tot_chans*sizeof(double));
	for (int j = 0; j < num_tot_chans; j += 1) 
	{
		fobs[j]=0;
		fobs_GHz[j] =0;
		for (int jj=j*args->num_chans_per_node; jj < (j+1)*args->num_chans_per_node; jj += 1)
		{
			fobs[j] += freq_axis_fullres[jj];
			fobs_GHz[j] += freq_axis_fullres_GHz[jj];
		}
		fobs[j] /= args->num_chans_per_node;
		fobs_GHz[j] /= args->num_chans_per_node;
	}

	//read data from file
	int corrs[16] = {3,4,5,6,7,8,10,11,12,14,15,16,18,19,21,22};
	char corr[5];
	char sb_s[6];
	char fname[120];
	size_t total_samples_per_file = (args->num_time_samples)*(args->num_chans_per_node)*(args->max_base)*2;
	size_t total_samples = (args->num_chans)*total_samples_per_file;
	printf(">>>%d,%d,%ld,%ld,%ld\n",args->num_time_samples,args->num_chans_per_node,args->max_base,total_samples_per_file,total_samples);
	cuComplex *data;//data is ordered sub-band x time x baseline x channel x polarization as complex 32-bit float (64-bit total)
	cudaMallocManaged(&data,total_samples*sizeof(cuComplex));
	
	FILE *fobj;
	int sb = -1;
	double mjd = 0.0;
	float dec = 0.0;
	size_t nread = 0;
	for (int i =0; i<16; i+=1){
		nread = 0;
		sprintf(corr,"h%02d_",corrs[i]);
		sprintf(sb_s,"sb%02d_",i);
		if (strlen(args->filedir) == 0)
		{
			strcpy(fname,args->path);
		       	strcpy(fname+strlen(args->path),"/lxd110");
			strcpy(fname+strlen(args->path)+strlen("/lxd110"),corr);
			strcpy(fname+strlen(args->path)+strlen("/lxd110")+strlen(corr),"/nsfrb_");
			if (args->sbname)
			{
				strcpy(fname+strlen(args->path)+strlen("/lxd110")+strlen(corr)+strlen("/nsfrb_"),sb_s);
				strcpy(fname+strlen(args->path)+strlen("/lxd110")+strlen(corr)+strlen("/nsfrb_")+strlen(sb_s),args->filelabel);
				strcpy(fname+strlen(args->path)+strlen("/lxd110")+strlen(corr)+strlen("/nsfrb_")+strlen(sb_s)+strlen(args->filelabel),".out");
			}
			else
			{
				strcpy(fname+strlen(args->path)+strlen("/lxd110")+strlen(corr)+strlen("/nsfrb_"),corr);
				strcpy(fname+strlen(args->path)+strlen("/lxd110")+strlen(corr)+strlen("/nsfrb_")+strlen(corr),args->filelabel);
                                strcpy(fname+strlen(args->path)+strlen("/lxd110")+strlen(corr)+strlen("/nsfrb_")+strlen(corr)+strlen(args->filelabel),".out");
			}
		}
		else
		{
                        strcpy(fname,args->filedir);
                        strcpy(fname+strlen(args->filedir),"/nsfrb_");
                        if (args->sbname)
                        {
                                strcpy(fname+strlen(args->filedir)+strlen("/nsfrb_"),sb_s);
                                strcpy(fname+strlen(args->filedir)+strlen("/nsfrb_")+strlen(sb_s),args->filelabel);
                                strcpy(fname+strlen(args->filedir)+strlen("/nsfrb_")+strlen(sb_s)+strlen(args->filelabel),".out");
                        }
                        else
                        {
				strcpy(fname+strlen(args->filedir)+strlen("/nsfrb_"),corr);
                                strcpy(fname+strlen(args->filedir)+strlen("/nsfrb_")+strlen(corr),args->filelabel);
                                strcpy(fname+strlen(args->filedir)+strlen("/nsfrb_")+strlen(corr)+strlen(args->filelabel),".out");
                        }
                }
		printf("Reading file %s\n",fname);
		fobj = fopen(fname,"rb");
		nread += fread(&mjd,sizeof(double),1,fobj)*sizeof(double); 
		nread += fread(&sb,sizeof(float),1,fobj)*sizeof(float);
		nread += fread(&dec,sizeof(float),1,fobj)*sizeof(float);
		printf("MJD:%f, ",mjd);
                printf("SB:%d, ",sb);
                printf("DEC:%f,",dec);
		fseek(fobj, sizeof(cuComplex)*total_samples_per_file*(args->gulp_offset),SEEK_CUR);
		nread += fread(data,sizeof(cuComplex),total_samples_per_file,fobj)*sizeof(cuComplex);
		fclose(fobj);
		printf("BYTES:%ld/%ld\n",nread,(16 + total_samples_per_file*sizeof(cuComplex)));
	}
	if (args->flat_field == 1)
	{
		printf("Overwriting vis with flat field 1 + 0j\n");
		for (int i=0; i<total_samples;i++)
		{
			data[i].x=1.0;
			data[i].y=0.0;
		}
	}


	//get UVW coordinates from file
	printf("Updating and reading UVW coords\n");
	char updateUVWcmd[strlen(baseUVWcmd)+20];
	strcpy(updateUVWcmd,baseUVWcmd); 
	sprintf(updateUVWcmd + strlen(baseUVWcmd),"%f",dec*pi/180);
	system(updateUVWcmd);
	printf("%s\n",updateUVWcmd);
	printf("%s\n",table_dir);
	printf("%s\n",ufname);

	fobj = fopen(ufname,"rb");
	double *U;
	printf("Allocating %ld bytes (%ld values)\n",(args->max_base)*sizeof(double),args->max_base);
	cudaMallocManaged(&U,(args->max_base)*sizeof(double));
	printf("Done allocating\n");
	nread = fread(U,sizeof(double),args->max_base,fobj)*sizeof(double);
	printf("Successfully read %ld bytes\n",nread);
	fclose(fobj);

	fobj = fopen(vfname,"rb");
        double *V;
        printf("Allocating %ld bytes (%ld values)\n",(args->max_base)*sizeof(double),args->max_base);
        cudaMallocManaged(&V,(args->max_base)*sizeof(double));
        printf("Done allocating\n");
        nread = fread(V,sizeof(double),args->max_base,fobj)*sizeof(double);
        printf("Successfully read %ld bytes\n",nread);
        fclose(fobj);

	fobj = fopen(wfname,"rb");
        double *W;
        printf("Allocating %ld bytes (%ld values)\n",(args->max_base)*sizeof(double),args->max_base);
        cudaMallocManaged(&W,(args->max_base)*sizeof(double));
        printf("Done allocating\n");
        nread = fread(W,sizeof(double),args->max_base,fobj)*sizeof(double);
        printf("Successfully read %ld bytes\n",nread);
        fclose(fobj);

        fobj = fopen(bfname,"rb");
        double *BLEN;
        printf("Allocating %ld bytes (%ld values)\n",(args->max_base)*sizeof(double),args->max_base);
        cudaMallocManaged(&BLEN,(args->max_base)*sizeof(double));
        printf("Done allocating\n");
        nread = fread(BLEN,sizeof(double),args->max_base,fobj)*sizeof(double);
        printf("Successfully read %ld bytes\n",nread);
        fclose(fobj);
	
	fobj = fopen(a1fname,"rb");
	uint8_t *ANT1;
	printf("Allocating %ld bytes (%ld values)\n",(args->max_base)*sizeof(uint8_t),args->max_base);
	cudaMallocManaged(&ANT1,(args->max_base)*sizeof(uint8_t));
	printf("Done allocating\n");
	nread = fread(ANT1,sizeof(uint8_t),args->max_base,fobj)*sizeof(uint8_t);
	printf("Successfully read %ld bytes\n",nread);
	fclose(fobj);

	fobj = fopen(a2fname,"rb");
        uint8_t *ANT2;
        printf("Allocating %ld bytes (%ld values)\n",(args->max_base)*sizeof(uint8_t),args->max_base);
        cudaMallocManaged(&ANT2,(args->max_base)*sizeof(uint8_t));
        printf("Done allocating\n");
        nread = fread(ANT2,sizeof(uint8_t),args->max_base,fobj)*sizeof(uint8_t);
        printf("Successfully read %ld bytes\n",nread);
        fclose(fobj);

	//get the maximum baseline length
	int uv_diag_idx;
	cublasStatus_t stat;
	cublasHandle_t handle;
	stat = cublasCreate(&handle);
	stat = cublasSetVector(args->max_base,sizeof(double),BLEN,1,BLEN,1);
	stat = cublasIdamax(handle,args->max_base,BLEN,1,&uv_diag_idx);
	stat = cublasGetVector(args->max_base,sizeof(double),BLEN,1,BLEN,1);
	if (stat != CUBLAS_STATUS_SUCCESS) {
		printf("FAILED\n");
	}
	double uv_diag = BLEN[uv_diag_idx];
	double pixel_resolution = (lambdaref_/uv_diag)/(args->pixperFWHM);
	double uv_resolution = 1/((args->gridsize)*pixel_resolution);
	double uv_max = uv_resolution*(args->gridsize)/2;
	double grid_res = 2*uv_max/(args->gridsize);
	printf("Maximum baseline length is %f meters; resolution is %f meters\n",uv_max,grid_res);
	cudaDeviceSynchronize();


	//uniform gridding
	unsigned int NTHREADS = 48;
	unsigned int NBLOCKS = num_tot_chans;
	unsigned int NBASEPERTHREAD = (args->max_base)/NTHREADS;
	unsigned int *i_indices,*j_indices,*i_conj_indices,*j_conj_indices,*tmp_indices;
	cudaMallocManaged(&i_indices,(args->max_base)*num_tot_chans*sizeof(unsigned int));
	cudaMallocManaged(&j_indices,(args->max_base)*num_tot_chans*sizeof(unsigned int));
	cudaMallocManaged(&i_conj_indices,(args->max_base)*num_tot_chans*sizeof(unsigned int));
	cudaMallocManaged(&j_conj_indices,(args->max_base)*num_tot_chans*sizeof(unsigned int));
	cudaMallocManaged(&tmp_indices,(args->max_base)*num_tot_chans*sizeof(unsigned int));

	cudaMemPrefetchAsync(U,(args->max_base)*sizeof(double),0,0);
	cudaMemPrefetchAsync(V,(args->max_base)*sizeof(double),0,0);
	cudaMemPrefetchAsync(W,(args->max_base)*sizeof(double),0,0);
	cudaMemPrefetchAsync(i_indices,(args->max_base)*num_tot_chans*sizeof(unsigned int),0,0);
	cudaMemPrefetchAsync(j_indices,(args->max_base)*num_tot_chans*sizeof(unsigned int),0,0);
	cudaMemPrefetchAsync(i_conj_indices,(args->max_base)*num_tot_chans*sizeof(unsigned int),0,0);
	cudaMemPrefetchAsync(j_conj_indices,(args->max_base)*num_tot_chans*sizeof(unsigned int),0,0);
	cudaMemPrefetchAsync(fobs_GHz,num_tot_chans*sizeof(double),0,0);
	//cudaMemPrefetchAsync(bweights,(args->max_base)*num_tot_chans*sizeof(int),0,0);
	//for (int i=0;i<num_tot_chans;i+=1)
	//{
	//	printf("FREQ %d: %f -- WAV %d: %f\n",i,fobs_GHz[i],i,C_GHZ_M/fobs_GHz[i]);
	//}
	uniform_grid_singlefreq<<<NBLOCKS, NTHREADS>>>(U, V, W, args->max_base, NBASEPERTHREAD, fobs_GHz, args->num_chans, args->num_chans_per_node, 
				args->gridsize, uv_max, grid_res, args->robust, i_indices, j_indices, i_conj_indices, j_conj_indices);
	cudaDeviceSynchronize();
	



	
	//briggs robust weighting
	unsigned int *Wk;
        double *vis_weights;
        double *bweights;
	cudaMemPrefetchAsync(args->flagbase,(args->nflagbase)*sizeof(int),0,0);
        cudaMemPrefetchAsync(args->flagchans,(args->nflagchans)*sizeof(int),0,0);
        cudaMemPrefetchAsync(args->flagcorrs,(args->nflagcorrs)*sizeof(int),0,0);
        cudaMemPrefetchAsync(args->flagants,(args->nflagants)*sizeof(int),0,0);
        cudaMemPrefetchAsync(ANT1,(args->max_base)*sizeof(unsigned int),0,0);
        cudaMemPrefetchAsync(ANT2,(args->max_base)*sizeof(unsigned int),0,0);


	if (args->briggs) {
		cudaMallocManaged(&Wk,(args->gridsize)*(args->gridsize)*num_tot_chans*sizeof(unsigned int));
		cudaMallocManaged(&vis_weights,(args->max_base)*num_tot_chans*sizeof(double));
		cudaMallocManaged(&bweights,(args->max_base)*num_tot_chans*sizeof(double));
	
		for (int i=0; i < (args->gridsize)*(args->gridsize)*num_tot_chans; i+=1){Wk[i] = 0;}
		for (int i=0; i < (args->max_base)*num_tot_chans; i+=1){vis_weights[i]=1.0; bweights[i]=0.0;}
 
		cudaMemPrefetchAsync(Wk,(args->gridsize)*(args->gridsize)*num_tot_chans*sizeof(unsigned int),0,0);
		cudaMemPrefetchAsync(vis_weights,(args->max_base)*num_tot_chans*sizeof(double),0,0);
		cudaMemPrefetchAsync(bweights,(args->max_base)*num_tot_chans*sizeof(double),0,0);


		briggs_weight_singlefreq<<<1, NBLOCKS>>>(args->max_base, args->gridsize, 
				i_indices, j_indices, bweights, args->robust, Wk, 
				vis_weights, args->flagbase, args->flagchans,
                                        args->flagcorrs, args->flagants, args->nflagbase, args->nflagchans,
                                        args->nflagcorrs, args->nflagants, ANT1, ANT2);
		cudaDeviceSynchronize();
		for (int i=0; i<(args->max_base)*num_tot_chans;i++)
		{
			printf("BASECHAN %d: BWEIGHT %f\n",i,bweights[i]);
		}
	}
	
	//apply grid
	cuDoubleComplex *vis_grid;
	cudaMallocManaged(&vis_grid,(args->gridsize)*(args->gridsize)*(args->num_time_samples)*num_tot_chans*sizeof(cuDoubleComplex));

	for (int i=0; i < (args->gridsize)*(args->gridsize)*(args->num_time_samples)*num_tot_chans; i+=1) {vis_grid[i].x=0; vis_grid[i].y=0;}
	
	cudaMemPrefetchAsync(vis_grid,(args->gridsize)*(args->gridsize)*(args->num_time_samples)*num_tot_chans*sizeof(cuDoubleComplex),0,0);

	grid_data<<<NBLOCKS, NTHREADS>>>(args->num_chans, args->num_time_samples, args->num_chans_per_node, args->max_base, 
					NBASEPERTHREAD, data, i_indices, j_indices, i_conj_indices, j_conj_indices, bweights, 
					args->gridsize,vis_grid, args->flagbase, args->flagchans, 
					args->flagcorrs, args->flagants, args->nflagbase, args->nflagchans, 
					args->nflagcorrs, args->nflagants, ANT1, ANT2);

	cudaDeviceSynchronize();


	fobj = fopen("/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/cudaimager/tmpvisgrid.bin","wb");
        fwrite(vis_grid, sizeof(cuDoubleComplex), (args->num_time_samples)*num_tot_chans*(args->gridsize)*(args->gridsize), fobj);
        fclose(fobj);


        if (args->briggs){
                cudaFree(Wk);
                cudaFree(vis_weights);
                cudaFree(bweights);
        }
        cudaFree(BLEN);
        cudaFree(U);
        cudaFree(V);
        cudaFree(W);
        cudaFree(data);

	//image
	int gridsize_out = (args->gridsize - 1)*2;
	cufftDoubleReal *image;
	//cufftDoubleReal *image_shift;
	cudaMallocManaged(&image,gridsize_out*gridsize_out*(args->num_time_samples)*num_tot_chans*sizeof(cufftDoubleReal));
	//cudaMallocManaged(&image_shift,(args->gridsize)*(args->gridsize)*(args->num_time_samples)*num_tot_chans*sizeof(cufftDoubleReal));
	//cufftDoubleComplex *vis_grid_shift;
        //cudaMallocManaged(&vis_grid_shift,(args->gridsize)*(args->gridsize)*(args->num_time_samples)*num_tot_chans*sizeof(cufftDoubleComplex));

	for (int i=0; i < gridsize_out*gridsize_out*(args->num_time_samples)*num_tot_chans; i+=1) {image[i] =0.0;}
        //for (int i=0; i < (args->gridsize)*(args->gridsize)*(args->num_time_samples)*num_tot_chans; i+=1) {image[i] =0.0;image_shift[i]=0.0; vis_grid_shift[i].x=0; vis_grid_shift[i].y=0;}

        cudaMemPrefetchAsync(vis_grid,(args->gridsize)*(args->gridsize)*(args->num_time_samples)*num_tot_chans*sizeof(cuDoubleComplex),0,0);
	//cudaMemPrefetchAsync(vis_grid_shift,(args->gridsize)*(args->gridsize)*(args->num_time_samples)*num_tot_chans*sizeof(cufftDoubleComplex),0,0);
	cudaMemPrefetchAsync(image,gridsize_out*gridsize_out*(args->num_time_samples)*num_tot_chans*sizeof(cufftDoubleReal),0,0);
        //cudaMemPrefetchAsync(image_shift,(args->gridsize)*(args->gridsize)*(args->num_time_samples)*num_tot_chans*sizeof(cufftDoubleReal),0,0);

        cufftHandle plan;
        int batchsize = num_tot_chans*(args->num_time_samples);

	//attach callback
	cufftCreate(&plan);
	//cufftResult status; 
	//cufftResult = cufftXtSetJITCallback(plan, "cufftJITCallbackStoreD", (void*)cufftJITCallbackStoreD, sizeof(cufftJITCallbackStoreD), CUFFT_CB_LD_REAL, (void **)(&(&(args->gridsize_out))));

        //data layout options
        int inembed[2];
        inembed[0] = (args->gridsize)*(args->gridsize);//Ntotal;
        inembed[1] = args->gridsize;
        int oembed[2];
        oembed[0] = gridsize_out*gridsize_out; //Nout;
        oembed[1] = gridsize_out;
        int istride = 1;
        int idist = (args->gridsize)*(args->gridsize);
        int ostride = 1;
        int odist = gridsize_out*gridsize_out;
        int narr[2];
        narr[0] = (args->gridsize);
        narr[1] = (args->gridsize);

        cufftPlanMany(&plan, 2, narr, inembed, istride, idist,
                oembed, ostride, odist, CUFFT_Z2D, batchsize);

	//we want to use the double-precision real-to-complex version
	//NTHREADS = num_tot_chans;
	//NBLOCKS = args->num_time_samples;
	//my2difftshift_complex<<<NBLOCKS, NTHREADS>>>(args->gridsize,1,(cufftDoubleComplex *)vis_grid,vis_grid_shift,0);
	//cudaDeviceSynchronize();
	
	cufftResult res = cufftExecZ2D(plan,vis_grid,image);
	cudaDeviceSynchronize();
        
	//my2difftshift<<<NBLOCKS, NTHREADS>>>(args->gridsize,1,image,image_shift,0);
        //cudaDeviceSynchronize();


	cudaMemcpy(image, image, (args->num_time_samples)*num_tot_chans*gridsize_out*gridsize_out*sizeof(cufftDoubleReal) , cudaMemcpyDeviceToHost );

	cufftDoubleReal *image_shift;
	cudaMallocManaged(&image_shift,(args->gridsize)*(args->gridsize)*(args->num_time_samples)*num_tot_chans*sizeof(cufftDoubleReal));
	my2difftshift_quad(args->gridsize, gridsize_out, (args->num_time_samples)*num_tot_chans, image,image_shift);


	fobj = fopen("/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/cudaimager/cuda_offline_imager/tmpimage.bin","wb");
        fwrite(image_shift, sizeof(cufftDoubleReal), (args->num_time_samples)*num_tot_chans*(args->gridsize)*(args->gridsize), fobj);
        fclose(fobj);
	/*
	
	cudaMemcpy(data, data,total_samples*sizeof(cuDoubleComplex) , cudaMemcpyDeviceToHost );
	cudaMemcpy(bweights, bweights,(args->max_base)*num_tot_chans*sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(i_indices, i_indices,(args->max_base)*num_tot_chans*(sizeof(unsigned int)),cudaMemcpyDeviceToHost);
	cudaMemcpy(i_conj_indices, i_conj_indices,(args->max_base)*num_tot_chans*(sizeof(unsigned int)),cudaMemcpyDeviceToHost);
        cudaMemcpy(j_indices, j_indices,(args->max_base)*num_tot_chans*(sizeof(unsigned int)),cudaMemcpyDeviceToHost);
        cudaMemcpy(j_conj_indices, j_conj_indices,(args->max_base)*num_tot_chans*(sizeof(unsigned int)),cudaMemcpyDeviceToHost);
        
	
	for (int tmpblockIdx =0; tmpblockIdx < NBLOCKS; tmpblockIdx++)
	{
		for (int tmpthreadIdx=0; tmpthreadIdx < 1; tmpthreadIdx++)
		{
			printf("STARTING BLOCK %d THREAD %d\n",tmpblockIdx,tmpthreadIdx);
			grid_data_CPU(args->num_chans, args->num_time_samples, args->num_chans_per_node, args->max_base, 
                                      NBASEPERTHREAD, data, i_indices, j_indices, i_conj_indices, j_conj_indices, bweights, 
                                      args->gridsize,vis_grid,tmpblockIdx,tmpthreadIdx);
			printf("FINISHED BLOCK %d THREAD %d\n: %f + i%f\n",tmpblockIdx,tmpthreadIdx,vis_grid[tmpblockIdx*(args->gridsize)*(args->gridsize) + (args->gridsize/2)*(args->gridsize) + (args->gridsize/2)].x, vis_grid[tmpblockIdx*(args->gridsize)*(args->gridsize) + (args->gridsize/2)*(args->gridsize) + (args->gridsize/2)].y);
		}
	}
	*/

	/*
	int i =0;
	int b =100;
	for (int j=0;j<num_tot_chans;j+=1)
	{
		i=b + j*args->max_base;
		printf("[%d] FREQ %f -- WAV %f: U=%f, V=%f, W=%f, uvmax=%f, i=%u, i_conj=%u, j=%u, j_conj=%u\n",j,fobs_GHz[j],C_GHZ_M/fobs_GHz[j],U[b]/(C_GHZ_M/fobs_GHz[j]),V[b]/(C_GHZ_M/fobs_GHz[j]),W[b]/(C_GHZ_M/fobs_GHz[j]),uv_max,i_indices[i],i_conj_indices[i],j_indices[i],j_conj_indices[i]);
	}
	*/

	
	/*
	for (int i=0;i<args->max_base;i+=1)
	{
		if (args->briggs){
			printf("IDX %d: U=%f, V=%f, W=%f, uvmax=%f, i=%u, i_conj=%u, j=%u, j_conj=%u, bweight=%f\n",i,U[i]/lambdaref_,V[i]/lambdaref_,W[i]/lambdaref_,uv_max,i_indices[i],i_conj_indices[i],j_indices[i],j_conj_indices[i],bweights[i]);
		}
		else {
			printf("IDX %d: U=%f, V=%f, W=%f, uvmax=%f, i=%u, i_conj=%u, j=%u, j_conj=%u\n",i,U[i]/lambdaref_,V[i]/lambdaref_,W[i]/lambdaref_,uv_max,i_indices[i],i_conj_indices[i],j_indices[i],j_conj_indices[i]);
		}
	}
	*/
	/*
	cudaMemcpy(vis_grid,vis_grid,(args->num_time_samples)*num_tot_chans*(args->gridsize)*(args->gridsize)*sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        cudaMemcpy(Wk,Wk,num_tot_chans*(args->gridsize)*(args->gridsize)*sizeof(unsigned int), cudaMemcpyDeviceToHost);
        fobj = fopen("/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/cudaimager/tmpvisgrid.bin","wb");
	fwrite(vis_grid, sizeof(cuDoubleComplex), (args->num_time_samples)*num_tot_chans*(args->gridsize)*(args->gridsize), fobj);
	fclose(fobj);

	for (int tmpblockIdx =0; tmpblockIdx < NBLOCKS; tmpblockIdx++)
        {
        	printf("FINISHED BLOCK %d\n:",tmpblockIdx);
                for (int i=0; i<(args->gridsize)*(args->gridsize);i++)
                {
                        if (Wk[tmpblockIdx*(args->gridsize)*(args->gridsize) + i]>0)
                        {
                        printf("        %d baselines -> %f + i%f\n",Wk[tmpblockIdx*(args->gridsize)*(args->gridsize) + i],vis_grid[tmpblockIdx*(args->gridsize)*(args->gridsize) + i].x, vis_grid[tmpblockIdx*(args->gridsize)*(args->gridsize)].y);
                        }
                }
        }
	*/

	cufftDestroy(plan);

	cudaFree(image);
	cudaFree(vis_grid);
	cudaFree(fobs_GHz);
	cudaFree(fobj);
	return 0;
	/*
	if (args->briggs){
		cudaFree(Wk);
		cudaFree(vis_weights);
		cudaFree(bweights);
	}
	cudaFree(image);
	cudaFree(image_shift);
	cudaFree(vis_grid_shift);
	cudaFree(vis_grid);
	cudaFree(fobs_GHz);
	cudaFree(fobs);
	cudaFree(BLEN);
	cudaFree(U);
	cudaFree(V);
	cudaFree(W);
	cudaFree(data);
	//cudaFree(args);
	cudaFree(fobj);
	return 0;
	*/
	/*
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
	std::cout << "Max error (real): " << maxError_r << std::endl;
	std::cout << "Max error (imag): " << maxError_i << std::endl;

	

	//GPU free
	cudaFree(data);
	cudaFree(U);
	cudaFree(V);
	cudaFree(W);
	cudaFree(BLEN);
	cudaFree(in_image);
	cudaFree(in_image_shift);
	cudaFree(incomp_image);
	cudaFree(out_image);
	cudaFree(out_image_shift);
	cudaFree(comp_image);
	//cudaFree(args);
	unsigned long long t2 = myCPUTimer();
	unsigned long long tottime = (t2 - t1);
	float membw = (Ntotal*sizeof(cufftDoubleReal) +Nout*sizeof(cufftDoubleComplex))/(tottime*1e3);
	float thruput = Ntotal/(tottime*1e3); 

	std::cout << "Total execution time (microseconds):" << tottime << std::endl;
	std::cout << "Effective memory bandwidth (GB/s):" << membw << std::endl;
	std::cout << "Effective throughput (GFLOPS):" << thruput << std::endl;
	return 0;
	*/
}
