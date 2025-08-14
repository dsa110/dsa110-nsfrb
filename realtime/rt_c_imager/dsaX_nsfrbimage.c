/* fringestops/calibrates and writes nsfrb data to disk
*/

#include <fftw3.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <time.h>
#include <arpa/inet.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <syslog.h>

#include "sock.h"
#include "tmutil.h"
#include "dada_client.h"
#include "dada_def.h"
#include "dada_hdu.h"
#include "multilog.h"
#include "ipcio.h"
#include "ipcbuf.h"
#include "dada_affinity.h"
#include "ascii_header.h"
#include "dsaX_def.h"
#include "dsaX_nsfrbimage.h"

#include <lapacke.h>
#include <cblas.h>
/* global variables */
int DEBUG = 0;
int NINTS_PER_FILE = 2250; // approx 300s





void dsaX_dbgpu_cleanup (dada_hdu_t * in, dada_hdu_t * out);
void setup_frequencies(double *freq_axis_fullres,double *freq_axis_fullres_GHz, double *freq_axis, double *freq_axis_GHz);
//void dsaX_dbgpu_cleanup (dada_hdu_t * in);

void usage()
{
  fprintf (stdout,
	   "dsaX_nsfrb [options]\n"
	   " -c core   bind process to CPU core\n"
	   " -d debug [default no]\n"
	   " -k in_key [default XGPU_BLOCK_KEY]\n"
	   " -o out_key [default XGPU_BLOCK_KEY]\n"
	   " -f filename base [default ~/tmp]\n"
	   " -s SB number to include in filename [default 0]\n"
	   " -t full path to fstable [if not provided will not fringestop]\n"
	   " -j number of frequency integrations to average [default 48]\n"
	   " -e declination to add to header [default 0.0]\n"
	   " -h        print usage\n");
}

void dsaX_dbgpu_cleanup (dada_hdu_t * in, dada_hdu_t * out)
{

  if (dada_hdu_unlock_read (in) < 0)
    {
      printf("could not unlock read on hdu_in\n");//syslog(LOG_ERR, "could not unlock read on hdu_in");
    }
  dada_hdu_destroy (in);

  if (dada_hdu_unlock_write (out) < 0)
    {
      printf("could not unlock write on hdu_out\n");//syslog(LOG_ERR, "could not unlock write on hdu_out");
    }
  dada_hdu_destroy (out);
  
}


// Data structure to hold command line arguments
typedef struct cmdargs_ {
        //input data
	int sb ;
	float dec;
        char *dada_inkey;
	char *dada_outkey;
        int num_time_samples;
        bool verbose ;
        bool search ;
        bool save ;
        long max_base;

        //injection parameters
        bool inject ;
        bool slowinject;
        double snr_inject ;
        double snr_min_inject ;
        double snr_max_inject ;
        double dm_inject;
        int width_inject ;
        int offsetRA_inject;
        int offsetDEC_inject;
        bool offline ;
        bool inject_noiseonly;
        bool inject_noiseless ;
        int num_inject ;
        bool flat_field ;
        bool gauss_field;
        bool point_field;

        //imaging parameters
        int num_chans ;
        int num_chans_per_node ;
        bool briggs ;
        float robust ;
        double sleeptime;
        double bmin ;
        double bmax ;
        bool wstack ;
        bool wstack_parallel;
        int Nlayers ;
        int gridsize ;
        int pixperFWHM ;


        //flagging parameters
        bool flagSWAVE;
        bool flagBPASS ;
        bool flagFRCBAND;
        bool flagBPASSBURST ;
        int *flagcorrs;
        int *flagants;
        int *flagchans;
        int *flagbase;
        int nflagcorrs ;
        int nflagants ;
        int nflagchans ;
        int nflagbase ;

        //processing parameters
	float rttimeout;
	char *logfile;

	//additional parameters
	int core ;
	int nfq ;

} cmdargs;

cmdargs *parseargs(cmdargs *args, int argc, char *argv[])
{
        /*
           Parses command line arguments and stores results in args struct
        */
	//default values
	//input data
        args->sb = 0;
        args->dec=0;
        args->num_time_samples = 25;
        args->verbose = 0;
        args->search = 0;
        args->save = 0;
        args->max_base = 4656;

        //injection parameters
        args->inject = 0;
        args->slowinject = 0;
        args->snr_inject =-1;
        args->snr_min_inject = 1e7;
        args->snr_max_inject = 1e8;
        args->dm_inject = -1;
        args->width_inject = -1;
        args->offsetRA_inject = 0;
        args->offsetDEC_inject = 0;
        args->offline = 0;
        args->inject_noiseonly = 0;
        args->inject_noiseless =0;
        args->num_inject = 0;
        args->flat_field = 0;
        args->gauss_field = 0;
        args->point_field = 0;

	//imaging parameters
        args->num_chans = 16;
        args->num_chans_per_node = 8;
        args->briggs = 0;
        args->robust = 0;
        args->sleeptime = 0;
        args->bmin = 20;
        args->bmax = 20000;
        args->wstack = 0;
        args->wstack_parallel = 0;
        args->Nlayers = 18;
        args->gridsize = 301;
        args->pixperFWHM = 3;


        //flagging parameters
        args->flagSWAVE = 0;
        args->flagBPASS = 0;
        args->flagFRCBAND = 0;
        args->flagBPASSBURST = 0;
        args->nflagcorrs = 0;
        args->nflagants = 0;
        args->nflagchans = 0;
        args->nflagbase = 0;

	
        //processing parameters
        args->rttimeout = 3.35;
	args->logfile = "";


        //additional parameters
        args->core = -1;
        args->nfq = 48;

        int i =0;
        while (i < argc)
        {
                if (argv[i][0] == 45 && argv[i][1] == 45)
                {
                        if (strcmp(argv[i] + 2,"sb") == 0) {args->sb = atoi(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"dec") == 0) {args->dec = atof(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"dada_inkey") == 0) {args->dada_inkey = argv[i+1];}
                        else if (strcmp(argv[i] + 2,"dada_outkey") == 0) {args->dada_outkey = argv[i+1];}
                        else if (strcmp(argv[i] + 2,"num_time_samples") == 0) {args->num_time_samples = atoi(argv[i+1]);}
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

				args->flagcorrs = (int *)malloc(tmp_i*sizeof(int)); //cudaMallocManaged(&(args->flagcorrs), tmp_i*sizeof(int));
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

                                args->flagants = (int *)malloc(tmp_i*sizeof(int)); //cudaMallocManaged(&(args->flagants), tmp_i*sizeof(int));
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

                                args->flagchans = (int *)malloc(tmp_i*sizeof(int)); //cudaMallocManaged(&(args->flagchans), tmp_i*sizeof(int));
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


                                args->flagbase = (int *)malloc(tmp_i*sizeof(int)); //cudaMallocManaged(&(args->flagbase), tmp_i*sizeof(int));
                                for (int j = 0; j < tmp_i; j+= 1)
                                {
                                        (args->flagbase)[j] = atoi(argv[i+1+j]);
                                }
                        }
			else if (strcmp(argv[i] + 2,"rttimeout") == 0) {args->rttimeout = atof(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"logfile") == 0) {args->logfile = argv[i+1];} 
			else if (strcmp(argv[i] + 2,"core") == 0) {args->core = atoi(argv[i+1]);}
			else if (strcmp(argv[i] + 2,"nfq") == 0) {args->nfq = atoi(argv[i+1]);}
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


void uniform_grid(double *U, double *V, double *W, long nbase,double *fobs_GHz, int num_chans_per_node, int gridsize, double uv_max, double grid_res, float robust, unsigned int *i_indices, unsigned int *j_indices, unsigned int *i_conj_indices, unsigned int *j_conj_indices)
{
        /*
           This function uniformly grids visibilities for the given observing frequency
        */
        //int fidx = blockIdx.x * nbase;
        //int bidx = threadIdx.x * nbase_thread;
        double lambda_m;// = C_GHZ_M/fobs_GHz[blockIdx.x];
        int shiftby = gridsize/2;

        //for (int i = bidx; i < bidx+nbase_thread; i+=1)
	for (int fidx=0; fidx<num_chans_per_node; fidx++)
	{
		lambda_m = C_GHZ_M/fobs_GHz[fidx];
		for (int i =0; i<nbase;i+=1)
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
}


void briggs_weight(long nbase, int gridsize, unsigned int *i_indices, unsigned int *j_indices, double *bweights, float robust, unsigned int *Wk, double *vis_weights, int *flagbase, int *flagchans, int *flagcorrs, int *flagants, int nflagbase, int nflagchans, int nflagcorrs, int nflagants, uint8_t *ANT1, uint8_t *ANT2, int num_chans_per_node)
{
        /*
           This function computes grid weights using Briggs robustness parameter
        */

	for (int f_i =0; f_i<num_chans_per_node; f_i++)
	{
		int fidx = f_i * nbase;
        	int Widx = f_i * (gridsize*gridsize);
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
}


void get_conj(size_t total_samples, float *data, float *data_conj)
{
	/*Fast complex conjugate of array with cblas
	 */

	//make array of -2di
	cblas_saxpy(total_samples, -2.0,
                                data + 1,
                                2,
                                data_conj + 1,
                                2);
	//add together
	cblas_saxpy(total_samples*2, 1.0,
		       data, 1,
		       data_conj, 1);
		
	
	//for (int i=0; i<total_samples; i++)
	//{
	//	printf("%f+j%f --> %f+j%f\n",data[i][0],data[i][1],data_conj[i][0],data_conj[i][1]);
	//}
}

void grid_data(int num_time_samples, int num_chans_per_node, int nbase, 
                fftwf_complex *data, fftwf_complex *data_conj, unsigned int *i_indices, unsigned int *j_indices,
                unsigned int *i_conj_indices, unsigned int *j_conj_indices,
                double *bweights, int gridsize, fftwf_complex *vis_grid,
                int *flagbase, int *flagchans, int *flagcorrs, int *flagants,
                int nflagbase, int nflagchans, int nflagcorrs, int nflagants,
                uint8_t *ANT1, uint8_t *ANT2, int sb, bool briggs, 
		double *BLEN, double bmin, double bmax)
{
        /*
           This function applies pre-computed weights to and grids visibilities.
           Note the grid should be arranged time x channel x U x V to properly
           interface with cufft.

           Input data is arranged subband x time x baseline x channel x polarization
           Input weights are arranged channel x baseline
         */
	bool flagflag =0;
	fftwf_complex bweight_complex;
	for (int chann=0; chann<num_chans_per_node;chann++)
	{
        	//indices for weights
        	int fidx = chann * nbase;

        	//check if channel is flagged
        	for (int k_f=0; k_f < nflagcorrs; k_f++)
        	{
			if (flagcorrs[k_f] == sb)
			{
				return;
			}
                	//if (flagcorrs[k_f]*num_chans_per_node <= blockIdx.x && blockIdx.x <= (k_f+1)*num_chans_per_node)
                	//{
                        //	return;
                	//}
        	}
        	for (int k_f=0; k_f < nflagchans; k_f++)
        	{
                	if (chann == k_f)
                	{
				flagflag=1;
                        	continue;
                	}
        	}
		if (flagflag==1)
                {
                        flagflag =0;
                	continue;
                }



        	//int bidx = threadIdx.x * nbase_thread;

        	//indices for data
        	//int sidx_data = chann/num_chans_per_node;
        	//sidx_data *= (num_time_samples * nbase * num_chans_per_node * 2);
        	int fidx_data = chann*2; //(chann % num_chans_per_node)*2; //channels
        	//int bidx_data = threadIdx.x * nbase_thread * num_chans_per_node * 2; //baseline

        	//indices for grid
        	int fidx_grid = chann * gridsize * gridsize;
        	int data_index = 0;
        	int weight_index = 0;
        	int grid_index = 0;
        	int grid_conj_index =0 ;
		for (int k=0; k<nbase; k++)
        	{
                	//check if baseline is flagged
                	for (int k_f=0; k_f<nflagbase; k_f++)
                	{
                        	if (k==flagbase[k_f])
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
                        	if (ANT1[k]==flagants[k_f] || ANT2[k]==flagants[k_f])
                        	{
                                	flagflag = 1;
                        	}
                	}
                	if (flagflag==1)
                	{
                        	flagflag =0;
                        	continue;
                	}
			if (BLEN[k]<=bmin || BLEN[k]>=bmax)
			{
				continue;
			}


			/*
			 * cublas
			 */
			weight_index = fidx +k;
			if (briggs && bweights != NULL) {bweight_complex[0] = bweights[weight_index];}
			else {bweight_complex[0] = 1.0;}
			bweight_complex[1] = 0.0;
			data_index = k*num_chans_per_node*2 + fidx_data;
			grid_index = fidx_grid + (i_indices[weight_index]*gridsize + j_indices[weight_index]);
			grid_conj_index = fidx_grid + (i_conj_indices[weight_index]*gridsize + j_conj_indices[weight_index]);
			//index
			cblas_caxpy(num_time_samples, bweight_complex,
				data + data_index,
				nbase*num_chans_per_node*2,
				vis_grid + grid_index,
				num_chans_per_node*gridsize*gridsize);
			cblas_caxpy(num_time_samples, bweight_complex,
                                data + data_index + 1,
                                nbase*num_chans_per_node*2,
                                vis_grid + grid_index,
                                num_chans_per_node*gridsize*gridsize);
			//conjugate index
                        cblas_caxpy(num_time_samples, bweight_complex,
                                data_conj + data_index,
                                nbase*num_chans_per_node*2,
                                vis_grid + grid_conj_index,
                                num_chans_per_node*gridsize*gridsize);
                        cblas_caxpy(num_time_samples, bweight_complex,
                                data_conj + data_index + 1,
                                nbase*num_chans_per_node*2,
                                vis_grid + grid_conj_index,
                                num_chans_per_node*gridsize*gridsize);

			/*
			 *
			 */

			
			/*
                	//weight/i/j index
                	weight_index = fidx + k;

                	for (int kt=0; kt<num_time_samples; kt++)
                	{
                        	//grid index
                        	grid_index = (kt*num_chans_per_node*gridsize*gridsize) + fidx_grid + (i_indices[weight_index]*gridsize + j_indices[weight_index]);
                        	grid_conj_index = (kt*num_chans_per_node*gridsize*gridsize) + fidx_grid + (i_conj_indices[weight_index]*gridsize + j_conj_indices[weight_index]);
                        	for (int kp=0; kp<2; kp++)
                        	{
                                	//data index
                                	data_index = (kt*nbase*num_chans_per_node*2) + (k*num_chans_per_node*2) + fidx_data + kp; //sidx_data + (kt*nbase*num_chans_per_node*2) + bidx_data + (k*num_chans_per_node*2) + fidx_data + kp;

                                	//skip if nan
                                	//if (isnan((double)(data[data_index].x)) || isnan((double)(data[data_index].y))
                                        //        || isnan(bweights[weight_index]))
                                	//{
                                        //	continue;
                                	//}

                                	//weight and add to grid
                                	if (briggs && bweights != NULL)
                                	{
                                        	vis_grid[grid_index][0] += data[data_index][0]*bweights[weight_index];
                                        	vis_grid[grid_index][1] += data[data_index][1]*bweights[weight_index];
                                        	vis_grid[grid_conj_index][0] += data[data_index][0]*bweights[weight_index];
                                        	vis_grid[grid_conj_index][1] -= data[data_index][1]*bweights[weight_index];
                                	}
                                	else
                                	{
                                        	vis_grid[grid_index][0] += data[data_index][0];
                                        	vis_grid[grid_index][1] += data[data_index][1];
                                        	vis_grid[grid_conj_index][0] += data[data_index][0];
                                        	vis_grid[grid_conj_index][1] -= data[data_index][1];
                                	}
                        	}

                	}
			*/
        	}
	}
}

void image_combine(int gridsize, int num_time_samples, int num_chans_per_node, fftwf_complex *vis_grid, double *image)
{
	/*
	 * Sums over the channel
	 */
	int imgidx;
        int grididx;
	for (int i=0; i<num_time_samples; i++)
        {
                for (int k=0; k<gridsize*gridsize; k++)
                {
                        imgidx = (i*num_chans_per_node*gridsize*gridsize) + k;
                        for (int j=0; j<num_chans_per_node; j++)
                        {
                                grididx = (i*num_chans_per_node*gridsize*gridsize) + (j*gridsize*gridsize) + k;
                                image[imgidx] += vis_grid[grididx][0];
                        }
                }
        }
}
/*
fftw_plan image_data(fftw_plan plan_for_fft, int gridsize, int num_time_samples, int num_chans_per_node, fftw_complex *vis_grid, double *image)
{
	 * fftw implemented 2d fft (backwards, unnormalized), summed over channels
	 * input array is time x channel x U x V
	 * output array is time x RA x DEC
	int imgidx;
	int grididx;
	for (int i=0; i<num_time_samples; i++)
	{
		for (int j=0; j<num_chans_per_node; j++)
		{
			grididx = (i*num_chans_per_node*gridsize*gridsize) + j*gridsize*gridsize;
			plan_for_fft = fftw_plan_dft_2d(gridsize, gridsize, vis_grid + grididx, vis_grid + grididx,
                        					FFTW_FORWARD, FFTW_MEASURE | FFTW_IN_PLACE);
			fftw_execute(plan_for_fft);
		}
	}
	for (int i=0; i<num_time_samples; i++)
	{	
		for (int k=0; k<gridsize*gridsize; k++)
		{
			imgidx = (i*num_chans_per_node*gridsize*gridsize) + k;
			for (int j=0; j<num_chans_per_node; j++)
			{
				grididx = (i*num_chans_per_node*gridsize*gridsize) + (j*gridsize*gridsize) + k;
				image[imgidx] += vis_grid[grididx][0];
			}
		}
	}		

	return fftw_plan
}
*/

//generic timer function from https://stackoverflow.com/questions/69136940/timing-kernel-execution-with-cpu-timers
unsigned long long myCPUTimer(void)
{
        struct timeval tv;
        gettimeofday(&tv,0);
        return ((tv.tv_sec*USECPSEC) + tv.tv_usec);
}



int main(int argc, char *argv[])
{
	sleep(180);
	
	FILE *fobj;
        //double pi = 22.0/7.0;
        cmdargs args_obj;
        cmdargs *args = &args_obj;
        //cudaMallocManaged(&args,sizeof(cmdargs));
        parseargs(args,argc,argv);
	bool uselogfile = (args->verbose) && (strlen(args->logfile)>0);

	if (uselogfile)
	{
		fobj = fopen(args->logfile,"w");
		fprintf(fobj,"START LOG NOW\n");
		fclose(fobj);
	}


	const char *cwd = getenv("NSFRBDIR");


	/*
        char baseUVWcmd[300];
        strcpy(baseUVWcmd, "conda init; conda activate casa310nsfrb; python ");
        strcat(baseUVWcmd, cwd);
        strcat(baseUVWcmd, "/realtime/rt_c_imager/_getbaselines.py --outdir ");
	*/
        char ufname[300];
        strcpy(ufname,cwd);
        strcat(ufname,"/realtime/rt_c_imager/U.bin");

        char vfname[300];
        strcpy(vfname,cwd);
        strcat(vfname,"/realtime/rt_c_imager/V.bin");

        char wfname[300];
        strcpy(wfname,cwd);
        strcat(wfname,"/realtime/rt_c_imager/W.bin");

        char bfname[300];
        strcpy(bfname,cwd);
        strcat(bfname,"/realtime/rt_c_imager/BLEN.bin");

        char a1fname[300];
        strcpy(a1fname,cwd);
        strcat(a1fname,"/realtime/rt_c_imager/ANT1.bin");

        char a2fname[300];
        strcpy(a2fname,cwd);
        strcat(a2fname,"/realtime/rt_c_imager/ANT2.bin");

	/*
        char outdirUVWcmd[300];
        strcpy(outdirUVWcmd,cwd);
        strcat(outdirUVWcmd,"/realtime/rt_c_imager/");
        strcat(baseUVWcmd, outdirUVWcmd);
        strcat(baseUVWcmd, " --pt_dec ");
	*/
        char imgfname[300];
        strcpy(imgfname,cwd);
        strcat(imgfname,"/realtime/rt_c_imager/tmpimage.bin");

	if (uselogfile)
        {
                fobj = fopen(args->logfile,"a");
                fprintf(fobj,"retrieved cwd=%s\n",cwd);
		//fprintf(fobj,"UVW command: %s\n",baseUVWcmd);
		fprintf(fobj,"%d ANTENNAS WILL BE FLAGGED\n",args->nflagants);
                fclose(fobj);
        }
	else{
        	printf("retrieved cwd=%s\n",cwd);
        	//printf("UVW command: %s\n",baseUVWcmd);
		printf("%d ANTENNAS WILL BE FLAGGED\n",args->nflagants);
	}



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
        double *fobs = (double *)malloc(num_tot_chans*sizeof(double)); //[num_tot_chans];
        double *fobs_GHz = (double *)malloc(num_tot_chans*sizeof(double)); //[num_tot_chans];
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



	//update and set UVW array
	//get UVW coordinates from file
        if (uselogfile) 
	{
		fobj = fopen(args->logfile,"a");
		fprintf(fobj,"Updating and reading UVW coords\n");
		fclose(fobj);
	}
	else
	{
		printf("Updating and reading UVW coords\n");
	}
	/*
        char updateUVWcmd[strlen(baseUVWcmd)+20];
        strcpy(updateUVWcmd,baseUVWcmd);
        sprintf(updateUVWcmd + strlen(baseUVWcmd),"%f",(args->dec)*pi/180);
        int res = system(updateUVWcmd);
	if (res!=0)
	{
		if (uselogfile)
        	{
                	fobj = fopen(args->logfile,"a");
                	fprintf(fobj,"UVW update failed\n");
			fprintf(fobj,"%s\n",updateUVWcmd);
			fclose(fobj);
		}
		else
		{
			printf("UVW update failed\n");//syslog(LOG_ERR, "UVW update failed\n");
			printf("%s\n",updateUVWcmd);
		}
		//dsaX_dbgpu_cleanup (hdu_in, hdu_out);
                return EXIT_FAILURE;
	}
	*/
	FILE *fobj_r;
	fobj_r = fopen(ufname,"rb");
        double *U;
	size_t nread;
        U = (double *)malloc((args->max_base)*sizeof(double));//cudaMallocManaged(&U,(args->max_base)*sizeof(double));
        nread = fread(U,sizeof(double),args->max_base,fobj_r)*sizeof(double);
        fclose(fobj_r);

        fobj_r = fopen(vfname,"rb");
        double *V;
        V = (double *)malloc((args->max_base)*sizeof(double));//cudaMallocManaged(&V,(args->max_base)*sizeof(double));
        nread = fread(V,sizeof(double),args->max_base,fobj_r)*sizeof(double);
        fclose(fobj_r);

        fobj_r = fopen(wfname,"rb");
        double *W;
        W = (double *)malloc((args->max_base)*sizeof(double));//cudaMallocManaged(&W,(args->max_base)*sizeof(double));
        nread = fread(W,sizeof(double),args->max_base,fobj_r)*sizeof(double);
        fclose(fobj_r);

        fobj_r = fopen(bfname,"rb");
        double *BLEN;
        BLEN = (double *)malloc((args->max_base)*sizeof(double));//cudaMallocManaged(&BLEN,(args->max_base)*sizeof(double));
        nread = fread(BLEN,sizeof(double),args->max_base,fobj_r)*sizeof(double);
        fclose(fobj_r);

        fobj_r = fopen(a1fname,"rb");
        uint8_t *ANT1;
        ANT1 = (uint8_t *)malloc((args->max_base)*sizeof(uint8_t)); //cudaMallocManaged(&ANT1,(args->max_base)*sizeof(unsigned int));
        nread = fread(ANT1,sizeof(uint8_t),args->max_base,fobj_r)*sizeof(uint8_t);
        fclose(fobj_r);

        fobj_r = fopen(a2fname,"rb");
        uint8_t *ANT2;
        ANT2 = (uint8_t *)malloc((args->max_base)*sizeof(uint8_t)); //cudaMallocManaged(&ANT2,(args->max_base)*sizeof(unsigned int));
        nread = fread(ANT2,sizeof(uint8_t),args->max_base,fobj_r)*sizeof(uint8_t);
        fclose(fobj_r);
	if (uselogfile)
        {
                fobj = fopen(args->logfile,"a");
                fprintf(fobj,"Read %ld bytes from baseline files\n",nread);
                fclose(fobj);
        }
        else
        {
		printf("Read %ld bytes from baseline files\n",nread);
        }


        //get the maximum baseline length
        int uv_diag_idx=0;
        double uv_diag=-1.0;
	bool flagflag = 0;
	for (int i=0; i<args->max_base; i++)
	{
		//check if baseline is flagged
                for (int k_f=0; k_f<args->nflagbase; k_f++)
                {
                	if (i==args->flagbase[k_f])
                        {
        	        	flagflag = 1;
                        }
                }
                if (flagflag==1)
                {
	                flagflag =0;
	                continue;
		}

                for (int k_f=0; k_f<(args->nflagants); k_f++)
                {
	                if (ANT1[i]==(args->flagants)[k_f] || ANT2[i]==(args->flagants)[k_f])
                        {
				if (uselogfile)
        			{
                			fobj = fopen(args->logfile,"a");
                			fprintf(fobj,"FLAGGING BASELINE WITH ANTENNAS %u & %u\n",ANT1[i],ANT2[i]);
					fclose(fobj);
        			}
				else
				{
					printf("FLAGGING BASELINE WITH ANTENNAS %u & %u\n",ANT1[i],ANT2[i]);
				}
        	                flagflag = 1;
                        }
                }
                if (flagflag==1)
		{
	        	flagflag =0;
        	        continue;
                }
                if (BLEN[i]<=args->bmin || BLEN[i]>=args->bmax)
                {
                        continue;
                }

		if (BLEN[i]>uv_diag) 
		{
			uv_diag_idx = i;
			uv_diag = BLEN[uv_diag_idx];
		}
	}
        double pixel_resolution = (lambdaref_/uv_diag)/(args->pixperFWHM);
        double uv_resolution = 1/((args->gridsize)*pixel_resolution);
        double uv_max = uv_resolution*(args->gridsize)/2;
        double grid_res = 2*uv_max/(args->gridsize);
	if (uselogfile)
        {
                fobj = fopen(args->logfile,"a");
                fprintf(fobj,"Maximum baseline length is %f meters; resolution is %f meters\n",uv_max,grid_res);
                fclose(fobj);
        }
	else
	{
		printf("Maximum baseline length is %f meters; resolution is %f meters\n",uv_max,grid_res);
	}

	//get grid indices
	if (uselogfile)
	{
		fobj = fopen(args->logfile,"a");
		fprintf(fobj,"Computing grid indices...\n");
		fclose(fobj);	
	}
	else
	{
		printf("Computing grid indices...\n");
	}
	unsigned int *i_indices,*j_indices,*i_conj_indices,*j_conj_indices;
        i_indices = (unsigned int *)malloc((args->max_base)*(args->num_chans_per_node)*sizeof(unsigned int));//cudaMallocManaged(&i_indices,(args->max_base)*num_tot_chans*sizeof(unsigned int));
        j_indices = (unsigned int *)malloc((args->max_base)*(args->num_chans_per_node)*sizeof(unsigned int));//cudaMallocManaged(&j_indices,(args->max_base)*num_tot_chans*sizeof(unsigned int));
        i_conj_indices = (unsigned int *)malloc((args->max_base)*(args->num_chans_per_node)*sizeof(unsigned int));//cudaMallocManaged(&i_conj_indices,(args->max_base)*num_tot_chans*sizeof(unsigned int));
        j_conj_indices = (unsigned int *)malloc((args->max_base)*(args->num_chans_per_node)*sizeof(unsigned int)); //cudaMallocManaged(&j_conj_indices,(args->max_base)*num_tot_chans*sizeof(unsigned int));
	uniform_grid(U, V, W, args->max_base,
			fobs_GHz + (args->sb*(args->num_chans_per_node)), 
			args->num_chans_per_node, 
			args->gridsize, uv_max, grid_res, 
			args->robust, i_indices, j_indices, 
			i_conj_indices,j_conj_indices);
	if (uselogfile)
        {
                fobj = fopen(args->logfile,"a");
                fprintf(fobj,"done\n");
		fclose(fobj);
        }
        else
        {
		printf("done\n");
	}


	//briggs robust weighting
        unsigned int *Wk;
        double *vis_weights;
        double *bweights;
	Wk = (unsigned int *)malloc((args->gridsize)*(args->gridsize)*(args->num_chans_per_node)*sizeof(unsigned int));//cudaMallocManaged(&Wk,(args->gridsize)*(args->gridsize)*num_tot_chans*sizeof(unsigned int));
        vis_weights = (double *)malloc((args->max_base)*(args->num_chans_per_node)*sizeof(double)); //cudaMallocManaged(&vis_weights,(args->max_base)*num_tot_chans*sizeof(double));
        bweights = (double *)malloc((args->max_base)*(args->num_chans_per_node)*sizeof(double)); //cudaMallocManaged(&bweights,(args->max_base)*num_tot_chans*sizeof(double));
	memset(Wk, 0, (args->gridsize)*(args->gridsize)*(args->num_chans_per_node)*sizeof(unsigned int));
	memset(vis_weights, 1.0, (args->max_base)*(args->num_chans_per_node)*sizeof(double));
	memset(bweights,0.0,(args->max_base)*(args->num_chans_per_node)*sizeof(double));


        if (args->briggs) {
		if (uselogfile)
        	{
                	fobj = fopen(args->logfile,"a");
                	fprintf(fobj,"Generating briggs weights...\n");
               		fclose(fobj);
        	}
        	else
        	{
                	printf("Generating briggs weights...\n");
		}

		briggs_weight(args->max_base, args->gridsize, i_indices, j_indices, 
				bweights, args->robust, Wk, vis_weights, 
				args->flagbase, args->flagchans, args->flagcorrs, args->flagants, 
				args->nflagbase, args->nflagchans, args->nflagcorrs, args->nflagants, 
				ANT1, ANT2, args->num_chans_per_node);

                /*for (int i=0; i<(args->max_base)*(args->num_chans_per_node);i++)
                {
                        printf("BASECHAN %d: BWEIGHT %f\n",i,bweights[i]);
                }*/
		if (uselogfile)
                {
                        fobj = fopen(args->logfile,"a");
                        fprintf(fobj,"Done\n");
                        fclose(fobj);
                }
                else
                {
                        printf("Done\n");
		}

        }



	//read data from dada buffer [adapted from dsaX_nsfrb.c]
        /* DADA defs */
        dada_hdu_t* hdu_in = 0;
        dada_hdu_t* hdu_out = 0;
        key_t in_key = XGPU_BLOCK_KEY;
        key_t out_key = XGPU_BLOCK_KEY;

        sscanf (args->dada_inkey, "%x", &in_key);
        sscanf (args->dada_outkey, "%x", &out_key);


	// DADA stuff
	if (uselogfile)
        {
        	fobj = fopen(args->logfile,"a");
                fprintf(fobj,"creating hdu\n");
		fclose(fobj);
	}
	else
	{
        	printf("creating hdu\n");//syslog (LOG_INFO, "creating hdu");
	}

        hdu_in  = dada_hdu_create ();
        dada_hdu_set_key (hdu_in, in_key);
        if (dada_hdu_connect (hdu_in) < 0) {
                if (uselogfile)
        	{
                	fobj = fopen(args->logfile,"a");
                	fprintf(fobj,"could not connect to dada buffer\n");
                	fclose(fobj);
        	}
        	else
        	{
                	printf("could not connect to dada buffer\n");//syslog (LOG_ERR,"could not connect to dada buffer");
		}
                return EXIT_FAILURE;
        }
        if (dada_hdu_lock_read (hdu_in) < 0) {
                if (uselogfile)
                {
                        fobj = fopen(args->logfile,"a");
                        fprintf(fobj,"could not lock to dada buffer\n");
                        fclose(fobj);
                }
                else
                {
                        printf("could not lock to dada buffer\n");//syslog (LOG_ERR,"could not lock to dada buffer");
		}
                return EXIT_FAILURE;
        }

        hdu_out  = dada_hdu_create ();
        dada_hdu_set_key (hdu_out, out_key);
        if (dada_hdu_connect (hdu_out) < 0) {
		if (uselogfile)
                {
                        fobj = fopen(args->logfile,"a");
                        fprintf(fobj,"could not connect to output dada buffer\n");
                        fclose(fobj);
                }
                else
                {
                	printf("could not connect to output  buffer\n"); //syslog (LOG_ERR,"could not connect to output  buffer");
		}
                return EXIT_FAILURE;
        }
        if (dada_hdu_lock_write(hdu_out) < 0) {
                if (uselogfile)
                {
                        fobj = fopen(args->logfile,"a");
                        fprintf(fobj,"could not lock to output dada buffer\n");
                        fclose(fobj);
                }
                else
                {
                        printf("could not lock to output buffer\n"); //syslog (LOG_ERR, "could not lock to output buffer");
		}
                return EXIT_FAILURE;
        }

        // Bind to cpu core
        if (args->core >= 0){
                if (uselogfile)
                {
                        fobj = fopen(args->logfile,"a");
                        fprintf(fobj,"binding to core %d\n",args->core);
                        fclose(fobj);
                }
                else
                {
                        printf("binding to core %d\n",args->core); //syslog(LOG_INFO,"binding to core %d", args->core);
		}
                if (dada_bind_thread_to_core(args->core) < 0) 
		{
			if (uselogfile)
                	{
                        	fobj = fopen(args->logfile,"a");
                        	fprintf(fobj,"failed to bind to core %d\n",args->core);
                        	fclose(fobj);
                	}
                	else
                	{
                        	printf("failed to bind to core %d\n",args->core);//syslog(LOG_ERR,"failed to bind to core %d", args->core);
			}
		}
		if (uselogfile)
                {
                        fobj = fopen(args->logfile,"a");
                        fprintf(fobj,"Done\n");
                        fclose(fobj);
                }
                else
                {
                        printf("Done\n");
                }
        }

        int observation_complete=0;
        uint64_t header_size = 0;

        // deal with headers
        char * header_in = ipcbuf_get_next_read (hdu_in->header_block, &header_size);
        if (!header_in)
        {
		if (uselogfile)
                {
                        fobj = fopen(args->logfile,"a");
                        fprintf(fobj,"could not read next header\n");
                        fclose(fobj);
                }
                else
                {
                        printf("could not read next header\n");
                }
                dsaX_dbgpu_cleanup (hdu_in, hdu_out);
                return EXIT_FAILURE;
        }
        if (ipcbuf_mark_cleared (hdu_in->header_block) < 0)
        {
                if (uselogfile)
                {
                        fobj = fopen(args->logfile,"a");
                        fprintf(fobj,"could not mark header block cleared\n");
                        fclose(fobj);
                }
                else
                {
                        printf("could not mark header block cleared\n");
                }
                dsaX_dbgpu_cleanup (hdu_in, hdu_out);
                return EXIT_FAILURE;
        }

        char * header_out = ipcbuf_get_next_write (hdu_out->header_block);
        if (!header_out)
        {
                if (uselogfile)
                {
                        fobj = fopen(args->logfile,"a");
                        fprintf(fobj,"could not get next header block [output]\n");
                        fclose(fobj);
                }
                else
                {
                        printf("could not get next header block [output]\n");
                }
                dsaX_dbgpu_cleanup (hdu_in, hdu_out);
                return EXIT_FAILURE;
        }
        memcpy (header_out, header_in, header_size);
        if (ipcbuf_mark_filled (hdu_out->header_block, header_size) < 0)
        {
                if (uselogfile)
                {
                        fobj = fopen(args->logfile,"a");
                        fprintf(fobj,"could not mark header block filled [output]\n");
                        fclose(fobj);
                }
                else
                {
                        printf("could not mark header block filled [output]\n");
                }

                dsaX_dbgpu_cleanup (hdu_in, hdu_out);
                return EXIT_FAILURE;
        }
	// data stuff
	uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
  	uint64_t block_out = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_out->data_block);
	if (uselogfile)
                {
                        fobj = fopen(args->logfile,"a");
                        fprintf(fobj,"done\n");
			fprintf(fobj,"allocating space for data...\n");
			fclose(fobj);
		}
	else
	{
		printf("done\n");
		printf("allocating space for data...\n");
	}

        uint64_t bytes_read = 0, block_id;
        uint64_t written;
        size_t total_samples = (args->num_time_samples)*(args->num_chans_per_node)*(args->max_base)*2;
        //char *block;// = (char *)malloc(sizeof(float)*total_samples*2);
        //float *data_f = (float *)malloc(sizeof(float)*total_samples*2);
        fftwf_complex *data;// = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*total_samples);
	fftwf_complex *data_conj = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*total_samples);
        fftwf_complex *vis_grid = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex)*(args->gridsize)*(args->gridsize)*(args->num_time_samples)*(args->num_chans_per_node));
        double *image = (double *)malloc(sizeof(double)*(args->gridsize)*(args->gridsize)*(args->num_time_samples));
	memset(data_conj,   0, sizeof(fftwf_complex)*total_samples*2);
        //memset(data_f, 0, sizeof(float)*total_samples*2);
        //memset(data,   0, sizeof(fftwf_complex)*total_samples*2);
	if (uselogfile)
                {
                        fobj = fopen(args->logfile,"a");
                        fprintf(fobj,"done\n");
                        fprintf(fobj,"Creating fft plan...\n");
                        fclose(fobj);
                }
        else
        {
                printf("done\n");
                printf("Creating fft plan...\n");
        }
        //float *data = (float *)malloc(sizeof(float)*25*4656*(384/nfq)*2*2);
        //memset(data, 0, 25*4656*(384/nfq)*2*2*sizeof(float));

        //int inidx, fsidx, outidx;
        //double mjd, mjd0;
        //int secs;

        //fft plan
        int n[] = {args->gridsize, args->gridsize};
        int idist = (args->gridsize)*(args->gridsize);
        int odist = (args->gridsize)*(args->gridsize);
        fftwf_plan plan_for_fft = fftwf_plan_many_dft(2, n, (args->num_time_samples)*(args->num_chans_per_node),
                                                    vis_grid, n, 1, idist,
                                                    vis_grid, n, 1, odist,
                                                    FFTW_FORWARD, FFTW_MEASURE);

        if (uselogfile)
                {
                        fobj = fopen(args->logfile,"a");
                        fprintf(fobj,"done\n");
                        fprintf(fobj,"starting observation\n");
                        fclose(fobj);
                }
        else
        {
                printf("done\n");
                printf("starting observation\n");
        }
        // start things
	//int iters =0;
	//int maxiters = 10;
	unsigned long long t2, tottime,t1;
	FILE *fobj_w;
  	while (!observation_complete) {
		t1 = myCPUTimer();
		if (uselogfile)
                {
                        fobj = fopen(args->logfile,"a");
                        fprintf(fobj,"Initializing data...");
                        fclose(fobj);
                }
        	else
        	{
                	printf("Initializing data...");
		}
		//re-initialize
		memset(vis_grid,  0, sizeof(fftwf_complex)*(args->gridsize)*(args->gridsize)*(args->num_time_samples)*(args->num_chans_per_node));
        	memset(image,  0, sizeof(double)*(args->gridsize)*(args->gridsize)*(args->num_time_samples));
		if (uselogfile)
                {
                        fobj = fopen(args->logfile,"a");
                        fprintf(fobj,"Done\n");
			fprintf(fobj,"Reading data...");
                        fclose(fobj);
                }
                else
                {
                        printf("Done\n");
			printf("Reading data...");
                }
    		
		// read block
		//block = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);
    		data = (fftwf_complex *)ipcio_open_block_read(hdu_in->data_block, &bytes_read, &block_id);//(fftwf_complex *)block;
		if (uselogfile)
                {
                        fobj = fopen(args->logfile,"a");
                        fprintf(fobj,"Done, read %ld bytes, expected %ld bytes\n",bytes_read,block_size);
                        fprintf(fobj,"Computing complex conjugate...\n");
                        fclose(fobj);
                }
                else
                {
                        printf("Done, read %ld bytes, expected %ld bytes\n",bytes_read,block_size);
                        printf("Computing complex conjugate...");
                }


		t2 = myCPUTimer();
                tottime = (t2 - t1);
                if (tottime>0.9*(1e6)*(args->rttimeout))
                {
			if (uselogfile)
                	{
                        	fobj = fopen(args->logfile,"a");
                        	fprintf(fobj,"main: timed out aftert READDATA: %lld microseconds\n",tottime);
                        	fclose(fobj);
                	}
                	else
                	{
                        	printf("main: timed out aftert READDATA: %lld microseconds\n",tottime);
                	}
			ipcio_close_block_read (hdu_in->data_block, bytes_read);
                        continue;
                }

		//compute complex conjugate
		get_conj(total_samples, (float *)data, (float *)data_conj);
		if (uselogfile)
                {
                        fobj = fopen(args->logfile,"a");
                        fprintf(fobj,"Done\n");
                        fprintf(fobj,"Gridding data...");
                        fclose(fobj);
                }
                else
                {
                        printf("Done\n");
                        printf("Gridding data...");
                }


		//grid data
		grid_data(args->num_time_samples, args->num_chans_per_node, args->max_base,
                	data, data_conj, i_indices, j_indices, i_conj_indices, j_conj_indices,
                	bweights, args->gridsize, vis_grid,
                	args->flagbase, args->flagchans, args->flagcorrs, args->flagants,
                	args->nflagbase, args->nflagchans, args->nflagcorrs, args->nflagants,
                	ANT1, ANT2, args->sb, args->briggs, BLEN, args->bmin, args->bmax);
		if (uselogfile)
                {
                        fobj = fopen(args->logfile,"a");
                        fprintf(fobj,"Done\n");
                        fclose(fobj);
                }
                else
                {
                        printf("Done\n");
                }

		t2 = myCPUTimer();
                tottime = (t2 - t1);
                if (tottime>0.9*(1e6)*(args->rttimeout))
                {
                        if (uselogfile)
                        {
                                fobj = fopen(args->logfile,"a");
                                fprintf(fobj,"main: timed out aftert GRIDDATA: %lld microseconds\n",tottime);
                                fclose(fobj);
                        }
                        else
                        {
                                printf("main: timed out aftert GRIDDATA: %lld microseconds\n",tottime);
                        }
			ipcio_close_block_read (hdu_in->data_block, bytes_read);
                        continue;
                }

		//image
		if (uselogfile)
                {
                        fobj = fopen(args->logfile,"a");
                        fprintf(fobj,"Imaging data...");
                        fclose(fobj);
                }
                else
                {
                        printf("Imaging data...");
                }


		plan_for_fft = fftwf_plan_many_dft(2, (const int *)n, (args->num_time_samples)*(args->num_chans_per_node),
                                                    vis_grid, NULL, 1, idist,
                                                    vis_grid, NULL, 1, odist,
                                                    FFTW_FORWARD, FFTW_MEASURE);
		fftwf_execute(plan_for_fft);
		image_combine(args->gridsize, args->num_time_samples,
                                args->num_chans_per_node, vis_grid, image);
		if (uselogfile)
                {
                        fobj = fopen(args->logfile,"a");
                        fprintf(fobj,"Done\n");
                        fclose(fobj);
                }
                else
                {
                        printf("Done\n");
                }

		t2 = myCPUTimer();
                tottime = (t2 - t1);
                if (tottime>0.9*(1e6)*(args->rttimeout))
                {
			if (uselogfile)
                        {
                                fobj = fopen(args->logfile,"a");
                                fprintf(fobj,"main: timed out aftert IMAGING: %lld microseconds\n",tottime);
                                fclose(fobj);
                        }
                        else
                        {
                                printf("main: timed out aftert IMAGING: %lld microseconds\n",tottime);
                        }
			ipcio_close_block_read (hdu_in->data_block, bytes_read);
                        continue;
                }

		// write to buffer
		if (uselogfile)
                {
                        fobj = fopen(args->logfile,"a");
                        fprintf(fobj,"Writing data...");
                        fclose(fobj);
                }
                else
                {
                        printf("Writing data...");
                }
		written = ipcio_write (hdu_out->data_block, (char *)image, block_out);
      		if (written < block_out)
        	{
          		if (uselogfile)
                        {
                                fobj = fopen(args->logfile,"a");
                                fprintf(fobj,"main: failed to write all data to datablock [output]\n");
                                fclose(fobj);
                        }
                        else
                        {
				printf("main: failed to write all data to datablock [output]\n");
                        }
          		dsaX_dbgpu_cleanup (hdu_in, hdu_out);
			ipcio_close_block_read (hdu_in->data_block, bytes_read);
          		return EXIT_FAILURE;
        	}
			
		//***for testing***//
		if (args->save)
		{
			fobj_w = fopen(imgfname,"wb");
        		fwrite(image, sizeof(double), (args->num_time_samples)*(args->gridsize)*(args->gridsize), fobj_w);
        		fclose(fobj_w);
		}
		if (uselogfile)
                {
                        fobj = fopen(args->logfile,"a");
                        fprintf(fobj,"Done\n");
                        fclose(fobj);
                }
                else
                {
                        printf("Done\n");
                }


		// close off loop
    		if (bytes_read < block_size) {observation_complete = 1;}

    		ipcio_close_block_read (hdu_in->data_block, bytes_read);

		t2 = myCPUTimer();
		tottime = (t2 - t1);
		if (uselogfile)
                {
                        fobj = fopen(args->logfile,"a");
                        fprintf(fobj,"Total execution time:%lld microseconds\n",tottime);
                        fclose(fobj);
                }
                else
                {
                        printf("Total execution time:%lld microseconds\n",tottime);
                }
  	}


	fftwf_destroy_plan(plan_for_fft);

	if (args->briggs){
                free(Wk);
                free(vis_weights);
                free(bweights);
        }
        free(BLEN);
        free(U);
        free(V);
        free(W);
	free(image);
	fftwf_free(vis_grid);
	//fftwf_free(data);

  	dsaX_dbgpu_cleanup(hdu_in, hdu_out);

}


