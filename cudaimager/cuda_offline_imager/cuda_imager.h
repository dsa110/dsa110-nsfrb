//constants
#define C_MHZ_M 299.79245800000004 
#define C_GHZ_M 0.29979245800000004


//filenames
char baseUVWcmd[] = "python /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/cudaimager/cuda_offline_imager/_getbaselines.py --pt_dec ";
char cwd[] = "/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb";
char table_dir[] = "/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-tables/";
char ufname[] = "/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/cudaimager/cuda_offline_imager/U.bin";
char vfname[] = "/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/cudaimager/cuda_offline_imager/V.bin";
char wfname[] = "/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/cudaimager/cuda_offline_imager/W.bin";
char bfname[] = "/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/cudaimager/cuda_offline_imager/BLEN.bin";
char a1fname[] = "/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/cudaimager/cuda_offline_imager/ANT1.bin";
char a2fname[] = "/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/cudaimager/cuda_offline_imager/ANT2.bin";


//frequency
const int NUM_CHANNELS = 768;
const int nchans = 16;
const int maxchans = nchans*NUM_CHANNELS/2;
double freq_axis_fullres[maxchans];
double freq_axis_fullres_GHz[maxchans];
double freq_axis[nchans];
double freq_axis_GHz[nchans];
double chanbw;
double fmin_;
double fmax_;
double fc_;
double lambdamin_;
double lambdamax_;
double lambdac_;
double lambdaref_;

