#include <stdlib.h>
#include <stdio.h>
#define USECPSEC 1000000ULL
#define C_MHZ_M 299.79245800000004 
#define C_GHZ_M 0.29979245800000004
#define NUM_CHANNELS 768
#define nchans 16
#define maxchans 6144

//filenames
/*char baseUVWcmd[] = "python /home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/realtime/rt_c_imager/_getbaselines.py --pt_dec ";
char cwd[] = "/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb";
char table_dir[] = "/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-tables/";
char ufname[] = "/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-tables/U.bin";
char vfname[] = "/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-tables/V.bin";
char wfname[] = "/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-tables/W.bin";
char bfname[] = "/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-tables/BLEN.bin";
char a1fname[] = "/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-tables/ANT1.bin";
char a2fname[] = "/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb-tables/ANT2.bin";
*/

//frequency
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


