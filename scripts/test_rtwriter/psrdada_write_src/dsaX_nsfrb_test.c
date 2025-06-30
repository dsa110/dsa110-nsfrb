/* fringestops/calibrates and writes nsfrb data to disk
*/

#include <stdio.h>
#include <stdlib.h>
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
/*
#include "dsaX_capture.h"
#include "dsaX_def.h"
*/
/* global variables */
int DEBUG = 0;
int NINTS_PER_FILE = 2250; // approx 300s

void dsaX_dbgpu_cleanup (dada_hdu_t * in, dada_hdu_t * out);
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
      syslog(LOG_ERR, "could not unlock read on hdu_in");
    }
  dada_hdu_destroy (in);

  if (dada_hdu_unlock_write (out) < 0)
    {
      syslog(LOG_ERR, "could not unlock write on hdu_out");
    }
  dada_hdu_destroy (out);
  
}


//void dsaX_dbgpu_cleanup (dada_hdu_t * in) {
//
//  if (dada_hdu_unlock_read (in) < 0)
//    {
//      syslog(LOG_ERR, "could not unlock read on hdu_in");
//    }
//  dada_hdu_destroy (in);
//
//}


int main (int argc, char *argv[]) {

  // startup syslog message
  // using LOG_LOCAL0
  openlog ("dsaX_nsfrb", LOG_CONS | LOG_PID | LOG_NDELAY, LOG_LOCAL0);
  syslog (LOG_NOTICE, "Program started by User %d", getuid ());
  
  /* DADA defs */
  dada_hdu_t* hdu_in = 0;
  dada_hdu_t* hdu_out = 0;
  multilog_t* log = 0;
  key_t in_key = 0xcaea;//XGPU_BLOCK_KEY;
  key_t out_key = 0xcaba;//XGPU_BLOCK_KEY;
  
  // command line
  int arg = 0;
  int core = -1;
  int nfq = 48;
  char fnam[300], fsnam[300], foutnam[300], finaloutnam[400];
  FILE *fout;
  int provided_fs = 0;
  int sb = 0;
  float decl = 0.0;
  sprintf(fnam,"./"); //"/home/ubuntu/tmp");
  
  while ((arg=getopt(argc,argv,"c:j:t:k:o:e:dh")) != -1)
    {
      switch (arg)
	{
	case 'c':
	  if (optarg)
	    {
	      core = atoi(optarg);
	      break;
	    }
	  else
	    {
	      printf ("ERROR: -c flag requires argument\n");
	      return EXIT_FAILURE;
	    }
	case 'k':
	  if (optarg)
	    {
	      if (sscanf (optarg, "%x", &in_key) != 1) {
		syslog(LOG_ERR, "could not parse key from %s\n", optarg);
		return EXIT_FAILURE;
	      }
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-k flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 'o':
	  if (optarg)
	    {
	      if (sscanf (optarg, "%x", &out_key) != 1) {
		syslog(LOG_ERR, "could not parse key from %s\n", optarg);
		return EXIT_FAILURE;
	      }
	      break;
	    }
	  else
	    {
	      syslog(LOG_ERR,"-o flag requires argument");
	      usage();
	      return EXIT_FAILURE;
	    }
	case 't':
	  strcpy(fsnam,optarg);
	  provided_fs = 1;
	  break;
	case 'd':
	  DEBUG=1;
	  break;
	case 'j':
	  nfq = atoi(optarg);
	  break;
	case 'e':
	  decl = atof(optarg);
	  break;
	case 'h':
	  usage();
	  return EXIT_SUCCESS;
	}
    }

  // DADA stuff
  syslog (LOG_INFO, "creating hdu");

  
  printf("hello world %x %x \n",in_key,out_key);
  hdu_in  = dada_hdu_create ();
  dada_hdu_set_key (hdu_in, in_key);
  if (dada_hdu_connect (hdu_in) < 0) {
    syslog (LOG_ERR,"could not connect to dada buffer");
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_read (hdu_in) < 0) {
    syslog (LOG_ERR,"could not lock to dada buffer");
    return EXIT_FAILURE;
  }
  
  

  hdu_out  = dada_hdu_create ();
  dada_hdu_set_key (hdu_out, out_key);
  if (dada_hdu_connect (hdu_out) < 0) {
    syslog (LOG_ERR,"could not connect to output  buffer");
    return EXIT_FAILURE;
  }
  if (dada_hdu_lock_write(hdu_out) < 0) {
    syslog (LOG_ERR, "could not lock to output buffer");
    return EXIT_FAILURE;
  }

  // Bind to cpu core
  if (core >= 0)
    {
      syslog(LOG_INFO,"binding to core %d", core);
      if (dada_bind_thread_to_core(core) < 0)
	syslog(LOG_ERR,"failed to bind to core %d", core);
    }

  int observation_complete=0;

  // more DADA stuff - deal with headers
  
  uint64_t header_size = 0;

  printf("HELLO AGAIN\n");
  // deal with headers
  
  char * header_in = ipcbuf_get_next_read (hdu_in->header_block, &header_size);
  if (!header_in)
    {
      syslog(LOG_ERR, "could not read next header");
      dsaX_dbgpu_cleanup (hdu_in, hdu_out);
      return EXIT_FAILURE;
    }
  if (ipcbuf_mark_cleared (hdu_in->header_block) < 0)
    {
      syslog (LOG_ERR, "could not mark header block cleared");
      dsaX_dbgpu_cleanup (hdu_in, hdu_out);
      return EXIT_FAILURE;
    }
  char * header_out = ipcbuf_get_next_write (hdu_out->header_block);
  if (!header_out)
    {
      syslog(LOG_ERR, "could not get next header block [output]");
      dsaX_dbgpu_cleanup (hdu_in, hdu_out);
      return EXIT_FAILURE;
    }
  memcpy (header_out, header_in, header_size);
  if (ipcbuf_mark_filled (hdu_out->header_block, header_size) < 0)
    {
      syslog (LOG_ERR, "could not mark header block filled [output]");
      dsaX_dbgpu_cleanup (hdu_in, hdu_out);
      return EXIT_FAILURE;
    }

  
//  // read the headers from the input HDUs and mark as cleared
//  char * header_in = ipcbuf_get_next_read (hdu_in->header_block, &header_size);
//  if (!header_in)
//    {
//      syslog(LOG_ERR, "main: could not read next header");
//      dsaX_dbgpu_cleanup (hdu_in);
//      return EXIT_FAILURE;
//    }
//  if (ipcbuf_mark_cleared (hdu_in->header_block) < 0)
//    {
//      syslog (LOG_ERR, "could not mark header block cleared");
//      dsaX_dbgpu_cleanup (hdu_in);
//      return EXIT_FAILURE;
//    }
//
  /* FRINGESTOPPING: process in blocks of 25 integrations.
     divide out at native resolution, then integrate. */
  // read fs table
  printf("HELLO FINALLY\n");
  FILE *fsin;
  // fstable has shape (25, 4656, 384, 2, 2)
  float *fstable = (float *)malloc(sizeof(float)*25*4656*384*2*2);
  
  // set up
  double fctr = 0.;
  int integration = 0, cyclectr = 0;
  
  // data stuff
  uint64_t block_size = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_in->data_block);
  uint64_t block_out = ipcbuf_get_bufsz ((ipcbuf_t *) hdu_out->data_block);
  printf("INPUT BLOCK SIZE%lu\n",block_size);
  uint64_t bytes_read = 0, block_id, written;
  char *block;
  float *fblock;
  float *data = (float *)malloc(sizeof(float)*25*4656*(384/nfq)*2*2);
  memset(data, 0, 25*4656*(384/nfq)*2*2*sizeof(float));
  int inidx, fsidx, outidx;
  int read_fstable = 0;
  double mjd, mjd0;
  int secs;

  clock_t start, end;
  double cpu_time_used;
  cpu_time_used = 0.0;
  // start things
  syslog(LOG_INFO, "starting observation");

  /*
  mjd = mjd0 + (double)(fctr*((double)(NINTS_PER_FILE))*4096.*32.768e-6/86400.);
  secs = (int)((mjd-60600.)*1440.);
  sprintf(foutnam,"%s_%d.out.tmp",fnam,secs);
  fout=fopen(foutnam,"wb");
  */
  while (!observation_complete) {
    
    // read block
    block = ipcio_open_block_read (hdu_in->data_block, &bytes_read, &block_id);
    fblock = (float *)(block);
    printf("BYTES READ ON EACH OBSERVATION:%lu\n",bytes_read);
    
    // read fstable if first integration
    // also read mjd of first spec
    /*
    if (read_fstable==0 && provided_fs==1) {
      fsin=fopen(fsnam,"rb");
      fread(fstable,sizeof(float),25*4656*384*2*2,fsin);
      fclose(fsin);
      
      
      read_fstable=1;
    }
    
    */ 
    // DO STUFF - from data to summed_vis

    // write into data
    for (int i=0;i<4656;i++) {
      for (int j=0;j<(384/nfq);j++) {
	for (int k=0;k<nfq;k++) {
	  for (int l=0;l<2;l++) {
	    inidx = 2*(i*384*2 + j*nfq*2 + k*2 + l);
	    fsidx = inidx + cyclectr*4656*384*2*2;
	    outidx = cyclectr*4656*(384/nfq)*2*2 + i*(384/nfq)*2*2 + j*4 + l*2;

	    // complex mult: (a+ib)*(c+id) = (ac-bd) + i(bc+ad)
	    /*if (provided_fs==1) {
	      data[outidx] += fblock[inidx]*fstable[fsidx] - fblock[inidx+1]*fstable[fsidx+1];
	      data[outidx+1] += fblock[inidx+1]*fstable[fsidx] + fblock[inidx]*fstable[fsidx+1];
	    }
	    else {
	      data[outidx] += fblock[inidx];
	      data[outidx+1] += fblock[inidx+1];
	    }*/
	    data[outidx] += fblock[inidx];
	    data[outidx+1] += fblock[inidx+1];
	    
	  }
	}
      }
    }
    
    // now loop stuff
    cyclectr++;
    integration++;

    //printf("IN LOOP\n");
    // check for end of block
    if (cyclectr==25) {
      start = clock();
      cyclectr=0;

      // write to buffer
      written = ipcio_write (hdu_out->data_block, (char *)data, block_out);
      if (written < block_out)
	{
	  syslog(LOG_ERR, "main: failed to write all data to datablock [output]");
	  dsaX_dbgpu_cleanup (hdu_in, hdu_out);
	  return EXIT_FAILURE;
	}
	
      
      // write out data
      //fwrite(data,sizeof(float),25*4656*(384/nfq)*2*2,fout);

      // zero out data
      memset(data, 0, 25*4656*(384/nfq)*2*2*sizeof(float));

      if (DEBUG) syslog(LOG_DEBUG,"Finished a block");
      end = clock();
      cpu_time_used += ((double) (end - start)) / CLOCKS_PER_SEC;
    }
    
    // check for end of file
    if (integration==NINTS_PER_FILE) {
      integration=0;
      fctr++;
      //fclose(fout);
      printf("TOTAL TIME:%lf\n",cpu_time_used);
      cpu_time_used = 0.0;
      /*
      syslog(LOG_INFO,"Closed file %s",foutnam);
      system(finaloutnam);

      mjd = mjd0 + (double)(fctr*((double)(NINTS_PER_FILE))*4096.*32.768e-6/86400.);
      secs = (int)((mjd-60600.)*1440.);
      sprintf(foutnam,"%s_%d.out.tmp",fnam,secs);
      sprintf(finaloutnam,"mv %s %s_%d.out",foutnam,fnam,secs);
      fout=fopen(foutnam,"wb");
      fwrite(&mjd,sizeof(double),1,fout);
      fwrite(&sb,sizeof(int),1,fout);
      fwrite(&decl,sizeof(float),1,fout);
      */
    }
       
    // close off loop
    //if (bytes_read < block_size)
    //  observation_complete = 1;
    ipcio_close_block_read (hdu_in->data_block, bytes_read);
  }

  printf("done\n");
  free(data);
  free(fstable);
  dsaX_dbgpu_cleanup(hdu_in, hdu_out);
 
}
