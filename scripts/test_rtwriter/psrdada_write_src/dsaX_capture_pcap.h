/***************************************************************************
 *  
 *    Copyright (C) 2009 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <errno.h>
#include <assert.h>
#include <netinet/in.h>
#include <signal.h>
#include <inttypes.h>
#include <sys/types.h>

#include "futils.h"
#include "dada_hdu.h"
#include "dada_pwc_main.h"
#include "multilog.h"
#include "ipcio.h"
#include "ascii_header.h"
#include "dada_udp.h"

#include "dsaX_def.h"

/* Number of UDP packets to be recived for a called to buffer_function */
#define NOTRECORDING 0
#define RECORDING 1

// structure for all threads
typedef struct {

  dada_hdu_t *      hdu;                // DADA Header + Data Unit
  uint64_t          hdu_bufsz;
  unsigned          block_open;        // if the current data block element is open
  char            * tblock;  
  uint64_t          tblock_idx;
  char            * temp_buffers;
  uint64_t        * temp_seq_byte;
  int               temp_idx;
  int               thread_id;
  uint64_t          block_start_byte;
  uint64_t          block_end_byte;
  uint64_t          block_count;
  int               nblocks_written;
  
  int               verbose;            // verbosity flag 

  // configuration for number of inputs
  unsigned int      num_inputs;         // number of antennas / inputs
  
  // packets
  uint64_t          packets_per_buffer;   // number of UDP packets per datablock buffer

  /* Packet and byte statistics */
  stats_t * packets;
  stats_t * bytes;

  uint64_t last_seq;                     // most recently received seq number

} dsaX_t;

// structure for stats thread
// both are shared between all recv structures and this one
// last_seq is also shared
typedef struct {

  /* Packet and byte statistics */
  stats_t * packets;
  stats_t * bytes;
  uint64_t * last_seq;                     // most recently received seq number

} dsaX_stats_t;


void signal_handler (int signalValue); 
void stats_thread(void * arg);
void control_thread(void * arg);
