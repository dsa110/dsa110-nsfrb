/***************************************************************************
 *  
 *    Copyright (C) 2009 by Andrew Jameson
 *    Licensed under the Academic Free License version 2.1
 * 
 ****************************************************************************/

#ifndef __DSAX_UDPDB_THREAD_H
#define __DSAX_UDPDB_THREAD_H

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

/* socket buffer for receiving udp data */
// this is initialised in each recv thread
typedef struct {

  int           fd;            // FD of the socket
  size_t        bufsz;         // size of socket buffer
  char *        buf;          // the socket buffer
  int           have_packet;   // 
  size_t        got;           // amount of data received

} dsaX_sock_t;

dsaX_sock_t * dsaX_init_sock ();
void dsaX_free_sock(dsaX_sock_t* b);

/* Number of UDP packets to be recived for a called to buffer_function */
#define NOTRECORDING 0
#define RECORDING 1

// structure for write thread
// tblock must be shared
typedef struct {

  dada_hdu_t *      hdu;                // DADA Header + Data Unit
  uint64_t          hdu_bufsz;
  unsigned          block_open;        // if the current data block element is open
  char            * block;             // pointer to current datablock buffer
  char            * tblock;            // area of memory to write to
  int               thread_id;

} dsaX_write_t;

// structure for stats thread
// both are shared between all recv structures and this one
// last_seq is also shared
typedef struct {

  /* Packet and byte statistics */
  stats_t * packets;
  stats_t * bytes;
  uint64_t * last_seq;                     // most recently received seq number

} dsaX_stats_t;


// structure for receive thread
// tblock, packets, bytes, last_seq, block_start_byte, block_end_byte, block_count, capture_started
typedef struct {

  multilog_t *      log;                // DADA logging interface
  int               verbose;            // verbosity flag 

  int               port;               // port to receive UDP data 
  int               control_port;       // port to receive control commands
  char *            interface;          // IP Address to accept packets on 

  // configuration for number of inputs
  unsigned int      num_inputs;         // number of antennas / inputs

  // datablock management
  uint64_t        * block_start_byte;  // seq_byte of first byte for the block
  uint64_t        * block_end_byte;    // seq_byte of first byte of final packet of the block
  uint64_t        * block_count;       // number of packets in this block  
  uint64_t          hdu_bufsz;
  char            * tblock;            // area of memory to write to
  
  // packets
  unsigned        * capture_started;      // flag for start of UDP data
  uint64_t          packets_per_buffer;   // number of UDP packets per datablock buffer

  /* Packet and byte statistics */
  stats_t * packets;
  stats_t * bytes;
  uint64_t rcv_sleeps;

  uint64_t * last_seq;                     // most recently received seq number
  struct   timeval timeout;
  int thread_id;

} udpdb_t;

void signal_handler (int signalValue); 
void stats_thread(void * arg);
void control_thread(void * arg);

#endif
