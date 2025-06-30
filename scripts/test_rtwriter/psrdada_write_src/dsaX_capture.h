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

#define DSAX_UDPDB_BUF_CLEAR = 0
#define DSAX_UDPDB_BUF_FULL = 1

/* socket buffer for receiving udp data */
typedef struct {

  int           fd;            // FD of the socket
  size_t        bufsz;         // size of socket buffer
  char *        buf;          // the socket buffer
  int           have_packet;   // 
  size_t        got;           // amount of data received

} dsaX_sock_t;

dsaX_sock_t * dsax_Xnit_sock ();

void dsaX_free_sock(dsaX_sock_t* b);

/* Number of UDP packets to be recived for a called to buffer_function */
#define NOTRECORDING 0
#define RECORDING 1

typedef struct {

  dada_hdu_t *      hdu;                // DADA Header + Data Unit
  multilog_t *      log;                // DADA logging interface
  int               verbose;            // verbosity flag 

  dsaX_sock_t *     sock;               // UDP socket for data capture
  int               port;               // port to receive UDP data 
  int               control_port;       // port to receive control commands
  char *            interface;          // IP Address to accept packets on 

  // configuration for number of inputs
  unsigned int      num_inputs;         // number of antennas / inputs

  // datablock management
  uint64_t          hdu_bufsz;
  unsigned          block_open;        // if the current data block element is open
  char            * block;             // pointer to current datablock buffer
  uint64_t          block_start_byte;  // seq_byte of first byte for the block
  uint64_t          block_end_byte;    // seq_byte of first byte of final packet of the block
  uint64_t          block_count;       // number of packets in this block
  char            * tblock;            // area of memory to write to
  
  // packets
  unsigned          capture_started;      // flag for start of UDP data
  uint64_t          packets_per_buffer;   // number of UDP packets per datablock buffer

  /* Packet and byte statistics */
  stats_t * packets;
  stats_t * bytes;

  uint64_t bytes_to_acquire;
  double mb_rcv_ps;
  double mb_drp_ps;
  double mb_free;
  double mb_total;
  uint64_t rcv_sleeps;

  uint64_t last_seq;                     // most recently received seq number
  uint64_t last_byte;                    // most recently received byte
  struct   timeval timeout; 

  uint64_t n_sleeps;
  uint64_t ooo_packets;

  int      recv_core;

} udpdb_t;


int dsaX_udpdb_init_receiver (udpdb_t * ctx);
void dsaX_udpdb_reset_receiver (udpdb_t * ctx);
int dsaX_udpdb_destroy_receiver (udpdb_t * ctx);
int dsaX_udpdb_open_buffer (udpdb_t * ctx);
int dsaX_udpdb_close_buffer (udpdb_t * ctx, uint64_t bytes_written, unsigned eod);
int dsaX_udpdb_new_buffer (udpdb_t * ctx);
int dsaX_udpdb_increment (udpdb_t * ctx);

// allocate required resources for data capture
int dsaX_udpdb_prepare (udpdb_t * ctx);

// move to a state where data acquisition can begin
time_t dsaX_dpdb_start (udpdb_t * ctx, char * header);

// main workhorse function to receive data for a single observation
void * dsaX_udpdb_receive_obs (void * ctx);

// close the datablock signifying end of data 
int udpdb_stop_function (udpdb_t* ctx);

void usage();
void signal_handler (int signalValue); 
void stats_thread(void * arg);
void control_thread(void * arg);

#endif
