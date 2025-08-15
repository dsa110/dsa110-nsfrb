#ifndef __DSAX_DEF_H
#define __DSAX_DEF_H

#include "dada_def.h"

// default dada block keys
#define TEST_BLOCK_KEY 0x0000aada // for capture program.
// 128*3*384*32*2=9437184 for 1 CHANG 1 SNAP 1 REORDER
// 128*3*384*32*2*4=37748736 for 4 CHANG 1 SNAP 1 REORDER
// 128*3*384*32*2*8=75497472 for 1 CHANG 1 SNAP 8 REORDER
#define CAPTURE_BLOCK_KEY 0x0000dada // for capture program.
// 128*3*384*32*2=9437184 for 1 CHANG 1 SNAP 1 REORDER
// 150994944 for doSnap
#define REORDER_BLOCK_KEY 0x0000eada // for reorder program.
// 589824 for doSnap
#define REORDER_BLOCK_KEY2 0x0000bada // for reorder program 2.
// 128*32*1536*16*2*2=402653184 1 REORDER
// 3221225472 for 8 REORDERS
#define XGPU_BLOCK_KEY 0x0000fada // for xgpu program. 
// 136*1536*2*8=3342336 
#define COPY_BLOCK_KEY 0x0000dbda // for split off data
#define BF_BLOCK_KEY 0x0000dcda // for beamformed data
#define BF_BLOCK_KEY2 0x0000bcda // for beamformed data testing
#define CAPTURED_BLOCK_KEY 0x0000abda // for capture program.
#define BEAMCAPTURE_BLOCK_KEY 0x0000bbda // for capture bf program.

// constants
#define PI 3.14159265359
#define CVAC 299792458.0

// default number of XGPU ints
#define NCORRINTS 128
#define NNATINTS 32 // native number of integrations
#define NREORDERS 1 // number of ints per reorder

// size of xgpu output
// TODO
#define XGPU_SIZE 835584 // size of single output vector (post-GPU)
#define XGPU_IN_INC 1 // size of input increment
#define NBASE 4656 // nant*(nant+1)/2
#define NPOL 2
#define NCHAN 1536 // regardless of NCHANG

// default port for packet capture
#define CAPTURE_PORT 4011

// default UDP packet dims
#define UDP_HEADER   8              // size of header/sequence number
#define UDP_DATA     4608           // obs bytes per packet
#define UDP_PAYLOAD  4616           // header + datasize

// number of channel groups to expect
#define NCHANG 1

// number of SNAPs to expect
#define NSNAPS 32

/* expect consecutive channel groups */
#define CHOFF 0 // offset in channels of first group

// default control ports
#define CAPTURE_CONTROL_PORT 11223
#define REORDER_CONTROL_PORT 11224
#define XGPU_CONTROL_PORT 11225
#define WRITEVIS_CONTROL_PORT 11226
#define TRIGGER_CONTROL_PORT 11227

#define NPACKETS_PER_CALL 2048
#define NPACKETS_PER_BLOCK 2048
#define NPACKETS_INTS 2048 // number of packets per xgpu int
#define NPACKETS_PER_FIL 2
#define NPACKETS 2048
#define NOUTBLOCKS 15 // number of input blocks stored by trigger
#define NANTS 96
#define NCHAN_PER_PACKET 384
#define NBEAMS 512

// for beamformer
//#define sep 1.0 // arcmin
#define NW 48 // number of weights per 384 chans. Also the number of channels formed
#define NANT 63
#define BEAM_OUT 23
#define NSTREAMS 4
#define NBP 8 // number of previous BPs to average

// for second corner turn
#define FIL_PORT0 6625 // port for first chan group
#define NCLIENTS 16 // number of client dbnic processes to expect
#define NSAMPS_PER_BLOCK 16384 // number of samples per block
#define NCHAN_FIL 768 // final number of filterbank chans
#define NBEAMS_PER_BLOCK 128 // number of beams to expect
#define NSAMPS_PER_TRANSMIT 512 // number of samples transmitted at one time
#define NBMS 512
#define P_SIZE 4108
#define NWAIT 100000

#endif 

