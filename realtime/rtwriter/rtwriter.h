#ifndef RTWRITER
#define RTWRITER
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <string.h>
#include <sys/shm.h>
/*
 * Opens a memory stream, writes data to it, and returns a pointer to the stream
 */

//Memory Mapping
int SHMID_INIT = -1;
struct rtwriter_obj {
        int shmid;
        size_t datasize;
        char *memaddr;
        int done;
	int init;
	double mjd;
	int sb;
	float dec;
};
struct rtwriter_obj *rtwrite(char *data, size_t datasize, int done, struct rtwriter_obj *rtobj);

const char *NSFRBDIR = "/home/ubuntu/msherman_nsfrb/DSA110-NSFRB-PROJECT/dsa110-nsfrb/";
int rtwriter_to_etcd(int shmid, size_t datasize, double mjd, int sb, float dec);

#endif
