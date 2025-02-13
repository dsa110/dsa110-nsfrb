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

int SHMID_INIT = -1;
struct rtwriter_obj {
        int shmid;
        size_t datasize;
        char *memaddr;
        int done;
	int init;
};

struct rtwriter_obj *rtwrite(char *data, size_t datasize, int done, struct rtwriter_obj *rtobj);

#endif
