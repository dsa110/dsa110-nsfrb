#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <string.h>
#include <sys/shm.h>
#include "rtwriter.h"

/*
 * Test script
 */



int main()
{

	// create a buffer
	size_t buffersize = 11;
	char *buffer = "helloworld";
	//strcpy(buffer,"helloworld");

	struct rtwriter_obj *rtobj = malloc(sizeof *rtobj);
       	printf("%p\n",rtobj);
	rtobj->init = 0;
	printf("%d\n",rtobj->init);
	
	rtobj=rtwrite(buffer, buffersize, 0, rtobj);//SHMID_INIT, buffersize, 0);
	printf("%d\n",rtobj->shmid);

	sleep(10);

	rtobj=rtwrite(buffer, buffersize, 1, rtobj); // shmid, buffersize, 1);
        printf("%d\n",rtobj->done);
	free(rtobj);

	return 0;

}

