#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <string.h>
#include <sys/shm.h>
#include "rtwriter.h"
/*
 * Opens a memory stream, writes data to it, and returns a pointer to the stream
 */


struct rtwriter_obj *rtwrite(char *data, size_t datasize, int done, struct rtwriter_obj *rtobj) //, int shmid, size_t datasize, int done, char *memaddr)
{


        //create shared memory segment if not already created
	printf("%d\n",rtobj->init);
	fflush(stdout);
        if ((rtobj->init)==0)
        {
		//struct rtwriter_obj *rtobj;
		printf("Creating new Shared Memory Block\n");
                rtobj->shmid = shmget(IPC_PRIVATE, datasize, 0x0777);
		printf("Error: %s\n",strerror(errno));
		
        
        	printf("Shared Memory ID: %d\n",rtobj->shmid);

        	//attach to mem address
        	rtobj->memaddr = shmat(rtobj->shmid, NULL, 0);
		printf("Error: %s\n",strerror(errno));
        	printf("Memory Address: %p\n",rtobj->memaddr);
        	printf("%ld\n",(long int)(rtobj->memaddr));
        	printf("Error: %s\n",strerror(errno));
		
		rtobj->datasize = datasize;
		rtobj->done = 0;
		rtobj->init = 1;
	}
	
	printf("Memory Address: %p\n",rtobj->memaddr);
        //write data to memory
        strcpy(rtobj->memaddr, data);
	printf("Data written %s\n",rtobj->memaddr);

        //sleep(30);

        //detach
        //shmdt(rtobj->memaddr);

        //destroy if prompted
        if (done==1)
        {
		rtobj->done = 1;
                printf("Deleting Shared Memory %d\n",rtobj->shmid);
		//detach
        	shmdt(rtobj->memaddr);
                shmctl(rtobj->shmid,IPC_RMID,NULL);
                return rtobj;
        }

        return rtobj;

}


