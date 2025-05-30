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



const char *fname = "/dataz/dsa110/nsfrb/dsa110-nsfrb-fast-visibilities/lxd110h04/nsfrb_sb01_175679.out";
int main()
{


	//open and read fast vis file
	FILE *fptr;
	size_t buffersize = 5*4656*8*2*2*4;
	size_t gulpsize = 5*buffersize;
	size_t headersize = 16;
	size_t maxread = (25*4656*8*2*2*4) + headersize;
	char *buffer = malloc(buffersize);
	size_t bytesread = 1;
	size_t totbytesread = 0;
	struct rtwriter_obj *rtobj = malloc(sizeof *rtobj);

	printf("Opening file...");
	fptr = fopen(fname,"rb");
	printf("Done!\n");
	
	//get header:sb, mjd, dec
	//unsigned long mjd_bkwrd =0 ;
        //unsigned int sb_bkwrd = 0;
        //unsigned int dec_bkwrd = 0;
	printf("Reading header...");
	totbytesread += fread(&rtobj->mjd,1,8,fptr);
	//flip byte order
	//char byte;
	//for (int i = 0; i < 64; i+=8)
	//{
	//	byte = (mjd_bkwrd>>i) & 0xFF;
	//	mjd |= (byte << (64 - i - 8));
	//}
	totbytesread += fread(&rtobj->sb,1,4,fptr);
	//for (int i = 0; i < 32; i+=8)
        //{
        //        byte = (sb_bkwrd>>i) & 0xFF;
        //        sb |= (byte << (32 - i - 8));
	//}
	totbytesread += fread(&rtobj->dec,1,4,fptr);
	//for (int i = 0; i < 64; i+=8)
        //{
        //        byte = (dec_bkwrd>>i) & 0xFF;
        //        dec |= (byte << (64 - i - 8));
	//}
	printf("Done: %f,%d,%f\n",rtobj->mjd,rtobj->sb,rtobj->dec);

	int offset = 1;
	fseek(fptr,headersize + offset*gulpsize,SEEK_SET);
	while (bytesread>0 && totbytesread<maxread) 
	{
		//read chunk of data
		bytesread = fread(buffer,1,buffersize,fptr);
		totbytesread += bytesread;
		printf("BYTES READ (%ld):%ld/%ld\n",bytesread,totbytesread,maxread);
		//write to memory
		rtobj=rtwrite(buffer, bytesread, 0, rtobj);
		
	}
	fclose(fptr);
	//wait for reader to finish
	sleep(60);
	
	rtwrite(NULL,0,1,rtobj);
	free(buffer);
	free(rtobj);
	/*
	// create a buffer
	size_t buffersize = 11;
	char buffer[] = {0x68, 0x65, 0x6C, 0x6C, 0x6F, 0x77, 0x6F, 0x72, 0x6C, 0x64, 0x0A};

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
	*/

}
