#include <stdio.h>
#include <sys/mman.h>
#include <string.h>
/*
 * Opens a memory stream, writes data to it, and returns a pointer to the stream
 */



int main()
{

	// create a buffer
	size_t buffersize = 11;
	char buffer[buffersize];
	// strcpy(buffer,"helloworld");

	// open a memory stream
	FILE *fptr = fmemopen(buffer, buffersize, "w+");
	printf("Pointer Address: %p\n",fptr);
	printf("Buffer Address: %p\n",buffer);
	
	//create memory map
	int PROT_FLAG = PROT_READ | PROT_WRITE;
	int MAP_FLAG = MAP_SHARED | MAP_ANONYMOUS;
	int fd = fileno(fptr);
	char *memaddr = mmap(NULL, buffersize, PROT_FLAG, MAP_FLAG, fd, 0);
	printf("Memory Address: %p\n", memaddr);
	printf("File descriptor: %d\n", fd);
	
	//write to memory
	memcpy(memaddr, "helloworld", buffersize);

	//write to stream
	//fprintf(fptr, "helloworld");
	while (1) {
	}
		
	//fclose(fptr);


	//now reopen stream and read contents
	//FILE *fptr2 = fmemopen(buffer, buffersize, "r");
	//char buffer2[buffersize];
	//fread(buffer2,1,buffersize,fptr2);
	//printf("Contents: %s\n",buffer2);
	//fclose(fptr2);

	//printf("%s\n",buffer);
	return 0;

}

