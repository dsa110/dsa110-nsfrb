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

	int status;
	status = Cetcd_put_rtwriter(12345,100);
	return status;

}

