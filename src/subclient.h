#ifndef SUBCLIENT_HEADER
#define SUBCLIENT_HEADER

#include <stdio.h>
#include <errno.h>
#include <sys/socket.h>
#include <unistd.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <string.h>
#include <ctype.h>
#include <netinet/tcp.h>
#include <fcntl.h>
#include <sys/types.h>
#include "server_helper.h"

/*
This file contains a helper function that opens a socket client to send a data stream to 
the Python process server.
*/

int subclient_send(unsigned char data_buffer[], 
		int data_size, const int subclientPORT,
		FILE *logfile)
		
{
	//create a socket
        int subclient_fd,subclient_socket,connectStatus,sendStatus;
        struct sockaddr_in outserveraddress;
	
	//make variables for out server
	outserveraddress.sin_family = AF_INET;
	//const int subclientPORT = 8843;
	outserveraddress.sin_port = htons(subclientPORT);
	outserveraddress.sin_addr.s_addr = htonl(INADDR_ANY);

	fprintf(logfile,"Opening Socket...");
        if ((subclient_fd = socket(AF_INET,SOCK_STREAM,0))<0)
        {
                fprintf(logfile,"cannot create socket");
                perror("cannot create socket");
                fclose(logfile);
                //update_pipestatus(argv[0]);
                return 0;
        }
	fprintf(logfile,"Done\n");
	fprintf(logfile,"Connecting to Server...");
	//connect to server
	if ((connectStatus = connect(subclient_fd, (struct sockaddr*)&outserveraddress, sizeof(outserveraddress))) < 0)
	{
		fprintf(logfile,"cannot connect to server");
		perror("cannot connect to server");
		fclose(logfile);
		//update_pipestatus(argv[0]);
		return 0;
	}
	fprintf(logfile,"Done\n");

	//send message
	fprintf(logfile,"Sending message...");
	char testmessage[255] = "Message from the server to the "
                       "client \'Hello Client\' ";
	if ((sendStatus = send(subclient_fd,testmessage,sizeof(testmessage),0)) < 0)
	{
		fprintf(logfile,"message send failed");
		perror("message send failed");
		fclose(logfile);
		//update_pipestatus(argv[0]);
		return 0;
	}
	fprintf(logfile,"Done\n");
	close(subclient_socket);	
	return 0;
}

#endif
