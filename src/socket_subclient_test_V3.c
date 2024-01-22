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


// test implementation of a subclient that sends data to a python server


int main(int argc, char const*argv[])
{
	//create a socket
        int subclient_fd,subclient_socket,connectStatus,sendStatus;
        struct sockaddr_in outserveraddress;
	
	//make variables for out server
	outserveraddress.sin_family = AF_INET;
	const int subclientPORT = 8600;
	outserveraddress.sin_port = htons(subclientPORT);
	outserveraddress.sin_addr.s_addr = htonl(INADDR_ANY);

	struct linger sl;
	sl.l_onoff = 1;
	sl.l_linger = 0;

	printf("Opening Socket...");
        if ((subclient_fd = socket(AF_INET,SOCK_STREAM,0))<0)
        {
                printf("cannot create socket");
                //perror("cannot create socket");
                //fclose(logfile);
                //update_pipestatus(argv[0]);
                return 0;
        }
	printf("Done\n");
	setsockopt(subclient_fd,SOL_SOCKET,SO_LINGER,&sl,sizeof(sl));
	//int stat = fcntl(subclient_fd,F_SETFL,fcntl(subclient_fd,F_GETFL,0) | O_NONBLOCK);
	//printf("%d\n",stat);
	printf("Connecting to Server...");
	//connect to server
	if ((connectStatus = connect(subclient_fd, (struct sockaddr*)&outserveraddress, sizeof(outserveraddress))) < 0)
	{
		printf("cannot connect to server");
		//perror("cannot connect to server");
		//fclose(logfile);
		//update_pipestatus(argv[0]);
		return 0;
	}
	printf("Done\n");

	//send message
	printf("Sending message...");
	char testmessage[255] = "Hello World";
	if ((sendStatus = (subclient_fd,testmessage,sizeof(testmessage),0)) < 0)
	{
		printf("message send failed");
		//perror("message send failed");
		//fclose(logfile);
		//update_pipestatus(argv[0]);
		return 0;
	}
	printf("Done\n");
	//close(subclient_socket);	
	return 0;
}
