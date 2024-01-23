
//from https://www.geeksforgeeks.org/simple-client-server-application-in-c/

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

int main(int argc, char const* argv[]) 
{ 
  
    // create server socket similar to what was done in 
    // client program 
    int servSockD = socket(AF_INET, SOCK_STREAM, 0);         
    //struct linger sl;
    //sl.l_onoff = 1;
    //sl.l_linger = 0;  
    //setsockopt(servSockD,SOL_SOCKET,SO_LINGER,&sl,sizeof(sl));

    // string store data to send to client 
    char serMsg[255] = "Message from the server to the "
                       "client \'Hello Client\' "; 
  
    // define server address 
    struct sockaddr_in servAddr; 
  
    servAddr.sin_family = AF_INET; 
    servAddr.sin_port = htons(8843); 
    servAddr.sin_addr.s_addr = INADDR_ANY; 
  
    // bind socket to the specified IP and port 
    bind(servSockD, (struct sockaddr*)&servAddr, 
         sizeof(servAddr)); 
  
    // listen for connections 
    listen(servSockD, 1); 
  
    // integer to hold client socket. 
    int clientSocket = accept(servSockD, NULL, NULL); 
  
    char strData[255];
    int recstatus;
    if ((recstatus = recv(clientSocket,strData,sizeof(strData),0))<0)
    {
        printf("Failed\n");
	close(clientSocket);
	return 0;
    }
    //while (recstatus == 0)
    //{
	//recstatus = recv(clientSocket,strData,sizeof(strData),0);
    //}
    printf("Received message size %d: %s\n",recstatus,strData);
  
    //close(clientSocket);
    return 0; 
}
