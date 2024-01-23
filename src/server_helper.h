#ifndef SERVER_HELPER_HEADER
#define SERVER_HELPER_HEADER
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
#include "server_helper.h"
#include "subclient.h"
#include <math.h>


//response messages
const char *error1 = "HTTP/1.1 405 Method Not Allowed\nContent-Type: text/plain\nContent-Length: 103\n\nServer received invalid command, only configured to respond to 'POST' commands. No data transferred.\n";
const char *error2 = "HTTP/1.1 400 Bad Request\nContent-Type: text/plain\nContent-Length: 57\n\nServer received invalid command, No data transferred.\n";
char invalidcommands[9][10] = {"GET","HEAD","DELETE","CONNECT","OPTIONS","TRACE","PATCH"};
const char *error3 = "HTTP/1.1 415 Unsupported Media Type\nContent-Type: text/plain\nContent-Length: 93\n\nServer received invalid content type, requires 'multipart/form-data', No data transferred.\n";
const char *error4 = "HTTP/1.1 400 Bad Request\nContent-Type: text/plain\nContent-Length: 50\n\nServer received empty file, No data transferred.\n";
const char *success = "HTTP/1.1 200 OK\nContent-Type: text/plain\nContent-Length: 14\n\nFile Received\n";
const char *expectresponse = "HTTP/1.1 100 Continue\n";
const unsigned char STARTBYTES[] = {0x93,0x4e,0x55,0x4d,0x50,0x59}; //Data starts with '\nNUMPY'
const unsigned char TESTSTRING[] = {0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x50, 0x01, 0xcd, 0x24, 0xa7, 0x34, 0xcf, 0x6e, 0xc0, 0xa8, 0x9b, 0xfe, 0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xe0, 0x0c, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x50, 0xc1, 0xa8, 0x9b, 0xfe, 0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xd0, 0x1f, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x48, 0x3c, 0xa7, 0x9e, 0x7f, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x58, 0xc1, 0xa8, 0x9b, 0xfe, 0x7f, 0x00, 0x00, 0xa0, 0x3c, 0x99, 0xa7, 0x01, 0x00, 0x00, 0x00, 0xd6, 0x0d, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2e, 0x32, 0x8f, 0x3b, 0x47, 0x68, 0x2e, 0x3c, 0xe0, 0x0c, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x50, 0xc1, 0xa8, 0x9b, 0xfe, 0x7f, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2e, 0x32, 0x2f, 0x85, 0x96, 0x5f};
const char *pipestatusfname = "/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/src/.pipestatus.txt";
const char *serverlogfname = "/home/ubuntu/proj/dsa110-shell/dsa110-nsfrb/src/server_log.txt";
const char *accepted_args = "-f-p-h";
const char *usage = "socket_server_test_V3.c usage:\n-h : print this message\n-f : output data to file provided by client\n-p [port] : send data to a server listening on the provided port number; if no port number is provided, the default port is 8843\n";
//HTTP REQUEST STRUCTURE: This struct will be populated with header values from the client request
struct REQUEST {
        char Host[32];
        char UserAgent[32];
        char Accept[16];
        int ContentLength;
        char Expect[16];
        char ContentType[20];
        unsigned char boundary[40];
        char message[512];
        char fname[128];
};

int update_pipestatus(char *fname) {
        FILE *pipestatus;
        pipestatus = fopen(pipestatusfname,"w");
        fprintf(pipestatus,"%s failed\n",fname);
        fclose(pipestatus);
        return 0;
}


//argument parsing
int server_parse_args(FILE* logfile, int argc, char *argv[], int flags[])

{
	int subclientPORT_tmp = 8843;
	int tofile = flags[0];
	int toport = flags[1];
	if (argc == 0)
        {
                fprintf(logfile,"sending data to stdout");
        }
        else
        {
                for (int i = 1; i < argc; i++)
                {
                        if ((strstr(accepted_args, argv[i]) == NULL) && (strcmp("-p", argv[i-1]) != 0))
                        {
                                printf("invalid argument\n");
                                printf("%s\n",usage);
                                update_pipestatus(argv[0]);
                                fclose(logfile);
                                exit(EXIT_FAILURE);
                        }
                        else if ((strcmp("-h", argv[i]) == 0))
                        {
                                printf("%s\n",usage);
                                update_pipestatus(argv[0]);
                                fclose(logfile);
                                exit(0);
                        }
                        else if ((strcmp("-f", argv[i]) == 0))
                        {
                                tofile = 1;
                                fprintf(logfile,"Will save output to file\n");
                        }
                        else if ((strcmp("-p", argv[i]) == 0))
                        {
                                toport = 1;
                                if ((i == argc-1) || strstr(accepted_args, argv[i+1]) != NULL)
                                {
                                        fprintf(logfile,"Will send data to server on default port %d\n",subclientPORT_tmp);
                                }
                                else if (strlen(argv[i+1]) != 4)
                                {
                                        printf("invalid port number\n");
                                        printf("%s\n",usage);
                                        update_pipestatus(argv[0]);
                                        fclose(logfile);
                                        exit(EXIT_FAILURE);
                                }
                                else
                                {
                                        subclientPORT_tmp = 0;
                                        int dig = 0;
                                        for (int j = 0; j < 4; j++)
                                        {
                                                //printf("%c\n",argv[i+1][j]);
                                                dig = isdigit(argv[i+1][j]);
                                                //printf("%d\n",dig);
                                                if (dig != 0)
                                                {
                                                        int digval = argv[i+1][j] - '0';
                                                        //printf("%d\n",digval);
                                                        subclientPORT_tmp += digval*pow(10,3-j);
                                                }
                                                else
                                                {
                                                        printf("invalid port number\n");
                                                        printf("%s\n",usage);
                                                        update_pipestatus(argv[0]);
                                                        fclose(logfile);
                                                        exit(EXIT_FAILURE);
                                                }
                                        }
                                        fprintf(logfile,"Will send data to server on port %d\n",subclientPORT_tmp);
                                }
                        }
                }
        }

	//update flags
	flags[0] = tofile;
	flags[1] = toport;
	return subclientPORT_tmp;

}


#endif
