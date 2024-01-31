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
extern int errno ;
/*
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
*/


int main(int argc, char *argv[]) {
	//create status file to report if command failed
	//FILE *pipestatus;
	
	//make log file to write output to, only want data written to stdout
        FILE *logfile;
        logfile = fopen(serverlogfname,"w");

        fprintf(logfile,"Hello, World!\n");//printf("Hello, World!\n");

	//check arguments
	int flags[3] = {0,0,0};
	
	const int subclientPORT = server_parse_args(logfile,argc,argv,flags);//subclientPORT_tmp;
	if (subclientPORT < 1000)
	{                                
		printf("invalid argument\n");
                printf("%s\n",usage);
                update_pipestatus(argv[0]);
                fclose(logfile);
                exit(EXIT_FAILURE);
	}
	int tofile = flags[0];
	int toport = flags[1];
	int tostdout = flags[2];
        //printf("%d %d %d %d\n",subclientPORT,tofile,toport,tostdout);
        //fclose(logfile);
        //exit(0);
	//make log file to write output to, only want data written to stdout
        //return 0;

	//Create server socket
	int server_fd,new_socket; 
	long valread;
	struct sockaddr_in address;
	int addrlen = sizeof(address);
	
	if ((server_fd = socket(AF_INET,SOCK_STREAM,0))<0)	
	{
		fprintf(logfile,"cannot create socket");
		perror("cannot create socket");
		fclose(logfile);
		update_pipestatus(argv[0]);
		return 0;
	}
	fprintf(logfile,"server socket created...\n");

	//create socket address
	const int PORT = 8080; //port for clients to reach, http
	
	memset((char *)&address, 0, sizeof(address));
	address.sin_family = AF_INET; //address family
	address.sin_addr.s_addr = htonl(INADDR_ANY); //use any ip address
	address.sin_port = htons(PORT); //specify port for servers

	fprintf(logfile,"socket address created...\n");

	//bind socket
	if (bind(server_fd,(struct sockaddr *)&address,sizeof(address)) < 0)
	{
		perror("bind failed");
		fclose(logfile);
		update_pipestatus(argv[0]);
		return 0;
	}
	fprintf(logfile,"socket bound...\n");	

	//open server for listening
	if (listen(server_fd, 3) < 0) //max number of pending connections is 3
	{
		perror("In listen");
		fclose(logfile);
		update_pipestatus(argv[0]);
		exit(EXIT_FAILURE);
	}
	fprintf(logfile,"opened server for listening...\n");
	int run = 0;
	while(run == 0)
	{
		fprintf(logfile,"\n+++++++ Waiting for new connection ++++++++\n\n");
		//accept connection
		if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen))<0)
		{
			perror("In accept");
			fclose(logfile);
			update_pipestatus(argv[0]);
			exit(EXIT_FAILURE);
		}
		//set no delay flag
		int one = 1;
		setsockopt(new_socket, SOL_TCP, TCP_NODELAY, &one, sizeof(one));
		//printf("started connection ok\n");
		//first find the header
		char buffer[1024];
		char response[1024];
		valread = read(new_socket,buffer,1024);
		char *command = strstr(buffer,"POST");
		char *commandalt = strstr(buffer,"PUT");
		if (command == NULL && commandalt == NULL)
		{

			fprintf(logfile,"Error, sending response:\n");
			//Figure out error type
			for (int i = 0; i < 9; i++)
			{
				command = strstr(buffer,invalidcommands[i]);
				if (command != NULL)
				{
					fprintf(logfile,"%s\n",error1);
					strcpy(response,error1);
					break;
				}
			}
			if (command == NULL)
			{
				fprintf(logfile,"%s\n",error2);
				strcpy(response,error2);
			}

			write(new_socket, response, strlen(response));

                	fprintf(logfile,"------------------Response sent-------------------\n");
                	//close socket
                	close(new_socket);
		}
		else
		{
			fprintf(logfile,"Received good command, proceeding...\n");
			//printf("valid command\n");
			//create REQUEST struct
                        struct REQUEST client_request;
			
			//command should point to start of POST; parse the header
			char *field;
			if (commandalt == NULL)
			{
				field = strstr(buffer,"POST") + 6;
			}
			else
			{
				field = strstr(buffer,"PUT") + 5;
			}
			//get the filename
			//char fname[128];
			int count = 0;
			while (!isspace(field[count]))
			{
				client_request.fname[count] = field[count];
				count ++;

			}

			fprintf(logfile,"%s",client_request.fname);
			//move to the first header
			//while (strcmp(&field[count],'\n')!=0)
			while (field[count] != '\n')
			{
				fprintf(logfile,"%c\n",field[count]);
				count++;
			}
			count ++;

			fprintf(logfile,"FINISHED: %s",&field[count]);
			

			//create REQUEST struct
			//struct REQUEST client_request;
			//bool endRequest = false;
			
			char dest[32];
			char *delim = ":";
			char *delimsemi = ";";
			char *eline;
			int ticker = 0;
			char *enddest;
			char arg[128];
			char *boundcomp ="boundary";
			//printf("starting arg reading...\n");
			while (strstr(&field[count],delim) != NULL)
			{
				//printf("----%s\n",&field[count]);
				int i = 1;
				while (i < 9)	
				{
					if (boundcomp[i-1] != field[count+i])
					{
						break;
					}
					i ++;
				}
				if (i == 9)
				{
					//printf("hit boundary\n");
					break;
				}

				//reset array
				memset(dest, 0, sizeof(dest));
				memset(arg, 0, sizeof(arg));

				//find header label
				enddest = strstr(&field[count],delim);
				strncpy(dest, &field[count], enddest-&field[count]);
				
				fprintf(logfile,"Reading Argument %s: ",dest);
				
				//find end of line
				eline = strchr(&field[count],'\n');
				//copy argument
				strncpy(arg,enddest+1,eline-enddest-1);


				//if last line before boundary, only read up to ;
				if (strstr(arg,delimsemi) != NULL)
				{
					memset(arg,0,sizeof(arg));
					eline = strstr(&field[count],delimsemi);
					strncpy(arg,enddest+1,eline-enddest-1);
				}
				
				fprintf(logfile,"%s\n",arg);
				//printf("%s\n",arg);				
				//increment
				count += eline - &field[count] + 1;
				ticker ++;


				//case statements to setup struct
				if (strstr(dest, "Host") != NULL)
				{
					strcpy(client_request.Host,arg);
				}
				else if (strstr(dest, "User-Agent") != NULL)
				{
					strcpy(client_request.UserAgent,arg);
				}
				else if (strstr(dest, "Accept") != NULL)
				{
					strcpy(client_request.Accept,arg);
				}
				else if (strstr(dest, "Expect") != NULL)
				{
					strcpy(client_request.Expect,arg);
				}
				else if (strstr(dest, "Content-Type") != NULL)
				{
					strcpy(client_request.ContentType,arg);
				}
				else if (strstr(dest, "Content-Length") != NULL)
				{
					client_request.ContentLength = atoi(arg);
				}
				else
				{
					fprintf(logfile,"Invalid Header--> %s: %s\n",dest,arg);
					//printf("Invalid Header--> %s: %s\n",dest,arg);
				}
			}
			fprintf(logfile,"finished obtaining arguments\n");
			//printf("finished getting args\n");
			
			
			if (commandalt == NULL) //received POST command
			{
				char *ctype = "multipart/form-data";
				if (strstr(client_request.ContentType, ctype) == NULL)
				{
					fprintf(logfile,"%s\n",error3);
					strcpy(response,error3);
					write(new_socket, response, strlen(response));

                        		fprintf(logfile,"------------------Response sent-------------------\n");
					fclose(logfile);
                        		//close socket
                        		close(new_socket);
					update_pipestatus(argv[0]);
					exit(EXIT_FAILURE);
				}


				//get boundary
				fprintf(logfile,"searching for boundary\n");
				char *bound = "boundary";
				char *delimnewline = "\n\n";
				if (strstr(&field[count],bound) != NULL)
				{
				
					memset(dest, 0, sizeof(dest));
					memset(arg, 0, sizeof(arg));

					//eline = strchr(&field[count],'\n');
					//for (int i = 0; i < 2+eline - &field[count+10]; i++)
					int idx = 0;
					for (int i = 0; i < 48; i++)
					{
						if ((field[count+10] != '\n'))// && (field[count+10] != '-'))//strchr(&field[count],'\n')==NULL)
						{
							client_request.boundary[idx] = field[count+10];//field[count+10+i];
							//fprintf(logfile,"%c",client_request.boundary[idx]);
							idx++;
						}
						count++;
					}

				
					//count += eline - &field[count] + 1;
				}
				fprintf(logfile,"finished getting boundary\n");
				//fprintf(logfile,"\nReceived Boundary: %s\n",client_request.boundary);
				//printf("finished getting boundary\n");	
				//check if an Expect 100 continue message is present
				//printf("%s\n",&field[count]);	
			}
			else //received PUT command
			{	
				fprintf(logfile,"received a PUT command, no boundary\n");
				//printf("PUT COMMAND CASE\n");
			}
				
			//look for expect-100 continue
			if (strstr(&field[count],"Expect: 100-continue") != NULL)
			{
				fprintf(logfile,"sending 100-continue...\n");
				//printf("sending 100-continue...\n");
				write(new_socket, expectresponse, strlen(expectresponse));

			}
	
			//Next message should be the file data
			int framesize = 512;//64;
			unsigned char data_buffer[framesize];
			//char data[4096];
			valread = read(new_socket,data_buffer,framesize);
			//fprintf(logfile,"%s\n",data_buffer);
			
			//check if the data is here
			unsigned char *startdata = &data_buffer[0];//strstr(data_buffer,"stream") + 7;
			int idx = 0;
			for (idx = 0; idx < 512; idx ++)
			{
				int j = 0;
				for (j = 0; j < sizeof(STARTBYTES); j++)
				{
					//printf("%x-%x ,",startdata[idx+j],STARTBYTES[j]);
					if (startdata[idx+j] != STARTBYTES[j])
					{
						break;
					}
					
				}
				if (j == sizeof(STARTBYTES))
				{
					//printf("FOUND START STRING!! %d\n",idx);
					break;
				}
			}
			if (idx == 512)
			{
				//printf("DIDNT FIND IT\n");
			}


			//printf("firststuff %c \n",startdata[0]);
			//printf("about to start iterating %x\n",startdata[idx]);
			/*
			for (int i = 0; i < 10; i ++)
			{
				printf("%x ",startdata[i]);
			}
			unsigned char tmp;
			tmp = startdata[20];
			startdata[20] = '\0';
			printf("ITER %s\n",startdata);
			startdata[20] = tmp;
			while ((isspace(startdata[idx])) && (strchr(&startdata[idx],'\0') == NULL))
			{
				idx++;
			}
			fprintf(logfile,"iterate until you reach data\n");
			*/
			//printf("nextstuff %c \n",startdata[idx]);
			/*
			if ((strchr(&startdata[idx],'\0') != NULL) || (strstr(startdata+idx,client_request.boundary) != NULL))
			{
				//keep reading until we get data
				//int maxtries = 10;
				//int iter = 0;

				memset(data_buffer, 0, sizeof(data_buffer));
				valread = read(new_socket,data_buffer,4096);
				//fprintf(logfile,"re-query: %s \n",data_buffer);
				//idx = 0;
				//fprintf(logfile,"stuff: %c \n",data_buffer[0]);
				if (isspace(data_buffer[0])) //NO DATA/EMPTY FILE
				{
                                	fprintf(logfile,"%s\n",error4);
                                	strcpy(response,error4);
                                	write(new_socket, response, strlen(response));

                                	fprintf(logfile,"------------------Response sent-------------------\n");
					fclose(logfile);
                                	//close socket
                                	close(new_socket);
                                	exit(EXIT_FAILURE);
				}
				startdata = &data_buffer[0];
			}
			else
			{
				startdata += idx;
			}

			//printf("%s \n", startdata);
			*/
			
			//ok we found the starting point, now get the full dataset
			//int framesize = 512;//64;//4096;
			unsigned char data[framesize];//client_request.ContentLength];
			int nframes = 16*51; //each frame is 4096 bytes//client_request.ContentLength/framesize + 1;
			fprintf(logfile,"number of frames to read: %d \n",nframes);
			FILE *fp;
			unsigned char fullfname[strlen(client_request.fname) + 2];
			if (tofile == 1)
			{
                        	strcpy(fullfname,"./");///dataz/dsa110/imaging/NSFRB_storage/NSFRB_buffer/");
                        	strcat(fullfname,client_request.fname);
                        	fprintf(logfile,"Here's the filename:%s\n",fullfname);
				//clear the file
				fp = fopen(fullfname,"w");
				//fprintf(fp, "");
				fclose(fp);
			}
			/*
			fprintf(logfile,"printingstuff...\n");

			for (int i = 0; i < 40; i++)
                        {
                                for (int k = 0; k < 8; k++)
                                {
                                        fprintf(logfile,"%d",!!((client_request.boundary[i] << k) & 0x80));
                                }
                                fprintf(logfile," %c\n",client_request.boundary[i]);
                        } */      




                        char *boundpoint = &client_request.boundary[0];
                        //fp = fopen(fullfname, "a");
			char hit_boundary = 0;
			//int numbytes = 0;
                        //for (int j = 0; j < nframes; j++)
			//FILE *tmpstdin;
			//tmpstdin = fopen("tmpstdin","a");
			int loops = 0;
			//fcntl(new_socket, F_SETFL, O_NONBLOCK);
			int totallength = valread;
			int subclient_fd = -1;
			//off_t thing = lseek(new_socket,0,SEEK_CUR);//tell(new_socket);
			//fprintf(logfile,"THIS IS THE CURRENT FILE OFFSET: %ld\n",thing);//ftell(new_socket)); 
			while (hit_boundary == 0)
			{
				if (data_buffer[idx] == TESTSTRING[0])
				{
					int j = 0;
					for (j = 0; j < sizeof(TESTSTRING); j ++)
					{
						if (data_buffer[idx+j] != TESTSTRING[j])
						{
							break;
						}

					}
					if (j == sizeof(TESTSTRING))
					{
						fprintf(logfile,"FOUND THE BAD STRING, WHY IS IT HERE\n");

					}

				}
				//thing = lseek(new_socket,0,SEEK_CUR);
				fprintf(logfile,"loop: %d, %d, %ld\n",loops,hit_boundary & 0x01,valread);
				//if (thing == -1)
				//{
				//	fprintf(logfile,"Error during read, returned -1; value of errno: %d\n%s\n",errno,strerror(errno));
				//}
				//printf("loop: %d, %d, %ld\n",loops,hit_boundary & 0x01,valread);
				loops ++;
				if (tofile == 1)
				{
					//open file
					fp = fopen(fullfname,"a");
				}
				//loop through data buffer 
				int offset = 0;
				if (strstr(data_buffer,"ENDFILE") != NULL)
				{
					offset = 7;
					hit_boundary = 1;
					
				}
				if (toport == 1)
				{
					//subclient_send(data_buffer,valread,subclientPORT,logfile);
					subclient_fd = subclient_send_persistent(data_buffer,valread,subclientPORT,logfile,subclient_fd);
				}
				if (tofile ==1 || tostdout==1)
				{
					for (int i = 0; i < valread-offset; i++)
					{
						if (tofile == 1)
						{
							fprintf(fp,"%c",data_buffer[idx + i]);
						}
					
						if (tostdout == 1)
						{
							printf("%2.2x",data_buffer[i]);
						}
					}
				}
				idx = 0;
	
				if (tofile == 1)
				{
					//close file
					fclose(fp);	
				}

				/*if (valread > 0 && valread < framesize)
				{
					fprintf(logfile,"Only read%ld bytes, but expected%d\n",valread,framesize);
				}
				else if (strstr(data_buffer,"ENDFILE") != NULL)//(valread == 0)
				{
					hit_boundary = 1;
				*/
					/*
					//check for an error
					valread = read(new_socket,data_buffer,0);
					if (valread == -1)
					{
						printf("Read returned this error: %s\n",strerror(errno));
						close(new_socket);
						fclose(logfile);
						exit(EXIT_FAILURE);
					
					}
					//check if more data
					valread = read(new_socket,data_buffer,framesize);
					if (valread == 0)
					{
						hit_boundary = 1;
					}*/
				//}
				if (hit_boundary == 0)
				{
					memset(data_buffer, 0, sizeof(data_buffer));
                                        valread = read(new_socket,data_buffer,framesize);
					if (valread == -1)
					{
					/*	fprintf(logfile,"Error during read, returned -1; value of errno: %d\n%s\n",errno,strerror(errno));
						if (errno==11)//strcmp(strerror(errno),"EAGAIN") == 0 || strcmp(strerror(errno),"EWOULDBLOCK") == 0)
						{
							while (valread == -1)
							{
								valread = read(new_socket,data_buffer,framesize);
							}
						}

						else
						{
							fclose(logfile);
                        				close(new_socket);
							update_pipestatus(argv[0]);
                                        		exit(EXIT_FAILURE);
						}

					}*/
						fclose(logfile);
                                                close(new_socket);
                                                update_pipestatus(argv[0]);
                                                exit(EXIT_FAILURE);
					}
					totallength += valread;
                                        //printf("%s \n",data_buffer);
                                        startdata = &data_buffer[0];
				}
				else
				{
					fprintf(logfile,"breaking...\n");
					fprintf(logfile,"%s\n",data_buffer);
					for (int m = 0; m < valread; m++)
					{
						fprintf(logfile,"%.2x",data_buffer[m]);
					}
					//thing = lseek(new_socket,0,SEEK_CUR);//tell(new_socket);
					//fprintf(logfile,"THIS IS THE OFFSET NOW: %ld\n",thing);
					
					break;
				}





			}

			
			//fflush(logfile);
			//fclose(tmpstdin);
			//printf("right here\n");
			fprintf(logfile,"%p\n",data_buffer);
			fprintf(logfile,"%s",success);
			//fprintf(logfile,"%d bytes written\n",numbytes);
                        ssize_t writeout = write(new_socket, success, strlen(success));
			//printf("%ld\n",writeout);
			fprintf(logfile,"data read: %d bytes\n",totallength);
                        fprintf(logfile,"------------------Success message sent-------------------\n");
                        //close socket
                        
			//flush log file
			fflush(logfile);
			fclose(logfile);
                        close(new_socket);
			run ++;
			//free(buffer2);
		}
		//fclose(logfile);
		//run += 1;
	}
	//fclose(logfile);
	return 0;
}
