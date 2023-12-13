#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <string.h>
#include <ctype.h>
#include <netinet/tcp.h>


const char *error1 = "HTTP/1.1 405 Method Not Allowed\nContent-Type: text/plain\nContent-Length: 103\n\nServer received invalid command, only configured to respond to 'POST' commands. No data transferred.\n";
const char *error2 = "HTTP/1.1 400 Bad Request\nContent-Type: text/plain\nContent-Length: 57\n\nServer received invalid command, No data transferred.\n";
char invalidcommands[9][10] = {"GET","HEAD","PUT","DELETE","CONNECT","OPTIONS","TRACE","PATCH"};
const char *error3 = "HTTP/1.1 415 Unsupported Media Type\nContent-Type: text/plain\nContent-Length: 93\n\nServer received invalid content type, requires 'multipart/form-data', No data transferred.\n";
const char *error4 = "HTTP/1.1 400 Bad Request\nContent-Type: text/plain\nContent-Length: 50\n\nServer received empty file, No data transferred.\n";
const char *success = "HTTP/1.1 200 OK\nContent-Type: text/plain\nContent-Length: 14\n\nFile Received\n";
const char *expectresponse = "HTTP/1.1 100 Continue";
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

int main() {
	//make log file to write output to, only want data written to stdout
	FILE *logfile;
	logfile = fopen("server_log.txt","w");

        fprintf(logfile,"Hello, World!\n");//printf("Hello, World!\n");
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
		return 0;
	}
	fprintf(logfile,"socket bound...\n");	

	//open server for listening
	if (listen(server_fd, 3) < 0) //max number of pending connections is 3
	{
		perror("In listen");
		fclose(logfile);
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
		if (command == NULL)
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
			char *field = strstr(buffer,"POST") + 6;
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
			if (strstr(&field[count],"Expect: 100-continue") != NULL)
			{
				fprintf(logfile,"sending 100-continue...\n");
				//printf("sending 100-continue...\n");
				write(new_socket, expectresponse, strlen(expectresponse));

			}
	
			//Next message should be the file data
			unsigned char data_buffer[4096];
			//char data[4096];
			valread = read(new_socket,data_buffer,4096);
			//fprintf(logfile,"%s\n",data_buffer);
			
			//check if the data is here
			unsigned char *startdata = strstr(data_buffer,"stream") + 7;
			int idx = 0;
			//printf("firststuff %c \n",startdata[0]);
			while ((isspace(startdata[idx])) && (strchr(&startdata[idx],'\0') == NULL))
			{
				idx++;
			}
			fprintf(logfile,"iterate until you reach data\n");
			//printf("nextstuff %c \n",startdata[idx]);
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

			
			//ok we found the starting point, now get the full dataset
			int framesize = 4096;
			unsigned char data[framesize];//client_request.ContentLength];
			int nframes = 16*51; //each frame is 4096 bytes//client_request.ContentLength/framesize + 1;
			fprintf(logfile,"number of frames to read: %d \n",nframes);
			unsigned char fullfname[128];
                        strcpy(fullfname,"./");///dataz/dsa110/imaging/NSFRB_storage/NSFRB_buffer/");
                        strcat(fullfname,client_request.fname);
                        fprintf(logfile,"Here's the filename:%s\n",fullfname);
			FILE *fp;			
			//clear the file
			fp = fopen(fullfname,"w");
			//fprintf(fp, "");
			fclose(fp);

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
			while (hit_boundary == 0)
			{
				//open file
				fp = fopen(fullfname,"a");
				
				//loop through data buffer 
				for (int i = 0; i < valread; i++)
				{
					fprintf(fp,"%c",data_buffer[i]);
					//printf("%x",data_buffer[i]);
					//printf("%c",data_buffer[i]);
					if (valread == 0)
					{
						hit_boundary = 1;
					}
					/*
					else
					{
						fprintf(logfile,"%2.2x",data_buffer[i]);
						//printf("%2.2x",data_buffer[i]);
					}*/

				}

				//close file
				fclose(fp);	


				if (hit_boundary == 0)
				{
					memset(data_buffer, 0, sizeof(data_buffer));
                                        valread = read(new_socket,data_buffer,framesize);
                                        //printf("%s \n",data_buffer);
                                        startdata = &data_buffer[0];
				}
				else
				{
					fprintf(logfile,"breaking...\n");
					break;
				}





			}

                        
			//fflush(logfile);
			//fclose(tmpstdin);
			fprintf(logfile,"%p\n",data_buffer);
			fprintf(logfile,"%s",success);
			//fprintf(logfile,"%d bytes written\n",numbytes);
                        ssize_t writeout = write(new_socket, success, strlen(success));
			printf("%ld\n",writeout);

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
