#include <stdio.h>
#include <sys/socket.h>
#include <unistd.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <string.h>
#include <ctype.h> 


const char *error1 = "HTTP/1.1 405 Method Not Allowed\nContent-Type: text/plain\nContent-Length: 103\n\nServer received invalid command, only configured to respond to 'POST' commands. No data transferred.\n";
const char *error2 = "HTTP/1.1 400 Bad Request\nContent-Type: text/plain\nContent-Length: 57\n\nServer received invalid command, No data transferred.\n";
char invalidcommands[9][10] = {"GET","HEAD","PUT","DELETE","CONNECT","OPTIONS","TRACE","PATCH"};
const char *error3 = "HTTP/1.1 415 Unsupported Media Type\nContent-Type: text/plain\nContent-Length: 93\n\nServer received invalid content type, requires 'multipart/form-data', No data transferred.\n";
const char *error4 = "HTTP/1.1 400 Bad Request\nContent-Type: text/plain\nContent-Length: 50\n\nServer received empty file, No data transferred.\n";
const char *success = "HTTP/1.1 200 OK\nContent-Type: text/plain\nContent-Length: 14\n\nFile Received\n";
const char *buffdir = "~/data/NSFRB_OFFLINE_PIPELINE/NSFRB_storage/NSFRB_buffer/";
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

        printf("Hello, World!\n");
        //return 0;

	//Create server socket
	int server_fd,new_socket; 
	long valread;
	struct sockaddr_in address;
	int addrlen = sizeof(address);
	
	if ((server_fd = socket(AF_INET,SOCK_STREAM,0))<0)	
	{
		printf("cannot create socket");
		perror("cannot create socket");
		return 0;
	}
	printf("server socket created...");

	//create socket address
	const int PORT = 8080; //port for clients to reach, http
	
	memset((char *)&address, 0, sizeof(address));
	address.sin_family = AF_INET; //address family
	address.sin_addr.s_addr = htonl(INADDR_ANY); //use any ip address
	address.sin_port = htons(PORT); //specify port for servers

	printf("socket address created...");

	//bind socket
	if (bind(server_fd,(struct sockaddr *)&address,sizeof(address)) < 0)
	{
		perror("bind failed");
		return 0;
	}
	printf("socket bound...");	

	//open server for listening
	if (listen(server_fd, 3) < 0) //max number of pending connections is 3
	{
		perror("In listen");
		exit(EXIT_FAILURE);
	}
	printf("opened server for listening...");

	while(1)
	{
		printf("\n+++++++ Waiting for new connection ++++++++\n\n");
		//accept connection
		if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen))<0)
		{
			perror("In accept");
			exit(EXIT_FAILURE);
		}


		//first find the header
		char buffer[1024];
		char response[1024];
		valread = read(new_socket,buffer,1024);
		char *command = strstr(buffer,"POST");
		if (command == NULL)
		{

			printf("Error, sending response:\n");
			//Figure out error type
			for (int i = 0; i < 9; i++)
			{
				command = strstr(buffer,invalidcommands[i]);
				if (command != NULL)
				{
					printf("%s\n",error1);
					strcpy(response,error1);
					break;
				}
			}
			if (command == NULL)
			{
				printf("%s\n",error2);
				strcpy(response,error2);
			}

			write(new_socket, response, strlen(response));

                	printf("------------------Response sent-------------------\n");
                	//close socket
                	close(new_socket);
		}
		else
		{
			printf("Received good command, proceeding...\n");

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

			printf("%s",client_request.fname);
			//move to the first header
			//while (strcmp(&field[count],'\n')!=0)
			while (field[count] != '\n')
			{
				printf("%c\n",field[count]);
				count++;
			}
			count ++;

			printf("FINISHED: %s",&field[count]);
			

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
			while (strstr(&field[count],delim) != NULL)
			{
				//reset array
				memset(dest, 0, sizeof(dest));
				memset(arg, 0, sizeof(arg));

				//find header label
				enddest = strstr(&field[count],delim);
				strncpy(dest, &field[count], enddest-&field[count]);
				
				printf("Reading Argument %s: ",dest);
				
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
				
				printf("%s\n",arg);
				
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
					printf("Invalid Header--> %s: %s\n",dest,arg);
				}
			}

			char *ctype = "multipart/form-data";
			if (strstr(client_request.ContentType, ctype) == NULL)
			{
				printf("%s\n",error3);
				strcpy(response,error3);
				write(new_socket, response, strlen(response));

                        	printf("------------------Response sent-------------------\n");
                        	//close socket
                        	close(new_socket);
				exit(EXIT_FAILURE);
			}
			//get boundary
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
						printf("%c",client_request.boundary[idx]);
						idx++;
					}
					count++;
				}

				
				//count += eline - &field[count] + 1;
			}

			printf("\nReceived Boundary: %s\n",client_request.boundary);
			
				
			//Next message should be the file data
			unsigned char data_buffer[4096];
			//char data[4096];
			valread = read(new_socket,data_buffer,4096);
			printf("%s\n",data_buffer);
			
			//check if the data is here
			unsigned char *startdata = strstr(data_buffer,"stream") + 7;
			int idx = 0;
			//printf("firststuff %c \n",startdata[0]);
			while ((isspace(startdata[idx])) && (strchr(&startdata[idx],'\0') == NULL))
			{
				idx++;
			}
			//printf("nextstuff %c \n",startdata[idx]);
			if ((strchr(&startdata[idx],'\0') != NULL) || (strstr(startdata+idx,client_request.boundary) != NULL))
			{
				//keep reading until we get data
				//int maxtries = 10;
				//int iter = 0;

				memset(data_buffer, 0, sizeof(data_buffer));
				valread = read(new_socket,data_buffer,4096);
				printf("re-query: %s \n",data_buffer);
				//idx = 0;
				printf("stuff: %c \n",data_buffer[0]);
				if (isspace(data_buffer[0])) //NO DATA/EMPTY FILE
				{
                                	printf("%s\n",error4);
                                	strcpy(response,error4);
                                	write(new_socket, response, strlen(response));

                                	printf("------------------Response sent-------------------\n");
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
			printf("number of frames to read: %d \n",nframes);
			unsigned char fullfname[128];
                        strcpy(fullfname,buffdir);//"/dataz/dsa110/imaging/NSFRB_storage/NSFRB_buffer/");
                        strcat(fullfname,client_request.fname);
                        printf("Here's the filename:%s\n",fullfname);
			FILE *fp;			
			//clear the file
			fp = fopen(fullfname,"w");
			//fprintf(fp, "");
			fclose(fp);


			printf("printingstuff...\n");

			for (int i = 0; i < 40; i++)
                        {
                                for (int k = 0; k < 8; k++)
                                {
                                        printf("%d",!!((client_request.boundary[i] << k) & 0x80));
                                }
                                printf(" %c\n",client_request.boundary[i]);
                        }       


                        char *boundpoint = &client_request.boundary[0];
                        //fp = fopen(fullfname, "a");
			char hit_boundary = 0;
                        //for (int j = 0; j < nframes; j++)
			while (hit_boundary == 0)
			{
				//open file
				fp = fopen(fullfname,"a");
				
				//loop through data buffer 
				for (int i = 0; i < valread; i++)
				{
					fprintf(fp,"%c",data_buffer[i]);
					if (valread == 48)
					{
						hit_boundary = 1;
					}

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
					printf("breaking...\n");
					break;
				}





			}
                        

			printf("%s",success);
                        write(new_socket, success, strlen(success));

                        printf("------------------Success message sent-------------------\n");
                        //close socket
                        
                        close(new_socket);

			//free(buffer2);
		}
	}
	return 0;
}
