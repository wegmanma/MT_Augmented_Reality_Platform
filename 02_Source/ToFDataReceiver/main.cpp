/*
	C ECHO client example using sockets
*/
#include <stdio.h>	//printf
#include <string.h>	//strlen
#include <sys/socket.h>	//socket
#include <arpa/inet.h>	//inet_addr
#include <unistd.h>

#define MAX 5*352*286 // Bytes

size_t receive_all(int socket_desc, char *client_message, int max_length) {
	int size_recv , total_size= 0;
	
	//loop
	while(total_size<max_length)
	{
		memset(client_message+total_size ,0 , 512);	//clear the variable
		size_recv =  recv(socket_desc , client_message+total_size , 512 , 0);
		if(size_recv < 0)
		{
			break;
		}
		else
		{
			total_size += size_recv;
		}
	}
	
	return total_size;
}

int main(int argc , char *argv[])
{
	int sock;
	struct sockaddr_in server;
	char server_data[MAX];
	
	//Create socket
	sock = socket(AF_INET , SOCK_STREAM , 0);
	if (sock == -1)
	{
		printf("Could not create socket");
	}
	puts("Socket created");
	
	server.sin_addr.s_addr = inet_addr("10.42.0.58");
	server.sin_family = AF_INET;
	server.sin_port = htons( 23999 );

	//Connect to remote server
	if (connect(sock , (struct sockaddr *)&server , sizeof(server)) < 0)
	{
		perror("connect failed. Error");
		return 1;
	}
	
	puts("Connected\n");
	
	//keep communicating with server
	while(1)
	{
		printf("recv\n");	
		//Receive a reply from the server
		size_t len = receive_all(sock , server_data , MAX);
		if( len < 0)
		{
			printf("recv failed");
			break;
		}
		
		printf("Server reply len: = %ld\n",len);
		uint16_t ampl[352*286];
		uint8_t conf[352*286];
		uint16_t radial[352*286];		

		int offset = 0;
		memcpy(ampl,server_data+offset,352*286*sizeof(uint16_t));
		offset += 352*286*sizeof(uint16_t);
		memcpy(conf,server_data+offset,352*286*sizeof(uint8_t));
		offset += 352*286*sizeof(uint8_t);
		memcpy(radial,server_data+offset,352*286*sizeof(uint16_t));
		float avgr_ampl = 0.0f;
		float avgr_radial = 0.0f;
		for (int i = 0; i < 352*286; i++) {
			avgr_ampl += ((float)ampl[i]);
			avgr_radial += ((float)radial[i]);
		}
		avgr_ampl = avgr_ampl/ 352*286;
		printf("Average: %f\n Radial: %f\n",avgr_ampl, avgr_radial);
	}
	
	close(sock);
	return 0;
}