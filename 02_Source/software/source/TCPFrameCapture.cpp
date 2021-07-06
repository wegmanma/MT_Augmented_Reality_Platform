#include <vector>
#include <iostream>
#include <thread>
#include <mutex>
#include <fstream>
#include <stdio.h>      //printf
#include <string.h>     //strlen
#include <sys/socket.h> //socket
#include <arpa/inet.h>  //inet_addr
#include <unistd.h>

#define MAX 5 * 352 * 286 // Bytes

#include "TCPFrameCapture.hpp"

size_t TCPFrameCapture::receive_all(int socket_desc, char *client_message, int max_length)
{
    int size_recv, total_size = 0;

    //loop
    while (total_size < max_length)
    {
        memset(client_message + total_size, 0, 512); //clear the variable
        size_recv = recv(socket_desc, client_message + total_size, 512, 0);
        if (size_recv < 0)
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

void TCPFrameCapture::start()
{
    buffers[0] = (uint16_t *)malloc(352 * 286 * 4 * sizeof(uint16_t));
    buffers[1] = (uint16_t *)malloc(352 * 286 * 4 * sizeof(uint16_t));
    write_buf_id = 0;
    running = true;
    tid = std::thread(&TCPFrameCapture::run, this);
}

void TCPFrameCapture::cleanup()
{
    running = false;
    tid.join();
    free(buffers[0]);
    free(buffers[1]);
}

uint16_t *TCPFrameCapture::getToFFrame()
{
    if (write_buf_id == 0)
    {
        return buffers[1];
    }
    else
    {
        return buffers[0];
    }
}

int TCPFrameCapture::lockMutex()
{
    if (write_buf_id == 0)
    {
        mtx[1].lock();
        return 1;
    }
    else
    {
        mtx[0].lock();
        return 0;
    }
}

void TCPFrameCapture::unlockMutex(int mtx_nr)
{
    mtx[mtx_nr].unlock();
}

void TCPFrameCapture::run()
{
    int sock;
    struct sockaddr_in server;
    char server_data[MAX];

    //Create socket
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == -1)
    {
        printf("Could not create socket");
    }
    puts("Socket created");

    server.sin_addr.s_addr = inet_addr("10.42.0.58");
    server.sin_family = AF_INET;
    server.sin_port = htons(23999);

    //Connect to remote server
    if (connect(sock, (struct sockaddr *)&server, sizeof(server)) < 0)
    {
        perror("connect failed. Error");
    }
    uint16_t ampl[352 * 286];
    uint8_t conf[352 * 286];
    uint16_t radial[352 * 286];
    int cnt = 0;
    while (running)
    {
        //Receive a reply from the server
        size_t len = receive_all(sock, server_data, MAX);
        if (len < 0)
        {
            printf("recv failed");
            break;
        }

        // printf("Server reply len: = %ld\n", len);

        int offset_src = 0;
        mtx[write_buf_id].lock();
        memcpy(ampl, server_data + offset_src, 352 * 286 * sizeof(uint16_t));
        offset_src += 352 * 286 * sizeof(uint16_t);
        memcpy(conf, server_data + offset_src, 352 * 286 * sizeof(uint8_t));
        offset_src += 352 * 286 * sizeof(uint8_t);
        memcpy(radial, server_data + offset_src, 352 * 286 * sizeof(uint16_t));

        if (cnt == 100)
        {
            std::cout << "WRITING IMAGE DATA!" << std::endl;
            std::ofstream radialData;
            radialData.open("radial.txt");
            std::ofstream amplitudeData;
            amplitudeData.open("ampl.txt");
            for (int i = 0; i < 352 * 286; i++)
            {
                if (i < ((352 * 286) - 1))
                {
                    radialData << radial[i] << ";";
                    amplitudeData << ampl[i] << ";";
                }
                else
                {
                    radialData << radial[i];
                    amplitudeData << ampl[i];
                }
            }
            radialData.close();
            amplitudeData.close();
        }
        cnt++;
        for (int i = 0; i < 352 * 286; i++)
        {
            buffers[write_buf_id][i * 4 + 0] = ampl[i];
            buffers[write_buf_id][i * 4 + 1] = ((uint16_t)conf[i]) << 8; //
            buffers[write_buf_id][i * 4 + 2] = radial[i];
        }
        if (write_buf_id == 0)
        {
            write_buf_id = 1;
            mtx[0].unlock();
        }
        else
        {
            write_buf_id = 0;
            mtx[1].unlock();
        }
    }

    close(sock);
}