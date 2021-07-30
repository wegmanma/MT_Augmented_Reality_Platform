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
#include <any>

#define MAX 11 * 352 * 286 // Bytes
#define SAVE_IMAGES_TO_DISK

#include "TCPFrameCapture.hpp"

void write_data(std::string filename, uint16_t *buffer, int n)
{
    std::cout << "WRITING IMAGE DATA! " << n << std::endl;
    std::ofstream fileData;
    std::string num = std::to_string(n);
    filename = "../data/ToFData/" + filename + "_movement_" + num + ".txt";
    fileData.open(filename);
    for (int i = 0; i < 352 * 286; i++)
    {
        if (i < ((352 * 286) - 1))
        {
            fileData << buffer[i] << ";";
        }
        else
        {
            fileData << buffer[i];
        }
    }
    fileData.close();
}

size_t TCPFrameCapture::receive_all(int socket_desc, char *client_message, int max_length)
{
    int size_recv, total_size = 0;

    //loop
    // total_size = recv(socket_desc, client_message + total_size, 1107392, 0);
    // if (size_recv != 12) {
    //     return 0;
    // }
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
    buffers[0] = (uint16_t *)malloc(352 * 286 * 6 * sizeof(uint16_t));
    buffers[1] = (uint16_t *)malloc(352 * 286 * 6 * sizeof(uint16_t));
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
        std::cout << "lockMutex(): Called (buf_id = 0)" << std::endl;
        m_lock[1].lockR();
        return 1;
    }
    else
    {
        std::cout << "lockMutex(): Called (buf_id = 1)" << std::endl;
        m_lock[0].lockR();
        return 0;
    }
}

void TCPFrameCapture::unlockMutex(int mtx_nr)
{
    std::cout << "unlockMutex(): Called (buf_id = " << mtx_nr << ")" << std::endl;
    m_lock[mtx_nr].unlockR();
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
    uint16_t x[352 * 286];
    uint16_t y[352 * 286];
    uint16_t z[352 * 286];
#ifdef SAVE_IMAGES_TO_DISK
    int cnt = 0;
    int n = 0;
#endif
    while (running)
    {
        //Receive a reply from the server
        size_t len = receive_all(sock, server_data, MAX);
        if (len < 0)
        {
            printf("recv failed");
            break;
        }
        if (len != 1107392)
            continue;

        // printf("Server reply len: = %ld\n", len);

        int offset_src = 0;
        m_lock[write_buf_id].lockW();
        memcpy(ampl, server_data + offset_src, 352 * 286 * sizeof(uint16_t));
        offset_src += 352 * 286 * sizeof(uint16_t);
        memcpy(conf, server_data + offset_src, 352 * 286 * sizeof(uint8_t));
        offset_src += 352 * 286 * sizeof(uint8_t);
        memcpy(radial, server_data + offset_src, 352 * 286 * sizeof(uint16_t));
        offset_src += 352 * 286 * sizeof(uint16_t);
        memcpy(x, server_data + offset_src, 352 * 286 * sizeof(uint16_t));
        offset_src += 352 * 286 * sizeof(uint16_t);
        memcpy(y, server_data + offset_src, 352 * 286 * sizeof(uint16_t));
        offset_src += 352 * 286 * sizeof(uint16_t);
        memcpy(z, server_data + offset_src, 352 * 286 * sizeof(uint16_t));
#ifdef SAVE_IMAGES_TO_DISK
        if (cnt >= 40)
        {
            if (ampl[0] != 0)
            {
                write_data("ampl", ampl, n);
                write_data("radial", radial, n);
                n++;
            }
        }

        if (cnt == 50)
        {
            std::cout << "======================================== " << n << std::endl;
            // write_data("x",x,n);
            // write_data("y",y,n);
            // write_data("z",z,n);

            cnt = 0;
        }
        cnt++;
#endif
        for (int i = 0; i < 352 * 286; i++)
        {
            buffers[write_buf_id][i * 4 + 0] = ampl[i];
            buffers[write_buf_id][i * 4 + 1] = ampl[i]; //((uint16_t)conf[i]) << 8; //
            buffers[write_buf_id][i * 4 + 2] = ampl[i]; // radial[i];
        }
        if (write_buf_id == 0)
        {
            write_buf_id = 1;
            m_lock[0].unlockW();
        }
        else
        {
            write_buf_id = 0;
            m_lock[1].unlockW();
        }
    }

    close(sock);
}