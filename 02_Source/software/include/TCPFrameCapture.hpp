#pragma once
#include <vector>
#include <thread>
#include <mutex>

#define NUM_BUFFERS 2

class TCPFrameCapture {


public:
    void start();

    void cleanup();

    uint16_t* getToFFrame();

    int lockMutex();

    void unlockMutex(int mtx_nr);

private:

    struct buffer {
        void* start;
        size_t                  length;
    };

    uint16_t * buffers[2];

    int write_buf_id;

    std::thread tid;

    bool running;

    std::mutex mtx[2];

    void run();

    size_t receive_all(int socket_desc, char *client_message, int max_length);

};