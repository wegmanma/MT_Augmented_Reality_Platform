#pragma once
#include <vector>
#include <thread>
#include <mutex>
#include "RWLock.hpp"

#define NUM_BUFFERS 2

class TCPFrameCapture {
mutable RWLock m_lock[2];	// mutable: can be modified even in const methods

public:
    void start(Computation *computation_p);

    void cleanup();

    uint16_t* getToFFrame();

    int lockMutex();

    void unlockMutex(int mtx_nr);

private:

    struct buffer {
        void* start;
        size_t                  length;
    };

    Computation* computation;

    uint16_t * buffers[2];

    int write_buf_id;

    std::thread tid;

    bool running;

    void run();

    size_t receive_all(int socket_desc, char *client_message, int max_length);

};