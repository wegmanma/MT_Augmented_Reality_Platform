#pragma once
#include <vector>
#include <thread>
#include <mutex>
#include "RWLock.hpp"

#include "computation.cuh"

extern "C" {
#include "framebuffer.h"
#include "scaler.h"
#include "setup_camera.h"
}

#define NUM_BUFFERS 2

class CudaCapture {
mutable RWLock m_lock[2];	// mutable: can be modified even in const methods

public:
    void start(Computation *computation_p);

    void cleanup();

    uint16_t* getRPiFrame(int buffer);

    int lockMutex();

    void unlockMutex(int mtx_nr);

private:

    struct buffer {
        void* start;
        size_t                  length;
    };

    AVDictionary *options;
    int errors;
    fContextStruct f_context;

    AVPacket *pInputPacket;
    AVPacket *pOutputPacket;
    AVFrame *pInputFrame;
    AVFrame *pOutputFrame;

    Computation* computation;

    uint16_t * buffers_h[2];
    uint16_t * buffers_d[2];

    int write_buf_id;

    std::thread tid;

    bool running;

    void run();

};