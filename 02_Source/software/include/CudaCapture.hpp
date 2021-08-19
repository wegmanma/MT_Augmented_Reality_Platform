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
std::mutex m_lock[2];

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
    uint16_t *pInputFrameBuffer_d;

    Computation* computation;

    uint16_t *image_x_h;
    uint16_t *image_y_h;
    uint16_t *image_x_d;
    uint16_t *image_y_d;

    uint16_t * buffers_h[2];
    uint16_t * buffers_d[2];

    uint16_t * temp_mem_1280x720x4uint16_0_d;
    uint16_t * temp_mem_1280x720x4uint16_1_d;

    int write_buf_id;

    std::thread tid;

    bool running;

    void run();

};