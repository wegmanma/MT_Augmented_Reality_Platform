#pragma once
#include <vector>
#include <thread>
#include <mutex>
#include "RWLock.hpp"

#include "computation.cuh"

#define NUM_BUFFERS 2

class TCPFrameCapture {
mutable RWLock m_lock[2];	// mutable: can be modified even in const methods

public:
    void start(Computation *computation_p);

    void cleanup();

    uint16_t* getToFFrame(int buffer);

    int lockMutex();

    void unlockMutex(int mtx_nr);

private:

    struct buffer {
        void* start;
        size_t                  length;
    };

    Computation* computation;

    uint16_t *image_x_h;
    uint16_t *image_y_h;
    uint16_t *image_x_d;
    uint16_t *image_y_d;

    float *cos_alpha_map_h;
    float *cos_alpha_map_d;

    float *temp_mem_265x205xfloat_0_d[4];

    uint16_t *ampl_h;
    uint8_t *conf_h;
    uint16_t *radial_h;
    uint16_t *ampl_d;
    uint8_t *conf_d;
    uint16_t *radial_d;

    uint16_t * buffers_h[2];
    uint16_t * buffers_d[2];

    CudaImage siftImage;
    SiftData siftData;
    float *memoryTmp;

    int write_buf_id;

    std::thread tid;

    bool running;

    void run();

    size_t receive_all(int socket_desc, char *client_message, int max_length);

};