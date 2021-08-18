#include <iostream>

#include <cuda_runtime.h>

#include "CudaCapture.hpp"

#define INPUT_DRIVER "video4linux2,v4l2"
#define INPUT_STREAM_URL "/dev/video0"
#define WIDTH 1280
#define HEIGHT 720

#define checkMsg(msg)       __checkMsg(msg, __FILE__, __LINE__)

inline void __checkMsg(const char *errorMessage, const char *file, const int line)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "checkMsg() CUDA error: %s in file <%s>, line %i : %s.\n", errorMessage, file, line, cudaGetErrorString(err));
    exit(-1);
  }
}

void CudaCapture::start(Computation* computation_p)
{
    computation = computation_p;
    cudaSetDeviceFlags(cudaDeviceMapHost);    

    buffers_h[0] = NULL; 
    buffers_h[1] = NULL;

    options = NULL;

    cudaHostAlloc((void **)&buffers_h[0],  WIDTH * HEIGHT * 4 * sizeof(uint16_t),  cudaHostAllocMapped);
    cudaHostAlloc((void **)&buffers_h[1], WIDTH * HEIGHT *  4 * sizeof(uint16_t), cudaHostAllocMapped);

    cudaHostGetDevicePointer((void **)&buffers_d[0] ,  (void *) buffers_h[0] , 0);
    cudaHostGetDevicePointer((void **)&buffers_d[1] , (void *) buffers_h[1], 0);

    write_buf_id = 0;
    running = true;
    tid = std::thread(&CudaCapture::run, this);
}

void CudaCapture::cleanup()
{
    running = false;
    tid.join();
    cudaFree(buffers_h[0]);
    cudaFree(buffers_h[1]);
}

uint16_t *CudaCapture::getRPiFrame(int buffer)
{

    return buffers_h[buffer];
  
}

int CudaCapture::lockMutex()
{
    if (write_buf_id == 0)
    {
        std::cout << "read-locking 1" << std::flush;
        m_lock[1].lockR();
        std::cout << " locked!" << std::endl;
        return 1;
    }
    else
    {
        std::cout << "read-locking 0" << std::flush;
        m_lock[0].lockR();
        std::cout << " locked!" << std::endl;
        return 0;
    }
}

void CudaCapture::unlockMutex(int mtx_nr)
{
    m_lock[mtx_nr].unlockR();
    std::cout << "read unlocked!" << mtx_nr << std::endl;
}

void CudaCapture::run()
{
    int lost_frames = 0;
    std::cout << "Started run() -> av_dict_set()" << std::endl;
    av_dict_set(&options, "framerate", "60", 0);
    std::cout << "1" << std::endl;
    av_dict_set(&options, "input_format", "bayer_rggb10", 0); // mjpeg, yuyv422, bayer_rggb10
    std::cout << "2" << std::endl;
    av_dict_set(&options, "video_size", "1280x720", 0); // 3264x2464, 640x480, 1280x720
    std::cout << "last av_dict_set()" << std::endl;
    errors = set_camera_mode(INPUT_STREAM_URL, 4);
    errors = setup_av_device(&f_context, INPUT_DRIVER);
    errors = setup_av_format_context(&f_context, AVFMT_FLAG_NONBLOCK, INPUT_STREAM_URL, &options);
    errors = find_streams_on_media(f_context.avInputFormatContext, NULL);
    errors = setup_input_codec(&f_context, 0, NULL);
    errors = allocate_packet_flow((void **)&pInputPacket);
    errors = allocate_frame_flow((void **)&pInputFrame);

    av_dump_format(f_context.avInputFormatContext, 0, INPUT_STREAM_URL, 0); // get stream info
    while (running)
    {
        lost_frames = get_last_frame(&f_context, pInputPacket, pInputFrame);   
        if (lost_frames == -1) continue;     
        computation->gpuConvertBayer10toRGB((uint16_t *) pInputFrame->data[0], buffers_d[write_buf_id], WIDTH, HEIGHT, AV_PIX_FMT_BAYER_RGGB10, 4);        
        cudaDeviceSynchronize();
        std::cout << "RPi: buffers_h right after filling: "<< buffers_h[write_buf_id][50*265*4+50*4+0] << " lost_frames: " << lost_frames << std::endl;

        if (write_buf_id == 0)
        {
            write_buf_id = 1;
            // m_lock[0].unlockW();
        }
        else
        {

            write_buf_id = 0;
            // m_lock[1].unlockW();
        }

    }

}