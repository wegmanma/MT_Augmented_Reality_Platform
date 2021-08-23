#include <iostream>

#include <cuda_runtime.h>

#include "CudaCapture.hpp"

#define INPUT_DRIVER "video4linux2,v4l2"
#define INPUT_STREAM_URL "/dev/video1"
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

    buffers_h[0] = NULL; 
    buffers_h[1] = NULL;
    image_x_h = NULL;
    image_y_h = NULL;

    options = NULL;

    cudaHostAlloc((void **)&buffers_h[0],  WIDTH * HEIGHT * 4 * sizeof(uint16_t),  cudaHostAllocMapped);
    cudaHostAlloc((void **)&buffers_h[1], WIDTH * HEIGHT *  4 * sizeof(uint16_t), cudaHostAllocMapped);
    cudaHostAlloc((void **)&image_x_h,  1280 * 720 * sizeof(uint16_t),  cudaHostAllocMapped);
    cudaHostAlloc((void **)&image_y_h, 1280 * 720 * sizeof(uint16_t), cudaHostAllocMapped);

    cudaHostGetDevicePointer((void **)&buffers_d[0] ,  (void *) buffers_h[0] , 0);
    cudaHostGetDevicePointer((void **)&buffers_d[1] , (void *) buffers_h[1], 0);
    cudaHostGetDevicePointer((void **)&image_x_d    ,  (void *) image_x_h   , 0);
    cudaHostGetDevicePointer((void **)&image_y_d    , (void *) image_y_h   , 0);

    FILE *datfile;
    char buff[256];
    sprintf(buff, "%s", "../data/x_corr_Raspi.dat");
    datfile = fopen(buff, "r");
    fread(&(image_x_h[0]), sizeof(__uint16_t), 1280 * 720, datfile);
    fclose(datfile);

    sprintf(buff, "%s", "../data/y_corr_Raspi.dat");
    datfile = fopen(buff, "r");
    fread(&(image_y_h[0]), sizeof(__uint16_t), 1280 * 720, datfile);
    fclose(datfile);

    cudaMalloc((void **)&temp_mem_1280x720x4uint16_0_d,1280 * 720*4 * sizeof(uint16_t));
    cudaMalloc((void **)&temp_mem_1280x720x4uint16_1_d,1280 * 720*4 * sizeof(uint16_t));
    checkMsg("Problem with RPI allocations:\n");
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
        // std::cout << "read-locking 1" << std::flush;
        m_lock[1].lock();
        // std::cout << " locked!" << std::endl;
        return 1;
    }
    else
    {
        // std::cout << "read-locking 0" << std::flush;
        m_lock[0].lock();
        // std::cout << " locked!" << std::endl;
        return 0;
    }
}

void CudaCapture::unlockMutex(int mtx_nr)
{
    m_lock[mtx_nr].unlock();
    // std::cout << "read unlocked!" << mtx_nr << std::endl;
}

void CudaCapture::run()
{
    cudaStream_t cudaCaptureStream;
    cudaStreamCreate(&cudaCaptureStream);
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
    // std::cout << "Entering while loop " << std::endl;
    while (running)
    {
        

        lost_frames = get_last_frame(&f_context, pInputPacket, pInputFrame);   
        if (lost_frames == -1) continue;   
        m_lock[write_buf_id].lock();  
        computation->gpuConvertBayer10toRGB((uint16_t *) pInputFrame->data[0], temp_mem_1280x720x4uint16_0_d, WIDTH, HEIGHT, AV_PIX_FMT_BAYER_RGGB10, 4, cudaCaptureStream); 

        computation->rpi_camera_undistort(buffers_d[write_buf_id],temp_mem_1280x720x4uint16_0_d,image_x_d,image_y_d, cudaCaptureStream);      
        cudaStreamSynchronize(cudaCaptureStream);
        // std::cout << "RPi: buffers_h right after filling: "<< buffers_h[write_buf_id][50*265*4+50*4+0] << " lost_frames: " << lost_frames << std::endl;

        if (write_buf_id == 0)
        {
            write_buf_id = 1;
            m_lock[0].unlock();
        }
        else
        {

            write_buf_id = 0;
            m_lock[1].unlock();
        }

    }
    cudaStreamDestroy(cudaCaptureStream);

}