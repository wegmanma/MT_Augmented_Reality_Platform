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
#include <fcntl.h>
#include <any>
#include <computation.cuh>

// #define PRINT_DATA

#define FRAME_LENGTH 11 * 352 * 286 // Bytes
#define checkMsg(msg) __checkMsg(msg, __FILE__, __LINE__)

inline void __checkMsg(const char *errorMessage, const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "checkMsg() CUDA error: %s in file <%s>, line %i : %s.\n", errorMessage, file, line, cudaGetErrorString(err));
        exit(-1);
    }
}
// #define SAVE_IMAGES_TO_DISK

#include "TCPFrameCapture.hpp"

void write_data(std::string filename, uint16_t *buffer, int n)
{
    std::cout << "WRITING IMAGE DATA! " << n << std::endl;
    std::ofstream fileData;
    std::string num = std::to_string(n);
    filename = "../data/ToFData/" + filename + "_chess_placement_" + num + ".txt";
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
    int size_to_recv = 512;
    //loop
    // total_size = recv(socket_desc, client_message + total_size, 1107392, 0);
    // if (size_recv != 12) {
    //     return 0;
    // }
    size_t bytes_sent = 0;
    // std::cout << "Sending request in TCP!" << std::endl;

    bytes_sent = send(socket_desc, "100", 4, 0);
    if (bytes_sent != 4)
        std::cout << "Problem sending request in TCP!" << std::endl;
    while (total_size < max_length)
    {
        // std::cout << "Begin: Total_size: " << total_size << " size_recv: " << size_recv << "size_to_recv" << size_to_recv << std::endl;
        memset(client_message + total_size, 0, size_to_recv); //clear the variable
        size_recv = recv(socket_desc, client_message + total_size, size_to_recv, 0);
        if (size_recv <= 0)
        {
            int size_recv, total_size = 0;
            int size_to_recv = 512;
            continue;
        }
        else
        {
            total_size += size_recv;
        }
        // if (max_length - total_size <= size_to_recv) {
        //     size_to_recv = max_length - total_size;
        // }
        // std::cout << "End: Total_size: " << total_size << " size_recv: " << size_recv << "size_to_recv" << size_to_recv << std::endl;
    }
    // std::cout << "First element has brightness: " << (int)client_message[0] << " total size: " << total_size << std::endl;
    return total_size;
}

void TCPFrameCapture::start(Computation *computation_p)
{
    computation = computation_p;
    // buffers[0] = (uint16_t *)malloc(352 * 286 * 6 * sizeof(uint16_t));
    // buffers[1] = (uint16_t *)malloc(352 * 286 * 6 * sizeof(uint16_t));
    // image_x = (uint16_t *)malloc(205 * 265 * sizeof(uint16_t));
    // image_y = (uint16_t *)malloc(205 * 265 * sizeof(uint16_t));
    cudaSetDeviceFlags(cudaDeviceMapHost);

    buffers_h[0] = NULL;
    buffers_h[1] = NULL;
    image_x_h = NULL;
    image_y_h = NULL;
    radial_h = NULL;
    ampl_h = NULL;
    conf_h = NULL;
    cos_alpha_map_h = NULL;

    cudaHostAlloc((void **)&buffers_h[0], 205 * 265 * 4 * sizeof(uint16_t), cudaHostAllocMapped);
    cudaHostAlloc((void **)&buffers_h[1], 205 * 265 * 4 * sizeof(uint16_t), cudaHostAllocMapped);
    cudaHostAlloc((void **)&image_x_h, 205 * 265 * sizeof(uint16_t), cudaHostAllocMapped);
    cudaHostAlloc((void **)&image_y_h, 205 * 265 * sizeof(uint16_t), cudaHostAllocMapped);
    cudaHostAlloc((void **)&radial_h, 352 * 286 * sizeof(uint16_t), cudaHostAllocMapped);
    cudaHostAlloc((void **)&x_h, 352 * 286 * sizeof(uint16_t), cudaHostAllocMapped);
    cudaHostAlloc((void **)&y_h, 352 * 286 * sizeof(uint16_t), cudaHostAllocMapped);
    cudaHostAlloc((void **)&z_h, 352 * 286 * sizeof(uint16_t), cudaHostAllocMapped);
    cudaHostAlloc((void **)&conf_h, 352 * 286 * sizeof(uint16_t), cudaHostAllocMapped);
    cudaHostAlloc((void **)&ampl_h, 352 * 286 * sizeof(uint16_t), cudaHostAllocMapped);
    cudaHostAlloc((void **)&cos_alpha_map_h, 205 * 265 * sizeof(float), cudaHostAllocMapped);
    cudaHostAlloc((void **)&temp_mem_265x205xfloat_nocache_h, 205 * 265 * sizeof(float), cudaHostAllocMapped);
    cudaHostAlloc((void **)&best_translation_h, sizeof(vec4), cudaHostAllocMapped);
    cudaHostAlloc((void **)&best_rotation_h, sizeof(mat4x4), cudaHostAllocMapped);
    cudaHostAlloc((void **)&ransac_dx_h, sizeof(float), cudaHostAllocMapped);
    cudaHostAlloc((void **)&ransac_dy_h, sizeof(float), cudaHostAllocMapped);

    cudaHostGetDevicePointer((void **)&buffers_d[0], (void *)buffers_h[0], 0);
    cudaHostGetDevicePointer((void **)&buffers_d[1], (void *)buffers_h[1], 0);
    cudaHostGetDevicePointer((void **)&image_x_d, (void *)image_x_h, 0);
    cudaHostGetDevicePointer((void **)&image_y_d, (void *)image_y_h, 0);
    cudaHostGetDevicePointer((void **)&ampl_d, (void *)ampl_h, 0);
    cudaHostGetDevicePointer((void **)&radial_d, (void *)radial_h, 0);
    cudaHostGetDevicePointer((void **)&x_d, (void *)x_h, 0);
    cudaHostGetDevicePointer((void **)&y_d, (void *)y_h, 0);
    cudaHostGetDevicePointer((void **)&z_d, (void *)z_h, 0);
    cudaHostGetDevicePointer((void **)&conf_d, (void *)conf_h, 0);
    cudaHostGetDevicePointer((void **)&cos_alpha_map_d, (void *)cos_alpha_map_h, 0);
    cudaHostGetDevicePointer((void **)&temp_mem_265x205xfloat_nocache_d, (void *)temp_mem_265x205xfloat_nocache_h, 0);
    cudaHostGetDevicePointer((void **)&best_translation_d, (void *)best_translation_h, 0);
    cudaHostGetDevicePointer((void **)&best_rotation_d, (void *)best_rotation_h, 0);
    cudaHostGetDevicePointer((void **)&ransac_dx_d, (void *)ransac_dx_h, 0);
    cudaHostGetDevicePointer((void **)&ransac_dy_d, (void *)ransac_dy_h, 0);

    cudaMalloc((void **)&temp_mem_265x205xfloat_0_d[0], 2 * 205 * 265 * sizeof(float));
    // checkMsg("Problem with cudaMalloc [0]");
    cudaMalloc((void **)&temp_mem_265x205xfloat_0_d[1], 2 * 205 * 265 * sizeof(float));
    // checkMsg("Problem with cudaMalloc [1]");
    cudaMalloc((void **)&temp_mem_265x205xfloat_0_d[2], 2 * 205 * 265 * sizeof(float));
    // checkMsg("Problem with cudaMalloc [2]");
    cudaMalloc((void **)&temp_mem_265x205xfloat_0_d[3], 2 * 205 * 265 * sizeof(float));
    cudaMalloc((void **)&temp_mem_265x205xfloat_0_d[4], 2 * 205 * 265 * sizeof(float));
    cudaMalloc((void **)&temp_mem_265x205xfloat_0_d[5], 2 * 205 * 265 * sizeof(float));

    cudaMalloc((void **)&index_list_d, 512 * 512 * sizeof(bool));
// checkMsg("Problem with cudaMalloc [3]");
#ifdef PRINT_DATA
    computation->InitSiftData(siftData[0], 512, 1, true, true);
    computation->InitSiftData(siftData[1], 512, 1, true, true);
#else
    computation->InitSiftData(siftData[0], 512, 1, false, true);
    computation->InitSiftData(siftData[1], 512, 1, false, true);
#endif
    siftImage.Allocate(256, 205, 256, false, NULL, NULL);
    memoryTmp = computation->AllocSiftTempMemory(256, 205, 5, true);

    FILE *datfile;
    char buff[256];
    sprintf(buff, "%s", "../data/x_corr_ToF.dat");
    datfile = fopen(buff, "r");
    fread(&(image_x_h[0]), sizeof(__uint16_t), 205 * 265, datfile);
    fclose(datfile);

    sprintf(buff, "%s", "../data/y_corr_ToF.dat");
    datfile = fopen(buff, "r");
    fread(&(image_y_h[0]), sizeof(__uint16_t), 205 * 265, datfile);
    fclose(datfile);

    sprintf(buff, "%s", "../data/cos_alpha_ToF.dat");
    datfile = fopen(buff, "r");
    fread(&(cos_alpha_map_h[0]), sizeof(float), 205 * 265, datfile);
    fclose(datfile);
    std::cout << "Float in cos_alpha[0]" << cos_alpha_map_h[0] << std::endl;
    write_buf_id = 0;
    running = true;
    tid = std::thread(&TCPFrameCapture::run, this);
}

void TCPFrameCapture::cleanup()
{
    running = false;
    tid.join();
    cudaFree(buffers_h[0]);
    cudaFree(buffers_h[1]);
}

uint16_t *TCPFrameCapture::getToFFrame(int buffer)
{

    return buffers_h[buffer];
}

bool TCPFrameCapture::getRotationTranslation(int buffer, mat4x4 rotation, vec4 translation)
{
    mat4x4_dup(rotation, rotation_buf[buffer]);
    translation[0] = translation_buf[buffer][0];
    translation[1] = translation_buf[buffer][1];
    translation[2] = translation_buf[buffer][2];
    translation[3] = translation_buf[buffer][3];
    bool ret = newdata;
    newdata = false;
    return ret;
}

int TCPFrameCapture::lockMutex()
{
    if (write_buf_id == 0)
    {
        // std::cout << "lockMutexR(): 1" << std::endl;
        m_lock[1].lockR();
        // std::cout << "locked R: 1" << std::endl;
        return 1;
    }
    else
    {
        // std::cout << "lockMutexR(): 0" << std::endl;
        m_lock[0].lockR();
        // std::cout << "locked R: 0" << std::endl;
        return 0;
    }
}

void TCPFrameCapture::unlockMutex(int mtx_nr)
{
    // std::cout << "unlockMutexR(): " << mtx_nr << std::endl;
    m_lock[mtx_nr].unlockR();
}

void TCPFrameCapture::run()
{
    cudaStream_t tcpCaptureStream;
    cudaStreamCreate(&tcpCaptureStream);
    int sock;
    struct sockaddr_in server;
    char server_data[FRAME_LENGTH + 4];

    //Create socket
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == -1)
    {
        printf("Could not create socket");
    }
    // puts("Socket created");

    server.sin_addr.s_addr = inet_addr("10.42.0.58");
    server.sin_family = AF_INET;
    server.sin_port = htons(23999);

    //Connect to remote server
    if (connect(sock, (struct sockaddr *)&server, sizeof(server)) < 0)
    {
        perror("connect failed. Error");
    }
#ifdef SAVE_IMAGES_TO_DISK
    int cnt = 0;
    int n = 0;
#endif
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);

    std::chrono::steady_clock::time_point endTime;
    std::chrono::steady_clock::time_point startTime;
    std::chrono::steady_clock::duration timeSpan;

    std::cout << "Start loop" << std::endl;
    double quality = 0.0f;
    int inliers_count = 0;
    int frame_count = 0;
    int outliers_count = 0;
    float x_traveled = 0.;
    float y_traveled = 0.;
    float z_traveled = 0.;
    float min_diag = 1.0;
    float initBlur = 0.0f;
    float thresh = 2.0f;
    float lowestScale = 0.0f;
    newdata = false;
    while (running)
    {
        // std::cout << "Start of loop" << std::endl;
        startTime = std::chrono::steady_clock::now();
        //Receive a reply from the server
        size_t len = receive_all(sock, server_data, FRAME_LENGTH);
        if (len < 0)
        {
            printf("recv failed");
            break;
        }
        if (len != 1107392)
            continue;

        // printf("Server reply len: = %ld\n", len);

        int offset_src = 0;
        
        // std::cout << "locking W" << write_buf_id << std::endl;
        memcpy(ampl_h, server_data + offset_src, 352 * 286 * sizeof(uint16_t));
        offset_src += 352 * 286 * sizeof(uint16_t);
        //memcpy(conf_h, server_data + offset_src, 352 * 286 * sizeof(uint8_t));
        for (int i = 0; i < 352 * 286; i++)
        {
            uint16_t *ptr = conf_h + i;
            *ptr = server_data[offset_src + i] * 255;
        }
        offset_src += 352 * 286 * sizeof(uint8_t);
        memcpy(radial_h, server_data + offset_src, 352 * 286 * sizeof(uint16_t));
        offset_src += 352 * 286 * sizeof(uint16_t);
        memcpy(x_h, server_data + offset_src, 352 * 286 * sizeof(uint16_t));
        offset_src += 352 * 286 * sizeof(uint16_t);
        memcpy(y_h, server_data + offset_src, 352 * 286 * sizeof(uint16_t));
        offset_src += 352 * 286 * sizeof(uint16_t);
        memcpy(z_h, server_data + offset_src, 352 * 286 * sizeof(uint16_t));

        // std::cout << "start undistorting" << std::endl;
        cudaStreamSynchronize(tcpCaptureStream);
        computation->tof_camera_undistort(temp_mem_265x205xfloat_0_d[1], radial_d, image_x_d, image_y_d, tcpCaptureStream, cos_alpha_map_d);
        // computation->tof_meanfilter_3x3(temp_mem_265x205xfloat_0_d[1],temp_mem_265x205xfloat_0_d[0], tcpCaptureStream);
        computation->tof_medianfilter_3x3(temp_mem_265x205xfloat_0_d[0],temp_mem_265x205xfloat_0_d[1], tcpCaptureStream);
        // computation->tof_meanfilter_3x3(temp_mem_265x205xfloat_0_d[1],temp_mem_265x205xfloat_0_d[0], tcpCaptureStream);
        // computation->tof_meanfilter_3x3(temp_mem_265x205xfloat_0_d[0],temp_mem_265x205xfloat_0_d[1], tcpCaptureStream);
        checkMsg("Error at tof_camera_undistort\n");
        computation->tof_camera_undistort(temp_mem_265x205xfloat_0_d[5], conf_d, image_x_d, image_y_d, tcpCaptureStream);
        // computation->tof_meanfilter_3x3(temp_mem_265x205xfloat_0_d[4],temp_mem_265x205xfloat_0_d[5], tcpCaptureStream);
        computation->tof_camera_undistort(temp_mem_265x205xfloat_0_d[5], ampl_d, image_x_d, image_y_d, tcpCaptureStream);
        computation->scale_float_to_float(siftImage.d_data, temp_mem_265x205xfloat_0_d[5], 256, 205, tcpCaptureStream);
        // computation->tof_camera_undistort(temp_mem_265x205xfloat_0_d[1],x_d,image_x_d,image_y_d, tcpCaptureStream);
        // computation->tof_camera_undistort(temp_mem_265x205xfloat_0_d[2],y_d,image_x_d,image_y_d, tcpCaptureStream);
        // computation->tof_camera_undistort(temp_mem_265x205xfloat_0_d[3],z_d,image_x_d,image_y_d, tcpCaptureStream);

        //computation->tof_meanfilter_3x3(temp_mem_265x205xfloat_0_d[1],temp_mem_265x205xfloat_0_d[0], tcpCaptureStream);
        //computation->tof_meanfilter_3x3(temp_mem_265x205xfloat_0_d[0],temp_mem_265x205xfloat_0_d[1], tcpCaptureStream);
        //computation->tof_sobel(temp_mem_265x205xfloat_0_d[2],NULL,temp_mem_265x205xfloat_0_d[0], tcpCaptureStream);
        //computation->tof_maxfilter_3x3(temp_mem_265x205xfloat_0_d[3],temp_mem_265x205xfloat_0_d[2], tcpCaptureStream);
        //computation->tof_fill_area(temp_mem_265x205xfloat_0_d[0],temp_mem_265x205xfloat_0_d[3],50,50,150.0, tcpCaptureStream);
        cudaStreamSynchronize(tcpCaptureStream);
        std::this_thread::sleep_for(std::chrono::seconds(5));
        // continue;
        // // computation->buffer_Float_to_uInt16x4(buffers_d[write_buf_id],siftImage.d_data,256,205, tcpCaptureStream);
        // checkMsg("Error at cudaStreamSynchronize\n");
        // computation->ExtractSift(siftData[write_buf_id], siftImage, 4, initBlur, thresh, 0.0f, false, memoryTmp, NULL, tcpCaptureStream);
        // checkMsg("Error at ExtractSift\n");
        // if (siftData[write_buf_id].numPts == 0)
        // {
// // 
        //     continue;
        // }
        // // std::cout << "Undistorted data, adding Deptht Info to Sift features" << std::endl;
        // computation->addDepthInfoToSift(siftData[write_buf_id], temp_mem_265x205xfloat_0_d[0], tcpCaptureStream, temp_mem_265x205xfloat_0_d[1], temp_mem_265x205xfloat_0_d[2], temp_mem_265x205xfloat_0_d[3], temp_mem_265x205xfloat_0_d[4]);
        // cudaStreamSynchronize(tcpCaptureStream);
        // // computation->buffer_Float_to_uInt16x4(buffers_d[write_buf_id],temp_mem_265x205xfloat_0_d[1],256,205, tcpCaptureStream);
        // if (write_buf_id == 0)
        // {
// 
        //     computation->MatchSiftData(siftData[0], siftData[1], tcpCaptureStream);
        //     computation->ransac2d(siftData[0], siftData[1], temp_mem_265x205xfloat_nocache_d, index_list_d, ransac_dx_d, ransac_dy_d, tcpCaptureStream);
        //     if ((sqrt((*ransac_dy_h) * (*ransac_dy_h) + (*ransac_dx_h) * (*ransac_dx_h)) > 1.0)||(0))
        //     {
        //         computation->findRotationTranslation_step0(siftData[0], temp_mem_265x205xfloat_nocache_d, index_list_d, best_rotation_d, best_translation_d, tcpCaptureStream);
        //         computation->findRotationTranslation_step1(siftData[0], temp_mem_265x205xfloat_nocache_d, index_list_d, best_rotation_d, best_translation_d, tcpCaptureStream);
        //         computation->findRotationTranslation_step2(siftData[0], temp_mem_265x205xfloat_nocache_d, index_list_d, best_rotation_d, best_translation_d, tcpCaptureStream);
        //     }
        //     else
        //     {
        //         mat4x4_identity(*best_rotation_h);
        //         *best_translation_h[0] = 0.0f;
        //         *best_translation_h[1] = 0.0f;
        //         *best_translation_h[2] = 0.0f;
        //     }
        //     // std::cout << "rot_trans" << std::endl;
        //     computation->ransacFromFoundRotationTranslation(siftData[0], siftData[1], best_rotation_d, best_translation_d, tcpCaptureStream);
        // }
        // else
        // {
// 
        //     computation->MatchSiftData(siftData[1], siftData[0], tcpCaptureStream);
        //     computation->ransac2d(siftData[1], siftData[0], temp_mem_265x205xfloat_nocache_d, index_list_d, ransac_dx_d, ransac_dy_d, tcpCaptureStream);
        //     if ((sqrt((*ransac_dy_h) * (*ransac_dy_h) + (*ransac_dx_h) * (*ransac_dx_h)) > 1.0)||(0))
        //     {
        //         computation->findRotationTranslation_step0(siftData[1], temp_mem_265x205xfloat_nocache_d, index_list_d, best_rotation_d, best_translation_d, tcpCaptureStream);
        //         computation->findRotationTranslation_step1(siftData[1], temp_mem_265x205xfloat_nocache_d, index_list_d, best_rotation_d, best_translation_d, tcpCaptureStream);
        //         computation->findRotationTranslation_step2(siftData[1], temp_mem_265x205xfloat_nocache_d, index_list_d, best_rotation_d, best_translation_d, tcpCaptureStream);
        //     }
        //     else
        //     {
                 int i, j;
                 for (i = 0; i < 4; ++i)
                     for (j = 0; j < 4; ++j) *(best_rotation_h[i][j]) = i == j ? 1.f : 0.f;
                 *(best_translation_h[0]) = 0.0f;
                 *(best_translation_h[1]) = 0.0f;
                 *(best_translation_h[2]) = 0.0f;
// 
        //     }
        //     // std::cout << "Num points: "  << siftData[1].numPts << "\n";
        //     computation->ransacFromFoundRotationTranslation(siftData[1], siftData[0], best_rotation_d, best_translation_d, tcpCaptureStream);
        // }
        // cudaStreamSynchronize(tcpCaptureStream);
        // //  std::cout << "Best matrix:" << std::endl;
        // // for (int i = 0; i<3; i++) {
        // //     for (int j = 0; j<3; j++) {
        // //         std::cout << *(best_rotation_h[j][i]) << " ";
        // //     }
        // //     std::cout << std::endl;
        // //  }
        // //   std::cout << "Translation:" << std::endl;
        // //  for (int i = 0; i<3; i++) {
        // //      std::cout << *(best_translation_h[i]) << " ";
        // //      std::cout << std::endl;
        // //  }
        // if (min_diag > *(best_rotation_h[0][0]))
        //     min_diag = *(best_rotation_h[0][0]);
        // if (min_diag > *(best_rotation_h[1][1]))
        //     min_diag = *(best_rotation_h[1][1]);
        // if (min_diag > *(best_rotation_h[2][2]))
        //     min_diag = *(best_rotation_h[2][2]);
// 
        // if ((*(best_rotation_h[0][0]) < 0.9995) || (*(best_rotation_h[1][1]) < 0.9995) || (*(best_rotation_h[2][2]) < 0.9995))
        // {
        //     outliers_count++;
        // }
        // else
        // {
        //     x_traveled += *(best_translation_h[0]);
        //     y_traveled += *(best_translation_h[1]);
        //     z_traveled += *(best_translation_h[2]);
        //     quality += ((double)(*(best_rotation_h[0][0]) + *(best_rotation_h[1][1]) + *(best_rotation_h[2][2]))) / 3.0;
        //     inliers_count++;
        // }

        // if ((frame_count > 0) && (frame_count % 100 == 0))
        // {
        //     // quality = quality/((double)inliers_count);
        //     std::cout << thresh << ";" << initBlur << ";" << lowestScale << ";" << quality / ((double)inliers_count) << ";" << x_traveled << ";" << y_traveled << ";" << z_traveled << ";" << outliers_count << ";" << min_diag << std::endl;
        //     // quality = 0;
        //     // inliers_count = 0;
        //     // outliers_count = 0;
        //     // thresh = thresh+0.5;
        //     // if (frame_count == 1000) exit(0);
        // }
        m_lock[write_buf_id].lockW();
        frame_count++;
        computation->drawSiftData(buffers_d[write_buf_id], siftImage, siftData[write_buf_id], 256, 205, tcpCaptureStream);
        // computation->buffer_Float_to_uInt16x4(buffers_d[write_buf_id],temp_mem_265x205xfloat_0_d[0],256,205, tcpCaptureStream);
        // cudaStreamSynchronize(tcpCaptureStream);
        // Processing finished, change buffer index.
        mat4x4_dup(rotation_buf[write_buf_id],*best_rotation_h);
        translation_buf[write_buf_id][0] = *best_translation_h[0];
        translation_buf[write_buf_id][1] = *best_translation_h[1];
        translation_buf[write_buf_id][2] = *best_translation_h[2];
        translation_buf[write_buf_id][3] = *best_translation_h[3];
        newdata = true;

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
        endTime = std::chrono::steady_clock::now();
        std::chrono::steady_clock::duration timeSpan = endTime - startTime;
        double nseconds = double(timeSpan.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
        // std::cout << "ToF Frame by frame time: " << nseconds << "s"<<  std::endl;
        // while(1) {
        //     std::this_thread::sleep_for(std::chrono::microseconds(5000));
        // }
    }

    close(sock);
    cudaStreamDestroy(tcpCaptureStream);
}