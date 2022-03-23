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
#include <iomanip>
#include <fcntl.h>
#include <any>
#include <computation.cuh>

#define ELEMENT 145*300

// #define PRINT_DATA

#define FRAME_LENGTH 5 * 352 * 286 // Bytes
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
#define SAVE_IMAGES_TO_DISK

#include "TCPFrameCapture.hpp"

void write_data(std::string filename, uint16_t *buffer, int n, int width, int height)
{
    // std::cout << "WRITING IMAGE DATA! " << n << std::endl;
    std::ofstream fileData;
    std::string num = std::to_string(n);
    filename = "../data/DemoImages/" + filename + "_demo_image_" + num + ".txt";
    fileData.open(filename);
    for (int i = 0; i < width * height; i++)
    {
        if (i < ((width * height) - 1))
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

size_t TCPFrameCapture::receive_all(int socket_desc, char *client_message, struct sockaddr_in servaddr)
{
    int size_recv, total_size = 0;
    int size_to_recv = 8192*4;
    //loop
    size_t bytes_sent = 0;
    // std::cout << "Sending request in TCP!" << std::endl;

    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 10000;
    if (setsockopt(socket_desc, SOL_SOCKET, SO_RCVTIMEO,&tv,sizeof(tv)) < 0) {
        perror("Error");
    }

    bytes_sent = sendto(socket_desc, "1", 2, MSG_CONFIRM, (const struct sockaddr *) &servaddr, sizeof(servaddr));
    if (bytes_sent != 2)
        std::cout << "Problem sending request in TCP!" << std::endl;
    // std::cout << "Waiting for data to come..." << std::endl;
    while (total_size < FRAME_LENGTH)
    {
        // memset(client_message + total_size, 0, size_to_recv); //clear the variable
        socklen_t len = sizeof(servaddr);
        size_recv = recvfrom(socket_desc, (char*)(client_message+ total_size), size_to_recv, 0, (struct sockaddr *) &servaddr,
                &len);
        if (size_recv <= 0)
        {
            // printf("Oh dear, something went wrong with read()! %s\n", strerror(errno));
            return 0;
        }
        else
        {
            total_size += size_recv;
        }
        if (FRAME_LENGTH - total_size <= size_to_recv) {
            size_to_recv = FRAME_LENGTH - total_size;
        }
        if (size_to_recv == 0) break;
    }
    // std::cout << "Element has brightness: " << (int)client_message[ELEMENT*2] << ":" << (int)client_message[ELEMENT*2+1] << " ";
    return total_size;
}

void TCPFrameCapture::start(Computation *computation_p)
{
    computation = computation_p;
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
    cudaHostAlloc((void **)&x_h, 205 * 265 * sizeof(uint16_t), cudaHostAllocMapped);
    cudaHostAlloc((void **)&y_h, 205 * 265 * sizeof(uint16_t), cudaHostAllocMapped);
    cudaHostAlloc((void **)&z_h, 205 * 265 * sizeof(uint16_t), cudaHostAllocMapped);
    cudaHostAlloc((void **)&conf_h, 352 * 286 * sizeof(uint16_t), cudaHostAllocMapped);
    cudaHostAlloc((void **)&ampl_h, 352 * 286 * sizeof(uint16_t), cudaHostAllocMapped);
    cudaHostAlloc((void **)&cos_alpha_map_h, 205 * 265 * sizeof(float), cudaHostAllocMapped);
    cudaHostAlloc((void **)&temp_mem_265x205xfloat_nocache_h, 205 * 265 * sizeof(float), cudaHostAllocMapped);
    cudaHostAlloc((void **)&best_translation_h, sizeof(vec4), cudaHostAllocMapped);
    cudaHostAlloc((void **)&best_rotation_h, sizeof(mat4x4), cudaHostAllocMapped);
    cudaHostAlloc((void **)&opt_translation_h, sizeof(vec4), cudaHostAllocMapped);
    cudaHostAlloc((void **)&opt_rotation_h, sizeof(mat4x4), cudaHostAllocMapped);
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
    cudaHostGetDevicePointer((void **)&opt_translation_d, (void *)opt_translation_h, 0);
    cudaHostGetDevicePointer((void **)&opt_rotation_d, (void *)opt_rotation_h, 0);
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

    computation->InitSiftData(siftData[0], 512, 1, false, false, true);
    computation->InitSiftData(siftData[1], 512, 1, false, false, true);

    siftImage.Allocate(256, 205, 256, false, NULL, NULL);
    memoryTmp = computation->AllocSiftTempMemory(256, 205, 5, true);
    computation->InitClusterSet(kMeansClusters, 20, false, false, true);
    computation->InitClusterSet(kMeansClusterStorage, 200, false, false, true);

    

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

void TCPFrameCapture::attachOutputBuffer(uint16_t* buffer){
    outputBuffers.push_back(buffer);
}

int TCPFrameCapture::getRotationTranslation(int buffer, mat4x4 rotation, quat translation)
{
    mat4x4_dup(rotation, rotation_buf[buffer]);
    // print_mat4x4("rotation",rotation);
    translation[0] = translation_buf[buffer][0];
    translation[1] = translation_buf[buffer][1];
    translation[2] = translation_buf[buffer][2];
    translation[3] = translation_buf[buffer][3];
    int ret = 0;
    if (newdata)
    {
        if (standing_still) ret = 2;
        else ret = 1;
    }

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
        int least, greatest;
    cudaDeviceGetStreamPriorityRange(&least, &greatest);
    // std::cout << "cuda priorities - least: " << least << " greatest: " << greatest << std::endl; 
    cudaStreamCreateWithPriority(&tcpCaptureStream, 0, greatest);

    int sockfd;
    char buffer[FRAME_LENGTH];
    struct sockaddr_in     servaddr;
   
    // Creating socket file descriptor
    if ( (sockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0 ) {
        perror("socket creation failed");
        exit(EXIT_FAILURE);
    }
   
    memset(&servaddr, 0, sizeof(servaddr));
       
    // Filling server information
    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(23999);
    servaddr.sin_addr.s_addr = inet_addr("10.42.0.58");

    float initBlur = 0.0f;
    float thresh = 2.0f;
    float lowestScale = 0.0f;
    newdata = false;
    standing_still = false;
    std::cout << "Start loop" << std::endl;
    while (running)
    {
        
        //Receive a reply from the server
        size_t len = receive_all(sockfd, buffer, servaddr);
        if (len < 0)
        {
            printf("recv failed");
            break;
        }
        if (len != 503360)
            continue;

        // If data is valid:  Copy data to 
        startTime = std::chrono::steady_clock::now();

        memcpy(ampl_h, buffer, 352 * 286 * sizeof(uint16_t));
        int offset_radial = 302016;
        memcpy(radial_h, buffer + offset_radial, 352 * 286 * sizeof(uint16_t));

        cudaStreamSynchronize(tcpCaptureStream);

        computation->tof_camera_undistort(temp_mem_265x205xfloat_0_d[1], radial_d, image_x_d, image_y_d, tcpCaptureStream, cos_alpha_map_d);
        computation->tof_medianfilter_3x3(temp_mem_265x205xfloat_0_d[0], temp_mem_265x205xfloat_0_d[1], tcpCaptureStream);
        checkMsg("Error at tof_camera_undistort\n");        
        computation->tof_camera_undistort(temp_mem_265x205xfloat_0_d[5], ampl_d, image_x_d, image_y_d, tcpCaptureStream);
        computation->scale_float_to_float(siftImage.d_data, temp_mem_265x205xfloat_0_d[5], 256, 205, tcpCaptureStream);
        checkMsg("Error at cudaStreamSynchronize\n");
        computation->ExtractSift(siftData[write_buf_id], siftImage, 4, initBlur, thresh, 0.0f, false, memoryTmp, NULL, tcpCaptureStream);
        checkMsg("Error at ExtractSift\n");
        if (siftData[write_buf_id].numPts <= 5)
        {
            std::cout << "Continuing, too few features found!" << std::endl;
            continue;
        }
        
        computation->addDepthInfoToSift(siftData[write_buf_id], temp_mem_265x205xfloat_0_d[0], tcpCaptureStream, temp_mem_265x205xfloat_0_d[1], temp_mem_265x205xfloat_0_d[2], temp_mem_265x205xfloat_0_d[3], temp_mem_265x205xfloat_0_d[4]); 
        computation->InitSeedPoints(kMeansClusters);
        if (write_buf_id == 0)
        {
            computation->MatchSiftData(siftData[0], siftData[1], tcpCaptureStream);
            computation->ransac2d(siftData[0], siftData[1], temp_mem_265x205xfloat_nocache_d, index_list_d, ransac_dx_d, ransac_dy_d, tcpCaptureStream);
            computation->findRotationTranslation_step0(siftData[0], temp_mem_265x205xfloat_nocache_d, index_list_d, best_rotation_d, best_translation_d, tcpCaptureStream);
            computation->findRotationTranslation_step1(siftData[0], temp_mem_265x205xfloat_nocache_d, index_list_d, best_rotation_d, best_translation_d, tcpCaptureStream);
            computation->findRotationTranslation_step2(siftData[0], temp_mem_265x205xfloat_nocache_d, index_list_d, best_rotation_d, best_translation_d, tcpCaptureStream);
            computation->ransacFromFoundRotationTranslation(siftData[0], siftData[1], best_rotation_d, best_translation_d, tcpCaptureStream);
            computation->findOptimalRotationTranslation(siftData[0], temp_mem_265x205xfloat_nocache_d, opt_rotation_d, opt_translation_d, tcpCaptureStream);
            computation->kMeansClustering(kMeansClusters,siftData[0]);
        }
        else
        {
            computation->MatchSiftData(siftData[1], siftData[0], tcpCaptureStream);
            computation->ransac2d(siftData[1], siftData[0], temp_mem_265x205xfloat_nocache_d, index_list_d, ransac_dx_d, ransac_dy_d, tcpCaptureStream);
            computation->findRotationTranslation_step0(siftData[1], temp_mem_265x205xfloat_nocache_d, index_list_d, best_rotation_d, best_translation_d, tcpCaptureStream);
            computation->findRotationTranslation_step1(siftData[1], temp_mem_265x205xfloat_nocache_d, index_list_d, best_rotation_d, best_translation_d, tcpCaptureStream);
            computation->findRotationTranslation_step2(siftData[1], temp_mem_265x205xfloat_nocache_d, index_list_d, best_rotation_d, best_translation_d, tcpCaptureStream);
            computation->ransacFromFoundRotationTranslation(siftData[1], siftData[0], best_rotation_d, best_translation_d, tcpCaptureStream);
            computation->findOptimalRotationTranslation(siftData[1], temp_mem_265x205xfloat_nocache_d, opt_rotation_d, opt_translation_d, tcpCaptureStream);
            computation->kMeansClustering(kMeansClusters,siftData[1]);
        }
        
        if ((sqrt((*ransac_dy_h) * (*ransac_dy_h) + (*ransac_dx_h) * (*ransac_dx_h)) > 0.1) || (0))
        {
            standing_still = false;
        }
        else
        {
            standing_still = true;
        }
        cudaStreamSynchronize(tcpCaptureStream);

        m_lock[write_buf_id].lockW();
        quat rot;
        quat trans;
        computation->StoreClusterPositions(kMeansClusters, kMeansClusterStorage, rot, trans, 10, 30, 0.5);
        for (uint16_t* i : outputBuffers) {
            computation->drawSiftData(i, siftImage, siftData[write_buf_id], 256, 205, tcpCaptureStream, kMeansClusterStorage);
        }
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                best_rotation[j][i] = *(opt_rotation_h[i][j]);
            }
        }
        // print_mat4x4("best_rotation",best_rotation);
        mat4x4_dup(rotation_buf[write_buf_id], best_rotation);
        translation_buf[write_buf_id][0] = *opt_translation_h[0];
        translation_buf[write_buf_id][1] = *opt_translation_h[1];
        translation_buf[write_buf_id][2] = *opt_translation_h[2];
        translation_buf[write_buf_id][3] = *opt_translation_h[3];
        // print_vec4("best_translation",translation_buf[write_buf_id]);
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

        timeSpan = endTime - startTime;

        double nseconds = double(timeSpan.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
 
        std::cout << "ToF FPS: " << 1/nseconds << std::endl;

    }
    cudaStreamDestroy(tcpCaptureStream);
}