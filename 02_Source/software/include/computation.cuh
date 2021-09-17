#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <vulkan/vulkan.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <array>
#include "linmath.h"
#include <libavutil/pixfmt.h>

#define NUM_SLICES 1


#define NUM_SCALES      5

// Scale down thread block width
#define SCALEDOWN_W    64 // 60 

// Scale down thread block height
#define SCALEDOWN_H    16 // 8

// Scale up thread block width
#define SCALEUP_W      64

// Scale up thread block height
#define SCALEUP_H       8

// Find point thread block width
#define MINMAX_W       30 //32 

// Find point thread block height
#define MINMAX_H        8 //16 
 
// Laplace thread block width
#define LAPLACE_W     128 // 56

// Laplace rows per thread
#define LAPLACE_H       4

// Number of laplace scales
#define LAPLACE_S   (NUM_SCALES+3)

// Laplace filter kernel radius
#define LAPLACE_R       4

#define LOWPASS_W      24 //56
#define LOWPASS_H      32 //16
#define LOWPASS_R       4

class CudaImage {
public:
  int width, height;
  int pitch;
  float *h_data;
  float *d_data;
  bool *d_pixelFlags;
  float *d_pixHeight;
  float *d_result;
  float *t_data;
  bool d_internalAlloc;
  bool h_internalAlloc;
public:
  CudaImage();
  ~CudaImage();
  void Allocate(int width, int height, int pitch, bool withHost, float *devMem = NULL, float *hostMem = NULL, bool withPixelFlags = false);
  void Download();
  void Readback();
  void InitTexture();
  void CopyToTexture(CudaImage &dst, bool host);
};

int iDivUp(int a, int b);
int iDivDown(int a, int b);
int iAlignUp(int a, int b);
int iAlignDown(int a, int b);

typedef struct {
  float xpos;
  float ypos;   
  float scale;
  float sharpness;
  float edgeness;
  float orientation;
  float score;
  float ambiguity;
  int match;
  float match_xpos;
  float match_ypos;
  float match_error;
  float subsampling;
  float x_3d;
  float y_3d;
  float z_3d;
  float conf;
  bool draw;
  float distance;
  float match_x_3d;
  float match_y_3d;
  float match_z_3d;
  float match_distance;
  float empty[3];
  float data[128];
} SiftPoint;

typedef struct {
  int numPts;         // Number of available Sift points
  int maxPts;         // Number of allocated Sift points
#ifdef MANAGEDMEM
  SiftPoint *m_data;  // Managed data
#else
  SiftPoint *h_data;  // Host (CPU) data
  SiftPoint *d_data;  // Device (GPU) data
#endif
} SiftData;

class Computation {


public:
//using linmat vectors for CUDA.
struct Vertex2 {
    vec4 pos;
    vec3 color;
    vec3 texcoords;

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription = {};

        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex2);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions = {};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex2, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex2, color);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex2, texcoords);
        return attributeDescriptions;
    }

};

    std::vector<uint32_t> indices; 
    
    size_t vertexBufSize;

    void* cudaDevVertptr;

    void* cudaDevIndexptr;

    int memHandleVertex;

    int memHandleIndex;

    int cudaUpdateVkVertexBufSemaphoreHandle;

    int vkUpdateCudaVertexBufSemaphoreHandle;

    uint8_t  vkDeviceUUID[VK_UUID_SIZE];

    double AnimTime=1.0f; 

    float* imagePtr; // temp pointer for handing images to host for debugging. 

    void initCuda();

    void cleanup();
    
void InitSiftData(SiftData &data, int numPoints, int numSlices, bool host, bool dev);

float *AllocSiftTempMemory(int width, int height, int numOctaves, bool scaleUp);

void ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, bool scaleUp, float *tempMemory, unsigned char * chardata);

void MatchSiftData(SiftData &data1, SiftData &data2, cudaStream_t stream);

void yuvToRGB_Convert(CudaImage &RGBImage, unsigned char *yuvdata);

void sharpenImage(CudaImage &greyscaleImage, unsigned char *yuvdata, float amount);

void LowPass_prepareKernel(void);

void tof_camera_undistort(float *dst, uint16_t *src, uint16_t *xCoordsPerPixel, uint16_t *yCoordsPerPixel, cudaStream_t stream, float *cosAlpha = NULL);

void rpi_camera_undistort(uint16_t *dst, uint16_t *src, uint16_t *xCoordsPerPixel, uint16_t *yCoordsPerPixel, cudaStream_t stream);

void tof_sobel(float *dst_mag, float *dst_phase, float *src, cudaStream_t stream);

void tof_maxfilter_3x3(float *dst_mag, float *src, cudaStream_t stream);

void tof_minfilter_3x3(float *dst_mag, float *src, cudaStream_t stream);

void tof_meanfilter_3x3(float *dst_mag, float *src, cudaStream_t stream);

void tof_fill_area(float *mask, float *src, int seed_x, int seed_y, float thresh, cudaStream_t stream);

void buffer_Float_to_uInt16x4(uint16_t *dst, float *src, int width, int height, cudaStream_t stream);

void buffer_Float_to_uInt16x4_SCALE(uint16_t *dst, float *src, int width, int height, cudaStream_t stream);

void buffer_uint16x4_to_Float(float *dst, uint16_t *src, int width, int height, cudaStream_t stream);

int8_t gpuConvertBayer10toRGB(uint16_t * src, uint16_t * dst, const int width, const int height, const enum AVPixelFormat format, const uint8_t bpp, cudaStream_t stream);

void drawSiftData(uint16_t *rgbImage, CudaImage &greyscaleImage, SiftData &siftData, int width, int height, cudaStream_t stream);

void FindHomography(SiftData &data, float *homography, int *numMatches, int numLoops, float minScore, float maxAmbiguity, float thresh);

void addDepthInfoToSift(SiftData &data, float* depthData, cudaStream_t stream, float *x, float *y, float *z, float *conf);

void findRotationTranslation_step0(SiftData &data, float *tempMemory, bool *index_list, mat4x4 *rotation, vec4 *translation, cudaStream_t stream);

void findRotationTranslation_step1(SiftData &data, float *tempMemory, bool *index_list, mat4x4 *rotation, vec4 *translation, cudaStream_t stream);

void findRotationTranslation_step2(SiftData &data, float *tempMemory, bool *index_list, mat4x4 *rotation, vec4 *translation, cudaStream_t stream);

void ransacFromFoundRotationTranslation(SiftData &data, SiftData &data_old, mat4x4 *rotation, vec4 *translation, cudaStream_t stream);

void findOptimalRotationTranslation(SiftData &data, float *tempMemory, mat4x4 rotation, vec4 translation, cudaStream_t stream);

private:

cudaStream_t streamToRun;

cudaExternalSemaphore_t cudaExtVkUpdateCudaVertexBufSemaphore;

cudaExternalSemaphore_t cudaExtCudaUpdateVkVertexBufSemaphore;



int setCudaVkDevice();

void FreeSiftTempMemory(float *memoryTmp);

void PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel);

void LowPass(CudaImage &res, CudaImage &src, float scale, unsigned char *chardata = NULL);

void LowPass_forSubImages(float *res, float *src, int width, int height);

int ExtractSiftLoop(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp, float *memorySub);

void ExtractSiftOctave(SiftData &siftData, CudaImage &img, int octave, float thresh, float lowestScale, float subsampling, float *memoryTmp);

void ScaleUp(CudaImage &res, CudaImage &src, unsigned char *chardata = NULL);

void ScaleDown(CudaImage &res, CudaImage &src, float variance);

void LaplaceMulti(cudaTextureObject_t texObj, CudaImage &baseImage, CudaImage *results, int octave);

void RescalePositions(SiftData &siftData, float scale);

void FindPointsMulti(CudaImage *sources, SiftData &siftData, float thresh, float edgeLimit, float factor, float lowestScale, float subsampling, int octave);

void ComputeOrientations(cudaTextureObject_t texObj, CudaImage &src, SiftData &siftData, int octave);

void ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData, float subsampling, int octave);
};