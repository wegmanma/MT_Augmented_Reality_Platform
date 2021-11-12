#include <cuda_runtime.h>
#include <cuda.h>
#include <helper_cuda.h>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <array>
#include <iomanip>
#include <limits>
#include "computation.cuh"
#include "common.h"
#include "svd3_cuda.cuh"

#include <curand.h>
#include <curand_kernel.h>

#define HEIGHT_ANGLE 0.87 / 2 //rad
#define WIDTH_ANGLE 1.04 / 2  //rad

// 16:9 display format.
#define GRIDSIZEX 1920
#define GRIDSIZEY 1080
#define GAUSSIANSIZE 25
#define MIN_SCORE 0.9

#define checkMsg(msg) __checkMsg(msg, __FILE__, __LINE__)
#define checkMsgNoFail(msg) __checkMsgNoFail(msg, __FILE__, __LINE__)

cudaExternalMemory_t cudaExtMemVertexBuffer;
cudaExternalMemory_t cudaExtMemIndexBuffer;
cudaExternalSemaphore_t cudaExtCudaUpdateVkVertexBufSemaphore;
cudaExternalSemaphore_t cudaExtVkUpdateCudaVertexBufSemaphore;

__constant__ int d_MaxNumPoints;
__device__ unsigned int d_PointCounter[2 * 8 * 2 + 1];
__constant__ float d_ScaleDownKernel[5];
__constant__ float d_LowPassKernel[2 * LOWPASS_R + 1];
__constant__ float d_GaussKernel[GAUSSIANSIZE * GAUSSIANSIZE];
__constant__ float d_LaplaceKernel[8 * 12 * 16];

int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }
int iDivDown(int a, int b) { return a / b; }
int iAlignUp(int a, int b) { return (a % b != 0) ? (a - a % b + b) : a; }
int iAlignDown(int a, int b) { return a - a % b; }

template <class T>
__device__ __inline__ T ShiftDown(T var, unsigned int delta, int width = 32)
{
#if (CUDART_VERSION >= 9000)
  return __shfl_down_sync(0xffffffff, var, delta, width);
#else
  return __shfl_down(var, delta, width);
#endif
}

template <class T>
__device__ __inline__ T ShiftUp(T var, unsigned int delta, int width = 32)
{
#if (CUDART_VERSION >= 9000)
  return __shfl_up_sync(0xffffffff, var, delta, width);
#else
  return __shfl_up(var, delta, width);
#endif
}

template <class T>
__device__ __inline__ T Shuffle(T var, unsigned int lane, int width = 32)
{
#if (CUDART_VERSION >= 9000)
  return __shfl_sync(0xffffffff, var, lane, width);
#else
  return __shfl(var, lane, width);
#endif
}

__device__ inline float clamp(float val, float mn, float mx)
{
  return (val >= mn) ? ((val <= mx) ? val : mx) : mn;
}

__global__ void CopyCharToImage(unsigned char *src, float *dst,
                                unsigned int width, unsigned int height)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx * 2 >= width)
  {
    return;
  }

  for (int i = 0; i < height; ++i)
  {
    int y0 = __uint2float_rn(src[i * width * 2 + idx * 4 + 1]);
    int y1 = __uint2float_rn(src[i * width * 2 + idx * 4 + 3]);

    dst[i * width + idx * 2 + 0] = clamp(y0, 0.0f, 255.0f);
    dst[i * width + idx * 2 + 1] = clamp(y1, 0.0f, 255.0f);
  }
}

#define MAP_HOST_DEVICE_POINTER 1

#define SHIFT_XAVIER 5
#define SHIFT_RESULT 0

#define LEFT(x, y, imgw) ((x)-1 + (y) * (imgw))
#define RIGHT(x, y, imgw) ((x) + 1 + (y) * (imgw))
#define TOP(x, y, imgw) ((x) + ((y)-1) * (imgw))
#define BOT(x, y, imgw) ((x) + ((y) + 1) * (imgw))
#define TL(x, y, imgw) ((x)-1 + ((y)-1) * (imgw))
#define BL(x, y, imgw) ((x)-1 + ((y) + 1) * (imgw))
#define TR(x, y, imgw) ((x) + 1 + ((y)-1) * (imgw))
#define BR(x, y, imgw) ((x) + 1 + ((y) + 1) * (imgw))

#define PIX(in, x, y, imgw) (in[((x) + (y) * (imgw))] >> SHIFT_XAVIER)

#define INTERPOLATE_H(in, x, y, w) ((in[LEFT(x, y, w)] >> SHIFT_XAVIER) / 2 + (in[RIGHT(x, y, w)] >> SHIFT_XAVIER) / 2)

#define INTERPOLATE_V(in, x, y, w) ((in[TOP(x, y, w)] >> SHIFT_XAVIER) / 2 + (in[BOT(x, y, w)] >> SHIFT_XAVIER) / 2)

#define INTERPOLATE_HV(in, x, y, w) ((in[LEFT(x, y, w)] >> SHIFT_XAVIER) / 4 + (in[RIGHT(x, y, w)] >> SHIFT_XAVIER) / 4 + (in[TOP(x, y, w)] >> SHIFT_XAVIER) / 4 + (in[BOT(x, y, w)] >> SHIFT_XAVIER) / 4)

#define INTERPOLATE_X(in, x, y, w) ((in[TL(x, y, w)] >> SHIFT_XAVIER) / 4 + (in[BL(x, y, w)] >> SHIFT_XAVIER) / 4 + (in[TR(x, y, w)] >> SHIFT_XAVIER) / 4 + (in[BR(x, y, w)] >> SHIFT_XAVIER) / 4)

#define RED 0
#define GREEN 1
#define BLUE 2

// ## WHITE BALANCE GAIN FACTORS (use with the White Patch method)
#define GAIN_RED 3
#define OFFSET_RED 55
#define GAIN_GREEN 2.8
#define OFFSET_GREEN 55
#define GAIN_BLUE 4.7
#define OFFSET_BLUE 55

// ## COLOR SATURATION FACTOR
/*
K is the saturation factor
K=1 means no change
K > 1 increases saturation
0<K<1 decreases saturation,  K=0 produces B&W ,  K<0 inverts color
*/
#define K 1

#define SATURATION_RED(R, G, B) ((0.299 + 0.701 * K) * R + (0.587 * (1 - K)) * G + (0.114 * (1 - K)) * B)
#define SATURATION_GREEN(R, G, B) ((0.299 * (1 - K)) * R + (0.587 + 0.413 * K) * G + (0.114 * (1 - K)) * B)
#define SATURATION_BLUE(R, G, B) ((0.299 * (1 - K)) * R + (0.587 * (1 - K)) * G + (0.114 + 0.886 * K) * B)

__global__ void bayer_to_rgb(uint16_t *in, uint16_t *out, const int imgw, const int imgh, const uint8_t bpp,
                             const int2 r, const int2 gr, const int2 gb, const int2 b)
{
  int x = 2 * ((blockDim.x * blockIdx.x) + threadIdx.x) + 1;
  int y = 2 * ((blockDim.y * blockIdx.y) + threadIdx.y) + 1;
  int elemCols = imgw * bpp;

  uint32_t r_color, g_color, b_color;

  if ((x + 2) < imgw && (x - 1) >= 0 && (y + 2) < imgh && (y - 1) >= 0)
  {
    /* Red */
    r_color = (PIX(in, x + r.x, y + r.y, imgw) >> SHIFT_RESULT);
    g_color = (INTERPOLATE_HV(in, x + r.x, y + r.y, imgw) >> SHIFT_RESULT);
    b_color = (INTERPOLATE_X(in, x + r.x, y + r.y, imgw) >> SHIFT_RESULT);
    //r_color = SATURATION_RED(r_color, g_color, b_color);
    //g_color = SATURATION_GREEN(r_color, g_color, b_color);
    //b_color = SATURATION_BLUE(r_color, g_color, b_color);
    r_color = ((r_color >> 1) < OFFSET_RED) ? 0 : (((r_color >> 1) - OFFSET_RED) * GAIN_RED);
    g_color = ((g_color >> 1) < OFFSET_GREEN) ? 0 : (((g_color >> 1) - OFFSET_GREEN) * GAIN_GREEN);
    b_color = ((b_color >> 1) < OFFSET_BLUE) ? 0 : (((b_color >> 1) - OFFSET_BLUE) * GAIN_BLUE);
    out[(y + r.y) * elemCols + (x + r.x) * bpp + RED] = (uint16_t)((r_color > 255) ? 255 : r_color) * 255;
    out[(y + r.y) * elemCols + (x + r.x) * bpp + GREEN] = (uint16_t)((g_color > 255) ? 255 : g_color) * 255;
    out[(y + r.y) * elemCols + (x + r.x) * bpp + BLUE] = (uint16_t)((b_color > 255) ? 255 : b_color) * 255;

    /* Green on a red line */
    r_color = (INTERPOLATE_H(in, x + gr.x, y + gr.y, imgw) >> SHIFT_RESULT);
    g_color = (PIX(in, x + gr.x, y + gr.y, imgw) >> SHIFT_RESULT);
    b_color = (INTERPOLATE_V(in, x + gr.x, y + gr.y, imgw) >> SHIFT_RESULT);
    //r_color = SATURATION_RED(r_color, g_color, b_color);
    //g_color = SATURATION_GREEN(r_color, g_color, b_color);
    //b_color = SATURATION_BLUE(r_color, g_color, b_color);
    r_color = ((r_color >> 1) < OFFSET_RED) ? 0 : (((r_color >> 1) - OFFSET_RED) * GAIN_RED);
    g_color = ((g_color >> 1) < OFFSET_GREEN) ? 0 : (((g_color >> 1) - OFFSET_GREEN) * GAIN_GREEN);
    b_color = ((b_color >> 1) < OFFSET_BLUE) ? 0 : (((b_color >> 1) - OFFSET_BLUE) * GAIN_BLUE);
    out[(y + gr.y) * elemCols + (x + gr.x) * bpp + RED] = (uint16_t)((r_color > 255) ? 255 : r_color) * 255;
    out[(y + gr.y) * elemCols + (x + gr.x) * bpp + GREEN] = (uint16_t)((g_color > 255) ? 255 : g_color) * 255;
    out[(y + gr.y) * elemCols + (x + gr.x) * bpp + BLUE] = (uint16_t)((b_color > 255) ? 255 : b_color) * 255;

    /* Green on a blue line */
    r_color = (INTERPOLATE_V(in, x + gb.x, y + gb.y, imgw) >> SHIFT_RESULT);
    g_color = (PIX(in, x + gb.x, y + gb.y, imgw) >> SHIFT_RESULT);
    b_color = (INTERPOLATE_H(in, x + gb.x, y + gb.y, imgw) >> SHIFT_RESULT);
    //r_color = SATURATION_RED(r_color, g_color, b_color);
    //g_color = SATURATION_GREEN(r_color, g_color, b_color);
    //b_color = SATURATION_BLUE(r_color, g_color, b_color);
    r_color = ((r_color >> 1) < OFFSET_RED) ? 0 : (((r_color >> 1) - OFFSET_RED) * GAIN_RED);
    g_color = ((g_color >> 1) < OFFSET_GREEN) ? 0 : (((g_color >> 1) - OFFSET_GREEN) * GAIN_GREEN);
    b_color = ((b_color >> 1) < OFFSET_BLUE) ? 0 : (((b_color >> 1) - OFFSET_BLUE) * GAIN_BLUE);
    out[(y + gb.y) * elemCols + (x + gb.x) * bpp + RED] = (uint16_t)((r_color > 255) ? 255 : r_color) * 255;
    out[(y + gb.y) * elemCols + (x + gb.x) * bpp + GREEN] = (uint16_t)((g_color > 255) ? 255 : g_color) * 255;
    out[(y + gb.y) * elemCols + (x + gb.x) * bpp + BLUE] = (uint16_t)((b_color > 255) ? 255 : b_color) * 255;

    /* Blue */
    r_color = (INTERPOLATE_X(in, x + b.x, y + b.y, imgw) >> SHIFT_RESULT);
    g_color = (INTERPOLATE_HV(in, x + b.x, y + b.y, imgw) >> SHIFT_RESULT);
    b_color = (PIX(in, x + b.x, y + b.y, imgw) >> SHIFT_RESULT);
    //r_color = SATURATION_RED(r_color, g_color, b_color)	;
    //g_color = SATURATION_GREEN(r_color, g_color, b_color);
    //b_color = SATURATION_BLUE(r_color, g_color, b_color);
    r_color = ((r_color >> 1) < OFFSET_RED) ? 0 : (((r_color >> 1) - OFFSET_RED) * GAIN_RED);
    g_color = ((g_color >> 1) < OFFSET_GREEN) ? 0 : (((g_color >> 1) - OFFSET_GREEN) * GAIN_GREEN);
    b_color = ((b_color >> 1) < OFFSET_BLUE) ? 0 : (((b_color >> 1) - OFFSET_BLUE) * GAIN_BLUE);
    out[(y + b.y) * elemCols + (x + b.x) * bpp + RED] = (uint16_t)((r_color > 255) ? 255 : r_color) * 255;
    out[(y + b.y) * elemCols + (x + b.x) * bpp + GREEN] = (uint16_t)((g_color > 255) ? 255 : g_color) * 255;
    out[(y + b.y) * elemCols + (x + b.x) * bpp + BLUE] = (uint16_t)((b_color > 255) ? 255 : b_color) * 255;

    if (bpp == 4)
    {
      out[y * elemCols + x * bpp + 3] = 0xFF;
      out[y * elemCols + (x + 1) * bpp + 3] = 0xFF;
      out[(y + 1) * elemCols + x * bpp + 3] = 0xFF;
      out[(y + 1) * elemCols + (x + 1) * bpp + 3] = 0xFF;
    }
  }
}

__global__ void ScaleDownKernel(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch)
{
  __shared__ float inrow[SCALEDOWN_W + 4];
  __shared__ float brow[5 * (SCALEDOWN_W / 2)];
  __shared__ int yRead[SCALEDOWN_H + 4];
  __shared__ int yWrite[SCALEDOWN_H + 4];
#define dx2 (SCALEDOWN_W / 2)
  const int tx = threadIdx.x;
  const int tx0 = tx + 0 * dx2;
  const int tx1 = tx + 1 * dx2;
  const int tx2 = tx + 2 * dx2;
  const int tx3 = tx + 3 * dx2;
  const int tx4 = tx + 4 * dx2;
  const int xStart = blockIdx.x * SCALEDOWN_W;
  const int yStart = blockIdx.y * SCALEDOWN_H;
  const int xWrite = xStart / 2 + tx;
  float k0 = d_ScaleDownKernel[0];
  float k1 = d_ScaleDownKernel[1];
  float k2 = d_ScaleDownKernel[2];
  if (tx < SCALEDOWN_H + 4)
  {
    int y = yStart + tx - 2;
    y = (y < 0 ? 0 : y);
    y = (y >= height ? height - 1 : y);
    yRead[tx] = y * pitch;
    yWrite[tx] = (yStart + tx - 4) / 2 * newpitch;
  }
  __syncthreads();
  int xRead = xStart + tx - 2;
  xRead = (xRead < 0 ? 0 : xRead);
  xRead = (xRead >= width ? width - 1 : xRead);

  int maxtx = min(dx2, width / 2 - xStart / 2);
  for (int dy = 0; dy < SCALEDOWN_H + 4; dy += 5)
  {
    {
      inrow[tx] = d_Data[yRead[dy + 0] + xRead];
      __syncthreads();
      if (tx < maxtx)
      {
        brow[tx4] = k0 * (inrow[2 * tx] + inrow[2 * tx + 4]) + k1 * (inrow[2 * tx + 1] + inrow[2 * tx + 3]) + k2 * inrow[2 * tx + 2];
        if (dy >= 4 && !(dy & 1))
          d_Result[yWrite[dy + 0] + xWrite] = k2 * brow[tx2] + k0 * (brow[tx0] + brow[tx4]) + k1 * (brow[tx1] + brow[tx3]);
      }
      __syncthreads();
    }
    if (dy < (SCALEDOWN_H + 3))
    {
      inrow[tx] = d_Data[yRead[dy + 1] + xRead];
      __syncthreads();
      if (tx < maxtx)
      {
        brow[tx0] = k0 * (inrow[2 * tx] + inrow[2 * tx + 4]) + k1 * (inrow[2 * tx + 1] + inrow[2 * tx + 3]) + k2 * inrow[2 * tx + 2];
        if (dy >= 3 && (dy & 1))
          d_Result[yWrite[dy + 1] + xWrite] = k2 * brow[tx3] + k0 * (brow[tx1] + brow[tx0]) + k1 * (brow[tx2] + brow[tx4]);
      }
      __syncthreads();
    }
    if (dy < (SCALEDOWN_H + 2))
    {
      inrow[tx] = d_Data[yRead[dy + 2] + xRead];
      __syncthreads();
      if (tx < maxtx)
      {
        brow[tx1] = k0 * (inrow[2 * tx] + inrow[2 * tx + 4]) + k1 * (inrow[2 * tx + 1] + inrow[2 * tx + 3]) + k2 * inrow[2 * tx + 2];
        if (dy >= 2 && !(dy & 1))
          d_Result[yWrite[dy + 2] + xWrite] = k2 * brow[tx4] + k0 * (brow[tx2] + brow[tx1]) + k1 * (brow[tx3] + brow[tx0]);
      }
      __syncthreads();
    }
    if (dy < (SCALEDOWN_H + 1))
    {
      inrow[tx] = d_Data[yRead[dy + 3] + xRead];
      __syncthreads();
      if (tx < maxtx)
      {
        brow[tx2] = k0 * (inrow[2 * tx] + inrow[2 * tx + 4]) + k1 * (inrow[2 * tx + 1] + inrow[2 * tx + 3]) + k2 * inrow[2 * tx + 2];
        if (dy >= 1 && (dy & 1))
          d_Result[yWrite[dy + 3] + xWrite] = k2 * brow[tx0] + k0 * (brow[tx3] + brow[tx2]) + k1 * (brow[tx4] + brow[tx1]);
      }
      __syncthreads();
    }
    if (dy < SCALEDOWN_H)
    {
      inrow[tx] = d_Data[yRead[dy + 4] + xRead];
      __syncthreads();
      if (tx < dx2 && xWrite < width / 2)
      {
        brow[tx3] = k0 * (inrow[2 * tx] + inrow[2 * tx + 4]) + k1 * (inrow[2 * tx + 1] + inrow[2 * tx + 3]) + k2 * inrow[2 * tx + 2];
        if (!(dy & 1))
          d_Result[yWrite[dy + 4] + xWrite] = k2 * brow[tx1] + k0 * (brow[tx4] + brow[tx3]) + k1 * (brow[tx0] + brow[tx2]);
      }
      __syncthreads();
    }
  }
}

__global__ void ScaleUpKernel(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch)
{
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  int x = blockIdx.x * SCALEUP_W + 2 * tx;
  int y = blockIdx.y * SCALEUP_H + 2 * ty;
  if (x < 2 * width && y < 2 * height)
  {
    int xl = blockIdx.x * (SCALEUP_W / 2) + tx;
    int yu = blockIdx.y * (SCALEUP_H / 2) + ty;
    int xr = min(xl + 1, width - 1);
    int yd = min(yu + 1, height - 1);
    float vul = d_Data[yu * pitch + xl];
    float vur = d_Data[yu * pitch + xr];
    float vdl = d_Data[yd * pitch + xl];
    float vdr = d_Data[yd * pitch + xr];
    d_Result[(y + 0) * newpitch + x + 0] = vul;
    d_Result[(y + 0) * newpitch + x + 1] = 0.50f * (vul + vur);
    d_Result[(y + 1) * newpitch + x + 0] = 0.50f * (vul + vdl);
    d_Result[(y + 1) * newpitch + x + 1] = 0.25f * (vul + vur + vdl + vdr);
  }
}

/* TODO: Adjust to correct YUYV alignment*/
__global__ void ScaleUpKernelCharYUYV(float *d_Result, unsigned char *d_Data, int width, int pitch, int height, int newpitch)
{
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  int x = blockIdx.x * SCALEUP_W + 2 * tx;
  int y = blockIdx.y * SCALEUP_H + 2 * ty;
  if (x < 2 * width && y < 2 * height)
  {
    int xl = blockIdx.x * (SCALEUP_W / 2) + tx;
    int yu = blockIdx.y * (SCALEUP_H / 2) + ty;
    int xr = min(xl + 1, width - 1);
    int yd = min(yu + 1, height - 1);
    float vul = d_Data[yu * pitch * 2 + xl * 2 + 1];
    float vur = d_Data[yu * pitch * 2 + xr * 2 + 1];
    float vdl = d_Data[yd * pitch * 2 + xl * 2 + 1];
    float vdr = d_Data[yd * pitch * 2 + xr * 2 + 1];
    d_Result[(y + 0) * newpitch + x + 0] = vul;
    d_Result[(y + 0) * newpitch + x + 1] = 0.50f * (vul + vur);
    d_Result[(y + 1) * newpitch + x + 0] = 0.50f * (vul + vdl);
    d_Result[(y + 1) * newpitch + x + 1] = 0.25f * (vul + vur + vdl + vdr);
  }
}

__global__ void LowPassBlock(float *d_Image, float *d_Result, int width, int pitch, int height)
{
  __shared__ float xrows[16][32];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int xp = blockIdx.x * LOWPASS_W + tx;
  const int yp = blockIdx.y * LOWPASS_H + ty;
  const int N = 16;
  float *k = d_LowPassKernel;
  int xl = max(min(xp - 4, width - 1), 0);
#pragma unroll
  for (int l = -8; l < 4; l += 4)
  {
    int ly = l + ty;
    int yl = max(min(yp + l + 4, height - 1), 0);
    float val = d_Image[yl * pitch + xl];
    val = k[4] * ShiftDown(val, 4) +
          k[3] * (ShiftDown(val, 5) + ShiftDown(val, 3)) +
          k[2] * (ShiftDown(val, 6) + ShiftDown(val, 2)) +
          k[1] * (ShiftDown(val, 7) + ShiftDown(val, 1)) +
          k[0] * (ShiftDown(val, 8) + val);
    xrows[ly + 8][tx] = val;
  }
  __syncthreads();
#pragma unroll
  for (int l = 4; l < LOWPASS_H; l += 4)
  {
    int ly = l + ty;
    int yl = min(yp + l + 4, height - 1);
    float val = d_Image[yl * pitch + xl];
    val = k[4] * ShiftDown(val, 4) +
          k[3] * (ShiftDown(val, 5) + ShiftDown(val, 3)) +
          k[2] * (ShiftDown(val, 6) + ShiftDown(val, 2)) +
          k[1] * (ShiftDown(val, 7) + ShiftDown(val, 1)) +
          k[0] * (ShiftDown(val, 8) + val);
    xrows[(ly + 8) % N][tx] = val;
    int ys = yp + l - 4;
    if (xp < width && ys < height && tx < LOWPASS_W)
      d_Result[ys * pitch + xp] = k[4] * xrows[(ly + 0) % N][tx] +
                                  k[3] * (xrows[(ly - 1) % N][tx] + xrows[(ly + 1) % N][tx]) +
                                  k[2] * (xrows[(ly - 2) % N][tx] + xrows[(ly + 2) % N][tx]) +
                                  k[1] * (xrows[(ly - 3) % N][tx] + xrows[(ly + 3) % N][tx]) +
                                  k[0] * (xrows[(ly - 4) % N][tx] + xrows[(ly + 4) % N][tx]);
    __syncthreads();
  }
  int ly = LOWPASS_H + ty;
  int ys = yp + LOWPASS_H - 4;
  if (xp < width && ys < height && tx < LOWPASS_W)
    d_Result[ys * pitch + xp] = k[4] * xrows[(ly + 0) % N][tx] +
                                k[3] * (xrows[(ly - 1) % N][tx] + xrows[(ly + 1) % N][tx]) +
                                k[2] * (xrows[(ly - 2) % N][tx] + xrows[(ly + 2) % N][tx]) +
                                k[1] * (xrows[(ly - 3) % N][tx] + xrows[(ly + 3) % N][tx]) +
                                k[0] * (xrows[(ly - 4) % N][tx] + xrows[(ly + 4) % N][tx]);
}

__global__ void LowPassBlockCharYUYV(unsigned char *d_Image, float *d_Result, int width, int pitch, int height)
{
  __shared__ float xrows[16][32];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int xp = blockIdx.x * LOWPASS_W + tx;
  const int yp = blockIdx.y * LOWPASS_H + ty;
  const int N = 16;
  float *k = d_LowPassKernel;
  int xl = max(min(xp - 4, width - 1), 0);
#pragma unroll
  for (int l = -8; l < 4; l += 4)
  {
    int ly = l + ty;
    int yl = max(min(yp + l + 4, height - 1), 0);
    float val = __uint2float_rn(d_Image[yl * pitch * 2 + xl * 2 + 1]);
    val = k[4] * ShiftDown(val, 4) +
          k[3] * (ShiftDown(val, 5) + ShiftDown(val, 3)) +
          k[2] * (ShiftDown(val, 6) + ShiftDown(val, 2)) +
          k[1] * (ShiftDown(val, 7) + ShiftDown(val, 1)) +
          k[0] * (ShiftDown(val, 8) + val);
    xrows[ly + 8][tx] = val;
  }
  __syncthreads();
#pragma unroll
  for (int l = 4; l < LOWPASS_H; l += 4)
  {
    int ly = l + ty;
    int yl = min(yp + l + 4, height - 1);
    float val = __uint2float_rn(d_Image[yl * pitch * 2 + xl * 2 + 1]);
    val = k[4] * ShiftDown(val, 4) +
          k[3] * (ShiftDown(val, 5) + ShiftDown(val, 3)) +
          k[2] * (ShiftDown(val, 6) + ShiftDown(val, 2)) +
          k[1] * (ShiftDown(val, 7) + ShiftDown(val, 1)) +
          k[0] * (ShiftDown(val, 8) + val);
    xrows[(ly + 8) % N][tx] = val;
    int ys = yp + l - 4;
    if (xp < width && ys < height && tx < LOWPASS_W)
      d_Result[ys * pitch + xp] = k[4] * xrows[(ly + 0) % N][tx] +
                                  k[3] * (xrows[(ly - 1) % N][tx] + xrows[(ly + 1) % N][tx]) +
                                  k[2] * (xrows[(ly - 2) % N][tx] + xrows[(ly + 2) % N][tx]) +
                                  k[1] * (xrows[(ly - 3) % N][tx] + xrows[(ly + 3) % N][tx]) +
                                  k[0] * (xrows[(ly - 4) % N][tx] + xrows[(ly + 4) % N][tx]);
    __syncthreads();
  }
  int ly = LOWPASS_H + ty;
  int ys = yp + LOWPASS_H - 4;
  if (xp < width && ys < height && tx < LOWPASS_W)
    d_Result[ys * pitch + xp] = k[4] * xrows[(ly + 0) % N][tx] +
                                k[3] * (xrows[(ly - 1) % N][tx] + xrows[(ly + 1) % N][tx]) +
                                k[2] * (xrows[(ly - 2) % N][tx] + xrows[(ly + 2) % N][tx]) +
                                k[1] * (xrows[(ly - 3) % N][tx] + xrows[(ly + 3) % N][tx]) +
                                k[0] * (xrows[(ly - 4) % N][tx] + xrows[(ly + 4) % N][tx]);
}

__global__ void sinewave_gen_kernel(Computation::Vertex2 *vertices, unsigned int width, unsigned int height, float time)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  // calculate uv coordinates
  float u = x / (float)width;
  float v = x / (float)height;
  u = u * 2.0f - 1.0f;
  v = v * 2.0f - 1.0f;

  if (y * width + x < 1920)
  {
    // write output vertex
    vertices[y * width + x].pos[0] = u;
    vertices[y * width + x].pos[1] = -1.0f;
    vertices[y * width + x].pos[2] = 0.0f;
    vertices[y * width + x].pos[3] = 1.0f;
    vertices[y * width + x].color[0] = 1.0;
    vertices[y * width + x].color[1] = 1.0;
    vertices[y * width + x].color[2] = 1.0f;
  }
  else if (y * width + x < 3840)
  {
    // write output vertex
    vertices[y * width + x].pos[0] = u;
    vertices[y * width + x].pos[1] = 1.0f;
    vertices[y * width + x].pos[2] = 0.0;
    vertices[y * width + x].pos[3] = 1.0f;
    vertices[y * width + x].color[0] = 1.0;
    vertices[y * width + x].color[1] = 1.0;
    vertices[y * width + x].color[2] = 1.0f;
  }
  else if (y * width + x < 4920)
  {
    // write output vertex
    vertices[y * width + x].pos[0] = -1.0;
    vertices[y * width + x].pos[1] = v;
    vertices[y * width + x].pos[2] = 0.0;
    vertices[y * width + x].pos[3] = 1.0f;
    vertices[y * width + x].color[0] = 1.0;
    vertices[y * width + x].color[1] = 1.0;
    vertices[y * width + x].color[2] = 1.0f;
  }
  else if (y * width + x < 6000)
  {
    // write output vertex
    if (y * width + x < 5760)
    {
      v = v - 2.0;
    }
    else
    {
      v = v + 1.557407;
    }
    vertices[y * width + x].pos[0] = 1.0;
    vertices[y * width + x].pos[1] = v;
    vertices[y * width + x].pos[2] = 0.0;
    vertices[y * width + x].pos[3] = 1.0f;
    vertices[y * width + x].color[0] = 1.0;
    vertices[y * width + x].color[1] = 1.0;
    vertices[y * width + x].color[2] = 1.0f;
  }
}

__global__ void init_index_kernel(int *indexbuffer)
{

  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; // 0 <= x <= 159

  if (x > GRIDSIZEX - 2)
    return;
  for (unsigned int y = 0; y < (GRIDSIZEY - 1); y++)
  {
    if ((x == 1918) && (y == 1078))
      printf("x: %d, y: %d, Buffer: %d: Upper Triangle: %d %d %d; lower triangle: %d %d %d\n", x, y, (y * (GRIDSIZEX - 1) + x) * 6, y * GRIDSIZEX + x, y * GRIDSIZEX + x + 1, (y + 1) * GRIDSIZEX + x + 1, (y + 1) * GRIDSIZEX + x + 1, (y + 1) * GRIDSIZEX + x, y * GRIDSIZEX + x);
    indexbuffer[(y * (GRIDSIZEX - 1) + x) * 6 + 0] = y * GRIDSIZEX + x;
    indexbuffer[(y * (GRIDSIZEX - 1) + x) * 6 + 1] = y * GRIDSIZEX + x + 1;
    indexbuffer[(y * (GRIDSIZEX - 1) + x) * 6 + 2] = (y + 1) * GRIDSIZEX + x + 1;
    indexbuffer[(y * (GRIDSIZEX - 1) + x) * 6 + 3] = (y + 1) * GRIDSIZEX + x + 1;
    indexbuffer[(y * (GRIDSIZEX - 1) + x) * 6 + 4] = (y + 1) * GRIDSIZEX + x;
    indexbuffer[(y * (GRIDSIZEX - 1) + x) * 6 + 5] = y * GRIDSIZEX + x;
  }
}

__global__ void init_grid_kernel(Computation::Vertex2 *vertices)
{

  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; // 0 <= x <= 159
  if (x > GRIDSIZEX - 1)
    return;

  float u = (float)x / (float)(GRIDSIZEX - 1);
  float v = 0;
  u = u * 2.0f - 1.0f;

  for (unsigned int y = 0; y < GRIDSIZEY; y++)
  {
    v = y / (float)(GRIDSIZEY - 1);
    v = v * 2.0f - 1.0f;
    //printf("x=%d, u=%f, y=%d, v=%f, pos_in_vector: %d\n", x, u, y, v, y*GRIDSIZEX+x);
    vertices[y * GRIDSIZEX + x].pos[0] = u;
    vertices[y * GRIDSIZEX + x].pos[1] = v;
    vertices[y * GRIDSIZEX + x].pos[2] = 0.0;
    vertices[y * GRIDSIZEX + x].pos[3] = 1.0f;
    vertices[y * GRIDSIZEX + x].color[0] = 1.0;
    vertices[y * GRIDSIZEX + x].color[1] = 1.0;
    vertices[y * GRIDSIZEX + x].color[2] = 1.0f;
    vertices[y * GRIDSIZEX + x].texcoords[0] = u;
    vertices[y * GRIDSIZEX + x].texcoords[0] = v;
  }
}

__global__ void clear_heightmap_kernel(float *d_Result, bool *d_Pixelflags, unsigned int width, unsigned int height)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= width)
    return;

  for (int i = 0; i < height; ++i)
  {
    d_Pixelflags[i * width + idx] = false;
    d_Result[i * width + idx] = 0.0f;
  }
}

__global__ void draw_heightmap_kernel(float *d_Result, bool *d_Pixelflags, SiftPoint *d_Sift, unsigned int width, unsigned int height, int numPoints, int maxPoints)
{
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Sift features
  if (idx >= maxPoints)
    return;

  if ((idx <= numPoints) && (d_Sift[idx].score > 0.0f))
  {
    //if (20.0>(d_Sift[idx].ypos-d_Sift[idx].match_ypos)*(d_Sift[idx].ypos-d_Sift[idx].match_ypos)) return;
    //if (d_Sift[idx].xpos<d_Sift[idx].match_xpos)  return;
    int x = d_Sift[idx].xpos;
    int y = d_Sift[idx].ypos;
    d_Pixelflags[x + width * y] = true;
    d_Result[x + width * y] = d_Sift[idx].match_error;
  }
  else
  {
    return;
  }
}

__global__ void interpolate_heightmap_kernel(float *d_HeightMap, bool *d_Pixelflags, float *d_pixHeight, unsigned int width, unsigned int height)
{

  unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x; // 0 <= x <= 159
  unsigned int idx_y = blockIdx.y * blockDim.y + threadIdx.y; // 0 <= x <= 159
  if (idx_x > width - 1)
    return;
  if (idx_y > height - 1)
    return;

  int x = idx_x;
  int y = idx_y;

  int weights = 1;
  float heights = 0.0f;

  //float max_height = 0.0f;
  for (int i = -30; i < 30; i++)
  {
    x = idx_x + i;
    if (x < 0)
      x = 0;
    if (x > width - 1)
      x = width - 1;
    for (int j = -30; j < 30; j++)
    {
      y = idx_y + j;
      if (y < 0)
        y = 0;
      if (y > height - 1)
        y = height - 1;
      if (d_Pixelflags[y * width + x])
      {
        heights += d_HeightMap[y * width + x];
        if ((i == 0) && (j == 0))
        {
          weights += 1;
        }
        else if ((i > -2) && (i < 2) && (j > -2) && (j < 2))
        {
          weights += 1;
        }
        else if ((i > -4) && (i < 4) && (j > -4) && (j < 4))
        {
          weights += 1;
        }
        else
        {
          weights += 1;
        }
      }
    }
  }
  d_pixHeight[(idx_y * width + idx_x)] = (heights) / (float)weights;
}

__global__ void GaussKernelBlock(float *res, float *src, int width, int height, int kernelSize)
{
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x; // image-pixel
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx_x < (kernelSize - 1) / 2)
    return;
  if (idx_y < (kernelSize - 1) / 2)
    return;
  if (idx_x > width - 1 - (kernelSize - 1) / 2)
    return;
  if (idx_y > height - 1 - (kernelSize - 1) / 2)
    return;

  float result = 0.0f;
  int x, y; // kernel-pixel
  for (int i = -((kernelSize - 1) / 2); i <= (kernelSize - 1) / 2; i++)
  {
    x = i + (kernelSize - 1) / 2;
    for (int j = -((kernelSize - 1) / 2); j <= (kernelSize - 1) / 2; j++)
    {
      y = j + (kernelSize - 1) / 2;
      result += (src[idx_x + i + (idx_y + j) * width] * d_GaussKernel[y * kernelSize + x]);
    }
  }
  // printf("result written on x = %d, y = %d", idx_x, idx_y);
  res[idx_x + (idx_y)*width] = result;
}

__global__ void visualize_features_kernel(Computation::Vertex2 *vertices, float *d_Heights, float *d_RGB, unsigned int width, unsigned int height)
{
  unsigned int idx_x = blockIdx.x * blockDim.x + threadIdx.x; // 0 <= x <= 159
  unsigned int idx_y = blockIdx.y * blockDim.y + threadIdx.y; // 0 <= x <= 159
  if (idx_x > GRIDSIZEX - 1)
    return;
  if (idx_y > GRIDSIZEY - 1)
    return;

  float r = d_RGB[idx_y * GRIDSIZEX * 3 + 3 * idx_x] / 255;
  float g = d_RGB[idx_y * GRIDSIZEX * 3 + 3 * idx_x + 1] / 255;
  float b = d_RGB[idx_y * GRIDSIZEX * 3 + 3 * idx_x + 2] / 255;
  float h = d_Heights[idx_y * GRIDSIZEX + idx_x];

  vertices[idx_y * GRIDSIZEX + (1919 - idx_x)].pos[2] = h / 50;
  vertices[idx_y * GRIDSIZEX + (1919 - idx_x)].color[0] = r;
  vertices[idx_y * GRIDSIZEX + (1919 - idx_x)].color[1] = g;
  vertices[idx_y * GRIDSIZEX + (1919 - idx_x)].color[2] = b;
}

__global__ void gpuConvertYUYVtoRGBfloat_kernel(unsigned char *src, float *dst,
                                                unsigned int width, unsigned int height)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx * 2 >= width)
  {
    return;
  }

  for (int i = 0; i < height; ++i)
  {
    int cb = src[i * width * 2 + idx * 4 + 0];
    int y0 = src[i * width * 2 + idx * 4 + 1];
    int cr = src[i * width * 2 + idx * 4 + 2];
    int y1 = src[i * width * 2 + idx * 4 + 3];

    dst[i * width * 3 + idx * 6 + 0] = clamp(1.164f * (y0 - 16) + 1.596f * (cr - 128), 0.0f, 255.0f);
    dst[i * width * 3 + idx * 6 + 1] = clamp(1.164f * (y0 - 16) - 0.813f * (cr - 128) - 0.391f * (cb - 128), 0.0f, 255.0f);
    dst[i * width * 3 + idx * 6 + 2] = clamp(1.164f * (y0 - 16) + 2.018f * (cb - 128), 0.0f, 255.0f);

    dst[i * width * 3 + idx * 6 + 3] = clamp(1.164f * (y1 - 16) + 1.596f * (cr - 128), 0.0f, 255.0f);
    dst[i * width * 3 + idx * 6 + 4] = clamp(1.164f * (y1 - 16) - 0.813f * (cr - 128) - 0.391f * (cb - 128), 0.0f, 255.0f);
    dst[i * width * 3 + idx * 6 + 5] = clamp(1.164f * (y1 - 16) + 2.018f * (cb - 128), 0.0f, 255.0f);
  }
}

typedef struct
{
  float x_3d;
  float y_3d;
  float z_3d;
  float match_x_3d;
  float match_y_3d;
  float match_z_3d;
  float xpos;
  float ypos;
  float distance;
  float match_xpos;
  float match_ypos;
  float match_distance;
} TempCoords;

__device__ __inline__ float ld_gbl_cg(const float *addr)
{
  float return_value;
  asm("ld.global.cg.f32 %0, [%1];"
      : "=f"(return_value)
      : "l"(addr));
  return return_value;
}

#define MIN_THRESH_RANSAC 64

__global__ void gpuFindRotationTranslation_step0(SiftPoint *point, float *tempMemory, bool *index_list, mat4x4 *rotation, vec4 *translation, int numPts)
{
  if (numPts == 0)
    return;
  int idx0 = blockIdx.x * blockDim.x + threadIdx.x;
  int idx1 = numPts + 1;
  int idx2 = numPts + 1;
  TempCoords centroids;
  TempCoords corr0;
  TempCoords corr1;
  TempCoords corr2;
  int num_inliners;
  float inliners_metric;
  mat4x4 rot;
  mat4x4 rot_t;
  vec4 t;

  centroids.x_3d = 0.0f;
  centroids.y_3d = 0.0f;
  centroids.z_3d = 0.0f;
  centroids.match_x_3d = 0.0f;
  centroids.match_y_3d = 0.0f;
  centroids.match_z_3d = 0.0f;

  num_inliners = -1.0;
  inliners_metric = -1.0f;
  tempMemory[6 + idx0 * 15 + 12] = -1.0;
  tempMemory[6 + idx0 * 15 + 13] = -1.0;
  if ((idx0 < numPts) && (point[idx0].score > MIN_SCORE))
  {
    point[idx0].draw = true;
    // Random triples of point matches
    curandState_t state;
    curand_init(idx0, /* the seed controls the sequence of random values that are produced */
                0,    /* the sequence number is only important with multiple cores */
                idx0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &state);
    int rng;

    while (idx1 > numPts)
    {
      rng = curand(&state) % numPts;
      if ((point[rng].score > MIN_SCORE) && (rng != idx0))
      {
        idx1 = rng;
      }
    }
    while (idx2 > numPts)
    {
      rng = curand(&state) % numPts;
      if ((point[rng].score > MIN_SCORE) && (rng != idx0) && (rng != idx1))
      {
        idx2 = rng;
      }
    }
    if ((idx2 <= numPts) && (idx1 <= numPts))
    {
      // calculate Centroids of all the points, center the 3-point "clouds" around the source (0,0,0)
      centroids.x_3d = (point[idx0].x_3d + point[idx1].x_3d + point[idx2].x_3d) / 3.0;
      centroids.y_3d = (point[idx0].y_3d + point[idx1].y_3d + point[idx2].y_3d) / 3.0;
      centroids.z_3d = (point[idx0].z_3d + point[idx1].z_3d + point[idx2].z_3d) / 3.0;
      centroids.match_x_3d = (point[idx0].match_x_3d + point[idx1].match_x_3d + point[idx2].match_x_3d) / 3.0;
      centroids.match_y_3d = (point[idx0].match_y_3d + point[idx1].match_y_3d + point[idx2].match_y_3d) / 3.0;
      centroids.match_z_3d = (point[idx0].match_z_3d + point[idx1].match_z_3d + point[idx2].match_z_3d) / 3.0;
      corr0.x_3d = point[idx0].x_3d - centroids.x_3d;
      corr0.y_3d = point[idx0].y_3d - centroids.y_3d;
      corr0.z_3d = point[idx0].z_3d - centroids.z_3d;
      corr0.match_x_3d = point[idx0].match_x_3d - centroids.match_x_3d;
      corr0.match_y_3d = point[idx0].match_y_3d - centroids.match_y_3d;
      corr0.match_z_3d = point[idx0].match_z_3d - centroids.match_z_3d;
      corr1.x_3d = point[idx1].x_3d - centroids.x_3d;
      corr1.y_3d = point[idx1].y_3d - centroids.y_3d;
      corr1.z_3d = point[idx1].z_3d - centroids.z_3d;
      corr1.match_x_3d = point[idx1].match_x_3d - centroids.match_x_3d;
      corr1.match_y_3d = point[idx1].match_y_3d - centroids.match_y_3d;
      corr1.match_z_3d = point[idx1].match_z_3d - centroids.match_z_3d;
      corr2.x_3d = point[idx2].x_3d - centroids.x_3d;
      corr2.y_3d = point[idx2].y_3d - centroids.y_3d;
      corr2.z_3d = point[idx2].z_3d - centroids.z_3d;
      corr2.match_x_3d = point[idx2].match_x_3d - centroids.match_x_3d;
      corr2.match_y_3d = point[idx2].match_y_3d - centroids.match_y_3d;
      corr2.match_z_3d = point[idx2].match_z_3d - centroids.match_z_3d;

      mat4x4 pos_new = {{corr0.x_3d, corr0.y_3d, corr0.z_3d, 0},
                        {corr1.x_3d, corr1.y_3d, corr1.z_3d, 0},
                        {corr2.x_3d, corr2.y_3d, corr2.z_3d, 0},
                        {0, 0, 0, 1}};
      mat4x4 pos_old = {{corr0.match_x_3d, corr0.match_y_3d, corr0.match_z_3d, 0},
                        {corr1.match_x_3d, corr1.match_y_3d, corr1.match_z_3d, 0},
                        {corr2.match_x_3d, corr2.match_y_3d, corr2.match_z_3d, 0},
                        {0, 0, 0, 1}};
      mat4x4 pos_old_trans;
      // Transpose
      int i, j;
      for (j = 0; j < 4; ++j)
        for (i = 0; i < 4; ++i)
          pos_old_trans[i][j] = pos_old[j][i];
      mat4x4 h; // covariance matrix
      // Matrix Multiplication
      int k, r, c;
      for (c = 0; c < 4; ++c)
        for (r = 0; r < 4; ++r)
        {
          h[c][r] = 0.f;
          for (k = 0; k < 4; ++k)
            h[c][r] += pos_new[k][r] * pos_old_trans[c][k];
        }
      mat4x4 u;
      mat4x4 s;
      mat4x4 s_det;
      mat4x4 v;
      for (i = 0; i < 4; ++i)
        for (j = 0; j < 4; ++j)
        {
          u[i][j] = i == j ? 1.f : 0.f;
          s[i][j] = i == j ? 1.f : 0.f;
          s_det[i][j] = i == j ? 1.f : 0.f;
          v[i][j] = i == j ? 1.f : 0.f;
        }

      svd(h[0][0], h[1][0], h[2][0],
          h[0][1], h[1][1], h[2][1],
          h[0][2], h[1][2], h[2][2],
          u[0][0], u[1][0], u[2][0],
          u[0][1], u[1][1], u[2][1],
          u[0][2], u[1][2], u[2][2],
          s[0][0], s[1][1], s[2][2],
          v[0][0], v[1][0], v[2][0],
          v[0][1], v[1][1], v[2][1],
          v[0][2], v[1][2], v[2][2]);
      mat4x4 u_t;
      // if (idx0 == 0)
      // printf("IDX0: %d, IDX1: %d, IDX2: %d, S[0][0] = %f, S[1][1] = %f, S[2][2] = %f\n",idx0,idx1, idx2, s[0][0], s[1][1], s[2][2]);
      // if (idx0 == 0)
      // printf("Point Pairs 0: %f->%f, %f->%f, %f->%f centriod: (%f, %f, %f) -> (%f, %f, %f) \n",
      // corr0.x_3d, corr0.match_x_3d,corr0.y_3d, corr0.match_y_3d,corr0.z_3d, corr0.match_z_3d,
      // centroids.x_3d,centroids.y_3d,centroids.z_3d,centroids.match_x_3d,centroids.match_y_3d,centroids.match_z_3d);
      // if (idx0 == 0)
      // printf("Point Pairs %d: %f->%f, %f->%f, %f->%f \n", idx1,
      // corr1.x_3d, corr1.match_x_3d,corr1.y_3d, corr1.match_y_3d,corr1.z_3d, corr1.match_z_3d);
      // if (idx0 == 0)
      // printf("Point Pairs %d: %f->%f, %f->%f, %f->%f \n", idx2,
      // corr2.x_3d, corr2.match_x_3d,corr2.y_3d, corr2.match_y_3d,corr2.z_3d, corr2.match_z_3d);
      for (j = 0; j < 4; ++j)
        for (i = 0; i < 4; ++i)
          u_t[i][j] = u[j][i];
      mat4x4 vu_t;
      for (c = 0; c < 4; ++c)
        for (r = 0; r < 4; ++r)
        {
          vu_t[c][r] = 0.f;
          for (k = 0; k < 4; ++k)
            vu_t[c][r] += v[k][r] * u_t[c][k];
        }

      float det = vu_t[0][0] * vu_t[1][1] * vu_t[2][2] + vu_t[1][0] * vu_t[2][1] * vu_t[0][2] +
                  vu_t[2][0] * vu_t[0][1] * vu_t[1][2] - vu_t[2][0] * vu_t[1][1] * vu_t[0][2] -
                  vu_t[1][0] * vu_t[0][1] * vu_t[2][2] - vu_t[0][0] * vu_t[2][1] * vu_t[1][2];

      s_det[2][2] = det;
      mat4x4 vs_det;
      for (c = 0; c < 4; ++c)
        for (r = 0; r < 4; ++r)
        {
          vs_det[c][r] = 0.f;
          for (k = 0; k < 4; ++k)
            vs_det[c][r] += v[k][r] * s_det[c][k];
        }
      for (c = 0; c < 4; ++c)
        for (r = 0; r < 4; ++r)
        {
          rot_t[c][r] = 0.f;
          for (k = 0; k < 4; ++k)
            rot_t[c][r] += vs_det[k][r] * u_t[c][k];
        }

      for (j = 0; j < 4; ++j)
        for (i = 0; i < 4; ++i)
          rot[i][j] = rot_t[j][i];
      vec4 p_dash = {centroids.match_x_3d, centroids.match_y_3d, centroids.match_z_3d, 1.0};
      vec4 rp_dash;
      for (j = 0; j < 4; ++j)
      {
        rp_dash[j] = 0.f;
        for (i = 0; i < 4; ++i)
          rp_dash[j] += rot[i][j] * p_dash[i];
      }
      t[0] = centroids.x_3d - rp_dash[0];
      t[1] = centroids.y_3d - rp_dash[1];
      t[2] = centroids.z_3d - rp_dash[2];
      t[3] = 1.0;
      mat4x4 check_res;
      for (c = 0; c < 4; ++c)
        for (r = 0; r < 4; ++r)
        {
          check_res[c][r] = 0.f;
          for (k = 0; k < 4; ++k)
            check_res[c][r] += rot[k][r] * pos_old[c][k];
        }

      det = rot[0][0] * rot[1][1] * rot[2][2] + rot[1][0] * rot[2][1] * rot[0][2] +
            rot[2][0] * rot[0][1] * rot[1][2] - rot[2][0] * rot[1][1] * rot[0][2] -
            rot[1][0] * rot[0][1] * rot[2][2] - rot[0][0] * rot[2][1] * rot[1][2];

      if ((det < 1.1f) && (det > 0.9f) && (rot[0][0] > 0.6) && (rot[1][1] > 0.6) && (rot[2][2] > 0.6))
      {
        float ssd_local;
        inliners_metric = 0.0f;
        for (int idx = 0; idx < numPts; idx++)
        {
          index_list[idx0 * 512 + idx] = false;
          if (point[idx].score > MIN_SCORE)
          {
            vec4 temp_vec_new;
            vec4 temp_vec_old;
            temp_vec_old[0] = ld_gbl_cg(&(point[idx].match_x_3d)); // - centroids.match_x_3d;
            temp_vec_old[1] = ld_gbl_cg(&(point[idx].match_y_3d)); // - centroids.match_y_3d;
            temp_vec_old[2] = ld_gbl_cg(&(point[idx].match_z_3d)); // - centroids.match_z_3d;
            temp_vec_old[3] = 1.0;
            temp_vec_new[0] = ld_gbl_cg(&(point[idx].x_3d));       // - centroids.x_3d;
            temp_vec_new[1] = ld_gbl_cg(&(point[idx].y_3d));       // - centroids.y_3d;
            temp_vec_new[2] = ld_gbl_cg(&(point[idx].z_3d));       // - centroids.z_3d;
            temp_vec_new[3] = 1.0;

            vec4 temp_vec_old_rot;
            //printf("i= %d temp vec 2: %f\n",idx,temp_vec_old[2]);
            for (j = 0; j < 4; ++j)
            {
              temp_vec_old_rot[j] = t[j];
              for (i = 0; i < 4; ++i)
                temp_vec_old_rot[j] += rot[i][j] * temp_vec_old[i];
            }

            // if (idx0 <= 5)
            //   printf("idx0: %d, idx %d: : temp_vec: %f->%f=%f, %f->%f=%f, %f->%f=%f\n", idx0, idx, temp_vec_old[0], temp_vec_old_rot[0], temp_vec_new[0], temp_vec_old[1], temp_vec_old_rot[1], temp_vec_new[1], temp_vec_old[2], temp_vec_old_rot[2], temp_vec_new[2]);
            ssd_local = 0.0f;
            ssd_local += (temp_vec_old_rot[0] - temp_vec_new[0]) * (temp_vec_old_rot[0] - temp_vec_new[0]);
            ssd_local += (temp_vec_old_rot[1] - temp_vec_new[1]) * (temp_vec_old_rot[1] - temp_vec_new[1]);
            ssd_local += (temp_vec_old_rot[2] - temp_vec_new[2]) * (temp_vec_old_rot[2] - temp_vec_new[2]);
            if (ssd_local < MIN_THRESH_RANSAC)
            {
                            inliners_metric += ssd_local;
              num_inliners++;
              index_list[idx0 * 512 + idx] = true;
            } 
          }
        }
        if (num_inliners == 0)
          inliners_metric = -1.0;
        else
          inliners_metric = inliners_metric / ((float)num_inliners);
      }
    }
  }
  // if (idx0 == 2)
  // printf("rot\n%f,%f,%f| tx=%f\n%f,%f,%f| tx=%f\n%f,%f,%f| tx=%f\n",
  //      rot[0][0], rot[1][0], rot[2][0], t[0],  // matrix 1st row
  //      rot[0][1], rot[1][1], rot[2][1], t[1],  // matrix 2nd row
  //      rot[0][2], rot[1][2], rot[2][2], t[2]); // matrix 3rd row)
  tempMemory[6 + idx0 * 15 + 0] = rot[0][0];
  tempMemory[6 + idx0 * 15 + 1] = rot[0][1];
  tempMemory[6 + idx0 * 15 + 2] = rot[0][2];
  tempMemory[6 + idx0 * 15 + 3] = rot[1][0];
  tempMemory[6 + idx0 * 15 + 4] = rot[1][1];
  tempMemory[6 + idx0 * 15 + 5] = rot[1][2];
  tempMemory[6 + idx0 * 15 + 6] = rot[2][0];
  tempMemory[6 + idx0 * 15 + 7] = rot[2][1];
  tempMemory[6 + idx0 * 15 + 8] = rot[2][2];
  tempMemory[6 + idx0 * 15 + 9] = t[0];
  tempMemory[6 + idx0 * 15 + 10] = t[1];
  tempMemory[6 + idx0 * 15 + 11] = t[2];
  tempMemory[6 + idx0 * 15 + 12] = (float)num_inliners;
  tempMemory[6 + idx0 * 15 + 13] = inliners_metric;
}

__global__ void gpuFindRotationTranslation_step1(SiftPoint *point, float *tempMemory, bool *index_list, mat4x4 *rotation, vec4 *translation, int numPts)
{
  float tmp_inliners;
  float tmp_metric;
  float max_inliners;
  float min_metric;
  int max_idx;
  int min_idx;
  min_metric = 10000;
  max_inliners = -1.f;
  for (int i = 0; i < numPts; i++)
  {
    tmp_inliners = ld_gbl_cg(&(tempMemory[6 + i * 15 + 12]));
    tmp_metric = ld_gbl_cg(&(tempMemory[6 + i * 15 + 13]));
    if ((tmp_inliners > max_inliners))
    {
      if (tempMemory[6 + i * 15 + 0] > 0.6)
      {
        max_inliners = tmp_inliners;
        max_idx = i;
      }
    }
    if ((tmp_metric < min_metric) && (tmp_inliners > 1.0) && (tmp_metric > 0.0f))
    {
      if (tempMemory[6 + i * 15 + 0] > 0.6)
      {
        min_metric = tmp_metric;
        min_idx = i;
      }
    }
  }
  // printf("======== BEST ONES: max_idx: %d, max_inliners %f, its matrix[0][0] %f, min_idx, %d, min_metric %f,its matrix[0][0]%f\n",
  //       max_idx, max_inliners, tempMemory[6 + max_idx * 15 + 0], min_idx, min_metric, tempMemory[6 + min_idx * 15 + 0]);
  tempMemory[0] = (float)min_idx;
  tempMemory[1] = (float)max_idx;
}

__global__ void gpuFindRotationTranslation_step2(SiftPoint *point, float *tempMemory, bool *index_list, mat4x4 *rotation, vec4 *translation, int numPts)
{
  int idx0 = (int)(ld_gbl_cg(&(tempMemory[0]))); // = blockIdx.x * blockDim.x + threadIdx.x;
  vec4 t;
  mat4x4 h;
  mat4x4 rot;
  mat4x4 rot_t;
  TempCoords centroids;
  int num_inliners;
  int i, j;
  int k, r, c;

  // printf("rot before: idx %d\n%f,%f,%f| tx=%f\n%f,%f,%f| tx=%f\n%f,%f,%f| tx=%f, num inliners: %f, metric: %f\n", idx0,
  //        tempMemory[6 + idx0 * 15 + 0], tempMemory[6 + idx0 * 15 + 1], tempMemory[6 + idx0 * 15 + 2], tempMemory[6 + idx0 * 15 + 9],                                                                   // matrix 1st row
  //        tempMemory[6 + idx0 * 15 + 3], tempMemory[6 + idx0 * 15 + 4], tempMemory[6 + idx0 * 15 + 5], tempMemory[6 + idx0 * 15 + 10],                                                                  // matrix 2nd row
  //        tempMemory[6 + idx0 * 15 + 6], tempMemory[6 + idx0 * 15 + 7], tempMemory[6 + idx0 * 15 + 8], tempMemory[6 + idx0 * 15 + 11], tempMemory[6 + idx0 * 15 + 12], tempMemory[6 + idx0 * 15 + 13]); // matrix 3rd row)

  for (i = 0; i < 4; ++i)
  {
    for (j = 0; j < 4; ++j)
    {
      rot[i][j] = i == j ? 1.f : 0.f;
      h[i][j] = i == j ? 1.f : 0.f;
    }
  }
  centroids.x_3d = 0.0f;
  centroids.y_3d = 0.0f;
  centroids.z_3d = 0.0f;
  centroids.match_x_3d = 0.0f;
  centroids.match_y_3d = 0.0f;
  centroids.match_z_3d = 0.0f;
  num_inliners = 0;
  // printf("min idx = %d, i: ",idx0);
  for (int i = 0; i < numPts; i++)
  {
    if (index_list[idx0 * 512 + i] == true)
    {
      // printf(" %d,",i);
      centroids.x_3d += (point[i].x_3d);
      centroids.y_3d += (point[i].y_3d);
      centroids.z_3d += (point[i].z_3d);
      centroids.match_x_3d += (point[i].match_x_3d);
      centroids.match_y_3d += (point[i].match_y_3d);
      centroids.match_z_3d += (point[i].match_z_3d);
      num_inliners++;
    }
  }
  // printf("\n");
  centroids.x_3d /= num_inliners;
  centroids.y_3d /= num_inliners;
  centroids.z_3d /= num_inliners;
  centroids.match_x_3d /= num_inliners;
  centroids.match_y_3d /= num_inliners;
  centroids.match_z_3d /= num_inliners;

  for (int i = 0; i < numPts; i++)
  {
    if (index_list[idx0 * 512 + i] == true)
    {
      h[0][0] += (point[i].x_3d - centroids.x_3d) * (point[i].match_x_3d - centroids.match_x_3d);
      h[1][0] += (point[i].x_3d - centroids.x_3d) * (point[i].match_y_3d - centroids.match_y_3d);
      h[2][0] += (point[i].x_3d - centroids.x_3d) * (point[i].match_z_3d - centroids.match_z_3d);
      h[0][1] += (point[i].y_3d - centroids.y_3d) * (point[i].match_x_3d - centroids.match_x_3d);
      h[1][1] += (point[i].y_3d - centroids.y_3d) * (point[i].match_y_3d - centroids.match_y_3d);
      h[2][1] += (point[i].y_3d - centroids.y_3d) * (point[i].match_z_3d - centroids.match_z_3d);
      h[0][2] += (point[i].z_3d - centroids.z_3d) * (point[i].match_x_3d - centroids.match_x_3d);
      h[1][2] += (point[i].z_3d - centroids.z_3d) * (point[i].match_y_3d - centroids.match_y_3d);
      h[2][2] += (point[i].z_3d - centroids.z_3d) * (point[i].match_z_3d - centroids.match_z_3d);
      // printf("idx0: %d index %d: distance: %f \n",idx0,i,sqrt((point[i].x_3d-point[i].match_x_3d)*(point[i].x_3d-point[i].match_x_3d)+
      //                                           (point[i].y_3d-point[i].match_y_3d)*(point[i].y_3d-point[i].match_y_3d)+
      //                                           (point[i].z_3d-point[i].match_z_3d)*(point[i].z_3d-point[i].match_z_3d)));
      point[i].draw = true;
    } else {
        // point[i].draw = false;
      }
  }
  mat4x4 u;
  mat4x4 s;
  mat4x4 s_det;
  mat4x4 v;
  for (i = 0; i < 4; ++i)
    for (j = 0; j < 4; ++j)
    {
      u[i][j] = i == j ? 1.f : 0.f;
      s[i][j] = i == j ? 1.f : 0.f;
      s_det[i][j] = i == j ? 1.f : 0.f;
      v[i][j] = i == j ? 1.f : 0.f;
    }
  svd(h[0][0], h[1][0], h[2][0],
      h[0][1], h[1][1], h[2][1],
      h[0][2], h[1][2], h[2][2],
      u[0][0], u[1][0], u[2][0],
      u[0][1], u[1][1], u[2][1],
      u[0][2], u[1][2], u[2][2],
      s[0][0], s[1][1], s[2][2],
      v[0][0], v[1][0], v[2][0],
      v[0][1], v[1][1], v[2][1],
      v[0][2], v[1][2], v[2][2]);
  mat4x4 u_t;
  for (j = 0; j < 4; ++j)
    for (i = 0; i < 4; ++i)
      u_t[i][j] = u[j][i];
  mat4x4 vu_t;
  for (c = 0; c < 4; ++c)
    for (r = 0; r < 4; ++r)
    {
      vu_t[c][r] = 0.f;
      for (k = 0; k < 4; ++k)
        vu_t[c][r] += v[k][r] * u_t[c][k];
    }

  float det = vu_t[0][0] * vu_t[1][1] * vu_t[2][2] + vu_t[1][0] * vu_t[2][1] * vu_t[0][2] +
              vu_t[2][0] * vu_t[0][1] * vu_t[1][2] - vu_t[2][0] * vu_t[1][1] * vu_t[0][2] -
              vu_t[1][0] * vu_t[0][1] * vu_t[2][2] - vu_t[0][0] * vu_t[2][1] * vu_t[1][2];

  s_det[2][2] = det;
  mat4x4 vs_det;
  for (c = 0; c < 4; ++c)
    for (r = 0; r < 4; ++r)
    {
      vs_det[c][r] = 0.f;
      for (k = 0; k < 4; ++k)
        vs_det[c][r] += v[k][r] * s_det[c][k];
    }
  for (c = 0; c < 4; ++c)
    for (r = 0; r < 4; ++r)
    {
      rot_t[c][r] = 0.f;
      for (k = 0; k < 4; ++k)
        rot_t[c][r] += vs_det[k][r] * u_t[c][k];
    }
  for (j = 0; j < 4; ++j)
    for (i = 0; i < 4; ++i)
      rot[i][j] = rot_t[j][i];
  vec4 p_dash = {centroids.match_x_3d, centroids.match_y_3d, centroids.match_z_3d, 1.0};
  vec4 rp_dash;
  for (j = 0; j < 4; ++j)
  {
    rp_dash[j] = 0.f;
    for (i = 0; i < 4; ++i)
      rp_dash[j] += rot[i][j] * p_dash[i];
  }
  // printf("Centroid diff before rematch: %f, %f, %f\n", centroids.x_3d-centroids.match_x_3d, centroids.y_3d- centroids.match_y_3d, centroids.z_3d-centroids.match_z_3d);
  t[0] = isnan(centroids.x_3d - rp_dash[0]) ? 0.0 : centroids.x_3d - rp_dash[0];
  t[1] = isnan(centroids.y_3d - rp_dash[1]) ? 0.0 : centroids.y_3d - rp_dash[1];
  t[2] = isnan(centroids.z_3d - rp_dash[2]) ? 0.0 : centroids.z_3d - rp_dash[2];
  t[3] = 1.0;
  *rotation[0][0] = isnan(rot[0][0]) ? 1.0 : rot[0][0];
  *rotation[0][1] = isnan(rot[0][1]) ? 0.0 : rot[0][1];
  *rotation[0][2] = isnan(rot[0][2]) ? 0.0 : rot[0][2];
  *rotation[1][0] = isnan(rot[1][0]) ? 0.0 : rot[1][0];
  *rotation[1][1] = isnan(rot[1][1]) ? 1.0 : rot[1][1];
  *rotation[1][2] = isnan(rot[1][2]) ? 0.0 : rot[1][2];
  *rotation[2][0] = isnan(rot[2][0]) ? 0.0 : rot[2][0];
  *rotation[2][1] = isnan(rot[2][1]) ? 0.0 : rot[2][1];
  *rotation[2][2] = isnan(rot[2][2]) ? 1.0 : rot[2][2];
  *translation[0] = t[0];
  *translation[1] = t[1];
  *translation[2] = t[2];
  // printf("rot after\n%f,%f,%f| tx=%f\n%f,%f,%f| tx=%f\n%f,%f,%f| tx=%f, num_inliers = %d \n",
  //        rot[0][0], rot[1][0], rot[2][0], t[0],                // matrix 1st row
  //        rot[0][1], rot[1][1], rot[2][1], t[1],                // matrix 2nd row
  //        rot[0][2], rot[1][2], rot[2][2], t[2], num_inliners); // matrix 3rd row)
}

#define EXP_LAMBDA 2.0f
__global__ void gpuRansac2d(SiftPoint *point, SiftPoint *point_old, float *tempMemory, bool *index_list, float *ransac_dx, float *ransac_dy, int numPts, int numPts_old)
{
  int idx0 = blockIdx.x * blockDim.x + threadIdx.x;
  tempMemory[idx0] = 0.0f;
  float sum_distances_metric = 0.0;
  if (point[idx0].score >= MIN_SCORE)
  {
    float dx, dy;
    float est_xpos, est_ypos;
    float temp_metric, temp_distance;
    dx = point[idx0].xpos - point[idx0].match_xpos;
    dy = point[idx0].ypos - point[idx0].match_ypos;
    for (int i = 0; i < numPts; i++)
    {
      if (point[i].score > MIN_SCORE)
      {
        est_xpos = point[i].xpos - dx;
        est_ypos = point[i].ypos - dy;
        temp_distance = sqrt((point[i].match_xpos - est_xpos) * (point[i].match_xpos - est_xpos) + (point[i].match_ypos - est_ypos) * (point[i].match_ypos - est_ypos));
        temp_metric = exp((-1.0 * temp_distance) / (EXP_LAMBDA));
        // if (idx0 == 0) printf("Idx0: 0 to i: %d, metric: %f\n", i, temp_metric);
        if (temp_metric >= 0.80)
          index_list[512 * idx0 + i] = true;
        else
          index_list[512 * idx0 + i] = false;
        sum_distances_metric += temp_metric;
      }
      else
      {
        index_list[512 * idx0 + i] = false;
      }
    }
  }
  tempMemory[idx0] = sum_distances_metric;
  __syncthreads();
  if (idx0 == 0)
  {
    float max_metric = 0.0f;
    float tmp_metric;
    int max_index;
    for (int i = 0; i < numPts; i++)
    {
      tmp_metric = ld_gbl_cg(&(tempMemory[i]));
      if (tmp_metric > max_metric)
      {
        max_metric = tmp_metric;
        max_index = i;
      }
    }
    tempMemory[512] = (float)(max_index);
  }
  __syncthreads();
  int tmp_index = (int)ld_gbl_cg(&(tempMemory[512]));

  if (idx0 == tmp_index)
  {
    float total_dx = 0.0f;
    float total_dy = 0.0f;
    int inliers_cnt = 0;
    for (int i = 0; i < numPts; i++)
    {
      if (index_list[512 * idx0 + i] == true)
      {
        inliers_cnt++;
        total_dx += point[i].xpos - point[i].match_xpos;
        total_dy += point[i].ypos - point[i].match_ypos;
      }
    }
    total_dx /= inliers_cnt;
    total_dy /= inliers_cnt;
    // printf("Index %d says hi! total inliers: %d of %d, dx: %f, dy%f\n", idx0, inliers_cnt, numPts, total_dx, total_dy);
    tempMemory[513] = total_dx;
    tempMemory[514] = total_dy;
    *ransac_dx = total_dx;
    *ransac_dy = total_dy;
  }
  __syncthreads();
  float total_dx = ld_gbl_cg(&(tempMemory[513]));
  float total_dy = ld_gbl_cg(&(tempMemory[514]));
  float est_xpos = point[idx0].xpos - total_dx;
  float est_ypos = point[idx0].ypos - total_dy;
  float temp_distance;
  float min_distance = 5000.0;
  int min_index = 0;
  for (int i = 0; i < numPts_old; i++)
  {
    temp_distance = sqrt((point_old[i].xpos - est_xpos) * (point_old[i].xpos - est_xpos) + (point_old[i].ypos - est_ypos) * (point_old[i].ypos - est_ypos));
    if (temp_distance < min_distance)
    {
      min_distance = temp_distance;
      min_index = i;
    }
  }
  float temp_metric = exp((-1.0 * min_distance) / (EXP_LAMBDA));
  point[idx0].ransac_match = min_index;
  point[idx0].ransac_score = temp_metric;
  point[idx0].ransac_xpos = point_old[min_index].xpos;
  point[idx0].ransac_ypos = point_old[min_index].ypos;
}

#define GAUSS_VARIANCE 100
__global__ void gpuRematchSiftPoints(SiftPoint *point_new, SiftPoint *point_old, mat4x4 *rotation, vec4 *translation, int numPts_new, int numPts_old)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numPts_new)
    return;
  point_new[idx].score = 1.0f;
  point_new[idx].match = -1;
  vec4 coords_new;
  coords_new[0] = point_new[idx].x_3d - *(translation[0]);
  coords_new[1] = point_new[idx].y_3d - *(translation[1]);
  coords_new[2] = point_new[idx].z_3d - *(translation[2]);
  coords_new[3] = 1.0f;
  vec4 coords_est;
  mat4x4 rot_trans;
  int i, j;
  for (j = 0; j < 4; ++j)
    for (i = 0; i < 4; ++i)
      rot_trans[i][j] = *(rotation[j][i]);
  rot_trans[3][3] = 1.0;
  // if (idx == 0)
  //   printf("Rot trans:\n %f, %f, %f, %f\n%f, %f, %f, %f\n%f, %f, %f, %f\n%f, %f, %f, %f\n",
  //          rot_trans[0][0], rot_trans[0][1], rot_trans[0][2], rot_trans[0][3],
  //          rot_trans[1][0], rot_trans[1][1], rot_trans[1][2], rot_trans[1][3],
  //          rot_trans[2][0], rot_trans[2][1], rot_trans[2][2], rot_trans[2][3],
  //          rot_trans[3][0], rot_trans[3][1], rot_trans[3][2], rot_trans[3][3]);

  for (j = 0; j < 4; ++j)
  {
    coords_est[j] = 0.f;
    for (i = 0; i < 4; ++i)
    {
      coords_est[j] += rot_trans[i][j] * coords_new[i];
    }
  }

  coords_est[0] = coords_est[0];//
  coords_est[1] = coords_est[1];//
  coords_est[2] = coords_est[2];//
  float d_x;
  float d_y;
  float d_z;
  float distance;
  float distance_min;
  distance_min = 100000;
  int idx_min = numPts_old+1;
  for (i = 0; i < numPts_old; i++)
  {
    d_x = point_old[i].x_3d - coords_est[0];
    d_y = point_old[i].y_3d - coords_est[1];
    d_z = point_old[i].z_3d - coords_est[2];
    distance = (d_x * d_x) + (d_y * d_y) + (d_z * d_z);
    if (distance < distance_min)
    {
      distance_min = distance;
      idx_min = i;
    }
  }
  // printf("idx: %d min distance to matching point: %f\n", idx, distance_min);

  if ((distance_min < 1.0f)&&(idx_min < numPts_old)) {
     point_new[idx].ransac_match = idx_min;
     float ransac_xpos = point_old[idx_min].xpos;
     float ransac_ypos = point_old[idx_min].ypos;
     float ransac_x_3d = point_old[idx_min].x_3d;
     float ransac_y_3d = point_old[idx_min].y_3d;
     float ransac_z_3d = point_old[idx_min].z_3d;
  //   point_new[idx].match_distance = point_old[idx_min].distance;
     point_new[idx].ransac_xpos_3d = ransac_xpos; // point_old[idx_min].xpos;
     point_new[idx].ransac_ypos_3d = ransac_ypos;
     point_new[idx].ransac_x_3d = ransac_x_3d;
     point_new[idx].ransac_y_3d = ransac_y_3d;
     point_new[idx].ransac_z_3d = ransac_z_3d;
     point_new[idx].draw = true;
  } else {
     point_new[idx].draw = false;
  }
  
  // point_new[idx].match_distance = coords_est[0];
  // float x_div = coords_est[1] / coords_est[0];
  // float x_tan = atan(x_div);
  // point_new[idx].ransac_xpos_3d = (-128 * (x_tan - (WIDTH_ANGLE))) / (WIDTH_ANGLE);
  // float y_div = coords_est[2] / coords_est[0];
  // float y_tan = atan(y_div);
  // point_new[idx].ransac_ypos_3d = (-102.5 * (y_tan - (HEIGHT_ANGLE))) / (HEIGHT_ANGLE);
  // point_new[idx].draw = true;

  // if (idx == 0)
  // {
  //   printf("orig point: %f, %f, %f ; estimated point: %f, %f, %f, x_div=%f, x_tan = %f\n",
  //          point_new[idx].x_3d, point_new[idx].y_3d, point_new[idx].z_3d, point_new[idx].match_x_3d, point_new[idx].match_y_3d, point_new[idx].match_z_3d, x_div, x_tan);
  //   printf("orig img: %f, %f, %f ; estimated img: %f, %f, %f\n",
  //          point_new[idx].xpos, point_new[idx].ypos, point_new[idx].distance, point_new[idx].match_xpos, point_new[idx].match_ypos, point_new[idx].match_distance);
  // }
}

__global__ void gpuFindOptimalRotationTranslation(SiftPoint *point, float *tempMemory, mat4x4 *rotation, vec4 *translation, int numPts)
{
  int idx0 = blockIdx.x * blockDim.x + threadIdx.x;
  mat4x4 rot;
  mat4x4 h;
  vec4 cent_old;
  vec4 cent_new;

    int idx;
    for (idx = 0; idx < 6; idx++)
    {
      int divisor = 0;
      if (idx == 0)
        cent_new[0] = 0.f;
      if (idx == 1)
        cent_new[1] = 0.f;
      if (idx == 2)
        cent_new[2] = 0.f;
      if (idx == 3)
        cent_old[0] = 0.f;
      if (idx == 4)
        cent_old[1] = 0.f;
      if (idx == 5)
        cent_old[2] = 0.f;
      for (int i = 0; i < numPts; i++)
      {
        if (point[i].draw == true)
        {
          divisor++;
          if (idx == 0)
            cent_new[0] += point[i].x_3d;
          if (idx == 1)
            cent_new[1] += point[i].y_3d;
          if (idx == 2)
            cent_new[2] += point[i].z_3d;
          if (idx == 3)
            cent_old[0] += point[i].ransac_x_3d;
          if (idx == 4)
            cent_old[1] += point[i].ransac_y_3d;
          if (idx == 5)
            cent_old[2] += point[i].ransac_z_3d;
        }
      }
      if (divisor == 0) {
        return;
      }
      if (idx == 0)
        cent_new[0] /= divisor;
      if (idx == 1)
        cent_new[1] /= divisor;
      if (idx == 2)
        cent_new[2] /= divisor;
      if (idx == 3)
        cent_old[0] /= divisor;
      if (idx == 4)
        cent_old[1] /= divisor;
      if (idx == 5)
        cent_old[2] /= divisor;
    }
    // printf("Centroid diff after rematch: %f, %f, %f\n", cent_new[0]-cent_old[0], cent_new[1]-cent_old[1], cent_new[2]-cent_old[2]);
    for (idx = 0; idx < 9; idx++)
    {
        h[0][0] = 0.f;
        h[1][0] = 0.f;
        h[2][0] = 0.f;
        h[0][1] = 0.f;
        h[1][1] = 0.f;
        h[2][1] = 0.f;
        h[0][2] = 0.f;
        h[1][2] = 0.f;
        h[2][2] = 0.f;
      for (int i = 0; i < numPts; i++)
      {
        if (point[i].draw == true)
        {
            h[0][0] += (point[i].x_3d - cent_new[0]) * (point[i].ransac_x_3d - cent_old[0]);
            h[1][0] += (point[i].x_3d - cent_new[0]) * (point[i].ransac_y_3d - cent_old[1]);
            h[2][0] += (point[i].x_3d - cent_new[0]) * (point[i].ransac_z_3d - cent_old[2]);
            h[0][1] += (point[i].y_3d - cent_new[1]) * (point[i].ransac_x_3d - cent_old[0]);
            h[1][1] += (point[i].y_3d - cent_new[1]) * (point[i].ransac_y_3d - cent_old[1]);
            h[2][1] += (point[i].y_3d - cent_new[1]) * (point[i].ransac_z_3d - cent_old[2]);
            h[0][2] += (point[i].z_3d - cent_new[2]) * (point[i].ransac_x_3d - cent_old[0]);
            h[1][2] += (point[i].z_3d - cent_new[2]) * (point[i].ransac_y_3d - cent_old[1]);
            h[2][2] += (point[i].z_3d - cent_new[2]) * (point[i].ransac_z_3d - cent_old[2]);
        }
      }
    }
    int i, j, k, r, c;
    mat4x4 u;
    mat4x4 s;
    mat4x4 s_det;
    mat4x4 v;
    for (i = 0; i < 4; ++i)
      for (j = 0; j < 4; ++j)
      {
        u[i][j] = i == j ? 1.f : 0.f;
        s[i][j] = i == j ? 1.f : 0.f;
        s_det[i][j] = i == j ? 1.f : 0.f;
        v[i][j] = i == j ? 1.f : 0.f;
      }

    svd(h[0][0], h[1][0], h[2][0],
        h[0][1], h[1][1], h[2][1],
        h[0][2], h[1][2], h[2][2],
        u[0][0], u[1][0], u[2][0],
        u[0][1], u[1][1], u[2][1],
        u[0][2], u[1][2], u[2][2],
        s[0][0], s[1][1], s[2][2],
        v[0][0], v[1][0], v[2][0],
        v[0][1], v[1][1], v[2][1],
        v[0][2], v[1][2], v[2][2]);
    mat4x4 u_t;
    for (j = 0; j < 4; ++j)
      for (i = 0; i < 4; ++i)
        u_t[i][j] = u[j][i];
    mat4x4 vu_t;
    for (c = 0; c < 4; ++c)
      for (r = 0; r < 4; ++r)
      {
        vu_t[c][r] = 0.f;
        for (k = 0; k < 4; ++k)
          vu_t[c][r] += v[k][r] * u_t[c][k];
      }

  float det = vu_t[0][0] * vu_t[1][1] * vu_t[2][2] + vu_t[1][0] * vu_t[2][1] * vu_t[0][2] +
              vu_t[2][0] * vu_t[0][1] * vu_t[1][2] - vu_t[2][0] * vu_t[1][1] * vu_t[0][2] -
              vu_t[1][0] * vu_t[0][1] * vu_t[2][2] - vu_t[0][0] * vu_t[2][1] * vu_t[1][2];
    s_det[2][2] = det;
    mat4x4 vs_det;
    for (c = 0; c < 4; ++c)
      for (r = 0; r < 4; ++r)
      {
        vs_det[c][r] = 0.f;
        for (k = 0; k < 4; ++k)
          vs_det[c][r] += v[k][r] * s_det[c][k];
      }
    mat4x4 rot_t;
    for (c = 0; c < 4; ++c)
      for (r = 0; r < 4; ++r)
      {
        rot_t[c][r] = 0.f;
        for (k = 0; k < 4; ++k)
          rot_t[c][r] += vs_det[k][r] * u_t[c][k];
      }

    for (j = 0; j < 4; ++j)
      for (i = 0; i < 4; ++i)
        rot[i][j] = rot_t[j][i];
  vec4 p_dash = {cent_old[0], cent_old[1], cent_old[2], 1.0};
  vec4 rp_dash;
  for (j = 0; j < 4; ++j)
  {
    rp_dash[j] = 0.f;
    for (i = 0; i < 4; ++i)
      rp_dash[j] += rot[i][j] * p_dash[i];
  }
  
  if (idx0 == 0)
  {
    //printf("centroids\n%f,%f,%f\n%f,%f,%f\n",
    //       cent_new[0], cent_new[1], cent_new[2],  // matrix 1st row
    //       cent_old[0], cent_old[1], cent_old[2]); // matrix 3rd row)
    //
    //printf("h\n%f,%f,%f\n%f,%f,%f\n%f,%f,%f\n",
    //       h[0][0], h[1][0], h[2][0],  // matrix 1st row
    //       h[0][1], h[1][1], h[2][1],  // matrix 2nd row
    //       h[0][2], h[1][2], h[2][2]); // matrix 3rd row)
    // printf("rot\n%f,%f,%f\n%f,%f,%f\n%f,%f,%f\n",
    //        rot[0][0], rot[1][0], rot[2][0],  // matrix 1st row
    //        rot[0][1], rot[1][1], rot[2][1],  // matrix 2nd row
    //        rot[0][2], rot[1][2], rot[2][2]); // matrix 3rd row)
  // t[0] = isnan(centroids.x_3d - rp_dash[0]) ? 0.0 : centroids.x_3d - rp_dash[0];
  // t[1] = isnan(centroids.y_3d - rp_dash[1]) ? 0.0 : centroids.y_3d - rp_dash[1];
  // t[2] = isnan(centroids.z_3d - rp_dash[2]) ? 0.0 : centroids.z_3d - rp_dash[2];
  // t[3] = 1.0;
  *rotation[0][0] = isnan(rot[0][0]) ? 1.0 : rot[0][0];
  *rotation[0][1] = isnan(rot[0][1]) ? 0.0 : rot[0][1];
  *rotation[0][2] = isnan(rot[0][2]) ? 0.0 : rot[0][2];
  *rotation[1][0] = isnan(rot[1][0]) ? 0.0 : rot[1][0];
  *rotation[1][1] = isnan(rot[1][1]) ? 1.0 : rot[1][1];
  *rotation[1][2] = isnan(rot[1][2]) ? 0.0 : rot[1][2];
  *rotation[2][0] = isnan(rot[2][0]) ? 0.0 : rot[2][0];
  *rotation[2][1] = isnan(rot[2][1]) ? 0.0 : rot[2][1];
  *rotation[2][2] = isnan(rot[2][2]) ? 1.0 : rot[2][2];
  *translation[0] = isnan(cent_new[0] - rp_dash[0]) ? 0.0 : cent_new[0] - rp_dash[0];
  *translation[1] = isnan(cent_new[1] - rp_dash[1]) ? 0.0 : cent_new[1] - rp_dash[1];
  *translation[2] = isnan(cent_new[2] - rp_dash[2]) ? 0.0 : cent_new[2] - rp_dash[2];
  *translation[3] = 1.0;
  }
}

__global__ void gpuUndistort(float *dst, uint16_t *src, uint16_t *xCoords, uint16_t *yCoords, int width_src, int height_src, int width_dst, int height_dst)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("hoi\n");
  if (idx >= width_dst)
  {
    return;
  }
  //int x = 50;
  //int y = 50;
  for (int i = 0; i < height_dst; ++i)
  {
    // Calculate new coordinates
    dst[i * width_dst + idx + 0] = src[yCoords[i * 265 + idx] * width_src + xCoords[i * 265 + idx] + 0];
  }
}

__global__ void gpuUndistortCosAlpha(float *dst, uint16_t *src, uint16_t *xCoords, uint16_t *yCoords, int width_src, int height_src, int width_dst, int height_dst, float *cosAlpha)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("hoi\n");
  if (idx >= width_dst)
  {
    return;
  }

  for (int i = 0; i < height_dst; ++i)
  {
    // Calculate new coordinates
    dst[i * width_dst + idx] = (float)(cosAlpha[i * 265 + idx] * src[yCoords[i * 265 + idx] * width_src + xCoords[i * 265 + idx]]);
  }
  // if (idx == 128)
  // {
  //   printf("Distance: %f\n", dst[100 * width_dst + 128]);
  // }
}

__global__ void gpuUndistort_uint16(uint16_t *dst, uint16_t *src, uint16_t *xCoords, uint16_t *yCoords, int width_src, int height_src, int width_dst, int height_dst)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("hoi\n");
  if (idx >= width_dst)
  {
    return;
  }

  for (int i = 0; i < height_dst; ++i)
  {
    // Calculate new coordinates
    // dst[i*width_dst*4+idx+0] = 255; // src[i*width_dst*4+idx+0];
    dst[i * width_dst * 4 + idx * 4] = src[yCoords[i * width_dst + idx] * width_src * 4 + xCoords[i * width_dst + idx] * 4 + 0];
    dst[i * width_dst * 4 + idx * 4 + 1] = src[yCoords[i * width_dst + idx] * width_src * 4 + xCoords[i * width_dst + idx] * 4 + 1];
    dst[i * width_dst * 4 + idx * 4 + 2] = src[yCoords[i * width_dst + idx] * width_src * 4 + xCoords[i * width_dst + idx] * 4 + 2];
    dst[i * width_dst * 4 + idx * 4 + 3] = 1;
  }
}

__global__ void gpuNormalizeToInt16(uint16_t *dst, float *src, const int width, const int height)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= width)
  {
    return;
  }
  for (int i = 0; i < height; ++i)
  {
    dst[i * width * 4 + idx * 4 + 0] = (uint16_t)(src[i * width + idx]);
    dst[i * width * 4 + idx * 4 + 1] = (uint16_t)(src[i * width + idx]);
    dst[i * width * 4 + idx * 4 + 2] = (uint16_t)(src[i * width + idx]);
    dst[i * width * 4 + idx * 4 + 3] = 255 * 255;
  }
}

__global__ void gpuDrawSiftData(uint16_t *dst, float *src, SiftPoint *d_sift, int nPoints, const int width, const int height)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < width)
  {
    for (int i = 0; i < height; ++i)
    {
      dst[i * width * 4 + idx * 4 + 0] = (uint16_t)(src[i * width + idx] * 255);
      dst[i * width * 4 + idx * 4 + 1] = (uint16_t)(src[i * width + idx] * 255);
      dst[i * width * 4 + idx * 4 + 2] = (uint16_t)(src[i * width + idx] * 255);
      dst[i * width * 4 + idx * 4 + 3] = 255 * 255;
    }
  }
  if (nPoints <= 0)
    return;
  __syncthreads();
#define FILTER_ZONE 500
  if (idx < nPoints)
  {
    if ((d_sift[idx].score > MIN_SCORE) && (d_sift[idx].draw == true) &&
        (((int)(d_sift[idx].ypos)) < height) && (((int)(d_sift[idx].match_ypos)) < height) &&
        (((int)(d_sift[idx].xpos)) < width) && (((int)(d_sift[idx].match_xpos)) < width))

    if ((d_sift[idx].score > MIN_SCORE) && (d_sift[idx].draw == true) &&
        (((int)(d_sift[idx].ypos)) < height) && (((int)(d_sift[idx].ransac_ypos)) < height) &&
        (((int)(d_sift[idx].xpos)) < width) && (((int)(d_sift[idx].ransac_xpos)) < width))
    {
      if (d_sift[idx].ypos < d_sift[idx].ransac_ypos)
      {
        for (int i = d_sift[idx].ypos; i < d_sift[idx].ransac_ypos - 1; i++)
          dst[((int)(i)) * width * 4 + ((int)(d_sift[idx].xpos)) * 4 + 0] = 255 * 255;
      }
      else
      {
        for (int i = d_sift[idx].ypos - 1; i > d_sift[idx].ransac_ypos; i--)
          dst[((int)(i)) * width * 4 + ((int)(d_sift[idx].xpos)) * 4 + 0] = 255 * 255;
      }
      if (d_sift[idx].xpos < d_sift[idx].ransac_xpos)
      {
        for (int i = d_sift[idx].xpos; i < d_sift[idx].ransac_xpos - 1; i++)
          dst[((int)(d_sift[idx].ransac_ypos)) * width * 4 + ((int)(i)) * 4 + 0] = 255 * 255;
      }
      else
      {
        for (int i = d_sift[idx].xpos - 1; i > d_sift[idx].ransac_xpos; i--)
          dst[((int)(d_sift[idx].ransac_ypos)) * width * 4 + ((int)(i)) * 4 + 0] = 255 * 255;
      }
    }
    if ((d_sift[idx].score > MIN_SCORE) && (d_sift[idx].draw == true) &&
    (((int)(d_sift[idx].ypos)) < height) && (((int)(d_sift[idx].ransac_ypos_3d)) < height) &&
    (((int)(d_sift[idx].xpos)) < width) && (((int)(d_sift[idx].ransac_xpos_3d)) < width))
    {
      if (d_sift[idx].ypos < d_sift[idx].ransac_ypos_3d)
      {
        for (int i = d_sift[idx].ypos; i < d_sift[idx].ransac_ypos_3d - 1; i++)
          dst[((int)(i)) * width * 4 + ((int)(d_sift[idx].xpos)) * 4 + 1] = 255 * 255;
      }
      else
      {
        for (int i = d_sift[idx].ypos - 1; i > d_sift[idx].ransac_ypos_3d; i--)
          dst[((int)(i)) * width * 4 + ((int)(d_sift[idx].xpos)) * 4 + 1] = 255 * 255;
      }
      if (d_sift[idx].xpos < d_sift[idx].ransac_xpos_3d)
      {
        for (int i = d_sift[idx].xpos; i < d_sift[idx].ransac_xpos_3d - 1; i++)
          dst[((int)(d_sift[idx].ransac_ypos_3d)) * width * 4 + ((int)(i)) * 4 + 1] = 255 * 255;
      }
      else
      {
        for (int i = d_sift[idx].xpos - 1; i > d_sift[idx].ransac_xpos_3d; i--)
          dst[((int)(d_sift[idx].ransac_ypos_3d)) * width * 4 + ((int)(i)) * 4 + 1] = 255 * 255;
      }
      //printf("New;%d;%f;%f;%f;%d;%d;%d\n", idx, d_sift[idx].xpos, d_sift[idx].ypos, d_sift[idx].distance, d_sift[idx].x_3d, d_sift[idx].y_3d, d_sift[idx].z_3d);
      //printf("Old;%d;%f;%f;%f;%d;%d;%d\n", idx, d_sift[idx].match_xpos, d_sift[idx].match_ypos, d_sift[idx].match_distance, d_sift[idx].match_x_3d, d_sift[idx].match_y_3d, d_sift[idx].match_z_3d);

      //else {
      //  //printf("Filtered - score = %f, scale= %f, sharpness= %f, ambiguity= %f, edgeness= %f\n",d_sift[idx].score, d_sift[idx].scale, d_sift[idx].sharpness, d_sift[idx].ambiguity, d_sift[idx].edgeness);
      //}
    }
  }
  __syncthreads();
  if (idx == 0)
  {
    dst[(int)(100 * width * 4 + (128) * 4 + 2)] = 255 * 255;
  }
}

// define USE_TEST_DATA

__global__ void gpuAddDepthInfoToSift(SiftPoint *data, float *depthData, int nPoints, float *x, float *y, float *z, float *conf)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nPoints)
  {
    return;
  }
// overwrite first three with test data
#ifdef USE_TEST_DATA
  if (idx == 0)
  {
    data[idx].match = idx;
    data[idx].score = 0.99;
    data[idx].match_xpos = 100.0;
    data[idx].match_ypos = 100.0;
    data[idx].match_distance = 1000.00;
    data[idx].match_x_3d = data[idx].match_distance;
    data[idx].match_y_3d = data[idx].match_distance * tan(WIDTH_ANGLE * (128 - data[idx].match_xpos) / 128);
    data[idx].match_z_3d = data[idx].match_distance * tan(HEIGHT_ANGLE * (102.5 - data[idx].match_ypos) / 102.5);
    data[idx].xpos = 102.99331315;
    data[idx].ypos = 124.96098776;
    data[idx].distance = 1192.66243651;
    data[idx].draw = true;
    data[idx].x_3d = data[idx].distance;
    data[idx].y_3d = data[idx].distance * tan(WIDTH_ANGLE * (128 - data[idx].xpos) / 128);
    data[idx].z_3d = data[idx].distance * tan(HEIGHT_ANGLE * (102.5 - data[idx].ypos) / 102.5);
    return;
  }
  if (idx == 1)
  {
    data[idx].match = idx;
    data[idx].score = 0.99;
    data[idx].match_xpos = 130.0;
    data[idx].match_ypos = 100.0;
    data[idx].match_distance = 800.00;
    data[idx].match_x_3d = data[idx].match_distance;
    data[idx].match_y_3d = data[idx].match_distance * tan(WIDTH_ANGLE * (128 - data[idx].match_xpos) / 128);
    data[idx].match_z_3d = data[idx].match_distance * tan(HEIGHT_ANGLE * (102.5 - data[idx].match_ypos) / 102.5);
    data[idx].xpos = 127.91434393;
    data[idx].ypos = 127.31468943 * 2;
    data[idx].distance = 993.15426132;
    data[idx].draw = true;
    data[idx].x_3d = data[idx].distance;
    data[idx].y_3d = data[idx].distance * tan(WIDTH_ANGLE * (128 - data[idx].xpos) / 128);
    data[idx].z_3d = data[idx].distance * tan(HEIGHT_ANGLE * (102.5 - data[idx].ypos) / 102.5);
    return;
  }
  if (idx == 2)
  {
    data[idx].match = idx;
    data[idx].score = 0.99;
    data[idx].match_xpos = 100.0;
    data[idx].match_ypos = 130.0;
    data[idx].match_distance = 1200.00;
    data[idx].match_x_3d = data[idx].match_distance;
    data[idx].match_y_3d = data[idx].match_distance * tan(WIDTH_ANGLE * (128 - data[idx].match_xpos) / 128);
    data[idx].match_z_3d = data[idx].match_distance * tan(HEIGHT_ANGLE * (102.5 - data[idx].match_ypos) / 102.5);
    data[idx].xpos = 98.05438935;
    data[idx].ypos = 151.20425739;
    data[idx].distance = 1369.64237417;
    data[idx].draw = true;
    data[idx].x_3d = data[idx].distance;
    data[idx].y_3d = data[idx].distance * tan(WIDTH_ANGLE * (128 - data[idx].xpos) / 128);
    data[idx].z_3d = data[idx].distance * tan(HEIGHT_ANGLE * (102.5 - data[idx].ypos) / 102.5);
    return;
  }
  if (idx == 3)
  {
    data[idx].match = idx;
    data[idx].score = 0.99;
    data[idx].match_xpos = 110.0;
    data[idx].match_ypos = 110.0;
    data[idx].match_distance = 1500.00;
    data[idx].match_x_3d = data[idx].match_distance;
    data[idx].match_y_3d = data[idx].match_distance * tan(WIDTH_ANGLE * (128 - data[idx].match_xpos) / 128);
    data[idx].match_z_3d = data[idx].match_distance * tan(HEIGHT_ANGLE * (102.5 - data[idx].match_ypos) / 102.5);
    data[idx].xpos = 108.91867911;
    data[idx].ypos = 136.2342138;
    data[idx].distance = 1679.43575987;
    data[idx].draw = true;
    data[idx].x_3d = data[idx].distance;
    data[idx].y_3d = data[idx].distance * tan(WIDTH_ANGLE * (128 - data[idx].xpos) / 128);
    data[idx].z_3d = data[idx].distance * tan(HEIGHT_ANGLE * (102.5 - data[idx].ypos) / 102.5);
    return;
  }
  if (idx == 4)
  {
    data[idx].match = idx;
    data[idx].score = 0.99;
    data[idx].match_xpos = 70.0;
    data[idx].match_ypos = 90.0;
    data[idx].match_distance = 900.00;
    data[idx].match_x_3d = data[idx].match_distance;
    data[idx].match_y_3d = data[idx].match_distance * tan(WIDTH_ANGLE * (128 - data[idx].match_xpos) / 128);
    data[idx].match_z_3d = data[idx].match_distance * tan(HEIGHT_ANGLE * (102.5 - data[idx].match_ypos) / 102.5);
    data[idx].xpos = 80.22861221;
    data[idx].ypos = 113.04208145;
    data[idx].distance = 1099.89691147;
    data[idx].draw = true;
    data[idx].x_3d = data[idx].distance;
    data[idx].y_3d = data[idx].distance * tan(WIDTH_ANGLE * (128 - data[idx].xpos) / 128);
    data[idx].z_3d = data[idx].distance * tan(HEIGHT_ANGLE * (102.5 - data[idx].ypos) / 102.5);
    return;
  }
  return;
#endif
  if ((depthData[(int)(data[idx].xpos) + 256 * (int)(data[idx].ypos)] > 60000) || (depthData[(int)(data[idx].xpos) + 256 * (int)(data[idx].ypos)] == 0))
  {
    data[idx].distance = 0.0;
    data[idx].x_3d = 0.0;
    data[idx].y_3d = 0.0;
    data[idx].z_3d = 0.0;
    data[idx].conf = 0.0;
  }
  else
  {
    data[idx].conf = conf[(int)(data[idx].xpos) + 256 * (int)(data[idx].ypos)];
    data[idx].distance = (depthData[(int)(data[idx].xpos) + 256 * (int)(data[idx].ypos)]) / 100.;
    data[idx].x_3d = (depthData[(int)(data[idx].xpos) + 256 * (int)(data[idx].ypos)]) / 100.;   //x[(int)(data[idx].xpos) + 256 * (int)(data[idx].ypos)]; //
    data[idx].y_3d = -(data[idx].xpos+128)/280*data[idx].distance; // data[idx].distance * tan(WIDTH_ANGLE * (128 - data[idx].xpos) / 128);      //y[(int)(data[idx].xpos) + 256 * (int)(data[idx].ypos)]; //
    data[idx].z_3d = (data[idx].ypos+102.5)/280*data[idx].distance; //data[idx].distance * tan(HEIGHT_ANGLE * (102.5 - data[idx].ypos) / 102.5); //z[(int)(data[idx].xpos) + 256 * (int)(data[idx].ypos)]; //
    // printf("distance = x_3d = %f, y_3d = %f, z_3d = %f, xpos = %f, ypos = %f\n", data[idx].x_3d, data[idx].y_3d, data[idx].z_3d, data[idx].xpos, data[idx].ypos);
  }
}

__global__ void gpuScaleFloat2Float(float *dst, float *src, const int width, const int height)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float min_value[1920];
  __shared__ float max_value[1920];
  if (idx >= width)
  {
    return;
  }
  min_value[idx] = 70000.0;
  max_value[idx] = -70000.0;

  for (int i = 0; i < height; ++i)
  {
    if (src[i * width + idx] < min_value[idx])
      min_value[idx] = src[i * width + idx];
    if (src[i * width + idx] > max_value[idx])
      max_value[idx] = src[i * width + idx];
  }
  __syncthreads();
  if (idx == 0)
  {
    for (int i = 0; i < width; i++)
    {
      if (min_value[0] > min_value[i])
        min_value[0] = min_value[i];
    }
    // printf("min value: %f\n", min_value[0]);
  }
  if (idx == 1)
  {
    for (int i = 0; i < width; i++)
    {
      if (max_value[0] < max_value[i])
        max_value[0] = max_value[i];
    }
    // printf("max value: %f\n", max_value[0]);
  }
  __syncthreads();
  //int x = 50;
  //int y = 50;
  for (int i = 0; i < height; ++i)
  {
    // Calculate new coordinates
    // if ((idx == 5)||(idx == 10)) {
    //   printf("Idx = %d, min_value = %f, max_value = %f\n", idx, min_value[0], max_value[0]);
    // }
    if (((src[i * width + idx + 0] - min_value[0])/2.0) > 255.0)
      dst[i * width + idx + 0] = 255.0;
    else
      dst[i * width + idx + 0] = (src[i * width + idx + 0] - min_value[0])/2.0; // ) / (max_value[0] - min_value[0]) * 255.0;
  }
}

__global__ void gpuNormalizeToInt16_SCALE(uint16_t *dst, float *src, const int width, const int height)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ float min_value[1920];
  __shared__ float max_value[1920];
  if (idx >= width)
  {
    return;
  }
  min_value[idx] = 70000.0;
  max_value[idx] = -70000.0;

  for (int i = 0; i < height; ++i)
  {
    if (src[i * width + idx] < min_value[idx])
      min_value[idx] = src[i * width + idx];
    if (src[i * width + idx] > max_value[idx])
      max_value[idx] = src[i * width + idx];
  }
  __syncthreads();
  if (idx == 0)
  {
    for (int i = 0; i < width; i++)
    {
      if (min_value[0] > min_value[i])
        min_value[0] = min_value[i];
    }
    // printf("min value: %f\n", min_value[0]);
  }
  if (idx == 1)
  {
    for (int i = 0; i < width; i++)
    {
      if (max_value[0] < max_value[i])
        max_value[0] = max_value[i];
    }
    // printf("max value: %f\n", max_value[0]);
  }
  __syncthreads();
  for (int i = 0; i < height; ++i)
  {
    dst[i * width * 4 + idx * 4 + 0] = (uint16_t)((src[i * width + idx] - min_value[0]) / (max_value[0] - min_value[0]));
    dst[i * width * 4 + idx * 4 + 1] = (uint16_t)((src[i * width + idx] - min_value[0]) / (max_value[0] - min_value[0]));
    dst[i * width * 4 + idx * 4 + 2] = (uint16_t)((src[i * width + idx] - min_value[0]) / (max_value[0] - min_value[0]));
    dst[i * width * 4 + idx * 4 + 3] = 255 * 255;
  }
}

__global__ void gpuNormalizeToFloat(float *dst, uint16_t *src, const int width, const int height)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= width)
  {
    return;
  }
  for (int i = 0; i < height; ++i)
  {
    dst[i * width + idx] = (float)(src[i * width * 4 + idx * 4 + 0]);
  }
}

__global__ void gpuSobelToF(float *dst_mag, float *dst_phase, float *src,
                            unsigned int width, unsigned int height)
{
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x; // image-pixel
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

  float result_x = 0;
  float result_y = 0;

  if ((idx_x != 0) && (idx_x != width - 1) && (idx_y != 0) && (idx_y != height - 1))
  {
    // sobel
    result_x += 1 * src[(idx_x - 1) + (idx_y - 1) * width];
    result_x += 2 * src[(idx_x - 1) + (idx_y)*width];
    result_x += 1 * src[(idx_x - 1) + (idx_y + 1) * width];
    result_x += -1 * src[(idx_x + 1) + (idx_y - 1) * width];
    result_x += -2 * src[(idx_x + 1) + (idx_y)*width];
    result_x += -1 * src[(idx_x + 1) + (idx_y + 1) * width];
    //
    result_y += 1 * src[(idx_x - 1) + (idx_y - 1) * width];
    result_y += 2 * src[(idx_x) + (idx_y - 1) * width];
    result_y += 1 * src[(idx_x + 1) + (idx_y - 1) * width];
    result_y += -1 * src[(idx_x - 1) + (idx_y + 1) * width];
    result_y += -2 * src[(idx_x) + (idx_y + 1) * width];
    result_y += -1 * src[(idx_x + 1) + (idx_y + 1) * width];
  }
  //result_x = result_x/9.0f;
  float result = (sqrt(result_x * result_x + result_y * result_y));

  // printf("result written on x = %d, y = %d", idx_x, idx_y);
  dst_mag[idx_x + (idx_y)*width] = result;
  if (dst_phase == NULL)
    return;
  if (result_x == 0.0)
    result_x = 0.000000001;
  dst_phase[idx_x + (idx_y)*width] = tan(result_y / result_x);
}

__global__ void gpuMaxfilterToF(float *dst, float *src,
                                unsigned int width, unsigned int height)
{
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x; // image-pixel
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

  float result = -65000.0;

  if ((idx_x != 0) && (idx_x != width - 1) && (idx_y != 0) && (idx_y != height - 1))
  {
    // sobel
    if (result < src[(idx_x - 1) + (idx_y - 1) * width])
      result = src[(idx_x - 1) + (idx_y - 1) * width];
    if (result < src[(idx_x - 1) + (idx_y + 0) * width])
      result = src[(idx_x - 1) + (idx_y + 0) * width];
    if (result < src[(idx_x - 1) + (idx_y + 1) * width])
      result = src[(idx_x - 1) + (idx_y + 1) * width];
    if (result < src[(idx_x + 0) + (idx_y - 1) * width])
      result = src[(idx_x + 0) + (idx_y - 1) * width];
    if (result < src[(idx_x + 0) + (idx_y + 0) * width])
      result = src[(idx_x + 0) + (idx_y + 0) * width];
    if (result < src[(idx_x + 0) + (idx_y + 1) * width])
      result = src[(idx_x + 0) + (idx_y + 1) * width];
    if (result < src[(idx_x + 1) + (idx_y - 1) * width])
      result = src[(idx_x + 1) + (idx_y - 1) * width];
    if (result < src[(idx_x + 1) + (idx_y + 0) * width])
      result = src[(idx_x + 1) + (idx_y + 0) * width];
    if (result < src[(idx_x + 1) + (idx_y + 1) * width])
      result = src[(idx_x + 1) + (idx_y + 1) * width];
  }

  dst[idx_x + (idx_y)*width] = result;
}

__global__ void gpuMinfilterToF(float *dst, float *src,
                                unsigned int width, unsigned int height)
{
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x; // image-pixel
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

  float result = 65000.0;

  if ((idx_x != 0) && (idx_x != width - 1) && (idx_y != 0) && (idx_y != height - 1))
  {
    // sobel
    if (result > src[(idx_x - 1) + (idx_y - 1) * width])
      result = src[(idx_x - 1) + (idx_y - 1) * width];
    if (result > src[(idx_x - 1) + (idx_y + 0) * width])
      result = src[(idx_x - 1) + (idx_y + 0) * width];
    if (result > src[(idx_x - 1) + (idx_y + 1) * width])
      result = src[(idx_x - 1) + (idx_y + 1) * width];
    if (result > src[(idx_x + 0) + (idx_y - 1) * width])
      result = src[(idx_x + 0) + (idx_y - 1) * width];
    if (result > src[(idx_x + 0) + (idx_y + 0) * width])
      result = src[(idx_x + 0) + (idx_y + 0) * width];
    if (result > src[(idx_x + 0) + (idx_y + 1) * width])
      result = src[(idx_x + 0) + (idx_y + 1) * width];
    if (result > src[(idx_x + 1) + (idx_y - 1) * width])
      result = src[(idx_x + 1) + (idx_y - 1) * width];
    if (result > src[(idx_x + 1) + (idx_y + 0) * width])
      result = src[(idx_x + 1) + (idx_y + 0) * width];
    if (result > src[(idx_x + 1) + (idx_y + 1) * width])
      result = src[(idx_x + 1) + (idx_y + 1) * width];
  }

  dst[idx_x + (idx_y)*width] = result;
}

__global__ void gpuMeanfilterToF(float *dst_mag, float *src,
                                 unsigned int width, unsigned int height)
{
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x; // image-pixel
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

  float result = 0;

  if ((idx_x != 0) && (idx_x != width - 1) && (idx_y != 0) && (idx_y != height - 1))
  {
    // sobel
    result += src[(idx_x - 1) + (idx_y - 1) * width];
    result += src[(idx_x - 1) + (idx_y)*width];
    result += src[(idx_x - 1) + (idx_y + 1) * width];
    result += src[(idx_x - 0) + (idx_y - 1) * width];
    result += src[(idx_x - 0) + (idx_y)*width];
    result += src[(idx_x - 0) + (idx_y + 1) * width];
    result += src[(idx_x + 1) + (idx_y - 1) * width];
    result += src[(idx_x + 1) + (idx_y)*width];
    result += src[(idx_x + 1) + (idx_y + 1) * width];
  }
  dst_mag[idx_x + (idx_y)*width] = result / 9.0;
}

__global__ void gpuMedianfilterToF(float *dst_mag, float *src,
                                   unsigned int width, unsigned int height)
{
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x; // image-pixel
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

  int idx_top, idx_bottom, idx_left, idx_right;
  if (idx_x == 0)
    idx_left = 1; //mirror at edge
  else
    idx_left = idx_x - 1;
  if (idx_x == width - 1)
    idx_right = width - 2; //mirror at edge
  else
    idx_right = idx_x + 1;
  if (idx_y == 0)
    idx_top = 1; //mirror at edge
  else
    idx_top = idx_y - 1;
  if (idx_y == height - 1)
    idx_bottom = height - 2; //mirror at edge
  else
    idx_bottom = idx_y + 1;

  float values[9];
  values[0] = src[(idx_left) + (idx_top)*width];
  values[1] = src[(idx_left) + (idx_y)*width];
  values[2] = src[(idx_left) + (idx_bottom)*width];
  values[3] = src[(idx_x) + (idx_top)*width];
  values[4] = src[(idx_x) + (idx_y)*width];
  values[5] = src[(idx_x) + (idx_bottom)*width];
  values[6] = src[(idx_right) + (idx_top)*width];
  values[7] = src[(idx_right) + (idx_y)*width];
  values[8] = src[(idx_right) + (idx_bottom)*width];
  bool flags[9];
  float min_val = values[0];
  float max_val = values[0];
  int min_idx = 0;
  int max_idx = 0;
  for (int i = 0; i < 9; i++)
  {
    flags[i] = true; // zero all the flags, currently all are possible median
    if (min_val > values[i])
    {
      min_val = values[i];
      min_idx = i;
    }
    if (max_val < values[i])
    {
      max_val = values[i];
      max_idx = i;
    }
  }
  flags[min_idx] = false;
  flags[max_idx] = false;
  // 7 values left
  min_val = max_val;
  max_val = 0;
  for (int i = 0; i < 9; i++)
  {
    if ((min_val > values[i]) && (flags[i]))
    {
      min_val = values[i];
      min_idx = i;
    }
    if ((max_val < values[i]) && (flags[i]))
    {
      max_val = values[i];
      max_idx = i;
    }
  }
  flags[min_idx] = false;
  flags[max_idx] = false;
  // 5 values left
  min_val = max_val;
  max_val = 0;
  for (int i = 0; i < 9; i++)
  {
    if ((min_val > values[i]) && (flags[i]))
    {
      min_val = values[i];
      min_idx = i;
    }
    if ((max_val < values[i]) && (flags[i]))
    {
      max_val = values[i];
      max_idx = i;
    }
  }
  flags[min_idx] = false;
  flags[max_idx] = false;
  // 3 values left
  min_val = max_val;
  max_val = 0;
  for (int i = 0; i < 9; i++)
  {
    if ((min_val > values[i]) && (flags[i]))
    {
      min_val = values[i];
      min_idx = i;
    }
    if ((max_val < values[i]) && (flags[i]))
    {
      max_val = values[i];
      max_idx = i;
    }
  }
  float median_val;
  for (int i = 0; i < 9; i++)
  {
    if ((flags[i]))
    {
      median_val = values[i];
    }
    
  }
  dst_mag[idx_x + (idx_y)*width] = median_val;
}

__global__ void gpuFillAreaToF(float *mask, float *src,
                               unsigned int width, unsigned int height, int seed_x, int seed_y, float thresh)
{
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x; // image-pixel
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

  float seed_value = src[(seed_x) + (seed_y)*width];

  if ((src[(idx_x) + (idx_y)*width] >= seed_value - thresh) && ((src[(idx_x) + (idx_y)*width] <= seed_value + thresh)))
  {
    mask[(idx_x) + (idx_y)*width] = 256.0 * 256.0 - 1;
  }
  else
  {
    mask[(idx_x) + (idx_y)*width] = 0.0;
  }
}
__global__ void gpuSharpenImageToGrayscale(unsigned char *src, float *dst,
                                           unsigned int width, unsigned int height, float amount)
{
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x; // image-pixel
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;

  float result_x = 0;
  float result_y = 0;

  if ((idx_x != 0) && (idx_x != width - 1) && (idx_y != 0) && (idx_y != height - 1))
  {
    // sobel
    result_x += 1 * src[(idx_x - 1) * 2 + 1 + (idx_y - 1) * width * 2];
    result_x += 2 * src[(idx_x - 1) * 2 + 1 + (idx_y)*width * 2];
    result_x += 1 * src[(idx_x - 1) * 2 + 1 + (idx_y + 1) * width * 2];
    result_x += -1 * src[(idx_x + 1) * 2 + 1 + (idx_y - 1) * width * 2];
    result_x += -2 * src[(idx_x + 1) * 2 + 1 + (idx_y)*width * 2];
    result_x += -1 * src[(idx_x + 1) * 2 + 1 + (idx_y + 1) * width * 2];
    //
    result_y += 1 * src[(idx_x - 1) * 2 + 1 + (idx_y - 1) * width * 2];
    result_y += 2 * src[(idx_x)*2 + 1 + (idx_y - 1) * width * 2];
    result_y += 1 * src[(idx_x + 1) * 2 + 1 + (idx_y - 1) * width * 2];
    result_y += -1 * src[(idx_x - 1) * 2 + 1 + (idx_y + 1) * width * 2];
    result_y += -2 * src[(idx_x)*2 + 1 + (idx_y + 1) * width * 2];
    result_y += -1 * src[(idx_x + 1) * 2 + 1 + (idx_y + 1) * width * 2];
    //for (int i = -1; i<=1; i++) {
    //  for (int j = -1; j<=1; j++) {
    //    result_x += src[(idx_x+i)*2+1+(idx_y+j)*width*2];
    //  }
    //}
  }
  //result_x = result_x/9.0f;
  float result = (src[idx_x * 2 + 1 + (idx_y)*width * 2] * (1 - amount)) + (amount * sqrt(result_x * result_x + result_y * result_y));
  result = (result - 80.0f) / 130.0f * 255.0f;
  if (result < 0)
    result = 0;
  if (result > 255)
    result = 255;

  // printf("result written on x = %d, y = %d", idx_x, idx_y);
  dst[idx_x + (idx_y)*width] = result;
}

__global__ void LaplaceMultiMem(float *d_Image, float *d_Result, int width, int pitch, int height, int octave)
{
  __shared__ float buff[(LAPLACE_W + 2 * LAPLACE_R) * LAPLACE_S];
  const int tx = threadIdx.x;
  const int xp = blockIdx.x * LAPLACE_W + tx;
  const int yp = blockIdx.y;
  float *data = d_Image + max(min(xp - LAPLACE_R, width - 1), 0);
  float temp[2 * LAPLACE_R + 1], kern[LAPLACE_S][LAPLACE_R + 1];
  if (xp < (width + 2 * LAPLACE_R))
  {
    for (int i = 0; i <= 2 * LAPLACE_R; i++)
      temp[i] = data[max(0, min(yp + i - LAPLACE_R, height - 1)) * pitch];
    for (int scale = 0; scale < LAPLACE_S; scale++)
    {
      float *buf = buff + (LAPLACE_W + 2 * LAPLACE_R) * scale;
      float *kernel = d_LaplaceKernel + octave * 12 * 16 + scale * 16;
      for (int i = 0; i <= LAPLACE_R; i++)
        kern[scale][i] = kernel[i];
      float sum = kern[scale][0] * temp[LAPLACE_R];
#pragma unroll
      for (int j = 1; j <= LAPLACE_R; j++)
        sum += kern[scale][j] * (temp[LAPLACE_R - j] + temp[LAPLACE_R + j]);
      buf[tx] = sum;
    }
  }
  __syncthreads();
  if (tx < LAPLACE_W && xp < width)
  {
    int scale = 0;
    float oldRes = kern[scale][0] * buff[tx + LAPLACE_R];
#pragma unroll
    for (int j = 1; j <= LAPLACE_R; j++)
      oldRes += kern[scale][j] * (buff[tx + LAPLACE_R - j] + buff[tx + LAPLACE_R + j]);
    for (int scale = 1; scale < LAPLACE_S; scale++)
    {
      float *buf = buff + (LAPLACE_W + 2 * LAPLACE_R) * scale;
      float res = kern[scale][0] * buf[tx + LAPLACE_R];
#pragma unroll
      for (int j = 1; j <= LAPLACE_R; j++)
        res += kern[scale][j] * (buf[tx + LAPLACE_R - j] + buf[tx + LAPLACE_R + j]);
      d_Result[(scale - 1) * height * pitch + yp * pitch + xp] = res - oldRes;
      oldRes = res;
    }
  }
}

__global__ void FindPointsMultiNew(float *d_Data0, SiftPoint *d_Sift, int width, int pitch, int height, float subsampling, float lowestScale, float thresh, float factor, float edgeLimit, int octave)
{
#define MEMWID (MINMAX_W + 2)
  __shared__ unsigned short points[2 * MEMWID];

  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
  {
    atomicMax(&d_PointCounter[2 * octave + 0], d_PointCounter[2 * octave - 1]);
    atomicMax(&d_PointCounter[2 * octave + 1], d_PointCounter[2 * octave - 1]);
  }
  int tx = threadIdx.x;
  int block = blockIdx.x / NUM_SCALES;
  int scale = blockIdx.x - NUM_SCALES * block;
  int minx = block * MINMAX_W;
  int maxx = min(minx + MINMAX_W, width);
  int xpos = minx + tx;
  int size = pitch * height;
  int ptr = size * scale + max(min(xpos - 1, width - 1), 0);

  int yloops = min(height - MINMAX_H * blockIdx.y, MINMAX_H);
  float maxv = 0.0f;
  for (int y = 0; y < yloops; y++)
  {
    int ypos = MINMAX_H * blockIdx.y + y;
    int yptr1 = ptr + ypos * pitch;
    float val = d_Data0[yptr1 + 1 * size];
    maxv = fmaxf(maxv, fabs(val));
  }
  //if (tx==0) printf("XXX1\n");
  if (!__any_sync(0xffffffff, maxv > thresh))
    return;
  //if (tx==0) printf("XXX2\n");

  int ptbits = 0;
  for (int y = 0; y < yloops; y++)
  {

    int ypos = MINMAX_H * blockIdx.y + y;
    int yptr1 = ptr + ypos * pitch;
    float d11 = d_Data0[yptr1 + 1 * size];
    if (__any_sync(0xffffffff, fabs(d11) > thresh))
    {

      int yptr0 = ptr + max(0, ypos - 1) * pitch;
      int yptr2 = ptr + min(height - 1, ypos + 1) * pitch;
      float d01 = d_Data0[yptr1];
      float d10 = d_Data0[yptr0 + 1 * size];
      float d12 = d_Data0[yptr2 + 1 * size];
      float d21 = d_Data0[yptr1 + 2 * size];

      float d00 = d_Data0[yptr0];
      float d02 = d_Data0[yptr2];
      float ymin1 = fminf(fminf(d00, d01), d02);
      float ymax1 = fmaxf(fmaxf(d00, d01), d02);
      float d20 = d_Data0[yptr0 + 2 * size];
      float d22 = d_Data0[yptr2 + 2 * size];
      float ymin3 = fminf(fminf(d20, d21), d22);
      float ymax3 = fmaxf(fmaxf(d20, d21), d22);
      float ymin2 = fminf(fminf(ymin1, fminf(fminf(d10, d12), d11)), ymin3);
      float ymax2 = fmaxf(fmaxf(ymax1, fmaxf(fmaxf(d10, d12), d11)), ymax3);

      float nmin2 = fminf(ShiftUp(ymin2, 1), ShiftDown(ymin2, 1));
      float nmax2 = fmaxf(ShiftUp(ymax2, 1), ShiftDown(ymax2, 1));
      float minv = fminf(fminf(nmin2, ymin1), ymin3);
      minv = fminf(fminf(minv, d10), d12);
      float maxv = fmaxf(fmaxf(nmax2, ymax1), ymax3);
      maxv = fmaxf(fmaxf(maxv, d10), d12);

      if (tx > 0 && tx < MINMAX_W + 1 && xpos <= maxx)
        ptbits |= ((d11 < fminf(-thresh, minv)) | (d11 > fmaxf(thresh, maxv))) << y;
    }
  }

  unsigned int totbits = __popc(ptbits);
  unsigned int numbits = totbits;
  for (int d = 1; d < 32; d <<= 1)
  {
    unsigned int num = ShiftUp(totbits, d);
    if (tx >= d)
      totbits += num;
  }
  int pos = totbits - numbits;
  for (int y = 0; y < yloops; y++)
  {
    int ypos = MINMAX_H * blockIdx.y + y;
    if (ptbits & (1 << y) && pos < MEMWID)
    {
      points[2 * pos + 0] = xpos - 1;
      points[2 * pos + 1] = ypos;
      pos++;
    }
  }

  totbits = Shuffle(totbits, 31);
  if (tx < totbits)
  {
    int xpos = points[2 * tx + 0];
    int ypos = points[2 * tx + 1];
    int ptr = xpos + (ypos + (scale + 1) * height) * pitch;
    float val = d_Data0[ptr];
    float *data1 = &d_Data0[ptr];
    float dxx = 2.0f * val - data1[-1] - data1[1];
    float dyy = 2.0f * val - data1[-pitch] - data1[pitch];
    float dxy = 0.25f * (data1[+pitch + 1] + data1[-pitch - 1] - data1[-pitch + 1] - data1[+pitch - 1]);
    float tra = dxx + dyy;
    float det = dxx * dyy - dxy * dxy;
    if (tra * tra < edgeLimit * det)
    {
      float edge = __fdividef(tra * tra, det);
      float dx = 0.5f * (data1[1] - data1[-1]);
      float dy = 0.5f * (data1[pitch] - data1[-pitch]);
      float *data0 = d_Data0 + ptr - height * pitch;
      float *data2 = d_Data0 + ptr + height * pitch;
      float ds = 0.5f * (data0[0] - data2[0]);
      float dss = 2.0f * val - data2[0] - data0[0];
      float dxs = 0.25f * (data2[1] + data0[-1] - data0[1] - data2[-1]);
      float dys = 0.25f * (data2[pitch] + data0[-pitch] - data2[-pitch] - data0[pitch]);
      float idxx = dyy * dss - dys * dys;
      float idxy = dys * dxs - dxy * dss;
      float idxs = dxy * dys - dyy * dxs;
      float idet = __fdividef(1.0f, idxx * dxx + idxy * dxy + idxs * dxs);
      float idyy = dxx * dss - dxs * dxs;
      float idys = dxy * dxs - dxx * dys;
      float idss = dxx * dyy - dxy * dxy;
      float pdx = idet * (idxx * dx + idxy * dy + idxs * ds);
      float pdy = idet * (idxy * dx + idyy * dy + idys * ds);
      float pds = idet * (idxs * dx + idys * dy + idss * ds);
      if (pdx < -0.5f || pdx > 0.5f || pdy < -0.5f || pdy > 0.5f || pds < -0.5f || pds > 0.5f)
      {
        pdx = __fdividef(dx, dxx);
        pdy = __fdividef(dy, dyy);
        pds = __fdividef(ds, dss);
      }
      float dval = 0.5f * (dx * pdx + dy * pdy + ds * pds);
      int maxPts = d_MaxNumPoints;
      float sc = powf(2.0f, (float)scale / NUM_SCALES) * exp2f(pds * factor);
      if (sc >= lowestScale)
      {
        atomicMax(&d_PointCounter[2 * octave + 0], d_PointCounter[2 * octave - 1]);
        unsigned int idx = atomicInc(&d_PointCounter[2 * octave + 0], 0x7fffffff);
        idx = (idx >= maxPts ? maxPts - 1 : idx);
        d_Sift[idx].xpos = xpos + pdx;
        d_Sift[idx].ypos = ypos + pdy;
        d_Sift[idx].scale = sc;
        d_Sift[idx].sharpness = val + dval;
        d_Sift[idx].edgeness = edge;
        d_Sift[idx].subsampling = subsampling;
      }
    }
  }
}
__global__ void RescalePositionsKernel(SiftPoint *d_sift, int numPts, float scale)
{
  int num = blockIdx.x * blockDim.x + threadIdx.x;
  if (num < numPts)
  {
    d_sift[num].xpos *= scale;
    d_sift[num].ypos *= scale;
    d_sift[num].scale *= scale;
  }
}

// With constant number of blocks
__global__ void ComputeOrientationsCONST(cudaTextureObject_t texObj, SiftPoint *d_Sift, int octave)
{
  __shared__ float hist[64];
  __shared__ float gauss[11];
  const int tx = threadIdx.x;

  int fstPts = min(d_PointCounter[2 * octave - 1], d_MaxNumPoints);
  int totPts = min(d_PointCounter[2 * octave + 0], d_MaxNumPoints);
  for (int bx = blockIdx.x + fstPts; bx < totPts; bx += gridDim.x)
  {

    float i2sigma2 = -1.0f / (2.0f * 1.5f * 1.5f * d_Sift[bx].scale * d_Sift[bx].scale);
    if (tx < 11)
      gauss[tx] = exp(i2sigma2 * (tx - 5) * (tx - 5));
    if (tx < 64)
      hist[tx] = 0.0f;
    __syncthreads();
    float xp = d_Sift[bx].xpos - 4.5f;
    float yp = d_Sift[bx].ypos - 4.5f;
    int yd = tx / 11;
    int xd = tx - yd * 11;
    float xf = xp + xd;
    float yf = yp + yd;
    if (yd < 11)
    {
      float dx = tex2D<float>(texObj, xf + 1.0, yf) - tex2D<float>(texObj, xf - 1.0, yf);
      float dy = tex2D<float>(texObj, xf, yf + 1.0) - tex2D<float>(texObj, xf, yf - 1.0);
      int bin = 16.0f * atan2f(dy, dx) / 3.1416f + 16.5f;
      if (bin > 31)
        bin = 0;
      float grad = sqrtf(dx * dx + dy * dy);
      atomicAdd(&hist[bin], grad * gauss[xd] * gauss[yd]);
    }
    __syncthreads();
    int x1m = (tx >= 1 ? tx - 1 : tx + 31);
    int x1p = (tx <= 30 ? tx + 1 : tx - 31);
    if (tx < 32)
    {
      int x2m = (tx >= 2 ? tx - 2 : tx + 30);
      int x2p = (tx <= 29 ? tx + 2 : tx - 30);
      hist[tx + 32] = 6.0f * hist[tx] + 4.0f * (hist[x1m] + hist[x1p]) + (hist[x2m] + hist[x2p]);
    }
    __syncthreads();
    if (tx < 32)
    {
      float v = hist[32 + tx];
      hist[tx] = (v > hist[32 + x1m] && v >= hist[32 + x1p] ? v : 0.0f);
    }
    __syncthreads();
    if (tx == 0)
    {
      float maxval1 = 0.0;
      float maxval2 = 0.0;
      int i1 = -1;
      int i2 = -1;
      for (int i = 0; i < 32; i++)
      {
        float v = hist[i];
        if (v > maxval1)
        {
          maxval2 = maxval1;
          maxval1 = v;
          i2 = i1;
          i1 = i;
        }
        else if (v > maxval2)
        {
          maxval2 = v;
          i2 = i;
        }
      }
      float val1 = hist[32 + ((i1 + 1) & 31)];
      float val2 = hist[32 + ((i1 + 31) & 31)];
      float peak = i1 + 0.5f * (val1 - val2) / (2.0f * maxval1 - val1 - val2);
      d_Sift[bx].orientation = 11.25f * (peak < 0.0f ? peak + 32.0f : peak);
      atomicMax(&d_PointCounter[2 * octave + 1], d_PointCounter[2 * octave + 0]);
      if (maxval2 > 0.8f * maxval1 && true)
      {
        float val1 = hist[32 + ((i2 + 1) & 31)];
        float val2 = hist[32 + ((i2 + 31) & 31)];
        float peak = i2 + 0.5f * (val1 - val2) / (2.0f * maxval2 - val1 - val2);
        unsigned int idx = atomicInc(&d_PointCounter[2 * octave + 1], 0x7fffffff);
        if (idx < d_MaxNumPoints)
        {
          d_Sift[idx].xpos = d_Sift[bx].xpos;
          d_Sift[idx].ypos = d_Sift[bx].ypos;
          d_Sift[idx].scale = d_Sift[bx].scale;
          d_Sift[idx].sharpness = d_Sift[bx].sharpness;
          d_Sift[idx].edgeness = d_Sift[bx].edgeness;
          d_Sift[idx].orientation = 11.25f * (peak < 0.0f ? peak + 32.0f : peak);
          ;
          d_Sift[idx].subsampling = d_Sift[bx].subsampling;
        }
      }
    }
    __syncthreads();
  }
}

__device__ float FastAtan2(float y, float x)
{
  float absx = abs(x);
  float absy = abs(y);
  float a = __fdiv_rn(min(absx, absy), max(absx, absy));
  float s = a * a;
  float r = ((-0.0464964749f * s + 0.15931422f) * s - 0.327622764f) * s * a + a;
  r = (absy > absx ? 1.57079637f - r : r);
  r = (x < 0 ? 3.14159274f - r : r);
  r = (y < 0 ? -r : r);
  return r;
}

__global__ void ExtractSiftDescriptorsCONSTNew(cudaTextureObject_t texObj, SiftPoint *d_sift, float subsampling, int octave)
{
  __shared__ float gauss[16];
  __shared__ float buffer[128];
  __shared__ float sums[4];

  const int tx = threadIdx.x; // 0 -> 16
  const int ty = threadIdx.y; // 0 -> 8
  const int idx = ty * 16 + tx;
  if (ty == 0)
    gauss[tx] = __expf(-(tx - 7.5f) * (tx - 7.5f) / 128.0f);

  int fstPts = min(d_PointCounter[2 * octave - 1], d_MaxNumPoints);
  int totPts = min(d_PointCounter[2 * octave + 1], d_MaxNumPoints);
  //if (tx==0 && ty==0)
  //  printf("%d %d %d %d\n", octave, fstPts, min(d_PointCounter[2*octave], d_MaxNumPoints), totPts);
  for (int bx = blockIdx.x + fstPts; bx < totPts; bx += gridDim.x)
  {

    buffer[idx] = 0.0f;
    __syncthreads();

    // Compute angles and gradients
    float theta = 2.0f * 3.1415f / 360.0f * d_sift[bx].orientation;
    float sina = __sinf(theta); // cosa -sina
    float cosa = __cosf(theta); // sina  cosa
    float scale = 12.0f / 16.0f * d_sift[bx].scale;
    float ssina = scale * sina;
    float scosa = scale * cosa;

    for (int y = ty; y < 16; y += 8)
    {
      float xpos = d_sift[bx].xpos + (tx - 7.5f) * scosa - (y - 7.5f) * ssina + 0.5f;
      float ypos = d_sift[bx].ypos + (tx - 7.5f) * ssina + (y - 7.5f) * scosa + 0.5f;
      float dx = tex2D<float>(texObj, xpos + cosa, ypos + sina) -
                 tex2D<float>(texObj, xpos - cosa, ypos - sina);
      float dy = tex2D<float>(texObj, xpos - sina, ypos + cosa) -
                 tex2D<float>(texObj, xpos + sina, ypos - cosa);
      float grad = gauss[y] * gauss[tx] * __fsqrt_rn(dx * dx + dy * dy);
      float angf = 4.0f / 3.1415f * FastAtan2(dy, dx) + 4.0f;

      int hori = (tx + 2) / 4 - 1; // Convert from (tx,y,angle) to bins
      float horf = (tx - 1.5f) / 4.0f - hori;
      float ihorf = 1.0f - horf;
      int veri = (y + 2) / 4 - 1;
      float verf = (y - 1.5f) / 4.0f - veri;
      float iverf = 1.0f - verf;
      int angi = angf;
      int angp = (angi < 7 ? angi + 1 : 0);
      angf -= angi;
      float iangf = 1.0f - angf;

      int hist = 8 * (4 * veri + hori); // Each gradient measure is interpolated
      int p1 = angi + hist;             // in angles, xpos and ypos -> 8 stores
      int p2 = angp + hist;
      if (tx >= 2)
      {
        float grad1 = ihorf * grad;
        if (y >= 2)
        { // Upper left
          float grad2 = iverf * grad1;
          atomicAdd(buffer + p1, iangf * grad2);
          atomicAdd(buffer + p2, angf * grad2);
        }
        if (y <= 13)
        { // Lower left
          float grad2 = verf * grad1;
          atomicAdd(buffer + p1 + 32, iangf * grad2);
          atomicAdd(buffer + p2 + 32, angf * grad2);
        }
      }
      if (tx <= 13)
      {
        float grad1 = horf * grad;
        if (y >= 2)
        { // Upper right
          float grad2 = iverf * grad1;
          atomicAdd(buffer + p1 + 8, iangf * grad2);
          atomicAdd(buffer + p2 + 8, angf * grad2);
        }
        if (y <= 13)
        { // Lower right
          float grad2 = verf * grad1;
          atomicAdd(buffer + p1 + 40, iangf * grad2);
          atomicAdd(buffer + p2 + 40, angf * grad2);
        }
      }
    }
    __syncthreads();

    // Normalize twice and suppress peaks first time
    float sum = buffer[idx] * buffer[idx];
    for (int i = 16; i > 0; i /= 2)
      sum += ShiftDown(sum, i);
    if ((idx & 31) == 0)
      sums[idx / 32] = sum;
    __syncthreads();
    float tsum1 = sums[0] + sums[1] + sums[2] + sums[3];
    tsum1 = min(buffer[idx] * rsqrtf(tsum1), 0.2f);

    sum = tsum1 * tsum1;
    for (int i = 16; i > 0; i /= 2)
      sum += ShiftDown(sum, i);
    if ((idx & 31) == 0)
      sums[idx / 32] = sum;
    __syncthreads();

    float tsum2 = sums[0] + sums[1] + sums[2] + sums[3];
    float *desc = d_sift[bx].data;
    desc[idx] = tsum1 * rsqrtf(tsum2);
    if (idx == 0)
    {
      d_sift[bx].xpos *= subsampling;
      d_sift[bx].ypos *= subsampling;
      d_sift[bx].scale *= subsampling;
    }
    __syncthreads();
  }
}

__global__ void CleanMatches(SiftPoint *sift1, int numPts1)
{
  const int p1 = min(blockIdx.x * 64 + threadIdx.x, numPts1 - 1);
  sift1[p1].score = 0.0f;
}

#define M7W 32
#define M7H 32
#define M7R 4
#define NRX 2
#define NDIM 128

__global__ void FindMaxCorr10(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2)
{
  __shared__ float4 buffer1[M7W * NDIM / 4];
  __shared__ float4 buffer2[M7H * NDIM / 4];
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bp1 = M7W * blockIdx.x;
  for (int j = ty; j < M7W; j += M7H / M7R)
  {
    int p1 = min(bp1 + j, numPts1 - 1);
    for (int d = tx; d < NDIM / 4; d += M7W)
      buffer1[j * NDIM / 4 + (d + j) % (NDIM / 4)] = ((float4 *)&sift1[p1].data)[d];
  }

  float max_score[NRX];
  float sec_score[NRX];
  int index[NRX];
  for (int i = 0; i < NRX; i++)
  {
    max_score[i] = 0.0f;
    sec_score[i] = 0.0f;
    index[i] = -1;
  }
  int idx = ty * M7W + tx;
  int ix = idx % (M7W / NRX);
  int iy = idx / (M7W / NRX);
  for (int bp2 = 0; bp2 < numPts2 - M7H + 1; bp2 += M7H)
  {
    for (int j = ty; j < M7H; j += M7H / M7R)
    {
      int p2 = min(bp2 + j, numPts2 - 1);
      for (int d = tx; d < NDIM / 4; d += M7W)
        buffer2[j * NDIM / 4 + d] = ((float4 *)&sift2[p2].data)[d];
    }
    __syncthreads();

    if (idx < M7W * M7H / M7R / NRX)
    {
      float score[M7R][NRX];
      for (int dy = 0; dy < M7R; dy++)
        for (int i = 0; i < NRX; i++)
          score[dy][i] = 0.0f;
      for (int d = 0; d < NDIM / 4; d++)
      {
        float4 v1[NRX];
        for (int i = 0; i < NRX; i++)
          v1[i] = buffer1[((M7W / NRX) * i + ix) * NDIM / 4 + (d + (M7W / NRX) * i + ix) % (NDIM / 4)];
        for (int dy = 0; dy < M7R; dy++)
        {
          float4 v2 = buffer2[(M7R * iy + dy) * (NDIM / 4) + d];
          for (int i = 0; i < NRX; i++)
          {
            score[dy][i] += v1[i].x * v2.x;
            score[dy][i] += v1[i].y * v2.y;
            score[dy][i] += v1[i].z * v2.z;
            score[dy][i] += v1[i].w * v2.w;
          }
        }
      }
      for (int dy = 0; dy < M7R; dy++)
      {
        for (int i = 0; i < NRX; i++)
        {
          if (score[dy][i] > max_score[i])
          {
            sec_score[i] = max_score[i];
            max_score[i] = score[dy][i];
            index[i] = min(bp2 + M7R * iy + dy, numPts2 - 1);
          }
          else if (score[dy][i] > sec_score[i])
            sec_score[i] = score[dy][i];
        }
      }
    }
    __syncthreads();
  }

  float *scores1 = (float *)buffer1;
  float *scores2 = &scores1[M7W * M7H / M7R];
  int *indices = (int *)&scores2[M7W * M7H / M7R];
  if (idx < M7W * M7H / M7R / NRX)
  {
    for (int i = 0; i < NRX; i++)
    {
      scores1[iy * M7W + (M7W / NRX) * i + ix] = max_score[i];
      scores2[iy * M7W + (M7W / NRX) * i + ix] = sec_score[i];
      indices[iy * M7W + (M7W / NRX) * i + ix] = index[i];
    }
  }
  __syncthreads();

  if (ty == 0)
  {
    float max_score = scores1[tx];
    float sec_score = scores2[tx];
    int index = indices[tx];
    for (int y = 0; y < M7H / M7R; y++)
      if (index != indices[y * M7W + tx])
      {
        if (scores1[y * M7W + tx] > max_score)
        {
          sec_score = max(max_score, sec_score);
          max_score = scores1[y * M7W + tx];
          index = indices[y * M7W + tx];
        }
        else if (scores1[y * M7W + tx] > sec_score)
          sec_score = scores1[y * M7W + tx];
        if (scores2[y * M7W + tx] > sec_score)
          sec_score = scores2[y * M7W + tx];
      }
    sift1[bp1 + tx].score = max_score;
    sift1[bp1 + tx].match = index;
    sift1[bp1 + tx].match_xpos = sift2[index].xpos;
    sift1[bp1 + tx].match_ypos = sift2[index].ypos;
    sift1[bp1 + tx].ambiguity = sec_score / (max_score + 1e-6f);
    sift1[bp1 + tx].match_x_3d = sift2[index].x_3d;
    sift1[bp1 + tx].match_y_3d = sift2[index].y_3d;
    sift1[bp1 + tx].match_z_3d = sift2[index].z_3d;
    sift1[bp1 + tx].match_distance = sift2[index].distance;
    sift1[bp1 + tx].ambiguity = sec_score / (max_score + 1e-6f);
  }
}

#define FMC_GH 512
#define FMC_BW 32
#define FMC_BH 32
#define FMC_BD 16
#define FMC_TW 1
#define FMC_TH 4
#define FMC_NW (FMC_BW / FMC_TW) //  32
#define FMC_NH (FMC_BH / FMC_TH) //   8
#define FMC_NT (FMC_NW * FMC_NH) // 256 = 8 warps

__device__ volatile int lock = 0;

__global__ void FindMaxCorr9(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2)
{
  __shared__ float4 siftParts1[FMC_BW * FMC_BD]; // 4*32*8 = 1024
  __shared__ float4 siftParts2[FMC_BH * FMC_BD]; // 4*32*8 = 1024
  //__shared__ float blksums[FMC_BW*FMC_BH];     // 32*32  = 1024
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int idx = ty * FMC_NW + tx;
  float4 *pts1 = 0, *pts2 = 0;
  if (idx < FMC_BW)
  {
    const int p1l = min(blockIdx.x * FMC_BW + idx, numPts1 - 1);
    pts1 = (float4 *)sift1[p1l].data;
  }
  float maxScore = -1.0f;
  float maxScor2 = -1.0f;
  int maxIndex = 0;
  for (int k = 0; k < min(FMC_GH, numPts2 - FMC_BH + 1); k += FMC_BH)
  {
    if (idx < FMC_BH)
    {
      const int p2l = min(blockIdx.y * FMC_GH + k + idx, numPts2 - 1);
      pts2 = (float4 *)sift2[p2l].data;
    }
    float sums[FMC_TW * FMC_TH];
    for (int i = 0; i < FMC_TW * FMC_TH; i++)
      sums[i] = 0.0f;

    if (idx < FMC_BW)
      for (int i = 0; i < FMC_BD / 2; i++)
        siftParts1[(i + 0) * FMC_BW + idx] = pts1[0 + i];
    if (idx < FMC_BH)
      for (int i = 0; i < FMC_BD / 2; i++)
        siftParts2[(i + 0) * FMC_BH + idx] = pts2[0 + i];
    __syncthreads();

    int b = FMC_BD / 2;
    for (int d = FMC_BD / 2; d < 32; d += FMC_BD / 2)
    {
      if (idx < FMC_BW)
        for (int i = 0; i < FMC_BD / 2; i++)
          siftParts1[(i + b) * FMC_BW + idx] = pts1[d + i];
      if (idx < FMC_BH)
        for (int i = 0; i < FMC_BD / 2; i++)
          siftParts2[(i + b) * FMC_BH + idx] = pts2[d + i];

      b ^= FMC_BD / 2;
      for (int i = 0; i < FMC_BD / 2; i++)
      {
        float4 v1[FMC_TW];
        for (int ix = 0; ix < FMC_TW; ix++)
          v1[ix] = siftParts1[(i + b) * FMC_BW + (tx * FMC_TW + ix)];
        for (int iy = 0; iy < FMC_TH; iy++)
        {
          float4 v2 = siftParts2[(i + b) * FMC_BH + (ty * FMC_TH + iy)];
          for (int ix = 0; ix < FMC_TW; ix++)
          {
            sums[iy * FMC_TW + ix] += v1[ix].x * v2.x;
            sums[iy * FMC_TW + ix] += v1[ix].y * v2.y;
            sums[iy * FMC_TW + ix] += v1[ix].z * v2.z;
            sums[iy * FMC_TW + ix] += v1[ix].w * v2.w;
          }
        }
      }
      __syncthreads();
    }

    b ^= FMC_BD / 2;
    for (int i = 0; i < FMC_BD / 2; i++)
    {
      float4 v1[FMC_TW];
      for (int ix = 0; ix < FMC_TW; ix++)
        v1[ix] = siftParts1[(i + b) * FMC_BW + (tx * FMC_TW + ix)];
      for (int iy = 0; iy < FMC_TH; iy++)
      {
        float4 v2 = siftParts2[(i + b) * FMC_BH + (ty * FMC_TH + iy)];
        for (int ix = 0; ix < FMC_TW; ix++)
        {
          sums[iy * FMC_TW + ix] += v1[ix].x * v2.x;
          sums[iy * FMC_TW + ix] += v1[ix].y * v2.y;
          sums[iy * FMC_TW + ix] += v1[ix].z * v2.z;
          sums[iy * FMC_TW + ix] += v1[ix].w * v2.w;
        }
      }
    }
    __syncthreads();

    float *blksums = (float *)siftParts1;
    for (int iy = 0; iy < FMC_TH; iy++)
      for (int ix = 0; ix < FMC_TW; ix++)
        blksums[(ty * FMC_TH + iy) * FMC_BW + (tx * FMC_TW + ix)] = sums[iy * FMC_TW + ix];
    __syncthreads();
    if (idx < FMC_BW)
    {
      for (int j = 0; j < FMC_BH; j++)
      {
        float sum = blksums[j * FMC_BW + idx];
        if (sum > maxScore)
        {
          maxScor2 = maxScore;
          maxScore = sum;
          maxIndex = min(blockIdx.y * FMC_GH + k + j, numPts2 - 1);
        }
        else if (sum > maxScor2)
          maxScor2 = sum;
      }
    }
    __syncthreads();
  }
  const int p1 = min(blockIdx.x * FMC_BW + idx, numPts1 - 1);
  if (idx == 0)
    while (atomicCAS((int *)&lock, 0, 1) != 0)
      ;
  __syncthreads();
  if (idx < FMC_BW)
  {
    float maxScor2Old = sift1[p1].ambiguity * (sift1[p1].score + 1e-6f);
    if (maxScore > sift1[p1].score)
    {
      maxScor2 = max(sift1[p1].score, maxScor2);
      sift1[p1].ambiguity = maxScor2 / (maxScore + 1e-6f);
      sift1[p1].score = maxScore;
      sift1[p1].match = maxIndex;
      sift1[p1].match_xpos = sift2[maxIndex].xpos;
      sift1[p1].match_ypos = sift2[maxIndex].ypos;
    }
    else if (maxScore > maxScor2Old)
      sift1[p1].ambiguity = maxScore / (sift1[p1].score + 1e-6f);
  }
  __syncthreads();
  if (idx == 0)
    atomicExch((int *)&lock, 0);
}

__global__ void FindMaxCorr8(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2)
{
  __shared__ float4 siftParts1[FMC_BW * FMC_BD]; // 4*32*8 = 1024
  __shared__ float4 siftParts2[FMC_BH * FMC_BD]; // 4*32*8 = 1024
  __shared__ float blksums[FMC_BW * FMC_BH];     // 32*32  = 1024
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int idx = ty * FMC_NW + tx;
  float4 *pts1 = 0, *pts2 = 0;
  if (idx < FMC_BW)
  {
    const int p1l = min(blockIdx.x * FMC_BW + idx, numPts1 - 1);
    pts1 = (float4 *)sift1[p1l].data;
  }
  float maxScore = -1.0f;
  float maxScor2 = -1.0f;
  int maxIndex = 0;
  for (int k = 0; k < min(FMC_GH, numPts2 - FMC_BH + 1); k += FMC_BH)
  {
    if (idx < FMC_BH)
    {
      const int p2l = min(blockIdx.y * FMC_GH + k + idx, numPts2 - 1);
      pts2 = (float4 *)sift2[p2l].data;
    }
    float sums[FMC_TW * FMC_TH];
    for (int i = 0; i < FMC_TW * FMC_TH; i++)
      sums[i] = 0.0f;
    for (int d = 0; d < 32; d += FMC_BD)
    {
      if (idx < FMC_BW)
        for (int i = 0; i < FMC_BD; i++)
          siftParts1[i * FMC_BW + idx] = pts1[d + i];
      if (idx < FMC_BH)
        for (int i = 0; i < FMC_BD; i++)
          siftParts2[i * FMC_BH + idx] = pts2[d + i];
      __syncthreads();

      for (int i = 0; i < FMC_BD; i++)
      {
        float4 v1[FMC_TW];
        for (int ix = 0; ix < FMC_TW; ix++)
          v1[ix] = siftParts1[i * FMC_BW + (tx * FMC_TW + ix)];
        for (int iy = 0; iy < FMC_TH; iy++)
        {
          float4 v2 = siftParts2[i * FMC_BH + (ty * FMC_TH + iy)];
          for (int ix = 0; ix < FMC_TW; ix++)
          {
            sums[iy * FMC_TW + ix] += v1[ix].x * v2.x;
            sums[iy * FMC_TW + ix] += v1[ix].y * v2.y;
            sums[iy * FMC_TW + ix] += v1[ix].z * v2.z;
            sums[iy * FMC_TW + ix] += v1[ix].w * v2.w;
          }
        }
      }
      __syncthreads();
    }
    //float *blksums = (float*)siftParts1;
    for (int iy = 0; iy < FMC_TH; iy++)
      for (int ix = 0; ix < FMC_TW; ix++)
        blksums[(ty * FMC_TH + iy) * FMC_BW + (tx * FMC_TW + ix)] = sums[iy * FMC_TW + ix];
    __syncthreads();
    if (idx < FMC_BW)
    {
      for (int j = 0; j < FMC_BH; j++)
      {
        float sum = blksums[j * FMC_BW + idx];
        if (sum > maxScore)
        {
          maxScor2 = maxScore;
          maxScore = sum;
          maxIndex = min(blockIdx.y * FMC_GH + k + j, numPts2 - 1);
        }
        else if (sum > maxScor2)
          maxScor2 = sum;
      }
    }
    __syncthreads();
  }
  const int p1 = min(blockIdx.x * FMC_BW + idx, numPts1 - 1);
  if (idx == 0)
    while (atomicCAS((int *)&lock, 0, 1) != 0)
      ;
  __syncthreads();
  if (idx < FMC_BW)
  {
    float maxScor2Old = sift1[p1].ambiguity * (sift1[p1].score + 1e-6f);
    if (maxScore > sift1[p1].score)
    {
      maxScor2 = max(sift1[p1].score, maxScor2);
      sift1[p1].ambiguity = maxScor2 / (maxScore + 1e-6f);
      sift1[p1].score = maxScore;
      sift1[p1].match = maxIndex;
      sift1[p1].match_xpos = sift2[maxIndex].xpos;
      sift1[p1].match_ypos = sift2[maxIndex].ypos;
    }
    else if (maxScore > maxScor2Old)
      sift1[p1].ambiguity = maxScore / (sift1[p1].score + 1e-6f);
  }
  __syncthreads();
  if (idx == 0)
    atomicExch((int *)&lock, 0);
}

__global__ void FindMaxCorr7(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2)
{
  __shared__ float siftParts1[17 * 64]; // features in columns
  __shared__ float siftParts2[16 * 64]; // one extra to avoid shared conflicts
  float4 *pts1 = (float4 *)siftParts1;
  float4 *pts2 = (float4 *)siftParts2;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int p1l = min(blockIdx.x * 16 + ty, numPts1 - 1);
  const float4 *p1l4 = (float4 *)sift1[p1l].data;
  float maxScore = -1.0f;
  float maxScor2 = -1.0f;
  int maxIndex = 0;
  for (int k = 0; k < 512 / 16; k++)
  {
    const int p2l = min(blockIdx.y * 512 + k * 16 + ty, numPts2 - 1);
    const float4 *p2l4 = (float4 *)sift2[p2l].data;
#define NUM 4
    float sum[NUM];
    if (ty < (16 / NUM))
      for (int l = 0; l < NUM; l++)
        sum[l] = 0.0f;
    __syncthreads();
    for (int i = 0; i < 2; i++)
    {
      pts1[17 * tx + ty] = p1l4[i * 16 + tx];
      pts2[16 * ty + tx] = p2l4[i * 16 + tx];
      __syncthreads();
      if (ty < (16 / NUM))
      {
#pragma unroll
        for (int j = 0; j < 16; j++)
        {
          float4 p1v = pts1[17 * j + tx];
#pragma unroll
          for (int l = 0; l < NUM; l++)
          {
            float4 p2v = pts2[16 * (ty + l * (16 / NUM)) + j];
            sum[l] += p1v.x * p2v.x;
            sum[l] += p1v.y * p2v.y;
            sum[l] += p1v.z * p2v.z;
            sum[l] += p1v.w * p2v.w;
          }
        }
      }
      __syncthreads();
    }
    float *sums = siftParts1;
    if (ty < (16 / NUM))
      for (int l = 0; l < NUM; l++)
        sums[16 * (ty + l * (16 / NUM)) + tx] = sum[l];
    __syncthreads();
    if (ty == 0)
    {
      for (int j = 0; j < 16; j++)
      {
        float sum = sums[16 * j + tx];
        if (sum > maxScore)
        {
          maxScor2 = maxScore;
          maxScore = sum;
          maxIndex = min(blockIdx.y * 512 + k * 16 + j, numPts2 - 1);
        }
        else if (sum > maxScor2)
          maxScor2 = sum;
      }
    }
    __syncthreads();
  }
  const int p1 = min(blockIdx.x * 16 + tx, numPts1 - 1);
  if (tx == 0 && ty == 0)
    while (atomicCAS((int *)&lock, 0, 1) != 0)
      ;
  __syncthreads();
  if (ty == 0)
  {
    float maxScor2Old = sift1[p1].ambiguity * (sift1[p1].score + 1e-6f);
    if (maxScore > sift1[p1].score)
    {
      maxScor2 = max(sift1[p1].score, maxScor2);
      sift1[p1].ambiguity = maxScor2 / (maxScore + 1e-6f);
      sift1[p1].score = maxScore;
      sift1[p1].match = maxIndex;
      sift1[p1].match_xpos = sift2[maxIndex].xpos;
      sift1[p1].match_ypos = sift2[maxIndex].ypos;
    }
    else if (maxScore > maxScor2Old)
      sift1[p1].ambiguity = maxScore / (sift1[p1].score + 1e-6f);
  }
  __syncthreads();
  if (tx == 0 && ty == 0)
    atomicExch((int *)&lock, 0);
}

__global__ void FindMaxCorr6(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2)
{
  //__shared__ float siftParts1[128*16]; // features in columns
  __shared__ float siftParts2[128 * 16]; // one extra to avoid shared conflicts
  __shared__ float sums[16 * 16];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int p1l = min(blockIdx.x * 16 + ty, numPts1 - 1);
  float *pt1l = sift1[p1l].data;
  float4 part1 = reinterpret_cast<float4 *>(pt1l)[tx];
  float maxScore = -1.0f;
  float maxScor2 = -1.0f;
  int maxIndex = 0;
  for (int k = 0; k < 512; k += 16)
  {
    const int p2l = min(blockIdx.y * 512 + k + ty, numPts2 - 1);
    float *pt2l = sift2[p2l].data;
    reinterpret_cast<float4 *>(siftParts2)[32 * ty + tx] = reinterpret_cast<float4 *>(pt2l)[tx];
    __syncthreads();
    for (int i = 0; i < 16; i++)
    {
      float4 part2 = reinterpret_cast<float4 *>(siftParts2)[32 * i + tx];
      float sum = part1.x * part2.x + part1.y * part2.y + part1.z * part2.z + part1.w * part2.w;
      sum += ShiftDown(sum, 16);
      sum += ShiftDown(sum, 8);
      sum += ShiftDown(sum, 4);
      sum += ShiftDown(sum, 2);
      sum += ShiftDown(sum, 1);
      if (tx == 0)
        sums[16 * i + ty] = sum;
    }
    __syncthreads();
    if (ty == 0 && tx < 16)
    {
      for (int j = 0; j < 16; j++)
      {
        float sum = sums[16 * j + tx];
        if (sum > maxScore)
        {
          maxScor2 = maxScore;
          maxScore = sum;
          maxIndex = min(blockIdx.y * 512 + k + j, numPts2 - 1);
        }
        else if (sum > maxScor2)
          maxScor2 = sum;
      }
    }
    __syncthreads();
  }
  if (tx == 0 && ty == 0)
    while (atomicCAS((int *)&lock, 0, 1) != 0)
      ;
  __syncthreads();
  if (ty == 0 && tx < 16)
  {
    const int p1 = min(blockIdx.x * 16 + tx, numPts1 - 1);
    float maxScor2Old = sift1[p1].ambiguity * (sift1[p1].score + 1e-6f);
    if (maxScore > sift1[p1].score)
    {
      maxScor2 = max(sift1[p1].score, maxScor2);
      sift1[p1].ambiguity = maxScor2 / (maxScore + 1e-6f);
      sift1[p1].score = maxScore;
      sift1[p1].match = maxIndex;
      sift1[p1].match_xpos = sift2[maxIndex].xpos;
      sift1[p1].match_ypos = sift2[maxIndex].ypos;
    }
    else if (maxScore > maxScor2Old)
      sift1[p1].ambiguity = maxScore / (sift1[p1].score + 1e-6f);
  }
  __syncthreads();
  if (tx == 0 && ty == 0)
    atomicExch((int *)&lock, 0);
}

__global__ void FindMaxCorr5(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2)
{
  __shared__ float siftParts1[17 * 16]; // features in columns
  __shared__ float siftParts2[17 * 16]; // one extra to avoid shared conflicts
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int p1l = min(blockIdx.x * 16 + ty, numPts1 - 1);
  const float *pt1l = sift1[p1l].data;
  float maxScore = -1.0f;
  float maxScor2 = -1.0f;
  int maxIndex = 0;
  for (int k = 0; k < 512 / 16; k++)
  {
    const int p2l = min(blockIdx.y * 512 + k * 16 + ty, numPts2 - 1);
    const float *pt2l = sift2[p2l].data;
    float sum = 0.0f;
    for (int i = 0; i < 8; i++)
    {
      siftParts1[17 * tx + ty] = pt1l[i * 16 + tx]; // load and transpose
      siftParts2[17 * tx + ty] = pt2l[i * 16 + tx];
      __syncthreads();
      for (int j = 0; j < 16; j++)
        sum += siftParts1[17 * j + tx] * siftParts2[17 * j + ty];
      __syncthreads();
    }
    float *sums = siftParts1;
    sums[16 * ty + tx] = sum;
    __syncthreads();
    if (ty == 0)
    {
      for (int j = 0; j < 16; j++)
      {
        float sum = sums[16 * j + tx];
        if (sum > maxScore)
        {
          maxScor2 = maxScore;
          maxScore = sum;
          maxIndex = min(blockIdx.y * 512 + k * 16 + j, numPts2 - 1);
        }
        else if (sum > maxScor2)
          maxScor2 = sum;
      }
    }
    __syncthreads();
  }
  const int p1 = min(blockIdx.x * 16 + tx, numPts1 - 1);
  if (tx == 0 && ty == 0)
    while (atomicCAS((int *)&lock, 0, 1) != 0)
      ;
  __syncthreads();
  if (ty == 0)
  {
    float maxScor2Old = sift1[p1].ambiguity * (sift1[p1].score + 1e-6f);
    if (maxScore > sift1[p1].score)
    {
      maxScor2 = max(sift1[p1].score, maxScor2);
      sift1[p1].ambiguity = maxScor2 / (maxScore + 1e-6f);
      sift1[p1].score = maxScore;
      sift1[p1].match = maxIndex;
      sift1[p1].match_xpos = sift2[maxIndex].xpos;
      sift1[p1].match_ypos = sift2[maxIndex].ypos;
      sift1[p1].match_x_3d = sift2[maxIndex].x_3d;
      sift1[p1].match_y_3d = sift2[maxIndex].y_3d;
      sift1[p1].match_z_3d = sift2[maxIndex].z_3d;
      sift1[p1].match_distance = sift2[maxIndex].distance;
    }
    else if (maxScore > maxScor2Old)
      sift1[p1].ambiguity = maxScore / (sift1[p1].score + 1e-6f);
  }
  __syncthreads();
  if (tx == 0 && ty == 0)
    atomicExch((int *)&lock, 0);
}

inline void __checkMsg(const char *errorMessage, const char *file, const int line)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err)
  {
    fprintf(stderr, "checkMsg() CUDA error: %s in file <%s>, line %i : %s.\n", errorMessage, file, line, cudaGetErrorString(err));
    exit(-1);
  }
}

inline void __checkMsgNoFail(const char *errorMessage, const char *file, const int line)
{
  cudaError_t err = cudaGetLastError();
  // if (cudaSuccess != err)
  // {
  //   fprintf(stderr, "checkMsg() CUDA warning: %s in file <%s>, line %i : %s.\n", errorMessage, file, line, cudaGetErrorString(err));
  // }
}

void Computation::initCuda()
{
  cudaSetDeviceFlags(cudaDeviceMapHost);
}

void Computation::cleanup()
{
}

int Computation::setCudaVkDevice()
{
  std::cout << "Setting CUDA-VK Device" << std::endl;
  int current_device = 0;
  int device_count = 0;
  int devices_prohibited = 0;
  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceCount(&device_count));
  if (device_count == 0)
  {
    std::cout << "CUDA error: no devices supporting CUDA." << std::endl;
    exit(EXIT_FAILURE);
  }
  // Find the GPU which is selected by Vulkan
  while (current_device < device_count)
  {
    std::cout << "Looking up device " << current_device << " of " << device_count << std::endl;
    cudaGetDeviceProperties(&deviceProp, current_device);
    if ((deviceProp.computeMode != cudaComputeModeProhibited))
    {
      // Compare the cuda device UUID with vulkan UUID
      int ret = memcmp(&deviceProp.uuid, &vkDeviceUUID, VK_UUID_SIZE);

      std::cout << "ret is " << ret << std::endl;
      printf("vkDeviceUUID    = 0x");
      for (int i = 0; i < VK_UUID_SIZE; i++)
      {
        printf("%x ", *(vkDeviceUUID + i));
      }
      printf("\n");
      printf("deviceProp.uuid = 0x");
      for (int i = 0; i < VK_UUID_SIZE; i++)
      {
        printf("%x ", *((deviceProp.uuid.bytes) + i));
      }
      printf("\n");
      if (ret == 0)
      {
        checkCudaErrors(cudaSetDevice(current_device));
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, current_device));
        std::cout << "GPU Device" << current_device << deviceProp.name << deviceProp.major << deviceProp.minor << std::endl;
        return current_device;
      }
    }
    else
    {
      devices_prohibited++;
    }
    current_device++;
  }
  if (devices_prohibited == device_count)
  {
    std::cout << "No Vulkan-CUDA Interop capable GPU found." << std::endl;
    exit(EXIT_FAILURE);
  }
  std::cout << "returning -1" << std::endl;
  return -1;
}

void Computation::LowPass_prepareKernel(void)
{
  int kernelSize{GAUSSIANSIZE};
  int range = kernelSize / 2;
  float gausskernel[kernelSize * kernelSize];

  double sigma = (float)GAUSSIANSIZE / 5.0;
  double r, s = 2.0 * sigma * sigma;

  // sum is for normalization
  double sum = 0.0;

  // generating NxN kernel
  for (int x = (-1 * range); x <= range; x++)
  {
    for (int y = (-1 * range); y <= range; y++)
    {
      r = sqrt(x * x + y * y);
      gausskernel[(x + range) + kernelSize * (y + range)] = (exp(-(r * r) / s)) / (M_PI * s);
      sum += gausskernel[(x + range) + kernelSize * (y + range)];
    }
  }
  // normalising the Kernel
  for (int i = 0; i < kernelSize; ++i)
    for (int j = 0; j < kernelSize; ++j)
      gausskernel[i + kernelSize * j] /= sum;

  for (int i = 0; i < kernelSize; ++i)
  {
    for (int j = 0; j < kernelSize; ++j)
      std::cout << gausskernel[i + kernelSize * j] << "\t";
    std::cout << std::endl;
  }

  cudaMemcpyToSymbol(d_GaussKernel, gausskernel, kernelSize * kernelSize * sizeof(float));
}

void Computation::LowPass_forSubImages(float *res, float *src, int width, int height)
{
  int kernelSize{GAUSSIANSIZE};

  dim3 block(16, 16, 1);
  dim3 grid(width / block.x + 1, height / block.y + 1, 1);

  GaussKernelBlock<<<grid, block, 0>>>(res, src, width, height, kernelSize);

  return;
}

void Computation::InitSiftData(SiftData &data, int numPoints, int numSlices, bool host, bool dev)
{
  data.numPts = 0;
  data.maxPts = numPoints * numSlices;
  int sz = sizeof(SiftPoint) * numPoints * numSlices;
#ifdef MANAGEDMEM
  cudaMallocManaged((void **)&data.m_data, sz);
#else
  data.h_data = NULL;
  if (host)
    data.h_data = (SiftPoint *)malloc(sz);
  data.d_data = NULL;
  if (dev)
    cudaMalloc((void **)&data.d_data, sz);
#endif
}

float *Computation::AllocSiftTempMemory(int width, int height, int numOctaves, bool scaleUp)
{
  const int nd = NUM_SCALES + 3;
  int w = width * (scaleUp ? 2 : 1);
  int h = height * (scaleUp ? 2 : 1);
  int p = iAlignUp(w, 128);
  int size = h * p;         // image sizes
  int sizeTmp = nd * h * p; // laplace buffer sizes
  for (int i = 0; i < numOctaves; i++)
  {
    w /= 2;
    h /= 2;
    int p = iAlignUp(w, 128);
    size += h * p;
    sizeTmp += nd * h * p;
  }
  float *memoryTmp = NULL;
  size_t pitch;
  size += sizeTmp;
  cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size + 4095) / 4096 * sizeof(float));
#ifdef VERBOSE
  printf("Allocated memory size: %d bytes\n", size);
  printf("Memory allocation time =      %.2f ms\n\n", timer.read());
#endif
  return memoryTmp;
}

void Computation::ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, bool scaleUp, float *tempMemory, unsigned char *chardata, cudaStream_t stream)
{
  unsigned int *d_PointCounterAddr;
  cudaGetSymbolAddress((void **)&d_PointCounterAddr, d_PointCounter);
  cudaMemset(d_PointCounterAddr, 0, (8 * 2 + 1) * sizeof(int));
  cudaMemcpyToSymbol(d_MaxNumPoints, &siftData.maxPts, sizeof(int)); // this seems okay, it's a cuda variable after all.

  const int nd = NUM_SCALES + 3;
  int w = img.width * (scaleUp ? 2 : 1);
  int h = img.height * (scaleUp ? 2 : 1);
  int p = iAlignUp(w, 128);
  int width = w, height = h;
  int size = h * p;         // image sizes
  int sizeTmp = nd * h * p; // laplace buffer sizes
  for (int i = 0; i < numOctaves; i++)
  {
    w /= 2;
    h /= 2;
    int p = iAlignUp(w, 128);
    size += h * p;
    sizeTmp += nd * h * p;
  }
  float *memoryTmp = tempMemory;
  size += sizeTmp;
  if (!tempMemory)
  {
    //size_t pitch;
    //cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size+4095)/4096*sizeof(float));
#ifdef VERBOSE
    printf("Allocated memory size: %d bytes\n", size);
#endif
  }
  float *memorySub = memoryTmp + sizeTmp;

  CudaImage lowImg;
  lowImg.Allocate(width, height, iAlignUp(width, 128), false, memorySub);

  if (!scaleUp)
  {
    float kernel[8 * 12 * 16];
    PrepareLaplaceKernels(numOctaves, 0.0f, kernel);
    cudaMemcpyToSymbolAsync(d_LaplaceKernel, kernel, 8 * 12 * 16 * sizeof(float));
    LowPass(lowImg, img, max(initBlur, 0.001f), chardata);
    ExtractSiftLoop(siftData, lowImg, numOctaves, 0.0f, thresh, lowestScale, 1.0f, memoryTmp, memorySub + height * iAlignUp(width, 128));
    cudaMemcpy(&siftData.numPts, &d_PointCounterAddr[2 * numOctaves], sizeof(int), cudaMemcpyDeviceToHost);
    checkMsg("Error at cudaMemcpy\n");
    siftData.numPts = (siftData.numPts < siftData.maxPts ? siftData.numPts : siftData.maxPts);
  }
  else
  {
    CudaImage upImg;
    upImg.Allocate(width, height, iAlignUp(width, 128), false, memoryTmp);
    ScaleUp(upImg, img, chardata);
    checkMsg("Error at ScaleUp\n");
    LowPass(lowImg, upImg, max(initBlur, 0.001f), NULL);
    checkMsg("Error at LowPass\n");
    float kernel[8 * 12 * 16];
    PrepareLaplaceKernels(numOctaves, 0.0f, kernel);
    checkMsg("Error at PrepareLaplace\n");
    cudaMemcpyToSymbolAsync(d_LaplaceKernel, kernel, 8 * 12 * 16 * sizeof(float));
    checkMsg("Error at cudaMemcpyToSymbolAsync\n");
    ExtractSiftLoop(siftData, lowImg, numOctaves, 1.0f, thresh, lowestScale * 2.0f, 1.0f, memoryTmp, memorySub + height * iAlignUp(width, 128));
    checkMsg("Error at ExtractSiftLoop\n");
    cudaDeviceSynchronize();
    checkMsg("Error at cudaDeviceSynchronize\n");
    cudaMemcpy(&siftData.numPts, &d_PointCounterAddr[2 * numOctaves], sizeof(int), cudaMemcpyDeviceToHost);
    printf("numpts: %d\n", siftData.numPts);
    siftData.numPts = 10;
    checkMsg("Error at cudaMemcpy\n");
    siftData.numPts = (siftData.numPts < siftData.maxPts ? siftData.numPts : siftData.maxPts);
    RescalePositions(siftData, 0.5f);
  }

  if (!tempMemory)
    cudaFree(memoryTmp);
#ifdef MANAGEDMEM
  cudaDeviceSynchronize());
#else
  if (siftData.h_data)
    cudaMemcpy(siftData.h_data, siftData.d_data, sizeof(SiftPoint) * siftData.numPts, cudaMemcpyDeviceToHost);
#endif
}

void Computation::FreeSiftTempMemory(float *memoryTmp)
{
  if (memoryTmp)
    cudaFree(memoryTmp);
}

void Computation::PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel)
{
  if (numOctaves > 1)
  {
    float totInitBlur = (float)sqrt(initBlur * initBlur + 0.5f * 0.5f) / 2.0f;
    PrepareLaplaceKernels(numOctaves - 1, totInitBlur, kernel);
  }
  float scale = pow(2.0f, -1.0f / NUM_SCALES);
  float diffScale = pow(2.0f, 1.0f / NUM_SCALES);
  for (int i = 0; i < NUM_SCALES + 3; i++)
  {
    float kernelSum = 0.0f;
    float var = scale * scale - initBlur * initBlur;
    for (int j = 0; j <= LAPLACE_R; j++)
    {
      kernel[numOctaves * 12 * 16 + 16 * i + j] = (float)expf(-(double)j * j / 2.0 / var);
      kernelSum += (j == 0 ? 1 : 2) * kernel[numOctaves * 12 * 16 + 16 * i + j];
    }
    for (int j = 0; j <= LAPLACE_R; j++)
      kernel[numOctaves * 12 * 16 + 16 * i + j] /= kernelSum;
    scale *= diffScale;
  }
}

void Computation::LowPass(CudaImage &res, CudaImage &src, float scale, unsigned char *chardata)
{

  float kernel[2 * LOWPASS_R + 1];
  static float oldScale = -1.0f;
  if (scale != oldScale)
  {
    float kernelSum = 0.0f;
    float ivar2 = 1.0f / (2.0f * scale * scale);
    for (int j = -LOWPASS_R; j <= LOWPASS_R; j++)
    {
      kernel[j + LOWPASS_R] = (float)expf(-(double)j * j * ivar2);
      kernelSum += kernel[j + LOWPASS_R];
    }
    for (int j = -LOWPASS_R; j <= LOWPASS_R; j++)
      kernel[j + LOWPASS_R] /= kernelSum;

    cudaMemcpyToSymbol(d_LowPassKernel, kernel, (2 * LOWPASS_R + 1) * sizeof(float));
    oldScale = scale;
  }
  int width = res.width;
  int pitch = res.pitch;
  int height = res.height;
  dim3 blocks(iDivUp(width, LOWPASS_W), iDivUp(height, LOWPASS_H));
#if 1
  dim3 threads(LOWPASS_W + 2 * LOWPASS_R, 4);
  if (chardata != NULL)
  {
    LowPassBlockCharYUYV<<<blocks, threads>>>(chardata, res.d_data, 1920, 1920, 1080);
  }
  else
  {
    LowPassBlock<<<blocks, threads>>>(src.d_data, res.d_data, width, pitch, height);
  }

#else
  dim3 threads(LOWPASS_W + 2 * LOWPASS_R, LOWPASS_H);
  LowPass<<<blocks, threads>>>(src.d_data, res.d_data, width, pitch, height);
#endif
  return;
}

int Computation::ExtractSiftLoop(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, float subsampling, float *memoryTmp, float *memorySub)
{
#ifdef VERBOSE
  TimerGPU timer(0);
#endif
  int w = img.width;
  int h = img.height;
  if (numOctaves > 1)
  {
    CudaImage subImg;
    int p = iAlignUp(w / 2, 128);
    subImg.Allocate(w / 2, h / 2, p, false, memorySub);
    ScaleDown(subImg, img, 0.5f);
    float totInitBlur = (float)sqrt(initBlur * initBlur + 0.5f * 0.5f) / 2.0f;
    ExtractSiftLoop(siftData, subImg, numOctaves - 1, totInitBlur, thresh, lowestScale, subsampling * 2.0f, memoryTmp, memorySub + (h / 2) * p);
  }
  ExtractSiftOctave(siftData, img, numOctaves, thresh, lowestScale, subsampling, memoryTmp);
#ifdef VERBOSE
  double totTime = timer.read();
  printf("ExtractSift time total =      %.2f ms %d\n\n", totTime, numOctaves);
#endif
  return 0;
}

void Computation::ExtractSiftOctave(SiftData &siftData, CudaImage &img, int octave, float thresh, float lowestScale, float subsampling, float *memoryTmp)
{
  const int nd = NUM_SCALES + 3;
#ifdef VERBOSE
  unsigned int *d_PointCounterAddr;
  safeCall(cudaGetSymbolAddress((void **)&d_PointCounterAddr, d_PointCounter));
  unsigned int fstPts, totPts;
  safeCall(cudaMemcpy(&fstPts, &d_PointCounterAddr[2 * octave - 1], sizeof(int), cudaMemcpyDeviceToHost));
  TimerGPU timer0;
#endif
  CudaImage diffImg[nd];
  int w = img.width;
  int h = img.height;
  int p = iAlignUp(w, 128);
  for (int i = 0; i < nd - 1; i++)
    diffImg[i].Allocate(w, h, p, false, memoryTmp + i * p * h);

  // Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = img.d_data;
  resDesc.res.pitch2D.width = img.width;
  resDesc.res.pitch2D.height = img.height;
  resDesc.res.pitch2D.pitchInBytes = img.pitch * sizeof(float);
  resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;
  // Create texture object
  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

#ifdef VERBOSE
  TimerGPU timer1;
#endif
  float baseBlur = pow(2.0f, -1.0f / NUM_SCALES);
  float diffScale = pow(2.0f, 1.0f / NUM_SCALES);
  LaplaceMulti(texObj, img, diffImg, octave);
  FindPointsMulti(diffImg, siftData, thresh, 10.0f, 1.0f / NUM_SCALES, lowestScale / subsampling, subsampling, octave);
#ifdef VERBOSE
  double gpuTimeDoG = timer1.read();
  TimerGPU timer4;
#endif
  ComputeOrientations(texObj, img, siftData, octave);
  ExtractSiftDescriptors(texObj, siftData, subsampling, octave);
  //OrientAndExtract(texObj, siftData, subsampling, octave);

  cudaDestroyTextureObject(texObj);
#ifdef VERBOSE
  double gpuTimeSift = timer4.read();
  double totTime = timer0.read();
  printf("GPU time : %.2f ms + %.2f ms + %.2f ms = %.2f ms\n", totTime - gpuTimeDoG - gpuTimeSift, gpuTimeDoG, gpuTimeSift, totTime);
  safeCall(cudaMemcpy(&totPts, &d_PointCounterAddr[2 * octave + 1], sizeof(int), cudaMemcpyDeviceToHost));
  totPts = (totPts < siftData.maxPts ? totPts : siftData.maxPts);
  if (totPts > 0)
    printf("           %.2f ms / DoG,  %.4f ms / Sift,  #Sift = %d\n", gpuTimeDoG / NUM_SCALES, gpuTimeSift / (totPts - fstPts), totPts - fstPts);
#endif
}

void Computation::ScaleDown(CudaImage &res, CudaImage &src, float variance)
{
  static float oldVariance = -1.0f;
  if (res.d_data == NULL || src.d_data == NULL)
  {
    printf("ScaleDown: missing data\n");
    return;
  }
  if (oldVariance != variance)
  {
    float h_Kernel[5];
    float kernelSum = 0.0f;
    for (int j = 0; j < 5; j++)
    {
      h_Kernel[j] = (float)expf(-(double)(j - 2) * (j - 2) / 2.0 / variance);
      kernelSum += h_Kernel[j];
    }
    for (int j = 0; j < 5; j++)
      h_Kernel[j] /= kernelSum;
    cudaMemcpyToSymbol(d_ScaleDownKernel, h_Kernel, 5 * sizeof(float));
    oldVariance = variance;
  }
#if 0
  dim3 blocks(iDivUp(src.width, SCALEDOWN_W), iDivUp(src.height, SCALEDOWN_H));
  dim3 threads(SCALEDOWN_W + 4, SCALEDOWN_H + 4);
  ScaleDownDenseShift<<<blocks, threads>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);
#else
  dim3 blocks(iDivUp(src.width, SCALEDOWN_W), iDivUp(src.height, SCALEDOWN_H));
  dim3 threads(SCALEDOWN_W + 4);
  ScaleDownKernel<<<blocks, threads>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);
#endif
  checkMsg("ScaleDown() execution failed\n");
  return;
}

void Computation::ScaleUp(CudaImage &res, CudaImage &src, unsigned char *chardata)
{
  if (res.d_data == NULL || src.d_data == NULL)
  {
    printf("ScaleUp: missing data\n");
    return;
  }
  dim3 blocks(iDivUp(res.width, SCALEUP_W), iDivUp(res.height, SCALEUP_H));
  dim3 threads(SCALEUP_W / 2, SCALEUP_H / 2);
  if (chardata != NULL)
  {
    ScaleUpKernelCharYUYV<<<blocks, threads>>>(res.d_data, chardata, src.width, src.pitch, src.height, res.pitch);
  }
  else
  {
    ScaleUpKernel<<<blocks, threads>>>(res.d_data, src.d_data, src.width, src.pitch, src.height, res.pitch);
  }

  checkMsg("ScaleUp() execution failed\n");
  return;
}

void Computation::RescalePositions(SiftData &siftData, float scale)
{
  dim3 blocks(iDivUp(siftData.numPts, 64));
  dim3 threads(64);
  RescalePositionsKernel<<<blocks, threads>>>(siftData.d_data, siftData.numPts, scale);
  checkMsg("RescalePositions() execution failed\n");
  return;
}

void Computation::LaplaceMulti(cudaTextureObject_t texObj, CudaImage &baseImage, CudaImage *results, int octave)
{
  int width = results[0].width;
  int pitch = results[0].pitch;
  int height = results[0].height;
#if 1
  dim3 threads(LAPLACE_W + 2 * LAPLACE_R);
  dim3 blocks(iDivUp(width, LAPLACE_W), height);
  LaplaceMultiMem<<<blocks, threads>>>(baseImage.d_data, results[0].d_data, width, pitch, height, octave);
#endif
#if 0
  dim3 threads(LAPLACE_W+2*LAPLACE_R, LAPLACE_S);
  dim3 blocks(iDivUp(width, LAPLACE_W), iDivUp(height, LAPLACE_H));
  LaplaceMultiMemTest<<<blocks, threads>>>(baseImage.d_data, results[0].d_data, width, pitch, height, octave);
#endif
#if 0
  dim3 threads(LAPLACE_W+2*LAPLACE_R, LAPLACE_S);
  dim3 blocks(iDivUp(width, LAPLACE_W), height);
  LaplaceMultiMemOld<<<blocks, threads>>>(baseImage.d_data, results[0].d_data, width, pitch, height, octave);
#endif
#if 0
  dim3 threads(LAPLACE_W+2*LAPLACE_R, LAPLACE_S);
  dim3 blocks(iDivUp(width, LAPLACE_W), height);
  LaplaceMultiTex<<<blocks, threads>>>(texObj, results[0].d_data, width, pitch, height, octave);
#endif
  checkMsg("LaplaceMultiMem() execution failed\n");
  return;
}

void Computation::FindPointsMulti(CudaImage *sources, SiftData &siftData, float thresh, float edgeLimit, float factor, float lowestScale, float subsampling, int octave)
{
  if (sources->d_data == NULL)
  {
    printf("FindPointsMulti: missing data\n");
    return;
  }
  int w = sources->width;
  int p = sources->pitch;
  int h = sources->height;

#if 0
  dim3 blocks(iDivUp(w, MINMAX_W)*NUM_SCALES, iDivUp(h, MINMAX_H));
  dim3 threads(MINMAX_W + 2, MINMAX_H);
  FindPointsMultiTest<<<blocks, threads>>>(sources->d_data, siftData.d_data, w, p, h, subsampling, lowestScale, thresh, factor, edgeLimit, octave);
#endif
#if 1
  dim3 blocks(iDivUp(w, MINMAX_W) * NUM_SCALES, iDivUp(h, MINMAX_H));
  dim3 threads(MINMAX_W + 2);
#ifdef MANAGEDMEM
  FindPointsMulti<<<blocks, threads>>>(sources->d_data, siftData.m_data, w, p, h, subsampling, lowestScale, thresh, factor, edgeLimit, octave);
#else
  FindPointsMultiNew<<<blocks, threads>>>(sources->d_data, siftData.d_data, w, p, h, subsampling, lowestScale, thresh, factor, edgeLimit, octave);
#endif
#endif
  checkMsg("FindPointsMultiNew() execution failed\n");
  return;
}

void Computation::ComputeOrientations(cudaTextureObject_t texObj, CudaImage &src, SiftData &siftData, int octave)
{
  dim3 blocks(512);
#ifdef MANAGEDMEM
  ComputeOrientationsCONST<<<blocks, threads>>>(texObj, siftData.m_data, octave);
#else
#if 1
  dim3 threads(11 * 11);
  ComputeOrientationsCONST<<<blocks, threads>>>(texObj, siftData.d_data, octave);
#else
  dim3 threads(256);
  ComputeOrientationsCONSTNew<<<blocks, threads>>>(src.d_data, src.width, src.pitch, src.height, siftData.d_data, octave);
#endif
#endif
  checkMsg("ComputeOrientations() execution failed\n");
  return;
}

void Computation::ExtractSiftDescriptors(cudaTextureObject_t texObj, SiftData &siftData, float subsampling, int octave)
{
  dim3 blocks(512);
  dim3 threads(16, 8);
#ifdef MANAGEDMEM
  ExtractSiftDescriptorsCONST<<<blocks, threads>>>(texObj, siftData.m_data, subsampling, octave);
#else
  ExtractSiftDescriptorsCONSTNew<<<blocks, threads>>>(texObj, siftData.d_data, subsampling, octave);
#endif
  checkMsg("ExtractSiftDescriptors() execution failed\n");
  return;
}

#define TESTHOMO_TESTS 16 // number of tests per block,  alt. 32, 32
#define TESTHOMO_LOOPS 16 // number of loops per block,  alt.  8, 16

__global__ void TestHomographies(float *d_coord, float *d_homo,
                                 int *d_counts, int numPts, float thresh2)
{
  __shared__ float homo[8 * TESTHOMO_LOOPS];
  __shared__ int cnts[TESTHOMO_TESTS * TESTHOMO_LOOPS];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int idx = blockIdx.y * blockDim.y + tx;
  const int numLoops = blockDim.y * gridDim.y;
  if (ty < 8 && tx < TESTHOMO_LOOPS)
    homo[tx * 8 + ty] = d_homo[idx + ty * numLoops];
  __syncthreads();
  float a[8];
  for (int i = 0; i < 8; i++)
    a[i] = homo[ty * 8 + i];
  int cnt = 0;
  for (int i = tx; i < numPts; i += TESTHOMO_TESTS)
  {
    float x1 = d_coord[i + 0 * numPts];
    float y1 = d_coord[i + 1 * numPts];
    float x2 = d_coord[i + 2 * numPts];
    float y2 = d_coord[i + 3 * numPts];
    float nomx = __fmul_rz(a[0], x1) + __fmul_rz(a[1], y1) + a[2];
    float nomy = __fmul_rz(a[3], x1) + __fmul_rz(a[4], y1) + a[5];
    float deno = __fmul_rz(a[6], x1) + __fmul_rz(a[7], y1) + 1.0f;
    float errx = __fmul_rz(x2, deno) - nomx;
    float erry = __fmul_rz(y2, deno) - nomy;
    float err2 = __fmul_rz(errx, errx) + __fmul_rz(erry, erry);
    if (err2 < __fmul_rz(thresh2, __fmul_rz(deno, deno)))
      cnt++;
  }
  int kty = TESTHOMO_TESTS * ty;
  cnts[kty + tx] = cnt;
  __syncthreads();
  int len = TESTHOMO_TESTS / 2;
  while (len > 0)
  {
    if (tx < len)
      cnts[kty + tx] += cnts[kty + tx + len];
    len /= 2;
    __syncthreads();
  }
  if (tx < TESTHOMO_LOOPS && ty == 0)
    d_counts[idx] = cnts[TESTHOMO_TESTS * tx];
  __syncthreads();
}

template <int size>
__device__ void InvertMatrix(float elem[size][size], float res[size][size])
{
  int indx[size];
  float b[size];
  float vv[size];
  for (int i = 0; i < size; i++)
    indx[i] = 0;
  int imax = 0;
  float d = 1.0;
  for (int i = 0; i < size; i++)
  { // find biggest element for each row
    float big = 0.0;
    for (int j = 0; j < size; j++)
    {
      float temp = fabs(elem[i][j]);
      if (temp > big)
        big = temp;
    }
    if (big > 0.0)
      vv[i] = 1.0 / big;
    else
      vv[i] = 1e16;
  }
  for (int j = 0; j < size; j++)
  {
    for (int i = 0; i < j; i++)
    {                                   // i<j
      float sum = elem[i][j];           // i<j (lower left)
      for (int k = 0; k < i; k++)       // k<i<j
        sum -= elem[i][k] * elem[k][j]; // i>k (upper right), k<j (lower left)
      elem[i][j] = sum;                 // i<j (lower left)
    }
    float big = 0.0;
    for (int i = j; i < size; i++)
    {                                   // i>=j
      float sum = elem[i][j];           // i>=j (upper right)
      for (int k = 0; k < j; k++)       // k<j<=i
        sum -= elem[i][k] * elem[k][j]; // i>k (upper right), k<j (lower left)
      elem[i][j] = sum;                 // i>=j (upper right)
      float dum = vv[i] * fabs(sum);
      if (dum >= big)
      {
        big = dum;
        imax = i;
      }
    }
    if (j != imax)
    { // imax>j
      for (int k = 0; k < size; k++)
      {
        float dum = elem[imax][k]; // upper right and lower left
        elem[imax][k] = elem[j][k];
        elem[j][k] = dum;
      }
      d = -d;
      vv[imax] = vv[j];
    }
    indx[j] = imax;
    if (elem[j][j] == 0.0) // j==j (upper right)
      elem[j][j] = 1e-16;
    if (j != (size - 1))
    {
      float dum = 1.0 / elem[j][j];
      for (int i = j + 1; i < size; i++) // i>j
        elem[i][j] *= dum;               // i>j (upper right)
    }
  }
  for (int j = 0; j < size; j++)
  {
    for (int k = 0; k < size; k++)
      b[k] = 0.0;
    b[j] = 1.0;
    int ii = -1;
    for (int i = 0; i < size; i++)
    {
      int ip = indx[i];
      float sum = b[ip];
      b[ip] = b[i];
      if (ii != -1)
        for (int j = ii; j < i; j++)
          sum -= elem[i][j] * b[j]; // i>j (upper right)
      else if (sum != 0.0)
        ii = i;
      b[i] = sum;
    }
    for (int i = size - 1; i >= 0; i--)
    {
      float sum = b[i];
      for (int j = i + 1; j < size; j++)
        sum -= elem[i][j] * b[j]; // i<j (lower left)
      b[i] = sum / elem[i][i];    // i==i (upper right)
    }
    for (int i = 0; i < size; i++)
      res[i][j] = b[i];
  }
}

__global__ void ComputeHomographies(float *coord, int *randPts, float *homo,
                                    int numPts)
{
  float a[8][8], ia[8][8];
  float b[8];
  const int bx = blockIdx.x;
  const int tx = threadIdx.x;
  const int idx = blockDim.x * bx + tx;
  const int numLoops = blockDim.x * gridDim.x;
  for (int i = 0; i < 4; i++)
  {
    int pt = randPts[i * numLoops + idx];
    float x1 = coord[pt + 0 * numPts];
    float y1 = coord[pt + 1 * numPts];
    float x2 = coord[pt + 2 * numPts];
    float y2 = coord[pt + 3 * numPts];
    float *row1 = a[2 * i + 0];
    row1[0] = x1;
    row1[1] = y1;
    row1[2] = 1.0;
    row1[3] = row1[4] = row1[5] = 0.0;
    row1[6] = -x2 * x1;
    row1[7] = -x2 * y1;
    float *row2 = a[2 * i + 1];
    row2[0] = row2[1] = row2[2] = 0.0;
    row2[3] = x1;
    row2[4] = y1;
    row2[5] = 1.0;
    row2[6] = -y2 * x1;
    row2[7] = -y2 * y1;
    b[2 * i + 0] = x2;
    b[2 * i + 1] = y2;
  }
  InvertMatrix<8>(a, ia);
  __syncthreads();
  for (int j = 0; j < 8; j++)
  {
    float sum = 0.0f;
    for (int i = 0; i < 8; i++)
      sum += ia[j][i] * b[i];
    homo[j * numLoops + idx] = sum;
  }
  __syncthreads();
}

void Computation::FindHomography(SiftData &data, float *homography, int *numMatches, int numLoops, float minScore, float maxAmbiguity, float thresh)
{
  *numMatches = 0;
  homography[0] = homography[4] = homography[8] = 1.0f;
  homography[1] = homography[2] = homography[3] = 0.0f;
  homography[5] = homography[6] = homography[7] = 0.0f;
#ifdef MANAGEDMEM
  SiftPoint *d_sift = data.m_data;
#else
  if (data.d_data == NULL)
    return;
  SiftPoint *d_sift = data.d_data;
#endif
  numLoops = iDivUp(numLoops, 16) * 16;
  int numPts = data.numPts;
  if (numPts < 8)
    return;
  int numPtsUp = iDivUp(numPts, 16) * 16;
  float *d_coord, *d_homo;
  int *d_randPts, *h_randPts;
  int randSize = 4 * sizeof(int) * numLoops;
  int szFl = sizeof(float);
  int szPt = sizeof(SiftPoint);
  cudaMalloc((void **)&d_coord, 4 * sizeof(float) * numPtsUp);
  cudaMalloc((void **)&d_randPts, randSize);
  cudaMalloc((void **)&d_homo, 8 * sizeof(float) * numLoops);
  h_randPts = (int *)malloc(randSize);
  float *h_scores = (float *)malloc(sizeof(float) * numPtsUp);
  float *h_ambiguities = (float *)malloc(sizeof(float) * numPtsUp);
  cudaMemcpy2D(h_scores, szFl, &d_sift[0].score, szPt, szFl, numPts, cudaMemcpyDeviceToHost);
  cudaMemcpy2D(h_ambiguities, szFl, &d_sift[0].ambiguity, szPt, szFl, numPts, cudaMemcpyDeviceToHost);
  int *validPts = (int *)malloc(sizeof(int) * numPts);
  int numValid = 0;
  for (int i = 0; i < numPts; i++)
  {
    if (h_scores[i] > minScore && h_ambiguities[i] < maxAmbiguity)
      validPts[numValid++] = i;
  }
  free(h_scores);
  free(h_ambiguities);
  if (numValid >= 8)
  {
    for (int i = 0; i < numLoops; i++)
    {
      int p1 = rand() % numValid;
      int p2 = rand() % numValid;
      int p3 = rand() % numValid;
      int p4 = rand() % numValid;
      while (p2 == p1)
        p2 = rand() % numValid;
      while (p3 == p1 || p3 == p2)
        p3 = rand() % numValid;
      while (p4 == p1 || p4 == p2 || p4 == p3)
        p4 = rand() % numValid;
      h_randPts[i + 0 * numLoops] = validPts[p1];
      h_randPts[i + 1 * numLoops] = validPts[p2];
      h_randPts[i + 2 * numLoops] = validPts[p3];
      h_randPts[i + 3 * numLoops] = validPts[p4];
    }
    cudaMemcpy(d_randPts, h_randPts, randSize, cudaMemcpyHostToDevice);
    cudaMemcpy2D(&d_coord[0 * numPtsUp], szFl, &d_sift[0].xpos, szPt, szFl, numPts, cudaMemcpyDeviceToDevice);
    cudaMemcpy2D(&d_coord[1 * numPtsUp], szFl, &d_sift[0].ypos, szPt, szFl, numPts, cudaMemcpyDeviceToDevice);
    cudaMemcpy2D(&d_coord[2 * numPtsUp], szFl, &d_sift[0].match_xpos, szPt, szFl, numPts, cudaMemcpyDeviceToDevice);
    cudaMemcpy2D(&d_coord[3 * numPtsUp], szFl, &d_sift[0].match_ypos, szPt, szFl, numPts, cudaMemcpyDeviceToDevice);
    ComputeHomographies<<<numLoops / 16, 16>>>(d_coord, d_randPts, d_homo, numPtsUp);
    cudaDeviceSynchronize();
    checkMsg("ComputeHomographies() execution failed\n");
    dim3 blocks(1, numLoops / TESTHOMO_LOOPS);
    dim3 threads(TESTHOMO_TESTS, TESTHOMO_LOOPS);
    TestHomographies<<<blocks, threads>>>(d_coord, d_homo, d_randPts, numPtsUp, thresh * thresh);
    cudaDeviceSynchronize();
    checkMsg("TestHomographies() execution failed\n");
    cudaMemcpy(h_randPts, d_randPts, sizeof(int) * numLoops, cudaMemcpyDeviceToHost);
    int maxIndex = -1, maxCount = -1;
    for (int i = 0; i < numLoops; i++)
      if (h_randPts[i] > maxCount)
      {
        maxCount = h_randPts[i];
        maxIndex = i;
      }
    *numMatches = maxCount;
    cudaMemcpy2D(homography, szFl, &d_homo[maxIndex], sizeof(float) * numLoops, szFl, 8, cudaMemcpyDeviceToHost);
  }
  free(validPts);
  free(h_randPts);
  cudaFree(d_homo);
  cudaFree(d_randPts);
  cudaFree(d_coord);
}

void CudaImage::Allocate(int w, int h, int p, bool host, float *devmem, float *hostmem, bool withPixelFlags)
{
  width = w;
  height = h;
  pitch = p;
  d_data = devmem;
  h_data = hostmem;
  t_data = NULL;
  if (devmem == NULL)
  {
    cudaMallocPitch((void **)&d_data, (size_t *)&pitch, (size_t)(sizeof(float) * width), (size_t)height);
    pitch /= sizeof(float);
    if (d_data == NULL)
      std::cout << "Failed to allocate device data" << std::endl;
    d_internalAlloc = true;
  }
  if (withPixelFlags == true)
  {
    cudaMallocPitch((void **)&d_pixelFlags, (size_t *)&pitch, (size_t)(sizeof(bool) * width), (size_t)height);
    cudaMallocPitch((void **)&d_pixHeight, (size_t *)&pitch, (size_t)(sizeof(float) * width), (size_t)height);
    cudaMallocPitch((void **)&d_result, (size_t *)&pitch, (size_t)(sizeof(float) * width), (size_t)height);
    if (d_pixelFlags == NULL)
      std::cout << "Failed to allocate device data" << std::endl;
  }
  if (host && hostmem == NULL)
  {
    h_data = (float *)malloc(sizeof(float) * pitch * height);
    h_internalAlloc = true;
  }
}

CudaImage::CudaImage() : width(0), height(0), d_data(NULL), h_data(NULL), t_data(NULL), d_internalAlloc(false), h_internalAlloc(false)
{
}

CudaImage::~CudaImage()
{
  if (d_internalAlloc && d_data != NULL)
    cudaFree(d_data);
  d_data = NULL;
  if (h_internalAlloc && h_data != NULL)
    free(h_data);
  h_data = NULL;
  if (t_data != NULL)
    cudaFreeArray((cudaArray *)t_data);
  t_data = NULL;
}

void CudaImage::Download()
{
  int p = sizeof(float) * pitch;
  if (d_data != NULL && h_data != NULL)
    cudaMemcpy2D(d_data, p, h_data, sizeof(float) * width, sizeof(float) * width, height, cudaMemcpyHostToDevice);
}

void CudaImage::Readback()
{
  int p = sizeof(float) * pitch;
  cudaMemcpy2D(h_data, sizeof(float) * width, d_data, p, sizeof(float) * width, height, cudaMemcpyDeviceToHost);
}

void CudaImage::InitTexture()
{
  cudaChannelFormatDesc t_desc = cudaCreateChannelDesc<float>();
  cudaMallocArray((cudaArray **)&t_data, &t_desc, pitch, height);
  if (t_data == NULL)
    std::cout << "Failed to allocated texture data" << std::endl;
}

void CudaImage::CopyToTexture(CudaImage &dst, bool host)
{
  if (dst.t_data == NULL)
  {
    std::cout << "Error CopyToTexture: No texture data" << std::endl;
    return;
  }
  if ((!host || h_data == NULL) && (host || d_data == NULL))
  {
    std::cout << "Error CopyToTexture: No source data" << std::endl;
    return;
  }
  if (host)
    cudaMemcpy2DToArray((cudaArray *)dst.t_data, 0, 0, h_data, sizeof(float) * pitch * dst.height, sizeof(float) * pitch * dst.height, 1, cudaMemcpyHostToDevice);
  else
    cudaMemcpy2DToArray((cudaArray *)dst.t_data, 0, 0, h_data, sizeof(float) * pitch * dst.height, sizeof(float) * pitch * dst.height, 1, cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
}

void Computation::MatchSiftData(SiftData &data1, SiftData &data2, cudaStream_t stream)
{
  int numPts1 = data1.numPts;
  int numPts2 = data2.numPts;
  // printf( "NumPts1 = %d, NumPts2 = %d\n",numPts1,numPts2);
  if (!numPts1 || !numPts2)
    return;
#ifdef MANAGEDMEM
  SiftPoint *sift1 = data1.m_data;
  SiftPoint *sift2 = data2.m_data;
#else
  if (data1.d_data == NULL || data2.d_data == NULL)
    return;
  SiftPoint *sift1 = data1.d_data;
  SiftPoint *sift2 = data2.d_data;
#endif

// Original version with correlation and maximization in two different kernels
// Global memory reguirement: O(N^2)
#if 0
  float *d_corrData; 
  int corrWidth = iDivUp(numPts2, 16)*16;
  int corrSize = sizeof(float)*numPts1*corrWidth;
  safeCall(cudaMalloc((void **)&d_corrData, corrSize));
#if 0 // K40c 10.9ms, 1080 Ti 3.8ms
  dim3 blocks1(numPts1, iDivUp(numPts2, 16));
  dim3 threads1(16, 16); // each block: 1 points x 16 points
  MatchSiftPoints<<<blocks1, threads1>>>(sift1, sift2, d_corrData, numPts1, numPts2);
#else // K40c 7.6ms, 1080 Ti 1.4ms
  dim3 blocks(iDivUp(numPts1,16), iDivUp(numPts2, 16));
  dim3 threads(16, 16); // each block: 16 points x 16 points
  MatchSiftPoints2<<<blocks, threads>>>(sift1, sift2, d_corrData, numPts1, numPts2);
#endif
  safeCall(cudaDeviceSynchronize());
  dim3 blocksMax(iDivUp(numPts1, 16));
  dim3 threadsMax(16, 16);
  FindMaxCorr<<<blocksMax, threadsMax>>>(d_corrData, sift1, sift2, numPts1, corrWidth, sizeof(SiftPoint));
  safeCall(cudaDeviceSynchronize());
  checkMsg("FindMaxCorr() execution failed\n");
  safeCall(cudaFree(d_corrData));
#endif

// Version suggested by Nicholas Lin with combined correlation and maximization
// Global memory reguirement: O(N)
#if 0 // K40c 51.2ms, 1080 Ti 9.6ms
  int block_dim = 16;
  float *d_corrData;
  int corrSize = numPts1 * block_dim * 2;
  safeCall(cudaMalloc((void **)&d_corrData, sizeof(float) * corrSize));
  dim3 blocks(iDivUp(numPts1, block_dim));
  dim3 threads(block_dim, block_dim); 
  FindMaxCorr3<<<blocks, threads >>>(d_corrData, sift1, sift2, numPts1, numPts2);
  safeCall(cudaDeviceSynchronize());
  checkMsg("FindMaxCorr3() execution failed\n");
  safeCall(cudaFree(d_corrData));
#endif

// Combined version with no global memory requirement using one 1 point per block
#if 0 // K40c 8.9ms, 1080 Ti 2.1ms, 2080 Ti 1.0ms
  dim3 blocksMax(numPts1);
  dim3 threadsMax(FMC2W, FMC2H);
  FindMaxCorr2<<<blocksMax, threadsMax>>>(sift1, sift2, numPts1, numPts2);
  safeCall(cudaDeviceSynchronize());
  checkMsg("FindMaxCorr2() execution failed\n");
#endif

// Combined version with no global memory requirement using one FMC2H points per block
#if 0 // K40c 9.2ms, 1080 Ti 1.3ms, 2080 Ti 1.1ms
  dim3 blocksMax2(iDivUp(numPts1, FMC2H));
  dim3 threadsMax2(FMC2W, FMC2H);
  FindMaxCorr4<<<blocksMax2, threadsMax2>>>(sift1, sift2, numPts1, numPts2);
  safeCall(cudaDeviceSynchronize());
  checkMsg("FindMaxCorr4() execution failed\n");
#endif

// Combined version with no global memory requirement using global locks
#if 1
  dim3 blocksMax3(iDivUp(numPts1, 16), iDivUp(numPts2, 512));
  dim3 threadsMax3(16, 16);
  CleanMatches<<<iDivUp(numPts1, 64), 64>>>(sift1, numPts1);
  int mode = 5;
  if (mode == 5) // K40c 5.0ms, 1080 Ti 1.2ms, 2080 Ti 0.83ms
    FindMaxCorr5<<<blocksMax3, threadsMax3, 0, stream>>>(sift1, sift2, numPts1, numPts2);
  else if (mode == 6)
  { // 2080 Ti 0.89ms
    threadsMax3 = dim3(32, 16);
    FindMaxCorr6<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
  }
  else if (mode == 7) // 2080 Ti 0.50ms
    FindMaxCorr7<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
  else if (mode == 8)
  { // 2080 Ti 0.45ms
    blocksMax3 = dim3(iDivUp(numPts1, FMC_BW), iDivUp(numPts2, FMC_GH));
    threadsMax3 = dim3(FMC_NW, FMC_NH);
    FindMaxCorr8<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
  }
  else if (mode == 9)
  { // 2080 Ti 0.46ms
    blocksMax3 = dim3(iDivUp(numPts1, FMC_BW), iDivUp(numPts2, FMC_GH));
    threadsMax3 = dim3(FMC_NW, FMC_NH);
    FindMaxCorr9<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
  }
  else if (mode == 10)
  { // 2080 Ti 0.24ms
    blocksMax3 = dim3(iDivUp(numPts1, M7W));
    threadsMax3 = dim3(M7W, M7H / M7R);
    FindMaxCorr10<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
  }
  cudaDeviceSynchronize();
  checkMsg("FindMaxCorr5() execution failed\n");
#endif
  //printf("Siftpoint number %d: x: %f, y: %f, score: %f\n",data1.numPts/2, data1.d_data[data1.numPts/2].xpos, data1.d_data[data1.numPts/2].ypos, data1.d_data[data1.numPts/2].score);
  if (data1.h_data != NULL)
  {
    float *h_ptr = &data1.h_data[0].score;
    float *d_ptr = &data1.d_data[0].score;
    cudaMemcpy2D(h_ptr, sizeof(SiftPoint), d_ptr, sizeof(SiftPoint), 5 * sizeof(float), data1.numPts, cudaMemcpyDeviceToHost);
  }

  return;
}

void Computation::yuvToRGB_Convert(CudaImage &RGBImage, unsigned char *yuvdata)
{

  int width = 1920;
  int height = 1080;
  dim3 blocks = dim3(1024);
  dim3 threads = dim3((RGBImage.width / 3 + 1024 - 1) / 1024);

  gpuConvertYUYVtoRGBfloat_kernel<<<blocks, threads>>>(yuvdata, RGBImage.d_data, width, height);
}

void Computation::sharpenImage(CudaImage &greyscaleImage, unsigned char *yuvdata, float amount)
{

  dim3 block(16, 16, 1);
  dim3 grid(1920 / block.x + 1, 1080 / block.y + 1, 1);

  gpuSharpenImageToGrayscale<<<grid, block, 0>>>(yuvdata, greyscaleImage.d_data, 1920, 1080, amount);
}

void Computation::drawSiftData(uint16_t *rgbImage, CudaImage &greyscaleImage, SiftData &siftData, int width, int height, cudaStream_t stream)
{
  dim3 blocks = dim3(2048);
  dim3 threads = dim3(1);
  gpuDrawSiftData<<<blocks, threads, 0, stream>>>(rgbImage, greyscaleImage.d_data, siftData.d_data, siftData.numPts, width, height);
  checkMsg("gpuDrawSiftData:\n");
}

void Computation::rpi_camera_undistort(uint16_t *dst, uint16_t *src, uint16_t *xCoordsPerPixel, uint16_t *yCoordsPerPixel, cudaStream_t stream)
{
  dim3 blocks = dim3(2048);
  dim3 threads = dim3(1);
  int width_dst = 1273;
  int height_dst = 709;
  int width_src = 1280;
  int height_src = 720;
  gpuUndistort_uint16<<<blocks, threads, 0, stream>>>(dst, src, xCoordsPerPixel, yCoordsPerPixel, width_src, height_src, width_dst, height_dst);
  checkMsg("Problem with RPI gpuUndistort:\n");
}

void Computation::tof_camera_undistort(float *dst, uint16_t *src, uint16_t *xCoordsPerPixel, uint16_t *yCoordsPerPixel, cudaStream_t stream, float *cosAlpha)
{

  dim3 blocks = dim3(512);
  dim3 threads = dim3(1);
  int width_dst = 256;
  int height_dst = 205;
  int width_src = 352;
  int height_src = 286;
  if (cosAlpha == NULL)
  {
    gpuUndistort<<<blocks, threads, 0, stream>>>(dst, src, xCoordsPerPixel, yCoordsPerPixel, width_src, height_src, width_dst, height_dst);
  }
  else
  {
    gpuUndistortCosAlpha<<<blocks, threads, 0, stream>>>(dst, src, xCoordsPerPixel, yCoordsPerPixel, width_src, height_src, width_dst, height_dst, cosAlpha);
  }
  checkMsg("Problem with TOF gpuUndistort:\n");
}

void Computation::buffer_Float_to_uInt16x4(uint16_t *dst, float *src, int width, int height, cudaStream_t stream)
{

  dim3 blocks = dim3(512);
  dim3 threads = dim3(1);
  gpuNormalizeToInt16<<<blocks, threads, 0, stream>>>(dst, src, width, height);
  checkMsg("Problem with gpuNormalizeToInt16:\n");
}

void Computation::buffer_Float_to_uInt16x4_SCALE(uint16_t *dst, float *src, int width, int height, cudaStream_t stream)
{

  dim3 blocks = dim3(512);
  dim3 threads = dim3(1);
  gpuNormalizeToInt16_SCALE<<<blocks, threads, 0, stream>>>(dst, src, width, height);
  checkMsg("Problem with gpuNormalizeToInt16:\n");
}

void Computation::scale_float_to_float(float *dst, float *src, int width, int height, cudaStream_t stream)
{

  dim3 blocks = dim3(1);
  dim3 threads = dim3(512);
  gpuScaleFloat2Float<<<blocks, threads, 0, stream>>>(dst, src, width, height);
  checkMsg("Problem with gpuScaleFloat2Float:\n");
}

void Computation::buffer_uint16x4_to_Float(float *dst, uint16_t *src, int width, int height, cudaStream_t stream)
{

  dim3 blocks = dim3(512);
  dim3 threads = dim3(1);
  gpuNormalizeToFloat<<<blocks, threads, 0, stream>>>(dst, src, width, height);
  checkMsg("Problem with gpuNormalizeToInt16:\n");
}

void Computation::tof_sobel(float *dst_mag, float *dst_phase, float *src, cudaStream_t stream)
{

  int width = 265;
  int height = 205;

  dim3 block(16, 16, 1);
  dim3 grid(265 / block.x + 1, 205 / block.y + 1, 1);
  gpuSobelToF<<<grid, block, 0, stream>>>(dst_mag, dst_phase, src, width, height);
  checkMsg("Problem with gpuSobelToF:\n");
}

void Computation::tof_maxfilter_3x3(float *dst_mag, float *src, cudaStream_t stream)
{

  int width = 256;
  int height = 205;

  dim3 block(16, 16, 1);
  dim3 grid(265 / block.x + 1, 205 / block.y + 1, 1);
  gpuMaxfilterToF<<<grid, block, 0, stream>>>(dst_mag, src, width, height);
  checkMsg("Problem with gpuMaxFilter:\n");
}

void Computation::tof_minfilter_3x3(float *dst_mag, float *src, cudaStream_t stream)
{

  int width = 256;
  int height = 205;

  dim3 block(16, 16, 1);
  dim3 grid(265 / block.x + 1, 205 / block.y + 1, 1);
  gpuMinfilterToF<<<grid, block, 0, stream>>>(dst_mag, src, width, height);
  checkMsg("Problem with gpuMaxFilter:\n");
}

void Computation::tof_meanfilter_3x3(float *dst_mag, float *src, cudaStream_t stream)
{

  int width = 256;
  int height = 205;

  dim3 block(16, 16, 1);
  dim3 grid(265 / block.x + 1, 205 / block.y + 1, 1);
  gpuMeanfilterToF<<<grid, block, 0, stream>>>(dst_mag, src, width, height);
  checkMsg("Problem with gpuMaxFilter:\n");
}

void Computation::tof_medianfilter_3x3(float *dst_mag, float *src, cudaStream_t stream)
{

  int width = 256;
  int height = 205;

  dim3 block(16, 16, 1);
  dim3 grid(265 / block.x + 1, 205 / block.y + 1, 1);
  gpuMedianfilterToF<<<grid, block, 0, stream>>>(dst_mag, src, width, height);
  checkMsg("Problem with gpuMaxFilter:\n");
}

void Computation::tof_fill_area(float *mask, float *src, int seed_x, int seed_y, float thresh, cudaStream_t stream)
{

  int width = 265;
  int height = 205;

  dim3 block(16, 16, 1);
  dim3 grid(265 / block.x + 1, 205 / block.y + 1, 1);
  gpuFillAreaToF<<<grid, block, 0, stream>>>(mask, src, width, height, seed_x, seed_y, thresh);
  checkMsg("Problem with gpuMaxFilter:\n");
}

int8_t Computation::gpuConvertBayer10toRGB(uint16_t *src, uint16_t *dst, const int width, const int height, const enum AVPixelFormat format, const uint8_t bpp, cudaStream_t stream)
{
  dim3 threads_p_block;
  dim3 blocks_p_grid;

  int2 pos_r;
  int2 pos_gr;
  int2 pos_gb;
  int2 pos_b;

  uint16_t *d_src = NULL;
  uint16_t *d_dst = NULL;

  unsigned int flags;

  size_t planeSize = width * height * sizeof(unsigned char);
  // checkMsg("Problem before cudaHostGetFlags:\n");
  bool srcIsMapped = (cudaHostGetFlags(&flags, (void *)src) == cudaSuccess) && (flags & cudaHostAllocMapped);

  checkMsgNoFail("Problem with cudaHostGetFlags:\n");
#if MAP_HOST_DEVICE_POINTER
  if (!srcIsMapped)
  {
    cudaHostRegister(src, planeSize * 2, cudaHostRegisterMapped);
  }

  cudaHostGetDevicePointer(&d_src, src, 0);
#else
  if (srcIsMapped)
  {
    d_src = src;
    cudaStreamAttachMemAsync(NULL, src, 0, cudaMemAttachGlobal);
  }
  else
  {
    cudaMalloc(&d_src, planeSize * 2);
    cudaMemcpy(d_src, src, planeSize * 2, cudaMemcpyHostToDevice);
  }
#endif

  d_dst = dst;
  // cudaStreamAttachMemAsync(NULL, dst, 0, cudaMemAttachGlobal);

  threads_p_block = dim3(32, 32);
  blocks_p_grid.x = (width / 2 +
                     threads_p_block.x - 1) /
                    threads_p_block.x;
  blocks_p_grid.y = (height / 2 +
                     threads_p_block.y - 1) /
                    threads_p_block.y;

  switch (format)
  {
  case AV_PIX_FMT_BAYER_BGGR10:
    pos_r = make_int2(0, 0);
    pos_gr = make_int2(1, 0);
    pos_gb = make_int2(0, 1);
    pos_b = make_int2(1, 1);
    break;
  case AV_PIX_FMT_BAYER_GBRG10:
    pos_r = make_int2(1, 0);
    pos_gr = make_int2(0, 0);
    pos_gb = make_int2(1, 1);
    pos_b = make_int2(0, 1);
    break;
  case AV_PIX_FMT_BAYER_GRBG10:
    pos_r = make_int2(0, 1);
    pos_gr = make_int2(1, 1);
    pos_gb = make_int2(0, 0);
    pos_b = make_int2(1, 0);
    break;
  case AV_PIX_FMT_BAYER_RGGB10:
    pos_r = make_int2(1, 1);
    pos_gr = make_int2(0, 1);
    pos_gb = make_int2(1, 0);
    pos_b = make_int2(0, 0);
    break;
  default:
    return RTN_ERROR;
    break;
  }

  bayer_to_rgb<<<blocks_p_grid, threads_p_block, 0, stream>>>(d_src, d_dst, width, height, bpp, pos_r, pos_gr, pos_gb, pos_b);
  checkMsg("Problem with bayer_to_rgb:\n");
  cudaStreamSynchronize(stream);

  // cudaStreamAttachMemAsync(NULL, dst, 0, cudaMemAttachHost);

#if !MAP_HOST_DEVICE_POINTER
  if (!srcIsMapped)
  {
    cudaFree(d_src);
  }
  else
  {
    cudaStreamAttachMemAsync(NULL, src, 0, cudaMemAttachHost);
  }
#endif

  // cudaStreamSynchronize(NULL);

  return RTN_SUCCESS;
}

void Computation::addDepthInfoToSift(SiftData &data, float *depthData, cudaStream_t stream, float *x, float *y, float *z, float *conf)
{
  dim3 blocks = dim3(512);
  dim3 threads = dim3(1);
#ifdef USE_TEST_DATA
  data.numPts = 5;
#endif
  gpuAddDepthInfoToSift<<<blocks, threads, 0, stream>>>(data.d_data, depthData, data.numPts, x, y, z, conf);
  checkMsg("Problem with gpuAddDepthInfoToSift:\n");
}

void Computation::findRotationTranslation_step0(SiftData &data, float *tempMemory, bool *index_list, mat4x4 *rotation, vec4 *translation, cudaStream_t stream)
{
  dim3 blocks = dim3(512);
  dim3 threads = dim3(1);
  gpuFindRotationTranslation_step0<<<blocks, threads, 0, stream>>>(data.d_data, tempMemory, index_list, rotation, translation, data.numPts);
  checkMsg("Problem with findRotationTranslation:\n");
}

void Computation::findRotationTranslation_step1(SiftData &data, float *tempMemory, bool *index_list, mat4x4 *rotation, vec4 *translation, cudaStream_t stream)
{
  dim3 blocks = dim3(1);
  dim3 threads = dim3(1);
  gpuFindRotationTranslation_step1<<<blocks, threads, 0, stream>>>(data.d_data, tempMemory, index_list, rotation, translation, data.numPts);
  checkMsg("Problem with findRotationTranslation:\n");
}

void Computation::findRotationTranslation_step2(SiftData &data, float *tempMemory, bool *index_list, mat4x4 *rotation, vec4 *translation, cudaStream_t stream)
{
  dim3 blocks = dim3(1);
  dim3 threads = dim3(1);
  gpuFindRotationTranslation_step2<<<blocks, threads, 0, stream>>>(data.d_data, tempMemory, index_list, rotation, translation, data.numPts);
  checkMsg("Problem with findRotationTranslation:\n");
}

void Computation::ransacFromFoundRotationTranslation(SiftData &data, SiftData &data_old, mat4x4 *rotation, vec4 *translation, cudaStream_t stream)
{
  dim3 blocks = dim3(512);
  dim3 threads = dim3(1);
  gpuRematchSiftPoints<<<blocks, threads, 0, stream>>>(data.d_data, data_old.d_data, rotation, translation, data.numPts, data_old.numPts);
  checkMsg("Problem with gpuRematchSiftPoints:\n");
}

void Computation::ransac2d(SiftData &data, SiftData &data_old, float *tempMemory, bool *index_list, float *dx, float *dy, cudaStream_t stream)
{
  dim3 blocks = dim3(1);
  dim3 threads = dim3(512);
  gpuRansac2d<<<blocks, threads, 0, stream>>>(data.d_data, data_old.d_data, tempMemory, index_list, dx, dy, data.numPts, data_old.numPts);
  checkMsg("Problem with findRotationTranslation:\n");
}

void Computation::findOptimalRotationTranslation(SiftData &data, float *tempMemory, mat4x4 *rotation, vec4 *translation, cudaStream_t stream)
{
  dim3 blocks = dim3(1);
  dim3 threads = dim3(1);
  gpuFindOptimalRotationTranslation<<<blocks, threads, 0, stream>>>(data.d_data, tempMemory, rotation, translation, data.numPts);
  checkMsg("Problem with findRotationTranslation:\n");
}