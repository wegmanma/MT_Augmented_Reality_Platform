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
#include "computation.cuh"

// 16:9 display format.
#define GRIDSIZEX 1920
#define GRIDSIZEY 1080
#define GAUSSIANSIZE 25

#define checkMsg(msg)       __checkMsg(msg, __FILE__, __LINE__)

cudaExternalMemory_t cudaExtMemVertexBuffer;
cudaExternalMemory_t cudaExtMemIndexBuffer;
cudaExternalSemaphore_t cudaExtCudaUpdateVkVertexBufSemaphore;
cudaExternalSemaphore_t cudaExtVkUpdateCudaVertexBufSemaphore;

__constant__ int d_MaxNumPoints;
__device__ unsigned int d_PointCounter[8*2+1];
__constant__ float d_ScaleDownKernel[5]; 
__constant__ float d_LowPassKernel[2*LOWPASS_R+1];
__constant__ float d_GaussKernel[GAUSSIANSIZE*GAUSSIANSIZE]; 
__constant__ float d_LaplaceKernel[8*12*16]; 



int iDivUp(int a, int b) { return (a%b != 0) ? (a/b + 1) : (a/b); }
int iDivDown(int a, int b) { return a/b; }
int iAlignUp(int a, int b) { return (a%b != 0) ?  (a - a%b + b) : a; }
int iAlignDown(int a, int b) { return a - a%b; }

template <class T>
__device__ __inline__ T ShiftDown(T var, unsigned int delta, int width = 32) {
#if (CUDART_VERSION >= 9000)
  return __shfl_down_sync(0xffffffff, var, delta, width);
#else
  return __shfl_down(var, delta, width);
#endif
}

template <class T>
__device__ __inline__ T ShiftUp(T var, unsigned int delta, int width = 32) {
#if (CUDART_VERSION >= 9000)
  return __shfl_up_sync(0xffffffff, var, delta, width);
#else
  return __shfl_up(var, delta, width);
#endif
}

template <class T>
__device__ __inline__ T Shuffle(T var, unsigned int lane, int width = 32) {
#if (CUDART_VERSION >= 9000)
  return __shfl_sync(0xffffffff, var, lane, width);
#else
  return __shfl(var, lane, width);
#endif
}

__device__ inline float clamp(float val, float mn, float mx)
{
	return (val >= mn)? ((val <= mx)? val : mx) : mn;
}


__global__ void CopyCharToImage(unsigned char *src, float *dst,
		unsigned int width, unsigned int height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx*2 >= width) {
		return;
	}

	for (int i = 0; i < height; ++i) {
		int y0 = __uint2float_rn(src[i*width*2+idx*4+1]);
		int y1 = __uint2float_rn(src[i*width*2+idx*4+3]);

		dst[i*width+idx*2+0] = clamp(y0, 0.0f, 255.0f);
		dst[i*width+idx*2+1] = clamp(y1, 0.0f, 255.0f);
	}
}



__global__ void ScaleDownKernel(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch)
{
  __shared__ float inrow[SCALEDOWN_W+4]; 
  __shared__ float brow[5*(SCALEDOWN_W/2)];
  __shared__ int yRead[SCALEDOWN_H+4];
  __shared__ int yWrite[SCALEDOWN_H+4];
  #define dx2 (SCALEDOWN_W/2)
  const int tx = threadIdx.x;
  const int tx0 = tx + 0*dx2;
  const int tx1 = tx + 1*dx2;
  const int tx2 = tx + 2*dx2;
  const int tx3 = tx + 3*dx2;
  const int tx4 = tx + 4*dx2;
  const int xStart = blockIdx.x*SCALEDOWN_W;
  const int yStart = blockIdx.y*SCALEDOWN_H;
  const int xWrite = xStart/2 + tx;
  float k0 = d_ScaleDownKernel[0];
  float k1 = d_ScaleDownKernel[1];
  float k2 = d_ScaleDownKernel[2];
  if (tx<SCALEDOWN_H+4) {
    int y = yStart + tx - 2; 
    y = (y<0 ? 0 : y);
    y = (y>=height ? height-1 : y);
    yRead[tx] = y*pitch;
    yWrite[tx] = (yStart + tx - 4)/2 * newpitch;
  }
  __syncthreads();
  int xRead = xStart + tx - 2;
  xRead = (xRead<0 ? 0 : xRead);
  xRead = (xRead>=width ? width-1 : xRead);

  int maxtx = min(dx2, width/2 - xStart/2);
  for (int dy=0;dy<SCALEDOWN_H+4;dy+=5) {
    {
      inrow[tx] = d_Data[yRead[dy+0] + xRead];
      __syncthreads();
      if (tx<maxtx) {
	brow[tx4] = k0*(inrow[2*tx]+inrow[2*tx+4]) + k1*(inrow[2*tx+1]+inrow[2*tx+3]) + k2*inrow[2*tx+2];
	if (dy>=4 && !(dy&1))
	  d_Result[yWrite[dy+0] + xWrite] = k2*brow[tx2] + k0*(brow[tx0]+brow[tx4]) + k1*(brow[tx1]+brow[tx3]);
      }
      __syncthreads();
    }
    if (dy<(SCALEDOWN_H+3)) {
      inrow[tx] = d_Data[yRead[dy+1] + xRead];
      __syncthreads();
      if (tx<maxtx) {
	brow[tx0] = k0*(inrow[2*tx]+inrow[2*tx+4]) + k1*(inrow[2*tx+1]+inrow[2*tx+3]) + k2*inrow[2*tx+2];
	if (dy>=3 && (dy&1))
	  d_Result[yWrite[dy+1] + xWrite] = k2*brow[tx3] + k0*(brow[tx1]+brow[tx0]) + k1*(brow[tx2]+brow[tx4]);
      }
      __syncthreads();
    }
    if (dy<(SCALEDOWN_H+2)) {
      inrow[tx] = d_Data[yRead[dy+2] + xRead];
      __syncthreads();
      if (tx<maxtx) {
	brow[tx1] = k0*(inrow[2*tx]+inrow[2*tx+4]) + k1*(inrow[2*tx+1]+inrow[2*tx+3]) + k2*inrow[2*tx+2];
	if (dy>=2 && !(dy&1))
	  d_Result[yWrite[dy+2] + xWrite] = k2*brow[tx4] + k0*(brow[tx2]+brow[tx1]) + k1*(brow[tx3]+brow[tx0]);
      }
      __syncthreads();
    }
    if (dy<(SCALEDOWN_H+1)) {
      inrow[tx] = d_Data[yRead[dy+3] + xRead];
      __syncthreads();
      if (tx<maxtx) {
	brow[tx2] = k0*(inrow[2*tx]+inrow[2*tx+4]) + k1*(inrow[2*tx+1]+inrow[2*tx+3]) + k2*inrow[2*tx+2];
	if (dy>=1 && (dy&1))
	  d_Result[yWrite[dy+3] + xWrite] = k2*brow[tx0] + k0*(brow[tx3]+brow[tx2]) + k1*(brow[tx4]+brow[tx1]);
      }
      __syncthreads();
    }
    if (dy<SCALEDOWN_H) {
      inrow[tx] = d_Data[yRead[dy+4] + xRead];
      __syncthreads();
      if (tx<dx2 && xWrite<width/2) {
	brow[tx3] = k0*(inrow[2*tx]+inrow[2*tx+4]) + k1*(inrow[2*tx+1]+inrow[2*tx+3]) + k2*inrow[2*tx+2];
	if (!(dy&1))
	  d_Result[yWrite[dy+4] + xWrite] = k2*brow[tx1] + k0*(brow[tx4]+brow[tx3]) + k1*(brow[tx0]+brow[tx2]);
      }
      __syncthreads();
    }
  }
}

__global__ void ScaleUpKernel(float *d_Result, float *d_Data, int width, int pitch, int height, int newpitch)
{
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  int x = blockIdx.x*SCALEUP_W + 2*tx;
  int y = blockIdx.y*SCALEUP_H + 2*ty;
  if (x<2*width && y<2*height) {
    int xl = blockIdx.x*(SCALEUP_W/2) + tx;
    int yu = blockIdx.y*(SCALEUP_H/2) + ty;
    int xr = min(xl + 1, width - 1);
    int yd = min(yu + 1, height - 1);
    float vul = d_Data[yu*pitch + xl];
    float vur = d_Data[yu*pitch + xr];
    float vdl = d_Data[yd*pitch + xl];
    float vdr = d_Data[yd*pitch + xr];
    d_Result[(y + 0)*newpitch + x + 0] = vul;
    d_Result[(y + 0)*newpitch + x + 1] = 0.50f*(vul + vur);
    d_Result[(y + 1)*newpitch + x + 0] = 0.50f*(vul + vdl);
    d_Result[(y + 1)*newpitch + x + 1] = 0.25f*(vul + vur + vdl + vdr);
  }
}

/* TODO: Adjust to correct YUYV alignment*/
__global__ void ScaleUpKernelCharYUYV(float *d_Result, unsigned char *d_Data, int width, int pitch, int height, int newpitch)
{
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  int x = blockIdx.x*SCALEUP_W + 2*tx;
  int y = blockIdx.y*SCALEUP_H + 2*ty;
  if (x<2*width && y<2*height) {
    int xl = blockIdx.x*(SCALEUP_W/2) + tx;
    int yu = blockIdx.y*(SCALEUP_H/2) + ty;
    int xr = min(xl + 1, width - 1);
    int yd = min(yu + 1, height - 1);
    float vul = d_Data[yu*pitch*2 + xl*2+1];
    float vur = d_Data[yu*pitch*2 + xr*2+1];
    float vdl = d_Data[yd*pitch*2 + xl*2+1];
    float vdr = d_Data[yd*pitch*2 + xr*2+1];
    d_Result[(y + 0)*newpitch + x + 0] = vul;
    d_Result[(y + 0)*newpitch + x + 1] = 0.50f*(vul + vur);
    d_Result[(y + 1)*newpitch + x + 0] = 0.50f*(vul + vdl);
    d_Result[(y + 1)*newpitch + x + 1] = 0.25f*(vul + vur + vdl + vdr);
  }
}

__global__ void LowPassBlock(float *d_Image, float *d_Result, int width, int pitch, int height)
{
  __shared__ float xrows[16][32];          
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int xp = blockIdx.x*LOWPASS_W + tx;
  const int yp = blockIdx.y*LOWPASS_H + ty;
  const int N = 16;
  float *k = d_LowPassKernel;
  int xl = max(min(xp - 4, width-1), 0);
#pragma unroll
  for (int l=-8;l<4;l+=4) {
    int ly = l + ty;
    int yl = max(min(yp + l + 4, height-1), 0);
    float val = d_Image[yl*pitch + xl];
    val = k[4]*ShiftDown(val, 4) +
      k[3]*(ShiftDown(val, 5) + ShiftDown(val, 3)) +
      k[2]*(ShiftDown(val, 6) + ShiftDown(val, 2)) +
      k[1]*(ShiftDown(val, 7) + ShiftDown(val, 1)) +
      k[0]*(ShiftDown(val, 8) + val);
    xrows[ly + 8][tx] = val;
  }
  __syncthreads();
#pragma unroll
  for (int l=4;l<LOWPASS_H;l+=4) {
    int ly = l + ty;
    int yl = min(yp + l + 4, height-1);
    float val = d_Image[yl*pitch + xl];
    val = k[4]*ShiftDown(val, 4) +
      k[3]*(ShiftDown(val, 5) + ShiftDown(val, 3)) +
      k[2]*(ShiftDown(val, 6) + ShiftDown(val, 2)) +
      k[1]*(ShiftDown(val, 7) + ShiftDown(val, 1)) +
      k[0]*(ShiftDown(val, 8) + val);
    xrows[(ly + 8)%N][tx] = val;
    int ys = yp + l - 4;
    if (xp<width && ys<height && tx<LOWPASS_W)
      d_Result[ys*pitch + xp] = k[4]*xrows[(ly + 0)%N][tx] +
		       k[3]*(xrows[(ly - 1)%N][tx] + xrows[(ly + 1)%N][tx]) +
		       k[2]*(xrows[(ly - 2)%N][tx] + xrows[(ly + 2)%N][tx]) +
		       k[1]*(xrows[(ly - 3)%N][tx] + xrows[(ly + 3)%N][tx]) +
		       k[0]*(xrows[(ly - 4)%N][tx] + xrows[(ly + 4)%N][tx]);
    __syncthreads();
  }
  int ly = LOWPASS_H + ty;
  int ys = yp + LOWPASS_H - 4;
  if (xp<width && ys<height && tx<LOWPASS_W)
    d_Result[ys*pitch + xp] = k[4]*xrows[(ly + 0)%N][tx] +
		     k[3]*(xrows[(ly - 1)%N][tx] + xrows[(ly + 1)%N][tx]) +
		     k[2]*(xrows[(ly - 2)%N][tx] + xrows[(ly + 2)%N][tx]) +
		     k[1]*(xrows[(ly - 3)%N][tx] + xrows[(ly + 3)%N][tx]) +
		     k[0]*(xrows[(ly - 4)%N][tx] + xrows[(ly + 4)%N][tx]);
}

__global__ void LowPassBlockCharYUYV(unsigned char *d_Image, float *d_Result, int width, int pitch, int height)
{
  __shared__ float xrows[16][32];          
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int xp = blockIdx.x*LOWPASS_W + tx;
  const int yp = blockIdx.y*LOWPASS_H + ty;
  const int N = 16;
  float *k = d_LowPassKernel;
  int xl = max(min(xp - 4, width-1), 0);
#pragma unroll
  for (int l=-8;l<4;l+=4) {
    int ly = l + ty;
    int yl = max(min(yp + l + 4, height-1), 0);
    float val = __uint2float_rn(d_Image[yl*pitch*2 + xl*2+1]);
    val = k[4]*ShiftDown(val, 4) +
      k[3]*(ShiftDown(val, 5) + ShiftDown(val, 3)) +
      k[2]*(ShiftDown(val, 6) + ShiftDown(val, 2)) +
      k[1]*(ShiftDown(val, 7) + ShiftDown(val, 1)) +
      k[0]*(ShiftDown(val, 8) + val);
    xrows[ly + 8][tx] = val;
  }
  __syncthreads();
#pragma unroll
  for (int l=4;l<LOWPASS_H;l+=4) {
    int ly = l + ty;
    int yl = min(yp + l + 4, height-1);
    float val = __uint2float_rn(d_Image[yl*pitch*2 + xl*2+1]);
    val = k[4]*ShiftDown(val, 4) +
      k[3]*(ShiftDown(val, 5) + ShiftDown(val, 3)) +
      k[2]*(ShiftDown(val, 6) + ShiftDown(val, 2)) +
      k[1]*(ShiftDown(val, 7) + ShiftDown(val, 1)) +
      k[0]*(ShiftDown(val, 8) + val);
    xrows[(ly + 8)%N][tx] = val;
    int ys = yp + l - 4;
    if (xp<width && ys<height && tx<LOWPASS_W)
      d_Result[ys*pitch + xp] = k[4]*xrows[(ly + 0)%N][tx] +
		       k[3]*(xrows[(ly - 1)%N][tx] + xrows[(ly + 1)%N][tx]) +
		       k[2]*(xrows[(ly - 2)%N][tx] + xrows[(ly + 2)%N][tx]) +
		       k[1]*(xrows[(ly - 3)%N][tx] + xrows[(ly + 3)%N][tx]) +
		       k[0]*(xrows[(ly - 4)%N][tx] + xrows[(ly + 4)%N][tx]);
    __syncthreads();
  }
  int ly = LOWPASS_H + ty;
  int ys = yp + LOWPASS_H - 4;
  if (xp<width && ys<height && tx<LOWPASS_W)
    d_Result[ys*pitch + xp] = k[4]*xrows[(ly + 0)%N][tx] +
		     k[3]*(xrows[(ly - 1)%N][tx] + xrows[(ly + 1)%N][tx]) +
		     k[2]*(xrows[(ly - 2)%N][tx] + xrows[(ly + 2)%N][tx]) +
		     k[1]*(xrows[(ly - 3)%N][tx] + xrows[(ly + 3)%N][tx]) +
		     k[0]*(xrows[(ly - 4)%N][tx] + xrows[(ly + 4)%N][tx]);
}

__global__ void sinewave_gen_kernel(Computation::Vertex2 *vertices, unsigned int width, unsigned int height, float time)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    // calculate uv coordinates
    float u = x / (float) width;
    float v = x / (float) height;
    u = u*2.0f - 1.0f;
    v = v*2.0f - 1.0f;

    if (y*width+x < 1920)
    {
        // write output vertex
        vertices[y*width+x].pos[0] = u;
        vertices[y*width+x].pos[1] = -1.0f;
        vertices[y*width+x].pos[2] = 0.0f;
        vertices[y*width+x].pos[3] = 1.0f;
        vertices[y*width+x].color[0] = 1.0;
        vertices[y*width+x].color[1] = 1.0;
        vertices[y*width+x].color[2] = 1.0f;
    } else if (y*width+x < 3840)
    {
        // write output vertex
        vertices[y*width+x].pos[0] = u;
        vertices[y*width+x].pos[1] = 1.0f;
        vertices[y*width+x].pos[2] = 0.0;
        vertices[y*width+x].pos[3] = 1.0f;
        vertices[y*width+x].color[0] = 1.0;
        vertices[y*width+x].color[1] = 1.0;
        vertices[y*width+x].color[2] = 1.0f;
    } else if (y*width+x < 4920)
        {
        // write output vertex
        vertices[y*width+x].pos[0] = -1.0;
        vertices[y*width+x].pos[1] = v;
        vertices[y*width+x].pos[2] = 0.0;
        vertices[y*width+x].pos[3] = 1.0f;
        vertices[y*width+x].color[0] = 1.0;
        vertices[y*width+x].color[1] = 1.0;
        vertices[y*width+x].color[2] = 1.0f;
    }else if (y*width+x < 6000)
        {
        // write output vertex
        if (y*width+x < 5760) {
          v = v-2.0;
        }
        else {
          v = v+1.557407;
        }
        vertices[y*width+x].pos[0] = 1.0;
        vertices[y*width+x].pos[1] = v;
        vertices[y*width+x].pos[2] = 0.0;
        vertices[y*width+x].pos[3] = 1.0f;
        vertices[y*width+x].color[0] = 1.0;
        vertices[y*width+x].color[1] = 1.0;
        vertices[y*width+x].color[2] = 1.0f;
    }
}

__global__ void init_index_kernel(int* indexbuffer) {

unsigned int x = blockIdx.x*blockDim.x + threadIdx.x; // 0 <= x <= 159

  if (x>GRIDSIZEX-2) return;
    for (unsigned int y = 0; y<(GRIDSIZEY-1);y++) {
      if ((x==1918)&&(y==1078))printf("x: %d, y: %d, Buffer: %d: Upper Triangle: %d %d %d; lower triangle: %d %d %d\n",x,y,(y*(GRIDSIZEX-1)+x)*6, y*GRIDSIZEX+x, y*GRIDSIZEX+x+1, (y+1)*GRIDSIZEX+x+1, (y+1)*GRIDSIZEX+x+1, (y+1)*GRIDSIZEX+x,y*GRIDSIZEX+x); 
      indexbuffer[(y*(GRIDSIZEX-1)+x)*6+0] = y*GRIDSIZEX+x;
      indexbuffer[(y*(GRIDSIZEX-1)+x)*6+1] = y*GRIDSIZEX+x+1;
      indexbuffer[(y*(GRIDSIZEX-1)+x)*6+2] = (y+1)*GRIDSIZEX+x+1;
      indexbuffer[(y*(GRIDSIZEX-1)+x)*6+3] = (y+1)*GRIDSIZEX+x+1;
      indexbuffer[(y*(GRIDSIZEX-1)+x)*6+4] = (y+1)*GRIDSIZEX+x;
      indexbuffer[(y*(GRIDSIZEX-1)+x)*6+5] = y*GRIDSIZEX+x;
    }  
 }


__global__ void init_grid_kernel(Computation::Vertex2 *vertices) {

    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x; // 0 <= x <= 159
    if (x>GRIDSIZEX-1) return;
    
    float u = (float) x /  (float)(GRIDSIZEX-1);
    float v = 0;
    u = u*2.0f - 1.0f;  

    for (unsigned int y = 0; y< GRIDSIZEY ;y++) {
        v = y / (float)(GRIDSIZEY-1);
        v = v*2.0f - 1.0f;
        //printf("x=%d, u=%f, y=%d, v=%f, pos_in_vector: %d\n", x, u, y, v, y*GRIDSIZEX+x);
        vertices[y*GRIDSIZEX+x].pos[0] = u;
        vertices[y*GRIDSIZEX+x].pos[1] = v;
        vertices[y*GRIDSIZEX+x].pos[2] = 0.0;
        vertices[y*GRIDSIZEX+x].pos[3] = 1.0f;
        vertices[y*GRIDSIZEX+x].color[0] = 1.0;
        vertices[y*GRIDSIZEX+x].color[1] = 1.0;
        vertices[y*GRIDSIZEX+x].color[2] = 1.0f;
        vertices[y*GRIDSIZEX+x].texcoords[0] = u;
        vertices[y*GRIDSIZEX+x].texcoords[0] = v;       
    }
}


__global__ void clear_heightmap_kernel(float *d_Result, bool *d_Pixelflags, unsigned int width, unsigned int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width) return;    

    for (int i = 0; i < height; ++i) {
    d_Pixelflags[i*width+idx] = false;	
		d_Result[i*width+idx] = 0.0f;
	}
}

__global__ void draw_heightmap_kernel(float *d_Result, bool *d_Pixelflags, SiftPoint *d_Sift, unsigned int width, unsigned int height, int numPoints, int maxPoints)
{
    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

    // Sift features
    if (idx >= maxPoints) return;

    if ((idx <= numPoints)&&(d_Sift[idx].score>0.0f)) { 
        //if (20.0>(d_Sift[idx].ypos-d_Sift[idx].match_ypos)*(d_Sift[idx].ypos-d_Sift[idx].match_ypos)) return; 
        //if (d_Sift[idx].xpos<d_Sift[idx].match_xpos)  return;
        int x = d_Sift[idx].xpos;
        int y = d_Sift[idx].ypos;
        d_Pixelflags[x+width*y] = true;
        d_Result[x+width*y] = d_Sift[idx].match_error; 
    } else {    
      return;
    } 
}

__global__ void interpolate_heightmap_kernel(float *d_HeightMap, bool *d_Pixelflags, float *d_pixHeight, unsigned int width, unsigned int height) {

  unsigned int idx_x = blockIdx.x*blockDim.x + threadIdx.x; // 0 <= x <= 159
  unsigned int idx_y = blockIdx.y*blockDim.y + threadIdx.y; // 0 <= x <= 159
  if (idx_x>width-1) return;
  if (idx_y>height-1) return;

int x = idx_x;
int y = idx_y;

int weights = 1;
float heights = 0.0f;

  //float max_height = 0.0f;
  for (int i = -30; i<30; i++) {
    x = idx_x+i;
    if (x<0) x=0;
    if (x>width-1) x=width-1;
    for (int j = -30; j<30; j++) {
      y=idx_y+j;
      if (y<0) y=0;
      if (y>height-1) y=height-1;
      if (d_Pixelflags[y*width+x]) {
        heights += d_HeightMap[y*width+x];
        if ((i==0)&&(j==0)) {
          weights += 1;
        } else  if ((i>-2)&&(i<2)&&(j>-2)&&(j<2)) {
          weights += 1;
        } else  if ((i>-4)&&(i<4)&&(j>-4)&&(j<4)) {
          weights += 1;
        } else {
          weights += 1;
        }    
      }
    }
  }
   d_pixHeight[(idx_y*width+idx_x)] = (heights)/(float)weights;
}

__global__ void  GaussKernelBlock(float *res, float *src, int width, int height, int kernelSize) {
  int idx_x = blockIdx.x*blockDim.x + threadIdx.x; // image-pixel
  int idx_y = blockIdx.y*blockDim.y + threadIdx.y; 

  if (idx_x < (kernelSize-1)/2) return;
  if (idx_y < (kernelSize-1)/2) return;
  if (idx_x > width-1-(kernelSize-1)/2) return;
  if (idx_y > height-1-(kernelSize-1)/2) return;

  float result = 0.0f;
  int x, y; // kernel-pixel
  for (int i = -((kernelSize-1)/2);i<=(kernelSize-1)/2;i++){
    x = i+(kernelSize-1)/2;
    for (int j = -((kernelSize-1)/2);j<=(kernelSize-1)/2;j++){
      y = j+(kernelSize-1)/2;
      result += (src[idx_x+i+(idx_y+j)*width]*d_GaussKernel[y*kernelSize+x]);
    }
  }
  // printf("result written on x = %d, y = %d", idx_x, idx_y);
  res[idx_x+(idx_y)*width] = result; 
}

__global__ void visualize_features_kernel(Computation::Vertex2 *vertices, float *d_Heights, float *d_RGB, unsigned int width, unsigned int height)
{
  unsigned int idx_x = blockIdx.x*blockDim.x + threadIdx.x; // 0 <= x <= 159
  unsigned int idx_y = blockIdx.y*blockDim.y + threadIdx.y; // 0 <= x <= 159
  if (idx_x>GRIDSIZEX-1) return;
  if (idx_y>GRIDSIZEY-1) return;

  float r = d_RGB[idx_y*GRIDSIZEX*3+3*idx_x]/255;
  float g = d_RGB[idx_y*GRIDSIZEX*3+3*idx_x+1]/255;
  float b = d_RGB[idx_y*GRIDSIZEX*3+3*idx_x+2]/255;    
  float h = d_Heights[idx_y*GRIDSIZEX+idx_x];

  vertices[idx_y*GRIDSIZEX+(1919-idx_x)].pos[2] = h/50;
  vertices[idx_y*GRIDSIZEX+(1919-idx_x)].color[0] = r;
  vertices[idx_y*GRIDSIZEX+(1919-idx_x)].color[1] = g;
  vertices[idx_y*GRIDSIZEX+(1919-idx_x)].color[2] = b;
}

__global__ void gpuConvertYUYVtoRGBfloat_kernel(unsigned char *src, float *dst,
		unsigned int width, unsigned int height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx*2 >= width) {
		return;
	}

	for (int i = 0; i < height; ++i) {
		int cb = src[i*width*2+idx*4+0];
		int y0 = src[i*width*2+idx*4+1];
		int cr = src[i*width*2+idx*4+2];
		int y1 = src[i*width*2+idx*4+3];

		dst[i*width*3+idx*6+0] = clamp(1.164f * (y0 - 16) + 1.596f * (cr - 128)                      , 0.0f, 255.0f);
		dst[i*width*3+idx*6+1] = clamp(1.164f * (y0 - 16) - 0.813f * (cr - 128) - 0.391f * (cb - 128), 0.0f, 255.0f);
		dst[i*width*3+idx*6+2] = clamp(1.164f * (y0 - 16)                       + 2.018f * (cb - 128), 0.0f, 255.0f);

		dst[i*width*3+idx*6+3] = clamp(1.164f * (y1 - 16) + 1.596f * (cr - 128)                      , 0.0f, 255.0f);
		dst[i*width*3+idx*6+4] = clamp(1.164f * (y1 - 16) - 0.813f * (cr - 128) - 0.391f * (cb - 128), 0.0f, 255.0f);
		dst[i*width*3+idx*6+5] = clamp(1.164f * (y1 - 16)                       + 2.018f * (cb - 128), 0.0f, 255.0f);
	}
}


__global__ void gpuUndistort(uint16_t *dst, uint16_t *src, uint16_t *xCoords, uint16_t *yCoords, int width_src, int height_src, int width_dst, int height_dst)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("hoi\n");
    if (idx >= width_dst) {
        return;
    }
  
    for (int i = 0; i < height_dst; ++i) {
        // Calculate new coordinates
       dst[i*width_dst*4+idx*3+0] =  src[yCoords[i*width_dst+idx]*width_src+xCoords[i*width_dst+idx]];
       dst[i*width_dst*4+idx*3+1] =  src[yCoords[i*width_dst+idx]*width_src+xCoords[i*width_dst+idx]];
       dst[i*width_dst*4+idx*3+2] =  src[yCoords[i*width_dst+idx]*width_src+xCoords[i*width_dst+idx]];
    }
}

__global__ void gpuUndistortCosAlpha(uint16_t *dst, uint16_t *src, uint16_t *xCoords, uint16_t *yCoords, int width_src, int height_src, int width_dst, int height_dst, uint16_t *cosAlpha)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("hoi\n");
    if (idx >= width_dst) {
        return;
    }
  
    for (int i = 0; i < height_dst; ++i) {
        // Calculate new coordinates
       dst[i*width_dst*4+idx*3+0] =  src[yCoords[i*width_dst+idx]*width_src+xCoords[i*width_dst+idx]];
       dst[i*width_dst*4+idx*3+1] =  src[yCoords[i*width_dst+idx]*width_src+xCoords[i*width_dst+idx]];
       dst[i*width_dst*4+idx*3+2] =  src[yCoords[i*width_dst+idx]*width_src+xCoords[i*width_dst+idx]];
    }
}

__global__ void gpuSharpenImageToGrayscale(unsigned char *src, float *dst,
		unsigned int width, unsigned int height, float amount)
{
  int idx_x = blockIdx.x*blockDim.x + threadIdx.x; // image-pixel
  int idx_y = blockIdx.y*blockDim.y + threadIdx.y; 

  float result_x = 0;
  float result_y = 0;

  if ((idx_x != 0)&&(idx_x != width-1)&&(idx_y != 0)&&(idx_y != height-1)) {
   // sobel
    result_x +=  1 * src[(idx_x-1)*2+1+(idx_y-1)*width*2];
    result_x +=  2 * src[(idx_x-1)*2+1+(idx_y)*width*2];
    result_x +=  1 * src[(idx_x-1)*2+1+(idx_y+1)*width*2];
    result_x += -1 * src[(idx_x+1)*2+1+(idx_y-1)*width*2];
    result_x += -2 * src[(idx_x+1)*2+1+(idx_y)*width*2];
    result_x += -1 * src[(idx_x+1)*2+1+(idx_y+1)*width*2];
//
    result_y +=  1 * src[(idx_x-1)*2+1+(idx_y-1)*width*2];
    result_y +=  2 * src[(idx_x)*2+1+(idx_y-1)*width*2];
    result_y +=  1 * src[(idx_x+1)*2+1+(idx_y-1)*width*2];
    result_y += -1 * src[(idx_x-1)*2+1+(idx_y+1)*width*2];
    result_y += -2 * src[(idx_x)*2+1+(idx_y+1)*width*2];
    result_y += -1 * src[(idx_x+1)*2+1+(idx_y+1)*width*2];
  //for (int i = -1; i<=1; i++) {
  //  for (int j = -1; j<=1; j++) {
  //    result_x += src[(idx_x+i)*2+1+(idx_y+j)*width*2];
  //  }
  //}
  }
  //result_x = result_x/9.0f;
  float result = (src[idx_x*2+1+(idx_y)*width*2]*(1-amount))+(amount*sqrt(result_x*result_x+result_y*result_y));
  result = (result-80.0f)/130.0f*255.0f;
  if (result < 0) result = 0;
  if (result > 255) result = 255;

  // printf("result written on x = %d, y = %d", idx_x, idx_y);
  dst[idx_x+(idx_y)*width] = result; 
}

__global__ void LaplaceMultiMem(float *d_Image, float *d_Result, int width, int pitch, int height, int octave)
{
  __shared__ float buff[(LAPLACE_W + 2*LAPLACE_R)*LAPLACE_S];
  const int tx = threadIdx.x;
  const int xp = blockIdx.x*LAPLACE_W + tx;
  const int yp = blockIdx.y;
  float *data = d_Image + max(min(xp - LAPLACE_R, width-1), 0);
  float temp[2*LAPLACE_R + 1], kern[LAPLACE_S][LAPLACE_R + 1];
  if (xp<(width + 2*LAPLACE_R)) {
    for (int i=0;i<=2*LAPLACE_R;i++)
      temp[i] = data[max(0, min(yp + i - LAPLACE_R, height - 1))*pitch];
    for (int scale=0;scale<LAPLACE_S;scale++) {
      float *buf = buff + (LAPLACE_W + 2*LAPLACE_R)*scale; 
      float *kernel = d_LaplaceKernel + octave*12*16 + scale*16; 
      for (int i=0;i<=LAPLACE_R;i++)
	kern[scale][i] = kernel[i];
      float sum = kern[scale][0]*temp[LAPLACE_R];
#pragma unroll      
      for (int j=1;j<=LAPLACE_R;j++)
	sum += kern[scale][j]*(temp[LAPLACE_R - j] + temp[LAPLACE_R + j]);
      buf[tx] = sum;
    }
  }
  __syncthreads();
  if (tx<LAPLACE_W && xp<width) {
    int scale = 0;
    float oldRes = kern[scale][0]*buff[tx + LAPLACE_R];
#pragma unroll
    for (int j=1;j<=LAPLACE_R;j++)
      oldRes += kern[scale][j]*(buff[tx + LAPLACE_R - j] + buff[tx + LAPLACE_R + j]); 
    for (int scale=1;scale<LAPLACE_S;scale++) {
      float *buf = buff + (LAPLACE_W + 2*LAPLACE_R)*scale; 
      float res = kern[scale][0]*buf[tx + LAPLACE_R];
#pragma unroll
      for (int j=1;j<=LAPLACE_R;j++)
	res += kern[scale][j]*(buf[tx + LAPLACE_R - j] + buf[tx + LAPLACE_R + j]); 
      d_Result[(scale-1)*height*pitch + yp*pitch + xp] = res - oldRes;
      oldRes = res;
    }
  }
}

__global__ void FindPointsMultiNew(float *d_Data0, SiftPoint *d_Sift, int width, int pitch, int height, float subsampling, float lowestScale, float thresh, float factor, float edgeLimit, int octave)
{
  #define MEMWID (MINMAX_W + 2)
  __shared__ unsigned short points[2*MEMWID];
  
  if (blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0) {
    atomicMax(&d_PointCounter[2*octave+0], d_PointCounter[2*octave-1]);
    atomicMax(&d_PointCounter[2*octave+1], d_PointCounter[2*octave-1]);
  }
  int tx = threadIdx.x;
  int block = blockIdx.x/NUM_SCALES; 
  int scale = blockIdx.x - NUM_SCALES*block;
  int minx = block*MINMAX_W;
  int maxx = min(minx + MINMAX_W, width);
  int xpos = minx + tx;
  int size = pitch*height;
  int ptr = size*scale + max(min(xpos-1, width-1), 0);

  int yloops = min(height - MINMAX_H*blockIdx.y, MINMAX_H);
  float maxv = 0.0f;
  for (int y=0;y<yloops;y++) {
    int ypos = MINMAX_H*blockIdx.y + y;
    int yptr1 = ptr + ypos*pitch;
    float val = d_Data0[yptr1 + 1*size];
    maxv = fmaxf(maxv, fabs(val));
  }
  //if (tx==0) printf("XXX1\n");
  if (!__any_sync(0xffffffff, maxv>thresh))
    return;
  //if (tx==0) printf("XXX2\n");
  
  int ptbits = 0;
  for (int y=0;y<yloops;y++) {

    int ypos = MINMAX_H*blockIdx.y + y;
    int yptr1 = ptr + ypos*pitch;
    float d11 = d_Data0[yptr1 + 1*size];
    if (__any_sync(0xffffffff, fabs(d11)>thresh)) {
    
      int yptr0 = ptr + max(0,ypos-1)*pitch;
      int yptr2 = ptr + min(height-1,ypos+1)*pitch;
      float d01 = d_Data0[yptr1];
      float d10 = d_Data0[yptr0 + 1*size];
      float d12 = d_Data0[yptr2 + 1*size];
      float d21 = d_Data0[yptr1 + 2*size];
      
      float d00 = d_Data0[yptr0];
      float d02 = d_Data0[yptr2];
      float ymin1 = fminf(fminf(d00, d01), d02);
      float ymax1 = fmaxf(fmaxf(d00, d01), d02);
      float d20 = d_Data0[yptr0 + 2*size];
      float d22 = d_Data0[yptr2 + 2*size]; 
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
      
      if (tx>0 && tx<MINMAX_W+1 && xpos<=maxx) 
	ptbits |= ((d11 < fminf(-thresh, minv)) | (d11 > fmaxf(thresh, maxv))) << y;
    }
  }
  
  unsigned int totbits = __popc(ptbits);
  unsigned int numbits = totbits;
  for (int d=1;d<32;d<<=1) {
    unsigned int num = ShiftUp(totbits, d);
    if (tx >= d)
      totbits += num;
  }
  int pos = totbits - numbits;
  for (int y=0;y<yloops;y++) {
    int ypos = MINMAX_H*blockIdx.y + y;
    if (ptbits & (1 << y) && pos<MEMWID) {
      points[2*pos + 0] = xpos - 1;
      points[2*pos + 1] = ypos;
      pos ++;
    }
  } 

  totbits = Shuffle(totbits, 31);
  if (tx<totbits) {
    int xpos = points[2*tx + 0];
    int ypos = points[2*tx + 1];
    int ptr = xpos + (ypos + (scale + 1)*height)*pitch;
    float val = d_Data0[ptr];
    float *data1 = &d_Data0[ptr];
    float dxx = 2.0f*val - data1[-1] - data1[1];
    float dyy = 2.0f*val - data1[-pitch] - data1[pitch];
    float dxy = 0.25f*(data1[+pitch+1] + data1[-pitch-1] - data1[-pitch+1] - data1[+pitch-1]);
    float tra = dxx + dyy;
    float det = dxx*dyy - dxy*dxy;
    if (tra*tra<edgeLimit*det) {
      float edge = __fdividef(tra*tra, det);
      float dx = 0.5f*(data1[1] - data1[-1]);
      float dy = 0.5f*(data1[pitch] - data1[-pitch]); 
      float *data0 = d_Data0 + ptr - height*pitch;
      float *data2 = d_Data0 + ptr + height*pitch;
      float ds = 0.5f*(data0[0] - data2[0]); 
      float dss = 2.0f*val - data2[0] - data0[0];
      float dxs = 0.25f*(data2[1] + data0[-1] - data0[1] - data2[-1]);
      float dys = 0.25f*(data2[pitch] + data0[-pitch] - data2[-pitch] - data0[pitch]);
      float idxx = dyy*dss - dys*dys;
      float idxy = dys*dxs - dxy*dss;   
      float idxs = dxy*dys - dyy*dxs;
      float idet = __fdividef(1.0f, idxx*dxx + idxy*dxy + idxs*dxs);
      float idyy = dxx*dss - dxs*dxs;
      float idys = dxy*dxs - dxx*dys;
      float idss = dxx*dyy - dxy*dxy;
      float pdx = idet*(idxx*dx + idxy*dy + idxs*ds);
      float pdy = idet*(idxy*dx + idyy*dy + idys*ds);
      float pds = idet*(idxs*dx + idys*dy + idss*ds);
      if (pdx<-0.5f || pdx>0.5f || pdy<-0.5f || pdy>0.5f || pds<-0.5f || pds>0.5f) {
	pdx = __fdividef(dx, dxx);
	pdy = __fdividef(dy, dyy);
	pds = __fdividef(ds, dss);
      }
      float dval = 0.5f*(dx*pdx + dy*pdy + ds*pds);
      int maxPts = d_MaxNumPoints;
      float sc = powf(2.0f, (float)scale/NUM_SCALES) * exp2f(pds*factor);
      if (sc>=lowestScale) {
	atomicMax(&d_PointCounter[2*octave+0], d_PointCounter[2*octave-1]); 
	unsigned int idx = atomicInc(&d_PointCounter[2*octave+0], 0x7fffffff);
	idx = (idx>=maxPts ? maxPts-1 : idx);
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
  int num = blockIdx.x*blockDim.x + threadIdx.x;
  if (num<numPts) {
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
  
  int fstPts = min(d_PointCounter[2*octave-1], d_MaxNumPoints);
  int totPts = min(d_PointCounter[2*octave+0], d_MaxNumPoints);  
  for (int bx = blockIdx.x + fstPts; bx < totPts; bx += gridDim.x) {
 
    float i2sigma2 = -1.0f/(2.0f*1.5f*1.5f*d_Sift[bx].scale*d_Sift[bx].scale);
    if (tx<11) 
      gauss[tx] = exp(i2sigma2*(tx-5)*(tx-5));
    if (tx<64)
      hist[tx] = 0.0f;
    __syncthreads();
    float xp = d_Sift[bx].xpos - 4.5f;
    float yp = d_Sift[bx].ypos - 4.5f;
    int yd = tx/11;
    int xd = tx - yd*11;
    float xf = xp + xd;
    float yf = yp + yd;
    if (yd<11) {
      float dx = tex2D<float>(texObj, xf+1.0, yf) - tex2D<float>(texObj, xf-1.0, yf); 
      float dy = tex2D<float>(texObj, xf, yf+1.0) - tex2D<float>(texObj, xf, yf-1.0); 
      int bin = 16.0f*atan2f(dy, dx)/3.1416f + 16.5f;
      if (bin>31)
	bin = 0;
      float grad = sqrtf(dx*dx + dy*dy);
      atomicAdd(&hist[bin], grad*gauss[xd]*gauss[yd]);
    }
    __syncthreads();
    int x1m = (tx>=1 ? tx-1 : tx+31);
    int x1p = (tx<=30 ? tx+1 : tx-31);
    if (tx<32) {
      int x2m = (tx>=2 ? tx-2 : tx+30);
      int x2p = (tx<=29 ? tx+2 : tx-30);
      hist[tx+32] = 6.0f*hist[tx] + 4.0f*(hist[x1m] + hist[x1p]) + (hist[x2m] + hist[x2p]);
    }
    __syncthreads();
    if (tx<32) {
      float v = hist[32+tx];
      hist[tx] = (v>hist[32+x1m] && v>=hist[32+x1p] ? v : 0.0f);
    }
    __syncthreads();
    if (tx==0) {
      float maxval1 = 0.0;
      float maxval2 = 0.0;
      int i1 = -1;
      int i2 = -1;
      for (int i=0;i<32;i++) {
	float v = hist[i];
	if (v>maxval1) {
	  maxval2 = maxval1;
	  maxval1 = v;
	  i2 = i1;
	  i1 = i;
	} else if (v>maxval2) {
	  maxval2 = v;
	  i2 = i;
	}
      }
      float val1 = hist[32+((i1+1)&31)];
      float val2 = hist[32+((i1+31)&31)];
      float peak = i1 + 0.5f*(val1-val2) / (2.0f*maxval1-val1-val2);
      d_Sift[bx].orientation = 11.25f*(peak<0.0f ? peak+32.0f : peak);
      atomicMax(&d_PointCounter[2*octave+1], d_PointCounter[2*octave+0]); 
      if (maxval2>0.8f*maxval1 && true) {
	float val1 = hist[32+((i2+1)&31)];
	float val2 = hist[32+((i2+31)&31)];
	float peak = i2 + 0.5f*(val1-val2) / (2.0f*maxval2-val1-val2);
	unsigned int idx = atomicInc(&d_PointCounter[2*octave+1], 0x7fffffff);
	if (idx<d_MaxNumPoints) {
	  d_Sift[idx].xpos = d_Sift[bx].xpos;
	  d_Sift[idx].ypos = d_Sift[bx].ypos;
	  d_Sift[idx].scale = d_Sift[bx].scale;
	  d_Sift[idx].sharpness = d_Sift[bx].sharpness;
	  d_Sift[idx].edgeness = d_Sift[bx].edgeness;
	  d_Sift[idx].orientation = 11.25f*(peak<0.0f ? peak+32.0f : peak);;
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
  float a = __fdiv_rn(min(absx, absy),  max(absx, absy));
  float s = a*a;
  float r = ((-0.0464964749f*s + 0.15931422f)*s - 0.327622764f)*s*a + a;
  r = (absy>absx ? 1.57079637f - r : r);
  r = (x<0 ? 3.14159274f - r : r);
  r = (y<0 ? -r : r);
  return r;
}

__global__ void ExtractSiftDescriptorsCONSTNew(cudaTextureObject_t texObj, SiftPoint *d_sift, float subsampling, int octave)
{
  __shared__ float gauss[16];
  __shared__ float buffer[128];
  __shared__ float sums[4];

  const int tx = threadIdx.x; // 0 -> 16
  const int ty = threadIdx.y; // 0 -> 8
  const int idx = ty*16 + tx;
  if (ty==0)
    gauss[tx] = __expf(-(tx-7.5f)*(tx-7.5f)/128.0f);

  int fstPts = min(d_PointCounter[2*octave-1], d_MaxNumPoints);
  int totPts = min(d_PointCounter[2*octave+1], d_MaxNumPoints);
  //if (tx==0 && ty==0)
  //  printf("%d %d %d %d\n", octave, fstPts, min(d_PointCounter[2*octave], d_MaxNumPoints), totPts); 
  for (int bx = blockIdx.x + fstPts; bx < totPts; bx += gridDim.x) {
    
    buffer[idx] = 0.0f;
    __syncthreads();

    // Compute angles and gradients
    float theta = 2.0f*3.1415f/360.0f*d_sift[bx].orientation;
    float sina = __sinf(theta);           // cosa -sina
    float cosa = __cosf(theta);           // sina  cosa
    float scale = 12.0f/16.0f*d_sift[bx].scale;
    float ssina = scale*sina; 
    float scosa = scale*cosa;
    
    for (int y=ty;y<16;y+=8) {
      float xpos = d_sift[bx].xpos + (tx-7.5f)*scosa - (y-7.5f)*ssina + 0.5f; 
      float ypos = d_sift[bx].ypos + (tx-7.5f)*ssina + (y-7.5f)*scosa + 0.5f;
      float dx = tex2D<float>(texObj, xpos+cosa, ypos+sina) - 
	tex2D<float>(texObj, xpos-cosa, ypos-sina);
      float dy = tex2D<float>(texObj, xpos-sina, ypos+cosa) - 
	tex2D<float>(texObj, xpos+sina, ypos-cosa);
      float grad = gauss[y]*gauss[tx] * __fsqrt_rn(dx*dx + dy*dy);
      float angf = 4.0f/3.1415f*FastAtan2(dy, dx) + 4.0f;
      
      int hori = (tx + 2)/4 - 1;      // Convert from (tx,y,angle) to bins      
      float horf = (tx - 1.5f)/4.0f - hori;
      float ihorf = 1.0f - horf;           
      int veri = (y + 2)/4 - 1;
      float verf = (y - 1.5f)/4.0f - veri;
      float iverf = 1.0f - verf;
      int angi = angf;
      int angp = (angi<7 ? angi+1 : 0);
      angf -= angi;
      float iangf = 1.0f - angf;
      
      int hist = 8*(4*veri + hori);   // Each gradient measure is interpolated 
      int p1 = angi + hist;           // in angles, xpos and ypos -> 8 stores
      int p2 = angp + hist;
      if (tx>=2) { 
	float grad1 = ihorf*grad;
	if (y>=2) {   // Upper left
	  float grad2 = iverf*grad1;
	  atomicAdd(buffer + p1, iangf*grad2);
	  atomicAdd(buffer + p2,  angf*grad2);
	}
	if (y<=13) {  // Lower left
	  float grad2 = verf*grad1;
	  atomicAdd(buffer + p1+32, iangf*grad2); 
	  atomicAdd(buffer + p2+32,  angf*grad2);
	}
      }
      if (tx<=13) { 
	float grad1 = horf*grad;
	if (y>=2) {    // Upper right
	  float grad2 = iverf*grad1;
	  atomicAdd(buffer + p1+8, iangf*grad2);
	  atomicAdd(buffer + p2+8,  angf*grad2);
	}
	if (y<=13) {   // Lower right
	  float grad2 = verf*grad1;
	  atomicAdd(buffer + p1+40, iangf*grad2);
	  atomicAdd(buffer + p2+40,  angf*grad2);
	}
      }
    }
    __syncthreads();
    
    // Normalize twice and suppress peaks first time
    float sum = buffer[idx]*buffer[idx];
    for (int i=16;i>0;i/=2)
      sum += ShiftDown(sum, i);
    if ((idx&31)==0)
      sums[idx/32] = sum;
    __syncthreads();
    float tsum1 = sums[0] + sums[1] + sums[2] + sums[3]; 
    tsum1 = min(buffer[idx] * rsqrtf(tsum1), 0.2f);
     
    sum = tsum1*tsum1; 
    for (int i=16;i>0;i/=2)
      sum += ShiftDown(sum, i);
    if ((idx&31)==0)
      sums[idx/32] = sum;
    __syncthreads();
    
    float tsum2 = sums[0] + sums[1] + sums[2] + sums[3];
    float *desc = d_sift[bx].data;
    desc[idx] = tsum1 * rsqrtf(tsum2);
    if (idx==0) {
      d_sift[bx].xpos *= subsampling;
      d_sift[bx].ypos *= subsampling;
      d_sift[bx].scale *= subsampling;
    }
    __syncthreads();
  }
}

__global__ void CleanMatches(SiftPoint *sift1, int numPts1)
{
  const int p1 = min(blockIdx.x*64 + threadIdx.x, numPts1-1);
  sift1[p1].score = 0.0f;
}

#define M7W   32
#define M7H   32
#define M7R    4
#define NRX    2
#define NDIM 128

__global__ void Find3DCorr(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2) {
  unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
  float x_limit {200.0f};
  int x_corr {0};//{600};
  int y_corr {0};//{-18};
  float dx = 0.0f;
  float dy = 0.0f;

  if (idx < numPts1){
    sift1[idx].match = -1;
    float corr = 0.0f;
    float best_corr = 100.0f;
    int   best_index = -1;
    float best_dx = x_corr+x_limit;
    float x1 = sift1[idx].xpos;
    float y1 = sift1[idx].ypos;
    float sharpness = sift1[idx].sharpness;
    float edgeness = sift1[idx].edgeness;
    float orientation = sift1[idx].orientation;
    int subsampling = sift1[idx].subsampling;
    float sharp_ratio = 0.0f;
    float edge_ratio = 0.0f;
    float ori_ratio = 0.0f;
    for (int i = 0; i<numPts2; i++) {
      //if ((idx==0)&&(sift2[i].xpos+x_corr>=1060)&&(sift2[i].xpos+x_corr<=1300)&&(sift2[i].ypos+y_corr>=560)&&(sift2[i].ypos+y_corr<=890)) printf("2; %f; -%f; %f; %f; %f; %d; %d; %d\n",sift2[i].xpos+x_corr,sift2[i].ypos+y_corr,sift2[i].orientation,sift2[i].sharpness,sift2[i].edgeness,(int)sift2[i].subsampling, i, numPts2);
      if (sift2[i].subsampling != subsampling) continue;
      dy = y1-sift2[i].ypos;
      if ((dy<(-3.0f+y_corr))||(dy>3.0f+y_corr)) continue; // abs(deltay) < 5, others do not pick 
      dx = x1-sift2[i].xpos;
      if ((dx<-1.0f+x_corr)||(dx>1.0f*x_limit+x_corr)) continue; // y1 < y2 
      dx = abs(dx);
      ori_ratio = (sift2[i].orientation - orientation); // best fit: near zero, 360-deg wrap around ignored for now.
      if ((ori_ratio<-10.0f)||(ori_ratio>10.0f)) continue; // y1 < y2 
      sharp_ratio = abs((sift2[i].sharpness / sharpness)-1.0f);
      edge_ratio = abs((sift2[i].edgeness / edgeness)-1.0f);
      corr = abs(sharp_ratio+edge_ratio);
      if (best_corr < corr*0.8) continue;
      //if ((best_dx != x_corr+x_limit)&&(best_dx > dx)) printf("dx = %f, best_dx = %f\n",dx,best_dx);
      if (best_dx < dx) continue;
      best_corr = corr;
      best_index = i;
      best_dx = dx;      
    }

  //if ((x1>=174)&&(x1<=213)&&(y1>=115)&&(y1<=187)) {
  //  printf("1; Small; 0; %f; -%f; %f; %f; %f; %d; %d; %d; %f\n",x1,y1,orientation,sharpness,edgeness,subsampling,best_index, numPts1, best_dx);
  //  if (best_index!=-1) printf("2; Small; 0; %f; -%f; %f; %f; %f; %d; %d; %d; %f\n",sift2[best_index].xpos+x_corr,sift2[best_index].ypos+y_corr,sift2[best_index].orientation,sift2[best_index].sharpness,sift2[best_index].edgeness,(int)sift2[best_index].subsampling, best_index, numPts2, best_dx);
  //}
  //if ((x1>=164)&&(x1<=230)&&(y1>=458)&&(y1<=542)) {
  //  printf("1; Medium; 0; %f; -%f; %f; %f; %f; %d; %d; %d; %f\n",x1,y1,orientation,sharpness,edgeness,subsampling,best_index, numPts1, best_dx);
  //  if (best_index!=-1) printf("2; Medium; 0; %f; -%f; %f; %f; %f; %d; %d; %d; %f\n",sift2[best_index].xpos+x_corr,sift2[best_index].ypos+y_corr,sift2[best_index].orientation,sift2[best_index].sharpness,sift2[best_index].edgeness,(int)sift2[best_index].subsampling, best_index, numPts2, best_dx);
  //}
  //if ((x1>=123)&&(x1<=212)&&(y1>=894)&&(y1<=1002)) {
  //  printf("1; Large; 0; %f; -%f; %f; %f; %f; %d; %d; %d; %f\n",x1,y1,orientation,sharpness,edgeness,subsampling,best_index, numPts1, best_dx);
  //  if (best_index!=-1) printf("2; Medium; 0; %f; -%f; %f; %f; %f; %d; %d; %d; %f\n",sift2[best_index].xpos+x_corr,sift2[best_index].ypos+y_corr,sift2[best_index].orientation,sift2[best_index].sharpness,sift2[best_index].edgeness,(int)sift2[best_index].subsampling, best_index, numPts2, best_dx);
  //}
//
  //if ((x1>=567)&&(x1<=606)&&(y1>=115)&&(y1<=187)) {
  //  printf("1; Small; 1; %f; -%f; %f; %f; %f; %d; %d; %d; %f\n",x1,y1,orientation,sharpness,edgeness,subsampling,best_index, numPts1, best_dx);
  //  if (best_index!=-1) printf("2; Small; 1; %f; -%f; %f; %f; %f; %d; %d; %d; %f\n",sift2[best_index].xpos+x_corr,sift2[best_index].ypos+y_corr,sift2[best_index].orientation,sift2[best_index].sharpness,sift2[best_index].edgeness,(int)sift2[best_index].subsampling, best_index, numPts2, best_dx);
  //}
  //if ((x1>=523)&&(x1<=589)&&(y1>=458)&&(y1<=542)) {
  //  printf("1; Medium; 1; %f; -%f; %f; %f; %f; %d; %d; %d; %f\n",x1,y1,orientation,sharpness,edgeness,subsampling,best_index, numPts1, best_dx);
  //  if (best_index!=-1) printf("2; Medium; 1; %f; -%f; %f; %f; %f; %d; %d; %d; %f\n",sift2[best_index].xpos+x_corr,sift2[best_index].ypos+y_corr,sift2[best_index].orientation,sift2[best_index].sharpness,sift2[best_index].edgeness,(int)sift2[best_index].subsampling, best_index, numPts2, best_dx);
  //}
  //if ((x1>=540)&&(x1<=629)&&(y1>=894)&&(y1<=1002)) {
  //  printf("1; Large; 1; %f; -%f; %f; %f; %f; %d; %d; %d; %f\n",x1,y1,orientation,sharpness,edgeness,subsampling,best_index, numPts1, best_dx);
  //  if (best_index!=-1) printf("2; Large; 1; %f; -%f; %f; %f; %f; %d; %d; %d; %f\n",sift2[best_index].xpos+x_corr,sift2[best_index].ypos+y_corr,sift2[best_index].orientation,sift2[best_index].sharpness,sift2[best_index].edgeness,(int)sift2[best_index].subsampling, best_index, numPts2, best_dx);
  //}
//
  //if ((x1>=1090)&&(x1<=1129)&&(y1>=115)&&(y1<=187)) {
  //  printf("1; Small; 2; %f; -%f; %f; %f; %f; %d; %d; %d; %f\n",x1,y1,orientation,sharpness,edgeness,subsampling,best_index, numPts1, best_dx);
  //  if (best_index!=-1) printf("2; Small; 2; %f; -%f; %f; %f; %f; %d; %d; %d; %f\n",sift2[best_index].xpos+x_corr,sift2[best_index].ypos+y_corr,sift2[best_index].orientation,sift2[best_index].sharpness,sift2[best_index].edgeness,(int)sift2[best_index].subsampling, best_index, numPts2, best_dx);
  //}
  //if ((x1>=1083)&&(x1<=1149)&&(y1>=458)&&(y1<=542)) {
  //  printf("1; Medium; 2; %f; -%f; %f; %f; %f; %d; %d; %d; %f\n",x1,y1,orientation,sharpness,edgeness,subsampling,best_index, numPts1, best_dx);
  //  if (best_index!=-1) printf("2; Medium; 2; %f; -%f; %f; %f; %f; %d; %d; %d; %f\n",sift2[best_index].xpos+x_corr,sift2[best_index].ypos+y_corr,sift2[best_index].orientation,sift2[best_index].sharpness,sift2[best_index].edgeness,(int)sift2[best_index].subsampling, best_index, numPts2, best_dx);
  //}
  //if ((x1>=1109)&&(x1<=1198)&&(y1>=894)&&(y1<=1002)) {
  //  printf("1; Large; 2; %f; -%f; %f; %f; %f; %d; %d; %d; %f\n",x1,y1,orientation,sharpness,edgeness,subsampling,best_index, numPts1, best_dx);
  //  if (best_index!=-1) printf("2; Large; 2; %f; -%f; %f; %f; %f; %d; %d; %d; %f\n",sift2[best_index].xpos+x_corr,sift2[best_index].ypos+y_corr,sift2[best_index].orientation,sift2[best_index].sharpness,sift2[best_index].edgeness,(int)sift2[best_index].subsampling, best_index, numPts2, best_dx);
  //}
//
  //if ((x1>=1688)&&(x1<=1727)&&(y1>=115)&&(y1<=187)) {
  //  printf("1; Small; 3; %f; -%f; %f; %f; %f; %d; %d; %d; %f\n",x1,y1,orientation,sharpness,edgeness,subsampling,best_index, numPts1, best_dx);
  //  if (best_index!=-1) printf("2; Small; 3; %f; -%f; %f; %f; %f; %d; %d; %d; %f\n",sift2[best_index].xpos+x_corr,sift2[best_index].ypos+y_corr,sift2[best_index].orientation,sift2[best_index].sharpness,sift2[best_index].edgeness,(int)sift2[best_index].subsampling, best_index, numPts2, best_dx);
  //}
  //if ((x1>=1688)&&(x1<=1754)&&(y1>=458)&&(y1<=542)) {
  //  printf("1; Medium; 3; %f; -%f; %f; %f; %f; %d; %d; %d; %f\n",x1,y1,orientation,sharpness,edgeness,subsampling,best_index, numPts1, best_dx);
  //  if (best_index!=-1) printf("2; Medium; 3; %f; -%f; %f; %f; %f; %d; %d; %d; %f\n",sift2[best_index].xpos+x_corr,sift2[best_index].ypos+y_corr,sift2[best_index].orientation,sift2[best_index].sharpness,sift2[best_index].edgeness,(int)sift2[best_index].subsampling, best_index, numPts2, best_dx);
  //}
  //if ((x1>=1617)&&(x1<=1706)&&(y1>=894)&&(y1<=1002)) {
  //  printf("1; Large; 3; %f; -%f; %f; %f; %f; %d; %d; %d; %f\n",x1,y1,orientation,sharpness,edgeness,subsampling,best_index, numPts1, best_dx);
  //  if (best_index!=-1) printf("2; Large; 3; %f; -%f; %f; %f; %f; %d; %d; %d; %f\n",sift2[best_index].xpos+x_corr,sift2[best_index].ypos+y_corr,sift2[best_index].orientation,sift2[best_index].sharpness,sift2[best_index].edgeness,(int)sift2[best_index].subsampling, best_index, numPts2, best_dx);
  //}
        
    
    //printf("dx = %f, best corr = %f\nori1 = %f, sharp1 = %f, edge1 = %f\nori2 = %f, sharp2 = %f, edge2 = %f\n",best_dx,best_corr,orientation,sharpness,edgeness,sift2[best_index].orientation, sift2[best_index].sharpness, sift2[best_index].edgeness);
    if ((best_corr < 0.5f)) {
      
      sift1[idx].score=1.0;
      sift1[idx].match_error=(sqrt((x1-sift2[best_index].xpos-x_corr)*(x1-sift2[best_index].xpos-x_corr)+(y1-sift2[best_index].ypos-y_corr)*(y1-sift2[best_index].ypos-y_corr)))/(5.0f);
      sift1[idx].match_xpos=sift2[best_index].xpos;
      sift1[idx].match_ypos=sift2[best_index].ypos;
      sift1[idx].match=best_index;
      //if (((dx < -35.0)||(dx>35.0))) printf("dx = %f, dy = %f\n",dx,dy);
    } else {
      sift1[idx].score=0.0;
    }
  }
}

__global__ void FindMaxCorr10(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2)
{
  __shared__ float4 buffer1[M7W*NDIM/4]; 
  __shared__ float4 buffer2[M7H*NDIM/4];       
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bp1 = M7W*blockIdx.x;
  for (int j=ty;j<M7W;j+=M7H/M7R) {    
    int p1 = min(bp1 + j, numPts1 - 1);
    for (int d=tx;d<NDIM/4;d+=M7W)
      buffer1[j*NDIM/4 + (d + j)%(NDIM/4)] = ((float4*)&sift1[p1].data)[d];
  }
      
  float max_score[NRX];
  float sec_score[NRX];
  int index[NRX];
  for (int i=0;i<NRX;i++) {
    max_score[i] = 0.0f;
    sec_score[i] = 0.0f;
    index[i] = -1;
  }
  int idx = ty*M7W + tx;
  int ix = idx%(M7W/NRX);
  int iy = idx/(M7W/NRX);
  for (int bp2=0;bp2<numPts2 - M7H + 1;bp2+=M7H) {
    for (int j=ty;j<M7H;j+=M7H/M7R) {      
      int p2 = min(bp2 + j, numPts2 - 1);
      for (int d=tx;d<NDIM/4;d+=M7W)
	buffer2[j*NDIM/4 + d] = ((float4*)&sift2[p2].data)[d];
    }
    __syncthreads();

    if (idx<M7W*M7H/M7R/NRX) {
      float score[M7R][NRX];                                    
      for (int dy=0;dy<M7R;dy++)
	for (int i=0;i<NRX;i++)
	  score[dy][i] = 0.0f;
      for (int d=0;d<NDIM/4;d++) {
	float4 v1[NRX];
	for (int i=0;i<NRX;i++) 
	  v1[i] = buffer1[((M7W/NRX)*i + ix)*NDIM/4 + (d + (M7W/NRX)*i + ix)%(NDIM/4)];
	for (int dy=0;dy<M7R;dy++) {
	  float4 v2 = buffer2[(M7R*iy + dy)*(NDIM/4) + d];    
	  for (int i=0;i<NRX;i++) {
	    score[dy][i] += v1[i].x*v2.x;
	    score[dy][i] += v1[i].y*v2.y;
	    score[dy][i] += v1[i].z*v2.z;
	    score[dy][i] += v1[i].w*v2.w;
	  }
	}
      }
      for (int dy=0;dy<M7R;dy++) {
	for (int i=0;i<NRX;i++) {
	  if (score[dy][i]>max_score[i]) {
	    sec_score[i] = max_score[i];
	    max_score[i] = score[dy][i];     
	    index[i] = min(bp2 + M7R*iy + dy, numPts2-1);
	  } else if (score[dy][i]>sec_score[i])
	    sec_score[i] = score[dy][i]; 
	}
      }
    }
    __syncthreads();
  }

  float *scores1 = (float*)buffer1;
  float *scores2 = &scores1[M7W*M7H/M7R];
  int *indices = (int*)&scores2[M7W*M7H/M7R];
  if (idx<M7W*M7H/M7R/NRX) {
    for (int i=0;i<NRX;i++) {
      scores1[iy*M7W + (M7W/NRX)*i + ix] = max_score[i];  
      scores2[iy*M7W + (M7W/NRX)*i + ix] = sec_score[i];  
      indices[iy*M7W + (M7W/NRX)*i + ix] = index[i];
    }
  }
  __syncthreads();
  
  if (ty==0) {
    float max_score = scores1[tx];
    float sec_score = scores2[tx];
    int index = indices[tx];
    for (int y=0;y<M7H/M7R;y++)
      if (index != indices[y*M7W + tx]) {
	if (scores1[y*M7W + tx]>max_score) {
	  sec_score = max(max_score, sec_score);
	  max_score = scores1[y*M7W + tx]; 
	  index = indices[y*M7W + tx];
	} else if (scores1[y*M7W + tx]>sec_score)
	  sec_score = scores1[y*M7W + tx];
    if (scores2[y*M7W + tx]>sec_score)
          sec_score = scores2[y*M7W + tx];
      }
    sift1[bp1 + tx].score = max_score;
    //sift1[bp1 + tx].distance = 
    sift1[bp1 + tx].match = index;
    sift1[bp1 + tx].match_xpos = sift2[index].xpos;
    sift1[bp1 + tx].match_ypos = sift2[index].ypos;
    sift1[bp1 + tx].ambiguity =  sec_score / (max_score + 1e-6f);
  }
}
  
#define FMC_GH  512
#define FMC_BW   32
#define FMC_BH   32
#define FMC_BD   16
#define FMC_TW    1
#define FMC_TH    4
#define FMC_NW   (FMC_BW/FMC_TW)   //  32
#define FMC_NH   (FMC_BH/FMC_TH)   //   8
#define FMC_NT   (FMC_NW*FMC_NH)   // 256 = 8 warps

__device__ volatile int lock = 0;

__global__ void FindMaxCorr9(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2)
{
  __shared__ float4 siftParts1[FMC_BW*FMC_BD]; // 4*32*8 = 1024
  __shared__ float4 siftParts2[FMC_BH*FMC_BD]; // 4*32*8 = 1024
  //__shared__ float blksums[FMC_BW*FMC_BH];     // 32*32  = 1024
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int idx = ty*FMC_NW + tx;
  float4 *pts1 = 0, *pts2 = 0;
  if (idx<FMC_BW) {
    const int p1l = min(blockIdx.x*FMC_BW + idx, numPts1-1);
    pts1 = (float4*)sift1[p1l].data;
  }
  float maxScore = -1.0f;
  float maxScor2 = -1.0f;
  int maxIndex = 0;
  for (int k=0;k<min(FMC_GH, numPts2 - FMC_BH + 1);k+=FMC_BH) {
    if (idx<FMC_BH) {
      const int p2l = min(blockIdx.y*FMC_GH + k + idx, numPts2-1);
      pts2 = (float4*)sift2[p2l].data;
    }
    float sums[FMC_TW*FMC_TH];
    for (int i=0;i<FMC_TW*FMC_TH;i++) 
      sums[i] = 0.0f;

    if (idx<FMC_BW)
      for (int i=0;i<FMC_BD/2;i++) 
	siftParts1[(i + 0)*FMC_BW + idx] = pts1[0 + i];
    if (idx<FMC_BH)
      for (int i=0;i<FMC_BD/2;i++) 
	siftParts2[(i + 0)*FMC_BH + idx] = pts2[0 + i];
    __syncthreads();
    
    int b = FMC_BD/2;
    for (int d=FMC_BD/2;d<32;d+=FMC_BD/2) {
      if (idx<FMC_BW)
	for (int i=0;i<FMC_BD/2;i++) 
	  siftParts1[(i + b)*FMC_BW + idx] = pts1[d + i];
      if (idx<FMC_BH)
	for (int i=0;i<FMC_BD/2;i++) 
	  siftParts2[(i + b)*FMC_BH + idx] = pts2[d + i];

      b ^= FMC_BD/2;
      for (int i=0;i<FMC_BD/2;i++) {
	float4 v1[FMC_TW];
	for (int ix=0;ix<FMC_TW;ix++)
	  v1[ix] = siftParts1[(i + b)*FMC_BW + (tx*FMC_TW + ix)];
	for (int iy=0;iy<FMC_TH;iy++) {
	  float4 v2 = siftParts2[(i + b)*FMC_BH + (ty*FMC_TH + iy)];
	  for (int ix=0;ix<FMC_TW;ix++) {
	    sums[iy*FMC_TW + ix] += v1[ix].x * v2.x;
	    sums[iy*FMC_TW + ix] += v1[ix].y * v2.y;
	    sums[iy*FMC_TW + ix] += v1[ix].z * v2.z;
	    sums[iy*FMC_TW + ix] += v1[ix].w * v2.w;
	  }
	}
      }
      __syncthreads();
    }
    
    b ^= FMC_BD/2;
    for (int i=0;i<FMC_BD/2;i++) {
      float4 v1[FMC_TW];
      for (int ix=0;ix<FMC_TW;ix++)
	v1[ix] = siftParts1[(i + b)*FMC_BW + (tx*FMC_TW + ix)];
      for (int iy=0;iy<FMC_TH;iy++) {
	float4 v2 = siftParts2[(i + b)*FMC_BH + (ty*FMC_TH + iy)];
	for (int ix=0;ix<FMC_TW;ix++) {
	  sums[iy*FMC_TW + ix] += v1[ix].x * v2.x;
	  sums[iy*FMC_TW + ix] += v1[ix].y * v2.y;
	  sums[iy*FMC_TW + ix] += v1[ix].z * v2.z;
	  sums[iy*FMC_TW + ix] += v1[ix].w * v2.w;
	}
      }
    }
    __syncthreads();
    
    float *blksums = (float*)siftParts1;
    for (int iy=0;iy<FMC_TH;iy++) 
      for (int ix=0;ix<FMC_TW;ix++) 
	blksums[(ty*FMC_TH + iy)*FMC_BW + (tx*FMC_TW + ix)] = sums[iy*FMC_TW + ix];
    __syncthreads();
    if (idx<FMC_BW) { 
      for (int j=0;j<FMC_BH;j++) {
	float sum = blksums[j*FMC_BW + idx];
	if (sum>maxScore) { 
	  maxScor2 = maxScore;
	  maxScore = sum;
	  maxIndex = min(blockIdx.y*FMC_GH + k + j, numPts2-1);
	} else if (sum>maxScor2)
	  maxScor2 = sum;
      }
    }
    __syncthreads();
  }
  const int p1 = min(blockIdx.x*FMC_BW + idx, numPts1-1);
  if (idx==0)
    while (atomicCAS((int *)&lock, 0, 1) != 0);
  __syncthreads();
  if (idx<FMC_BW) {
    float maxScor2Old = sift1[p1].ambiguity*(sift1[p1].score + 1e-6f);
    if (maxScore>sift1[p1].score) {
      maxScor2 = max(sift1[p1].score, maxScor2);
      sift1[p1].ambiguity = maxScor2 / (maxScore + 1e-6f);
      sift1[p1].score = maxScore;
      sift1[p1].match = maxIndex;
      sift1[p1].match_xpos = sift2[maxIndex].xpos;
      sift1[p1].match_ypos = sift2[maxIndex].ypos;
    } else if (maxScore>maxScor2Old)
      sift1[p1].ambiguity = maxScore / (sift1[p1].score + 1e-6f);
  }
  __syncthreads();
  if (idx==0)
    atomicExch((int* )&lock, 0);
}

__global__ void FindMaxCorr8(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2)
{
  __shared__ float4 siftParts1[FMC_BW*FMC_BD]; // 4*32*8 = 1024
  __shared__ float4 siftParts2[FMC_BH*FMC_BD]; // 4*32*8 = 1024
  __shared__ float blksums[FMC_BW*FMC_BH];     // 32*32  = 1024
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int idx = ty*FMC_NW + tx;
  float4 *pts1 = 0, *pts2 = 0;
  if (idx<FMC_BW) {
    const int p1l = min(blockIdx.x*FMC_BW + idx, numPts1-1);
    pts1 = (float4*)sift1[p1l].data;
  }
  float maxScore = -1.0f;
  float maxScor2 = -1.0f;
  int maxIndex = 0;
  for (int k=0;k<min(FMC_GH, numPts2 - FMC_BH + 1);k+=FMC_BH) {
    if (idx<FMC_BH) {
      const int p2l = min(blockIdx.y*FMC_GH + k + idx, numPts2-1);
      pts2 = (float4*)sift2[p2l].data;
    }
    float sums[FMC_TW*FMC_TH];
    for (int i=0;i<FMC_TW*FMC_TH;i++) 
      sums[i] = 0.0f;
    for (int d=0;d<32;d+=FMC_BD) {
      if (idx<FMC_BW)
	for (int i=0;i<FMC_BD;i++) 
	  siftParts1[i*FMC_BW + idx] = pts1[d + i];
      if (idx<FMC_BH)
	for (int i=0;i<FMC_BD;i++) 
	  siftParts2[i*FMC_BH + idx] = pts2[d + i];
      __syncthreads();
      
      for (int i=0;i<FMC_BD;i++) {
	float4 v1[FMC_TW];
	for (int ix=0;ix<FMC_TW;ix++)
	  v1[ix] = siftParts1[i*FMC_BW + (tx*FMC_TW + ix)];
	for (int iy=0;iy<FMC_TH;iy++) {
	  float4 v2 = siftParts2[i*FMC_BH + (ty*FMC_TH + iy)];
	  for (int ix=0;ix<FMC_TW;ix++) {
	    sums[iy*FMC_TW + ix] += v1[ix].x * v2.x;
	    sums[iy*FMC_TW + ix] += v1[ix].y * v2.y;
	    sums[iy*FMC_TW + ix] += v1[ix].z * v2.z;
	    sums[iy*FMC_TW + ix] += v1[ix].w * v2.w;
	  }
	}
      }
      __syncthreads();
    }
    //float *blksums = (float*)siftParts1;
    for (int iy=0;iy<FMC_TH;iy++) 
      for (int ix=0;ix<FMC_TW;ix++) 
	blksums[(ty*FMC_TH + iy)*FMC_BW + (tx*FMC_TW + ix)] = sums[iy*FMC_TW + ix];
    __syncthreads();
    if (idx<FMC_BW) { 
      for (int j=0;j<FMC_BH;j++) {
	float sum = blksums[j*FMC_BW + idx];
	if (sum>maxScore) { 
	  maxScor2 = maxScore;
	  maxScore = sum;
	  maxIndex = min(blockIdx.y*FMC_GH + k + j, numPts2-1);
	} else if (sum>maxScor2)
	  maxScor2 = sum;
      }
    }
    __syncthreads();
  }
  const int p1 = min(blockIdx.x*FMC_BW + idx, numPts1-1);
  if (idx==0)
    while (atomicCAS((int *)&lock, 0, 1) != 0);
  __syncthreads();
  if (idx<FMC_BW) {
    float maxScor2Old = sift1[p1].ambiguity*(sift1[p1].score + 1e-6f);
    if (maxScore>sift1[p1].score) {
      maxScor2 = max(sift1[p1].score, maxScor2);
      sift1[p1].ambiguity = maxScor2 / (maxScore + 1e-6f);
      sift1[p1].score = maxScore;
      sift1[p1].match = maxIndex;
      sift1[p1].match_xpos = sift2[maxIndex].xpos;
      sift1[p1].match_ypos = sift2[maxIndex].ypos;
    } else if (maxScore>maxScor2Old)
      sift1[p1].ambiguity = maxScore / (sift1[p1].score + 1e-6f);
  }
  __syncthreads();
  if (idx==0)
    atomicExch((int* )&lock, 0);
}

__global__ void FindMaxCorr7(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2)
{
  __shared__ float siftParts1[17*64]; // features in columns
  __shared__ float siftParts2[16*64]; // one extra to avoid shared conflicts
  float4 *pts1 = (float4*)siftParts1;
  float4 *pts2 = (float4*)siftParts2;
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int p1l = min(blockIdx.x*16 + ty, numPts1-1);
  const float4 *p1l4 = (float4*)sift1[p1l].data;
  float maxScore = -1.0f;
  float maxScor2 = -1.0f;
  int maxIndex = 0;
  for (int k=0;k<512/16;k++) {
    const int p2l = min(blockIdx.y*512 + k*16 + ty, numPts2-1);
    const float4 *p2l4 = (float4*)sift2[p2l].data;
#define NUM 4
    float sum[NUM];
    if (ty<(16/NUM))
      for (int l=0;l<NUM;l++)
	sum[l] = 0.0f;
    __syncthreads();
    for (int i=0;i<2;i++) {
      pts1[17*tx + ty] = p1l4[i*16 + tx];
      pts2[16*ty + tx] = p2l4[i*16 + tx];
      __syncthreads(); 
      if (ty<(16/NUM)) {
#pragma unroll
	for (int j=0;j<16;j++) {
	  float4 p1v = pts1[17* j + tx];
#pragma unroll
	  for (int l=0;l<NUM;l++) {
	    float4 p2v = pts2[16*(ty + l*(16/NUM)) +  j];
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
    if (ty<(16/NUM))
      for (int l=0;l<NUM;l++) 
	sums[16*(ty + l*(16/NUM)) + tx] = sum[l];
    __syncthreads();
    if (ty==0) { 
      for (int j=0;j<16;j++) {
	float sum = sums[16*j + tx];
	if (sum>maxScore) { 
	  maxScor2 = maxScore;
	  maxScore = sum;
	  maxIndex = min(blockIdx.y*512 +  k*16 + j, numPts2-1);
	} else if (sum>maxScor2)
	  maxScor2 = sum;
      }
    }
    __syncthreads();
  }
  const int p1 = min(blockIdx.x*16 + tx, numPts1-1);
  if (tx==0 && ty==0)
    while (atomicCAS((int *)&lock, 0, 1) != 0);
  __syncthreads();
  if (ty==0) {
    float maxScor2Old = sift1[p1].ambiguity*(sift1[p1].score + 1e-6f);
    if (maxScore>sift1[p1].score) {
      maxScor2 = max(sift1[p1].score, maxScor2);
      sift1[p1].ambiguity = maxScor2 / (maxScore + 1e-6f);
      sift1[p1].score = maxScore;
      sift1[p1].match = maxIndex;
      sift1[p1].match_xpos = sift2[maxIndex].xpos;
      sift1[p1].match_ypos = sift2[maxIndex].ypos;
    } else if (maxScore>maxScor2Old)
      sift1[p1].ambiguity = maxScore / (sift1[p1].score + 1e-6f);
  }
  __syncthreads();
  if (tx==0 && ty==0)
    atomicExch((int* )&lock, 0);
}

__global__ void FindMaxCorr6(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2)
{
  //__shared__ float siftParts1[128*16]; // features in columns
  __shared__ float siftParts2[128*16]; // one extra to avoid shared conflicts
  __shared__ float sums[16*16];
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int p1l = min(blockIdx.x*16 + ty, numPts1-1);
  float *pt1l = sift1[p1l].data;
  float4 part1 = reinterpret_cast<float4*>(pt1l)[tx];
  float maxScore = -1.0f;
  float maxScor2 = -1.0f;
  int maxIndex = 0;
  for (int k=0;k<512;k+=16) {
    const int p2l = min(blockIdx.y*512 + k + ty, numPts2-1);
    float *pt2l = sift2[p2l].data;
    reinterpret_cast<float4*>(siftParts2)[32*ty + tx] = reinterpret_cast<float4*>(pt2l)[tx];
    __syncthreads();
    for (int i=0;i<16;i++) {
      float4 part2 = reinterpret_cast<float4*>(siftParts2)[32*i  + tx];
      float sum = part1.x*part2.x + part1.y*part2.y + part1.z*part2.z + part1.w*part2.w;
      sum += ShiftDown(sum, 16);
      sum += ShiftDown(sum, 8);
      sum += ShiftDown(sum, 4);
      sum += ShiftDown(sum, 2);
      sum += ShiftDown(sum, 1);
      if (tx==0)
	sums[16*i + ty] = sum;
    }
    __syncthreads();
    if (ty==0 && tx<16) { 
      for (int j=0;j<16;j++) {
	float sum = sums[16*j + tx];
	if (sum>maxScore) { 
	  maxScor2 = maxScore;
	  maxScore = sum;
	  maxIndex = min(blockIdx.y*512 +  k + j, numPts2-1);
	} else if (sum>maxScor2)
	  maxScor2 = sum;
      }
    }
    __syncthreads();
  }
  if (tx==0 && ty==0)
    while (atomicCAS((int *)&lock, 0, 1) != 0);
  __syncthreads();
  if (ty==0 && tx<16) {
    const int p1 = min(blockIdx.x*16 + tx, numPts1-1);
    float maxScor2Old = sift1[p1].ambiguity*(sift1[p1].score + 1e-6f);
    if (maxScore>sift1[p1].score) {
      maxScor2 = max(sift1[p1].score, maxScor2);
      sift1[p1].ambiguity = maxScor2 / (maxScore + 1e-6f);
      sift1[p1].score = maxScore;
      sift1[p1].match = maxIndex;
      sift1[p1].match_xpos = sift2[maxIndex].xpos;
      sift1[p1].match_ypos = sift2[maxIndex].ypos;
    } else if (maxScore>maxScor2Old)
      sift1[p1].ambiguity = maxScore / (sift1[p1].score + 1e-6f);
  }
  __syncthreads();
  if (tx==0 && ty==0)
    atomicExch((int* )&lock, 0);
}
 
__global__ void FindMaxCorr5(SiftPoint *sift1, SiftPoint *sift2, int numPts1, int numPts2)
{
  __shared__ float siftParts1[17*16]; // features in columns
  __shared__ float siftParts2[17*16]; // one extra to avoid shared conflicts
  const int tx = threadIdx.x;
  const int ty = threadIdx.y;
  const int p1l = min(blockIdx.x*16 + ty, numPts1-1);
  const float *pt1l = sift1[p1l].data;
  float maxScore = -1.0f;
  float maxScor2 = -1.0f;
  int maxIndex = 0;
  for (int k=0;k<512/16;k++) {
    const int p2l = min(blockIdx.y*512 + k*16 + ty, numPts2-1);
    const float *pt2l = sift2[p2l].data;
    float sum = 0.0f;
    for (int i=0;i<8;i++) {
      siftParts1[17*tx + ty] = pt1l[i*16 + tx]; // load and transpose
      siftParts2[17*tx + ty] = pt2l[i*16 + tx];
      __syncthreads();
      for (int j=0;j<16;j++)
	sum += siftParts1[17*j + tx] * siftParts2[17*j + ty];
      __syncthreads();
    }
    float *sums = siftParts1;
    sums[16*ty + tx] = sum;
    __syncthreads();
    if (ty==0) { 
      for (int j=0;j<16;j++) {
	float sum = sums[16*j + tx];
	if (sum>maxScore) { 
	  maxScor2 = maxScore;
	  maxScore = sum;
	  maxIndex = min(blockIdx.y*512 +  k*16 + j, numPts2-1);
	} else if (sum>maxScor2)
	  maxScor2 = sum;
      }
    }
    __syncthreads();
  }
  const int p1 = min(blockIdx.x*16 + tx, numPts1-1);
  if (tx==0 && ty==0)
    while (atomicCAS((int *)&lock, 0, 1) != 0);
  __syncthreads();
  if (ty==0) {
    float maxScor2Old = sift1[p1].ambiguity*(sift1[p1].score + 1e-6f);
    if (maxScore>sift1[p1].score) {
      maxScor2 = max(sift1[p1].score, maxScor2);
      sift1[p1].ambiguity = maxScor2 / (maxScore + 1e-6f);
      sift1[p1].score = maxScore;
      sift1[p1].match = maxIndex;
      sift1[p1].match_xpos = sift2[maxIndex].xpos;
      sift1[p1].match_ypos = sift2[maxIndex].ypos;
    } else if (maxScore>maxScor2Old)
      sift1[p1].ambiguity = maxScore / (sift1[p1].score + 1e-6f);
  }
  __syncthreads();
  if (tx==0 && ty==0)
    atomicExch((int* )&lock, 0);
}

inline void __checkMsg(const char *errorMessage, const char *file, const int line)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "checkMsg() CUDA error: %s in file <%s>, line %i : %s.\n", errorMessage, file, line, cudaGetErrorString(err));
    exit(-1);
  }
}

void Computation::initCuda()
    {
        
        setCudaVkDevice();
        cudaVkImportVertexMem();
        cudaInitVertexMem();
        cudaInitIndexMem();
        cudaVkImportSemaphore();
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
    if (device_count == 0) {
        std::cout << "CUDA error: no devices supporting CUDA." << std::endl;
         exit(EXIT_FAILURE);
    }
    // Find the GPU which is selected by Vulkan
    while (current_device < device_count) {
        std::cout << "Looking up device " << current_device << " of " << device_count << std::endl;
        cudaGetDeviceProperties(&deviceProp, current_device);
        if ((deviceProp.computeMode != cudaComputeModeProhibited)) {
            // Compare the cuda device UUID with vulkan UUID
            int ret = memcmp(&deviceProp.uuid, &vkDeviceUUID, VK_UUID_SIZE);

            std::cout << "ret is " << ret << std::endl;
            printf("vkDeviceUUID    = 0x");
            for (int i = 0; i<VK_UUID_SIZE; i++) {
                printf("%x ",*(vkDeviceUUID+i));
            }
            printf("\n");
            printf("deviceProp.uuid = 0x");
            for (int i = 0; i<VK_UUID_SIZE; i++) {
                printf("%x ",*((deviceProp.uuid.bytes)+i));
            }
            printf("\n");
            if (ret == 0)
            {
                checkCudaErrors(cudaSetDevice(current_device));
                checkCudaErrors(cudaGetDeviceProperties(&deviceProp, current_device));
                std::cout << "GPU Device" << current_device << deviceProp.name << deviceProp.major << deviceProp.minor << std::endl;
                return current_device;
            }
        } else {
          devices_prohibited++;
        }
        current_device++;
    }
    if (devices_prohibited == device_count) {
        std::cout << "No Vulkan-CUDA Interop capable GPU found." << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "returning -1" << std::endl;
    return -1;
}


void Computation::cudaVkImportVertexMem()
{
    cudaExternalMemoryHandleDesc cudaExtMemHandleDesc;
    memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));
    cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    cudaExtMemHandleDesc.handle.fd = memHandleVertex;
    cudaExtMemHandleDesc.size = sizeof(Vertex2) * vertexBufSize;
    checkCudaErrors(cudaImportExternalMemory(&cudaExtMemVertexBuffer, &cudaExtMemHandleDesc));
    cudaExternalMemoryBufferDesc cudaExtBufferDesc;
    cudaExtBufferDesc.offset = 0;
    cudaExtBufferDesc.size = sizeof(Vertex2) * vertexBufSize;
    cudaExtBufferDesc.flags = 0;
    checkCudaErrors(cudaExternalMemoryGetMappedBuffer(&cudaDevVertptr, cudaExtMemVertexBuffer, &cudaExtBufferDesc));


    cudaExtMemHandleDesc.size = (1080*1920*6)*sizeof(int);
    cudaExtMemHandleDesc.handle.fd = memHandleIndex;
    checkCudaErrors(cudaImportExternalMemory(&cudaExtMemIndexBuffer, &cudaExtMemHandleDesc));
    cudaExtBufferDesc.size = (1080*1920*6)*sizeof(int);
    checkCudaErrors(cudaExternalMemoryGetMappedBuffer(&cudaDevIndexptr, cudaExtMemIndexBuffer, &cudaExtBufferDesc));
}


void Computation::cudaVkSemaphoreSignal(cudaExternalSemaphore_t &extSemaphore)
{
    cudaExternalSemaphoreSignalParams extSemaphoreSignalParams;
    memset(&extSemaphoreSignalParams, 0, sizeof(extSemaphoreSignalParams));
    extSemaphoreSignalParams.params.fence.value = 0;
    extSemaphoreSignalParams.flags              = 0;
    checkCudaErrors(cudaSignalExternalSemaphoresAsync(&extSemaphore, &extSemaphoreSignalParams, 1, streamToRun));
}


void Computation::cudaVkSemaphoreWait(cudaExternalSemaphore_t &extSemaphore)
{
    cudaExternalSemaphoreWaitParams extSemaphoreWaitParams;
    memset(&extSemaphoreWaitParams, 0, sizeof(extSemaphoreWaitParams));
    extSemaphoreWaitParams.params.fence.value = 0;
    extSemaphoreWaitParams.flags              = 0;
    checkCudaErrors(cudaWaitExternalSemaphoresAsync(&extSemaphore, &extSemaphoreWaitParams, 1, streamToRun));
}


void Computation::cudaInitVertexMem()
{
    std::cout << "initializing VertexMemory" << std::endl;
    checkCudaErrors(cudaStreamCreate(&streamToRun));
    dim3 block(120, 1, 1);

    dim3 grid(16, 1, 1);
    Vertex2 *vertices = (Vertex2*) cudaDevVertptr;

    init_grid_kernel<<<grid, block, 0, streamToRun>>>(vertices);
    checkCudaErrors(cudaStreamSynchronize(streamToRun));
}

void Computation::cudaInitIndexMem()
{
    std::cout << "initializing IndexMemory" << std::endl;
    checkCudaErrors(cudaStreamCreate(&streamToRun));
    dim3 block(120, 1, 1);

    dim3 grid(16, 1, 1);
    int* indexbuffer =  (int*)cudaDevIndexptr;

    init_index_kernel<<<grid, block, 0, streamToRun>>>(indexbuffer);
    checkCudaErrors(cudaStreamSynchronize(streamToRun));
}


void Computation::cudaUpdateVertexBuffer(CudaImage &heightMap, CudaImage &colorData)
{
    cudaVkSemaphoreWait(cudaExtVkUpdateCudaVertexBufSemaphore);
    dim3 block(16, 16, 1);
    int mesh_width = 1920;
    int mesh_height = 1080;
    dim3 grid(mesh_width/block.x+1, mesh_height/block.y+1, 1);
    Vertex2 *pos = (Vertex2*) cudaDevVertptr;
    visualize_features_kernel<<<grid, block, 0, streamToRun>>>(pos, heightMap.d_result ,colorData.d_data,1920,1080);
    cudaVkSemaphoreSignal(cudaExtCudaUpdateVkVertexBufSemaphore);
}

void Computation::drawHeightMap(CudaImage &heightMap, SiftData &siftData) {
  dim3 blocks2 = dim3(1024);
  dim3 threads2 = dim3((heightMap.width + 1024 - 1) / 1024);
  dim3 blocks1 = dim3(256);
  dim3 threads1 = dim3(siftData.numPts/256+1);

  clear_heightmap_kernel<<<blocks2, threads2>>>(heightMap.d_data, heightMap.d_pixelFlags, heightMap.width, heightMap.height);
  draw_heightmap_kernel<<<blocks1, threads1>>>(heightMap.d_data, heightMap.d_pixelFlags, siftData.d_data, heightMap.width, heightMap.height, siftData.numPts, siftData.maxPts);
  dim3 block(16, 16, 1);
  int mesh_width = 1920;
  int mesh_height = 1080;
  dim3 grid(mesh_width/block.x+1, mesh_height/block.y+1, 1);
  
  interpolate_heightmap_kernel<<<grid, block, 0>>>(heightMap.d_data, heightMap.d_pixelFlags, heightMap.d_pixHeight,  heightMap.width, heightMap.height);
  LowPass_forSubImages(heightMap.d_result,heightMap.d_pixHeight,heightMap.width,heightMap.height);
}

void Computation::LowPass_prepareKernel(void) {
  int kernelSize{GAUSSIANSIZE};
  int range = kernelSize/2;
  float gausskernel[kernelSize*kernelSize];

  double sigma = (float)GAUSSIANSIZE/5.0; 
  double r, s = 2.0 * sigma * sigma; 
  
  // sum is for normalization 
  double sum = 0.0; 
  
  // generating NxN kernel 
  for (int x = (-1*range); x <= range; x++) { 
      for (int y = (-1*range); y <= range; y++) { 
          r = sqrt(x * x + y * y); 
          gausskernel[(x + range)+kernelSize*(y + range)] = (exp(-(r * r) / s)) / (M_PI * s); 
          sum += gausskernel[(x + range)+kernelSize*(y + range)]; 
      } 
  } 
     // normalising the Kernel 
    for (int i = 0; i < kernelSize; ++i) 
        for (int j = 0; j < kernelSize; ++j) 
            gausskernel[i+kernelSize*j] /= sum; 

  for (int i = 0; i < kernelSize; ++i) { 
        for (int j = 0; j < kernelSize; ++j) 
            std::cout << gausskernel[i+kernelSize*j] << "\t"; 
        std::cout << std::endl; 
  } 

  cudaMemcpyToSymbol(d_GaussKernel, gausskernel, kernelSize*kernelSize*sizeof(float));
}

void Computation::LowPass_forSubImages(float *res, float *src, int width, int height)
{
  int kernelSize{GAUSSIANSIZE};
  
  dim3 block(16, 16, 1);
  dim3 grid(width/block.x+1, height/block.y+1, 1);

  GaussKernelBlock<<<grid, block, 0>>>(res, src, width, height, kernelSize);
  
  return; 
}


void Computation::cudaVkImportSemaphore()
{
    cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;
    memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));
    externalSemaphoreHandleDesc.type      = cudaExternalSemaphoreHandleTypeOpaqueFd;
    externalSemaphoreHandleDesc.handle.fd = cudaUpdateVkVertexBufSemaphoreHandle;
    externalSemaphoreHandleDesc.flags = 0;
    checkCudaErrors(cudaImportExternalSemaphore(&cudaExtCudaUpdateVkVertexBufSemaphore, &externalSemaphoreHandleDesc));
    memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));
    externalSemaphoreHandleDesc.type      = cudaExternalSemaphoreHandleTypeOpaqueFd;
    externalSemaphoreHandleDesc.handle.fd = vkUpdateCudaVertexBufSemaphoreHandle;
    externalSemaphoreHandleDesc.flags = 0;
    checkCudaErrors(cudaImportExternalSemaphore(&cudaExtVkUpdateCudaVertexBufSemaphore, &externalSemaphoreHandleDesc));
    printf("CUDA Imported Vulkan semaphore\n");
}

void Computation::InitSiftData(SiftData &data, int numPoints, int numSlices, bool host, bool dev)
{
  data.numPts = 0;
  data.maxPts = numPoints*numSlices;
  int sz = sizeof(SiftPoint)*numPoints*numSlices;
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
  int w = width*(scaleUp ? 2 : 1); 
  int h = height*(scaleUp ? 2 : 1);
  int p = iAlignUp(w, 128);
  int size = h*p;                 // image sizes
  int sizeTmp = nd*h*p;           // laplace buffer sizes
  for (int i=0;i<numOctaves;i++) {
    w /= 2;
    h /= 2;
    int p = iAlignUp(w, 128);
    size += h*p;
    sizeTmp += nd*h*p; 
  }
  float *memoryTmp = NULL; 
  size_t pitch;
  size += sizeTmp;
  cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size+4095)/4096*sizeof(float));
#ifdef VERBOSE
  printf("Allocated memory size: %d bytes\n", size);
  printf("Memory allocation time =      %.2f ms\n\n", timer.read());
#endif
  return memoryTmp;
}

void Computation::ExtractSift(SiftData &siftData, CudaImage &img, int numOctaves, double initBlur, float thresh, float lowestScale, bool scaleUp, float *tempMemory, unsigned char *chardata) 
{
  unsigned int *d_PointCounterAddr;
  cudaGetSymbolAddress((void**)&d_PointCounterAddr, d_PointCounter);
  cudaMemset(d_PointCounterAddr, 0, (8*2+1)*sizeof(int));
  cudaMemcpyToSymbol(d_MaxNumPoints, &siftData.maxPts, sizeof(int)); // this seems okay, it's a cuda variable after all.

  const int nd = NUM_SCALES + 3;
  int w = img.width*(scaleUp ? 2 : 1);
  int h = img.height*(scaleUp ? 2 : 1);
  int p = iAlignUp(w, 128);
  int width = w, height = h;
  int size = h*p;                 // image sizes
  int sizeTmp = nd*h*p;           // laplace buffer sizes
  for (int i=0;i<numOctaves;i++) {
    w /= 2;
    h /= 2;
    int p = iAlignUp(w, 128);
    size += h*p;
    sizeTmp += nd*h*p; 
  }
  float *memoryTmp = tempMemory; 
  size += sizeTmp;
  if (!tempMemory) {
    //size_t pitch;
    //cudaMallocPitch((void **)&memoryTmp, &pitch, (size_t)4096, (size+4095)/4096*sizeof(float));
#ifdef VERBOSE
    printf("Allocated memory size: %d bytes\n", size);
#endif
  }
  float *memorySub = memoryTmp + sizeTmp;
  
  CudaImage lowImg;
  lowImg.Allocate(width, height, iAlignUp(width, 128), false, memorySub);



  if (!scaleUp) {
    float kernel[8*12*16];
    PrepareLaplaceKernels(numOctaves, 0.0f, kernel);
    cudaMemcpyToSymbolAsync(d_LaplaceKernel, kernel, 8*12*16*sizeof(float));
    LowPass(lowImg, img, max(initBlur, 0.001f), chardata);
    ExtractSiftLoop(siftData, lowImg, numOctaves, 0.0f, thresh, lowestScale, 1.0f, memoryTmp, memorySub + height*iAlignUp(width, 128));
    cudaMemcpy(&siftData.numPts, &d_PointCounterAddr[2*numOctaves], sizeof(int), cudaMemcpyDeviceToHost); 
    siftData.numPts = (siftData.numPts<siftData.maxPts ? siftData.numPts : siftData.maxPts);
  } else {
    CudaImage upImg;
    upImg.Allocate(width, height, iAlignUp(width, 128), false, memoryTmp);
    ScaleUp(upImg, img, chardata);
    LowPass(lowImg, upImg, max(initBlur, 0.001f),NULL);
    float kernel[8*12*16];
    PrepareLaplaceKernels(numOctaves, 0.0f, kernel);
    cudaMemcpyToSymbolAsync(d_LaplaceKernel, kernel, 8*12*16*sizeof(float));
    ExtractSiftLoop(siftData, lowImg, numOctaves, 0.0f, thresh, lowestScale*2.0f, 1.0f, memoryTmp, memorySub + height*iAlignUp(width, 128));
    cudaMemcpy(&siftData.numPts, &d_PointCounterAddr[2*numOctaves], sizeof(int), cudaMemcpyDeviceToHost); 
    siftData.numPts = (siftData.numPts<siftData.maxPts ? siftData.numPts : siftData.maxPts);
    RescalePositions(siftData, 0.5f);
  } 
  
  if (!tempMemory)
    cudaFree(memoryTmp);
#ifdef MANAGEDMEM
  cudaDeviceSynchronize());
#else
  if (siftData.h_data)
    cudaMemcpy(siftData.h_data, siftData.d_data, sizeof(SiftPoint)*siftData.numPts, cudaMemcpyDeviceToHost);
#endif
}

void Computation::FreeSiftTempMemory(float *memoryTmp)
{
  if (memoryTmp)
    cudaFree(memoryTmp);
}

void Computation::PrepareLaplaceKernels(int numOctaves, float initBlur, float *kernel)
{
  if (numOctaves>1) {
    float totInitBlur = (float)sqrt(initBlur*initBlur + 0.5f*0.5f) / 2.0f;
    PrepareLaplaceKernels(numOctaves-1, totInitBlur, kernel);
  }
  float scale = pow(2.0f, -1.0f/NUM_SCALES);
  float diffScale = pow(2.0f, 1.0f/NUM_SCALES);
  for (int i=0;i<NUM_SCALES+3;i++) {
    float kernelSum = 0.0f;
    float var = scale*scale - initBlur*initBlur;
    for (int j=0;j<=LAPLACE_R;j++) {
      kernel[numOctaves*12*16 + 16*i + j] = (float)expf(-(double)j*j/2.0/var);
      kernelSum += (j==0 ? 1 : 2)*kernel[numOctaves*12*16 + 16*i + j]; 
    }
    for (int j=0;j<=LAPLACE_R;j++)
      kernel[numOctaves*12*16 + 16*i + j] /= kernelSum;
    scale *= diffScale;
  }
}

void Computation::LowPass(CudaImage &res, CudaImage &src, float scale, unsigned char *chardata)
{


  float kernel[2*LOWPASS_R+1];
  static float oldScale = -1.0f;
  if (scale!=oldScale) {
    float kernelSum = 0.0f;
    float ivar2 = 1.0f/(2.0f*scale*scale);
    for (int j=-LOWPASS_R;j<=LOWPASS_R;j++) {
      kernel[j+LOWPASS_R] = (float)expf(-(double)j*j*ivar2);
      kernelSum += kernel[j+LOWPASS_R]; 
    }
    for (int j=-LOWPASS_R;j<=LOWPASS_R;j++) 
      kernel[j+LOWPASS_R] /= kernelSum;

    
    cudaMemcpyToSymbol(d_LowPassKernel, kernel, (2*LOWPASS_R+1)*sizeof(float));
    oldScale = scale;
  }  
  int width = res.width;
  int pitch = res.pitch;
  int height = res.height;
  dim3 blocks(iDivUp(width, LOWPASS_W), iDivUp(height, LOWPASS_H));
#if 1
  dim3 threads(LOWPASS_W+2*LOWPASS_R, 4); 
  if (chardata != NULL) {
    LowPassBlockCharYUYV<<<blocks, threads>>>(chardata, res.d_data, 1920, 1920, 1080);
  }  
  else {
    LowPassBlock<<<blocks, threads>>>(src.d_data, res.d_data, width, pitch, height);
  }
  
#else
  dim3 threads(LOWPASS_W+2*LOWPASS_R, LOWPASS_H);
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
  if (numOctaves>1) {
    CudaImage subImg;
    int p = iAlignUp(w/2, 128);
    subImg.Allocate(w/2, h/2, p, false, memorySub); 
    ScaleDown(subImg, img, 0.5f);
    float totInitBlur = (float)sqrt(initBlur*initBlur + 0.5f*0.5f) / 2.0f;
    ExtractSiftLoop(siftData, subImg, numOctaves-1, totInitBlur, thresh, lowestScale, subsampling*2.0f, memoryTmp, memorySub + (h/2)*p);
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
  safeCall(cudaGetSymbolAddress((void**)&d_PointCounterAddr, d_PointCounter));
  unsigned int fstPts, totPts;
  safeCall(cudaMemcpy(&fstPts, &d_PointCounterAddr[2*octave-1], sizeof(int), cudaMemcpyDeviceToHost)); 
  TimerGPU timer0;
#endif
  CudaImage diffImg[nd];
  int w = img.width; 
  int h = img.height;
  int p = iAlignUp(w, 128);
  for (int i=0;i<nd-1;i++) 
    diffImg[i].Allocate(w, h, p, false, memoryTmp + i*p*h); 

  // Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = img.d_data;
  resDesc.res.pitch2D.width = img.width;
  resDesc.res.pitch2D.height = img.height;
  resDesc.res.pitch2D.pitchInBytes = img.pitch*sizeof(float);  
  resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();
  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0]   = cudaAddressModeClamp;
  texDesc.addressMode[1]   = cudaAddressModeClamp;
  texDesc.filterMode       = cudaFilterModeLinear;
  texDesc.readMode         = cudaReadModeElementType;
  texDesc.normalizedCoords = 0;
  // Create texture object
  cudaTextureObject_t texObj = 0;
  cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

#ifdef VERBOSE
  TimerGPU timer1;
#endif
  float baseBlur = pow(2.0f, -1.0f/NUM_SCALES);
  float diffScale = pow(2.0f, 1.0f/NUM_SCALES);
  LaplaceMulti(texObj, img, diffImg, octave); 
  FindPointsMulti(diffImg, siftData, thresh, 10.0f, 1.0f/NUM_SCALES, lowestScale/subsampling, subsampling, octave);
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
  printf("GPU time : %.2f ms + %.2f ms + %.2f ms = %.2f ms\n", totTime-gpuTimeDoG-gpuTimeSift, gpuTimeDoG, gpuTimeSift, totTime);
  safeCall(cudaMemcpy(&totPts, &d_PointCounterAddr[2*octave+1], sizeof(int), cudaMemcpyDeviceToHost));
  totPts = (totPts<siftData.maxPts ? totPts : siftData.maxPts);
  if (totPts>0) 
    printf("           %.2f ms / DoG,  %.4f ms / Sift,  #Sift = %d\n", gpuTimeDoG/NUM_SCALES, gpuTimeSift/(totPts-fstPts), totPts-fstPts); 
#endif
}

void Computation::ScaleDown(CudaImage &res, CudaImage &src, float variance)
{
  static float oldVariance = -1.0f;
  if (res.d_data==NULL || src.d_data==NULL) {
    printf("ScaleDown: missing data\n");
    return;
  }
  if (oldVariance!=variance) {
    float h_Kernel[5];
    float kernelSum = 0.0f;
    for (int j=0;j<5;j++) {
      h_Kernel[j] = (float)expf(-(double)(j-2)*(j-2)/2.0/variance);      
      kernelSum += h_Kernel[j];
    }
    for (int j=0;j<5;j++)
      h_Kernel[j] /= kernelSum;  
    cudaMemcpyToSymbol(d_ScaleDownKernel, h_Kernel, 5*sizeof(float));
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
  if (res.d_data==NULL || src.d_data==NULL) {
    printf("ScaleUp: missing data\n");
    return;
  }
  dim3 blocks(iDivUp(res.width, SCALEUP_W), iDivUp(res.height, SCALEUP_H));
  dim3 threads(SCALEUP_W/2, SCALEUP_H/2);
  if (chardata != NULL) {
    ScaleUpKernelCharYUYV<<<blocks, threads>>>(res.d_data, chardata, src.width, src.pitch, src.height, res.pitch);
  }  
  else {
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
  dim3 threads(LAPLACE_W+2*LAPLACE_R);
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
  if (sources->d_data==NULL) {
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
  dim3 blocks(iDivUp(w, MINMAX_W)*NUM_SCALES, iDivUp(h, MINMAX_H));
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
  dim3 threads(11*11);
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

void CudaImage::Allocate(int w, int h, int p, bool host, float *devmem, float *hostmem, bool withPixelFlags) 
{
  width = w;
  height = h; 
  pitch = p; 
  d_data = devmem;
  h_data = hostmem; 
  t_data = NULL; 
  if (devmem==NULL) {
    cudaMallocPitch((void **)&d_data, (size_t*)&pitch, (size_t)(sizeof(float)*width), (size_t)height);
    pitch /= sizeof(float);
    if (d_data==NULL) 
      std::cout << "Failed to allocate device data" << std::endl;
    d_internalAlloc = true;
  }
  if (withPixelFlags == true) {
    cudaMallocPitch((void **)&d_pixelFlags, (size_t*)&pitch, (size_t)(sizeof(bool)*width), (size_t)height);
    cudaMallocPitch((void **)&d_pixHeight, (size_t*)&pitch, (size_t)(sizeof(float)*width), (size_t)height);
    cudaMallocPitch((void **)&d_result, (size_t*)&pitch, (size_t)(sizeof(float)*width), (size_t)height);
    if (d_pixelFlags==NULL) 
      std::cout << "Failed to allocate device data" << std::endl;
  }
  if (host && hostmem==NULL) {
    h_data = (float *)malloc(sizeof(float)*pitch*height);
    h_internalAlloc = true;
  }
}

CudaImage::CudaImage() : 
  width(0), height(0), d_data(NULL), h_data(NULL), t_data(NULL), d_internalAlloc(false), h_internalAlloc(false)
{

}

CudaImage::~CudaImage()
{
  if (d_internalAlloc && d_data!=NULL) 
    cudaFree(d_data);
  d_data = NULL;
  if (h_internalAlloc && h_data!=NULL) 
    free(h_data);
  h_data = NULL;
  if (t_data!=NULL) 
    cudaFreeArray((cudaArray *)t_data);
  t_data = NULL;
}
  
void CudaImage::Download()  
{
  int p = sizeof(float)*pitch;
  if (d_data!=NULL && h_data!=NULL) 
    cudaMemcpy2D(d_data, p, h_data, sizeof(float)*width, sizeof(float)*width, height, cudaMemcpyHostToDevice);

}

void CudaImage::Readback()
{
  int p = sizeof(float)*pitch;
  cudaMemcpy2D(h_data, sizeof(float)*width, d_data, p, sizeof(float)*width, height, cudaMemcpyDeviceToHost);
}

void CudaImage::InitTexture()
{
  cudaChannelFormatDesc t_desc = cudaCreateChannelDesc<float>(); 
  cudaMallocArray((cudaArray **)&t_data, &t_desc, pitch, height); 
  if (t_data==NULL)
    std::cout << "Failed to allocated texture data" << std::endl;

}
 
void CudaImage::CopyToTexture(CudaImage &dst, bool host)
{
  if (dst.t_data==NULL) {
    std::cout << "Error CopyToTexture: No texture data" << std::endl;
    return;
  }
  if ((!host || h_data==NULL) && (host || d_data==NULL)) {
    std::cout << "Error CopyToTexture: No source data" << std::endl;
    return;
  }
  if (host)
    cudaMemcpy2DToArray((cudaArray *)dst.t_data, 0, 0,  h_data, sizeof(float)*pitch*dst.height, sizeof(float)*pitch*dst.height, 1, cudaMemcpyHostToDevice);
  else
    cudaMemcpy2DToArray((cudaArray *)dst.t_data, 0, 0,  h_data, sizeof(float)*pitch*dst.height, sizeof(float)*pitch*dst.height, 1, cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();

}

void Computation::MatchSiftData(SiftData &data1, SiftData &data2)
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
  if (data1.d_data==NULL || data2.d_data==NULL)
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
  int mode = 11;
  if (mode==5)// K40c 5.0ms, 1080 Ti 1.2ms, 2080 Ti 0.83ms
    FindMaxCorr5<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
  else if (mode==6) {                    // 2080 Ti 0.89ms
    threadsMax3 = dim3(32, 16);
    FindMaxCorr6<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
  } else if (mode==7)                    // 2080 Ti 0.50ms  
    FindMaxCorr7<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
  else if (mode==8) {                    // 2080 Ti 0.45ms
    blocksMax3 = dim3(iDivUp(numPts1, FMC_BW), iDivUp(numPts2, FMC_GH));
    threadsMax3 = dim3(FMC_NW, FMC_NH);
    FindMaxCorr8<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
  } else if (mode==9) {                  // 2080 Ti 0.46ms
    blocksMax3 = dim3(iDivUp(numPts1, FMC_BW), iDivUp(numPts2, FMC_GH));
    threadsMax3 = dim3(FMC_NW, FMC_NH);
    FindMaxCorr9<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
  } else if (mode==10) {                 // 2080 Ti 0.24ms
    blocksMax3 = dim3(iDivUp(numPts1, M7W));
    threadsMax3 = dim3(M7W, M7H/M7R);
    FindMaxCorr10<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
  } else if (mode==11) {                 // 2080 Ti 0.24ms
    dim3 blocksMax3 = dim3(256);
    dim3 threadsMax3 = dim3(numPts1/256+1);
    Find3DCorr<<<blocksMax3, threadsMax3>>>(sift1, sift2, numPts1, numPts2);
 
  }
  cudaDeviceSynchronize();
  checkMsg("FindMaxCorr5() execution failed\n");
#endif
  //printf("Siftpoint number %d: x: %f, y: %f, score: %f\n",data1.numPts/2, data1.d_data[data1.numPts/2].xpos, data1.d_data[data1.numPts/2].ypos, data1.d_data[data1.numPts/2].score);
  if (data1.h_data!=NULL) {
    float *h_ptr = &data1.h_data[0].score;
    float *d_ptr = &data1.d_data[0].score;
    cudaMemcpy2D(h_ptr, sizeof(SiftPoint), d_ptr, sizeof(SiftPoint), 5*sizeof(float), data1.numPts, cudaMemcpyDeviceToHost);
  }

  return;
}


void Computation::yuvToRGB_Convert(CudaImage &RGBImage, unsigned char *yuvdata) {
  
  int width = 1920;
  int height = 1080;
  dim3 blocks =   dim3(1024);
  dim3 threads =  dim3((RGBImage.width/3 + 1024 - 1) / 1024);

  gpuConvertYUYVtoRGBfloat_kernel<<<blocks, threads>>>(yuvdata, RGBImage.d_data, width, height);
}

void Computation::sharpenImage(CudaImage &greyscaleImage, unsigned char *yuvdata, float amount) {
  
  dim3 block(16, 16, 1);
  dim3 grid(1920/block.x+1, 1080/block.y+1, 1);

  gpuSharpenImageToGrayscale<<<grid, block, 0>>>(yuvdata, greyscaleImage.d_data, 1920, 1080, amount);
}

void Computation::tof_camera_undistort(uint16_t *dst, uint16_t *src, uint16_t *xCoordsPerPixel, uint16_t *yCoordsPerPixel, uint16_t *cosAlpha) {
  
    dim3 blocks =   dim3(512);
    dim3 threads =  dim3(1);
    int width_dst = 265;
    int height_dst = 205;
    int width_src = 352;
    int height_src = 286;
    if (cosAlpha == NULL) {
      gpuUndistort<<<blocks, threads>>>(dst, src, xCoordsPerPixel, yCoordsPerPixel, width_src, height_src, width_dst, height_dst);
    } else {
      gpuUndistortCosAlpha<<<blocks, threads>>>(dst, src, xCoordsPerPixel, yCoordsPerPixel, width_src, height_src, width_dst, height_dst, cosAlpha);
    }
    cudaDeviceSynchronize();
    checkMsg("Problem with gpuUndistort:\n");
  }