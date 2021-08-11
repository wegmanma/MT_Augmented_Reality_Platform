#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <getopt.h>             /* getopt_long() */
#include <thread>
#include <mutex>
#include <fcntl.h>              /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <malloc.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <chrono>

#include <linux/videodev2.h>

#define CLEAR(x) memset (&(x), 0, sizeof (x))
#define ARRAY_SIZE(a)   (sizeof(a)/sizeof((a)[0]))

static const struct option
long_options [] = {
    { "count",      required_argument,      NULL,           'c' },
    { "device",     required_argument,      NULL,           'd' },
    { "format",     required_argument,      NULL,           'f' },
    { "field",      required_argument,      NULL,           'F' },
    { "help",       no_argument,            NULL,           'h' },
    { "mmap",       no_argument,            NULL,           'm' },
    { "output",     required_argument,      NULL,           'o' },
    { "read",       no_argument,            NULL,           'r' },
    { "size",       required_argument,      NULL,           's' },
    { "userp",      no_argument,            NULL,           'u' },
    { "zcopy",      no_argument,            NULL,           'z' },
    { 0, 0, 0, 0 }
};

static const char short_options [] = "c:d:f:F:hmo:rs:uz";

static struct {
    const char *name;
    unsigned int fourcc;
} pixel_formats[] = {
    { "RGB332", V4L2_PIX_FMT_RGB332 },
    { "RGB555", V4L2_PIX_FMT_RGB555 },
    { "RGB565", V4L2_PIX_FMT_RGB565 },
    { "RGB555X", V4L2_PIX_FMT_RGB555X },
    { "RGB565X", V4L2_PIX_FMT_RGB565X },
    { "BGR24", V4L2_PIX_FMT_BGR24 },
    { "RGB24", V4L2_PIX_FMT_RGB24 },
    { "BGR32", V4L2_PIX_FMT_BGR32 },
    { "RGB32", V4L2_PIX_FMT_RGB32 },
    { "Y8", V4L2_PIX_FMT_GREY },
    { "Y10", V4L2_PIX_FMT_Y10 },
    { "Y12", V4L2_PIX_FMT_Y12 },
    { "Y16", V4L2_PIX_FMT_Y16 },
    { "UYVY", V4L2_PIX_FMT_UYVY },
    { "VYUY", V4L2_PIX_FMT_VYUY },
    { "YUYV", V4L2_PIX_FMT_YUYV },
    { "YVYU", V4L2_PIX_FMT_YVYU },
    { "NV12", V4L2_PIX_FMT_NV12 },
    { "NV21", V4L2_PIX_FMT_NV21 },
    { "NV16", V4L2_PIX_FMT_NV16 },
    { "NV61", V4L2_PIX_FMT_NV61 },
    { "NV24", V4L2_PIX_FMT_NV24 },
    { "NV42", V4L2_PIX_FMT_NV42 },
    { "SBGGR8", V4L2_PIX_FMT_SBGGR8 },
    { "SGBRG8", V4L2_PIX_FMT_SGBRG8 },
    { "SGRBG8", V4L2_PIX_FMT_SGRBG8 },
    { "SRGGB8", V4L2_PIX_FMT_SRGGB8 },
    { "SBGGR10_DPCM8", V4L2_PIX_FMT_SBGGR10DPCM8 },
    { "SGBRG10_DPCM8", V4L2_PIX_FMT_SGBRG10DPCM8 },
    { "SGRBG10_DPCM8", V4L2_PIX_FMT_SGRBG10DPCM8 },
    { "SRGGB10_DPCM8", V4L2_PIX_FMT_SRGGB10DPCM8 },
    { "SBGGR10", V4L2_PIX_FMT_SBGGR10 },
    { "SGBRG10", V4L2_PIX_FMT_SGBRG10 },
    { "SGRBG10", V4L2_PIX_FMT_SGRBG10 },
    { "SRGGB10", V4L2_PIX_FMT_SRGGB10 },
    { "SBGGR12", V4L2_PIX_FMT_SBGGR12 },
    { "SGBRG12", V4L2_PIX_FMT_SGBRG12 },
    { "SGRBG12", V4L2_PIX_FMT_SGRBG12 },
    { "SRGGB12", V4L2_PIX_FMT_SRGGB12 },
    { "DV", V4L2_PIX_FMT_DV },
    { "MJPEG", V4L2_PIX_FMT_MJPEG },
    { "MPEG", V4L2_PIX_FMT_MPEG },
};

static struct {
    const char *name;
    unsigned int field;
} fields[] = {
    { "ANY", V4L2_FIELD_ANY },
    { "NONE", V4L2_FIELD_NONE },
    { "TOP", V4L2_FIELD_TOP },
    { "BOTTOM", V4L2_FIELD_BOTTOM },
    { "INTERLACED", V4L2_FIELD_INTERLACED },
    { "SEQ_TB", V4L2_FIELD_SEQ_TB },
    { "SEQ_BT", V4L2_FIELD_SEQ_BT },
    { "ALTERNATE", V4L2_FIELD_ALTERNATE },
    { "INTERLACED_TB", V4L2_FIELD_INTERLACED_TB },
    { "INTERLACED_BT", V4L2_FIELD_INTERLACED_BT },
};

typedef enum {
    IO_METHOD_READ,
    IO_METHOD_MMAP,
    IO_METHOD_USERPTR,
} io_method;

struct buffer {
    void *                  start;
    size_t                  length;
};

class CudaCapture {


public:

unsigned char *  cuda_out_buffer = NULL;

void CudaCaptureInit(int device);

void cleanup();

void unlockMutex();

void lockMutex();

private:

char            dev_name[12]; 
io_method        io;
int              fd;
struct buffer *  buffers;
unsigned int     n_buffers;
unsigned int     width;
unsigned int     height;
unsigned int     count;

bool             cuda_zero_copy;
const char *     file_name;
unsigned int     pixel_format;
unsigned int     field;

bool running; 
std::thread tid;
std::mutex mtx;

std::chrono::steady_clock::time_point startTime;
std::chrono::steady_clock::time_point endTime;
std::chrono::steady_clock::duration timeSpan;

void
errno_exit                      (const char *           s);

int
xioctl                          (int                    fd,
                                 int                    request,
                                 void *                 arg);

void run();

void
process_image                   (void *           p);

int
read_frame                      (void);

void
mainloop                        (void);

void
stop_capturing                  (void);

void
start_capturing                 (void);

void
uninit_device                   (void);

void
init_read                       (unsigned int           buffer_size);

void
init_mmap                       (void);

void
init_userp                      (unsigned int           buffer_size);

void
init_device                     (void);

void
close_device                    (void);

void
open_device                     (void);

void
init_cuda                       (void);

void
usage                           (FILE *                 fp,
                                 int                    argc,
                                 char **                argv);

unsigned int v4l2_format_code(const char *name);
unsigned int v4l2_field_code(const char *name);
};