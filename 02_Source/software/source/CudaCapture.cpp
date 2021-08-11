/*
 * Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

/*
 *  V4L2 video capture example
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <iostream>

#include <getopt.h>             /* getopt_long() */

#include <fcntl.h>              /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <malloc.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <thread>
#include <asm/types.h>          /* for videodev2.h */

#include <linux/videodev2.h>

#include <pthread.h>  // for renaming the c++ thread

#include <cuda_runtime.h>


#include "CudaCapture.hpp"




void
CudaCapture::errno_exit                      (const char *           s)
{
    fprintf (stderr, "%s error %d, %s\n",
            s, errno, strerror (errno));

    exit (EXIT_FAILURE);
}

int
CudaCapture::xioctl                          (int                    fd,
                                 int                    request,
                                 void *                 arg)
{
    int r;

    do r = ioctl (fd, request, arg);
    while (-1 == r && EINTR == errno);

    return r;
}

void
CudaCapture::process_image                   (void *           p)
{
    //gpuConvertYUYVtoRGB ((unsigned char *) p, cuda_out_buffer, width, height);
    cuda_out_buffer = (unsigned char *) p;
}

int
CudaCapture::read_frame                      (void)
{
    struct v4l2_buffer buf;
    unsigned int i;

    switch (io) {
        case IO_METHOD_READ:
            if (-1 == read (fd, buffers[0].start, buffers[0].length)) {
                switch (errno) {
                    case EAGAIN:
                        return 0;

                    case EIO:
                        /* Could ignore EIO, see spec. */

                        /* fall through */

                    default:
                        errno_exit ("read");
                }
            }

            process_image (buffers[0].start);

            break;

        case IO_METHOD_MMAP:
            CLEAR (buf);

            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_MMAP;

            if (-1 == xioctl (fd, VIDIOC_DQBUF, &buf)) {
                switch (errno) {
                    case EAGAIN:
                        return 0;

                    case EIO:
                        /* Could ignore EIO, see spec. */

                        /* fall through */

                    default:
                        errno_exit ("VIDIOC_DQBUF");
                }
            }

            assert (buf.index < n_buffers);

            process_image (buffers[buf.index].start);

            if (-1 == xioctl (fd, VIDIOC_QBUF, &buf))
                errno_exit ("VIDIOC_QBUF");

            break;

        case IO_METHOD_USERPTR:
            CLEAR (buf);

            buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
            buf.memory = V4L2_MEMORY_USERPTR;

            if (-1 == xioctl (fd, VIDIOC_DQBUF, &buf)) {
                switch (errno) {
                    case EAGAIN:
                        return 0;

                    case EIO:
                        /* Could ignore EIO, see spec. */

                        /* fall through */

                    default:
                        errno_exit ("VIDIOC_DQBUF");
                }
            }

            for (i = 0; i < n_buffers; ++i)
                if (buf.m.userptr == (unsigned long) buffers[i].start
                        && buf.length == buffers[i].length)
                    break;

            assert (i < n_buffers);
            process_image ((void *) buf.m.userptr);

            if (-1 == xioctl (fd, VIDIOC_QBUF, &buf))
                errno_exit ("VIDIOC_QBUF");

            break;
    }

    return 1;
}

void
CudaCapture::mainloop                        (void)
{
    while (running) {
        startTime = std::chrono::steady_clock::now();
        
        count--;
        for (;;) {
            fd_set fds;
            struct timeval tv;
            int r;

            FD_ZERO (&fds);
            FD_SET (fd, &fds);

            /* Timeout. */
            tv.tv_sec = 2;
            tv.tv_usec = 0;

            r = select (fd + 1, &fds, NULL, NULL, &tv);

            if (-1 == r) {
                if (EINTR == errno)
                    continue;

                errno_exit ("select");
            }

            if (0 == r) {
                fprintf (stderr, "select timeout\n");
                continue; //exit (EXIT_FAILURE);
            }

            if (read_frame ())
                break;

            /* EAGAIN - continue select loop. */
        }

        endTime = std::chrono::steady_clock::now();

        std::chrono::steady_clock::duration timeSpan = endTime - startTime;

        double nseconds = double(timeSpan.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;

        //std::cout << dev_name << ": Capturing and conversions    " << 1/nseconds << "fps" <<  std::endl;
    }
}

void
CudaCapture::stop_capturing                  (void)
{
    enum v4l2_buf_type type;
    printf("stopping to capture\n");
    switch (io) {
        case IO_METHOD_READ:
            /* Nothing to do. */
            break;

        case IO_METHOD_MMAP:
        case IO_METHOD_USERPTR:
            type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

            if (-1 == xioctl (fd, VIDIOC_STREAMOFF, &type))
                errno_exit ("VIDIOC_STREAMOFF");

            break;
    }
}

void
CudaCapture::start_capturing                 (void)
{
    unsigned int i;
    enum v4l2_buf_type type;

    switch (io) {
        case IO_METHOD_READ:
            /* Nothing to do. */
            break;

        case IO_METHOD_MMAP:
            for (i = 0; i < n_buffers; ++i) {
                struct v4l2_buffer buf;

                CLEAR (buf);

                buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                buf.memory      = V4L2_MEMORY_MMAP;
                buf.index       = i;

                if (-1 == xioctl (fd, VIDIOC_QBUF, &buf))
                    errno_exit ("VIDIOC_QBUF");
            }

            type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

            if (-1 == xioctl (fd, VIDIOC_STREAMON, &type))
                errno_exit ("VIDIOC_STREAMON");

            break;

        case IO_METHOD_USERPTR:
            for (i = 0; i < n_buffers; ++i) {
                struct v4l2_buffer buf;

                CLEAR (buf);

                buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
                buf.memory      = V4L2_MEMORY_USERPTR;
                buf.index       = i;
                buf.m.userptr   = (unsigned long) buffers[i].start;
                buf.length      = buffers[i].length;

                if (-1 == xioctl (fd, VIDIOC_QBUF, &buf))
                    errno_exit ("VIDIOC_QBUF");
            }

            type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

            if (-1 == xioctl (fd, VIDIOC_STREAMON, &type))
                errno_exit ("VIDIOC_STREAMON");

            break;
    }
}

void
CudaCapture::uninit_device                   (void)
{
    unsigned int i;

    switch (io) {
        case IO_METHOD_READ:
            free (buffers[0].start);
            break;

        case IO_METHOD_MMAP:
            for (i = 0; i < n_buffers; ++i)
                if (-1 == munmap (buffers[i].start, buffers[i].length))
                    errno_exit ("munmap");
            break;

        case IO_METHOD_USERPTR:
            for (i = 0; i < n_buffers; ++i) {
                if (cuda_zero_copy) {
                    cudaFree (buffers[i].start);
                } else {
                    free (buffers[i].start);
                }
            }
            break;
    }

    free (buffers);

    if (cuda_zero_copy) {
        cudaFree (cuda_out_buffer);
    }
}

void
CudaCapture::init_read                       (unsigned int           buffer_size)
{
    buffers = (struct buffer *) calloc (1, sizeof (*buffers));

    if (!buffers) {
        fprintf (stderr, "Out of memory\n");
        exit (EXIT_FAILURE);
    }

    buffers[0].length = buffer_size;
    buffers[0].start = malloc (buffer_size);

    if (!buffers[0].start) {
        fprintf (stderr, "Out of memory\n");
        exit (EXIT_FAILURE);
    }
}

void
CudaCapture::init_mmap                       (void)
{
    struct v4l2_requestbuffers req;

    CLEAR (req);

    req.count               = 4;
    req.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory              = V4L2_MEMORY_MMAP;

    if (-1 == xioctl (fd, VIDIOC_REQBUFS, &req)) {
        if (EINVAL == errno) {
            fprintf (stderr, "%s does not support "
                    "memory mapping\n", dev_name);
            exit (EXIT_FAILURE);
        } else {
            errno_exit ("VIDIOC_REQBUFS");
        }
    }

    if (req.count < 2) {
        fprintf (stderr, "Insufficient buffer memory on %s\n",
                dev_name);
        exit (EXIT_FAILURE);
    }

    buffers = (struct buffer *) calloc (req.count, sizeof (*buffers));

    if (!buffers) {
        fprintf (stderr, "Out of memory\n");
        exit (EXIT_FAILURE);
    }

    for (n_buffers = 0; n_buffers < req.count; ++n_buffers) {
        struct v4l2_buffer buf;

        CLEAR (buf);

        buf.type        = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory      = V4L2_MEMORY_MMAP;
        buf.index       = n_buffers;

        if (-1 == xioctl (fd, VIDIOC_QUERYBUF, &buf))
            errno_exit ("VIDIOC_QUERYBUF");

        buffers[n_buffers].length = buf.length;
        buffers[n_buffers].start =
            mmap (NULL /* start anywhere */,
                    buf.length,
                    PROT_READ | PROT_WRITE /* required */,
                    MAP_SHARED /* recommended */,
                    fd, buf.m.offset);

        if (MAP_FAILED == buffers[n_buffers].start)
            errno_exit ("mmap");
    }
}

void
CudaCapture::init_userp                      (unsigned int           buffer_size)
{
    struct v4l2_requestbuffers req;
    unsigned int page_size;

    page_size = getpagesize ();
    buffer_size = (buffer_size + page_size - 1) & ~(page_size - 1);

    CLEAR (req);

    req.count               = 4;
    req.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory              = V4L2_MEMORY_USERPTR;

    if (-1 == xioctl (fd, VIDIOC_REQBUFS, &req)) {
        if (EINVAL == errno) {
            fprintf (stderr, "%s does not support "
                    "user pointer i/o\n", dev_name);
            exit (EXIT_FAILURE);
        } else {
            errno_exit ("VIDIOC_REQBUFS");
        }
    }

    buffers = (struct buffer *) calloc (4, sizeof (*buffers));

    if (!buffers) {
        fprintf (stderr, "Out of memory\n");
        exit (EXIT_FAILURE);
    }

    for (n_buffers = 0; n_buffers < 4; ++n_buffers) {
        buffers[n_buffers].length = buffer_size;
        printf("Buffer size: %d",buffer_size);
        if (cuda_zero_copy) {
            cudaMallocManaged (&buffers[n_buffers].start, buffer_size, cudaMemAttachGlobal);
        } else {
            buffers[n_buffers].start = memalign (/* boundary */ page_size,
                    buffer_size);
        }

        if (!buffers[n_buffers].start) {
            fprintf (stderr, "Out of memory\n");
            exit (EXIT_FAILURE);
        }
    }
}

void
CudaCapture::init_device                     (void)
{
    struct v4l2_capability cap;
    struct v4l2_cropcap cropcap;
    struct v4l2_crop crop;
    struct v4l2_format fmt;
    unsigned int min;

    if (-1 == xioctl (fd, VIDIOC_QUERYCAP, &cap)) {
        if (EINVAL == errno) {
            fprintf (stderr, "%s is no V4L2 device\n",
                    dev_name);
            exit (EXIT_FAILURE);
        } else {
            errno_exit ("VIDIOC_QUERYCAP");
        }
    }

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        fprintf (stderr, "%s is no video capture device\n",
                dev_name);
        exit (EXIT_FAILURE);
    }

    switch (io) {
        case IO_METHOD_READ:
            if (!(cap.capabilities & V4L2_CAP_READWRITE)) {
                fprintf (stderr, "%s does not support read i/o\n",
                        dev_name);
                exit (EXIT_FAILURE);
            }

            break;

        case IO_METHOD_MMAP:
        case IO_METHOD_USERPTR:
            if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
                fprintf (stderr, "%s does not support streaming i/o\n",
                        dev_name);
                exit (EXIT_FAILURE);
            }

            break;
    }


    /* Select video input, video standard and tune here. */


    CLEAR (cropcap);

    cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (0 == xioctl (fd, VIDIOC_CROPCAP, &cropcap)) {
        crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        crop.c = cropcap.defrect; /* reset to default */

        if (-1 == xioctl (fd, VIDIOC_S_CROP, &crop)) {
            switch (errno) {
                case EINVAL:
                    /* Cropping not supported. */
                    break;
                default:
                    /* Errors ignored. */
                    break;
            }
        }
    } else {
        /* Errors ignored. */
    }


    CLEAR (fmt);

    fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = width;
    fmt.fmt.pix.height      = height;
    fmt.fmt.pix.pixelformat = pixel_format;
    fmt.fmt.pix.field       = field;

    if (-1 == xioctl (fd, VIDIOC_S_FMT, &fmt))
        errno_exit ("VIDIOC_S_FMT");

    /* Note VIDIOC_S_FMT may change width and height. */

    /* Buggy driver paranoia. */
    min = fmt.fmt.pix.width * 2;
    if (fmt.fmt.pix.bytesperline < min)
        fmt.fmt.pix.bytesperline = min;
    min = fmt.fmt.pix.bytesperline * fmt.fmt.pix.height;
    if (fmt.fmt.pix.sizeimage < min)
        fmt.fmt.pix.sizeimage = min;

    switch (io) {
        case IO_METHOD_READ:
            init_read (fmt.fmt.pix.sizeimage);
            break;

        case IO_METHOD_MMAP:
            init_mmap ();
            break;

        case IO_METHOD_USERPTR:
            init_userp (fmt.fmt.pix.sizeimage);
            break;
    }
}

void
CudaCapture::close_device                    (void)
{
    if (-1 == close (fd))
        errno_exit ("close");

    fd = -1;
}

void
CudaCapture::open_device                     (void)
{
    struct stat st;

    if (-1 == stat (dev_name, &st)) {
        fprintf (stderr, "Cannot identify '%s': %d, %s\n",
                dev_name, errno, strerror (errno));
        exit (EXIT_FAILURE);
    }

    if (!S_ISCHR (st.st_mode)) {
        fprintf (stderr, "%s is no device\n", dev_name);
        exit (EXIT_FAILURE);
    }

    fd = open (dev_name, O_RDWR /* required */ | O_NONBLOCK, 0);

    if (-1 == fd) {
        fprintf (stderr, "Cannot open '%s': %d, %s\n",
                dev_name, errno, strerror (errno));
        exit (EXIT_FAILURE);
    }
}

void
CudaCapture::init_cuda                       (void)
{
    /* Check unified memory support. */
    if (cuda_zero_copy) {
        cudaDeviceProp devProp;
        cudaGetDeviceProperties (&devProp, 0);
        if (!devProp.managedMemory) {
            printf ("CUDA device does not support managed memory.\n");
            cuda_zero_copy = false;
        }
    }

    /* Allocate output buffer. */
    size_t size = width * height*3;
    if (cuda_zero_copy) {
        cudaMallocManaged (&cuda_out_buffer, size, cudaMemAttachGlobal);
    } else {
        cuda_out_buffer = (unsigned char *) malloc (size);
    }

    cudaDeviceSynchronize ();
}

void
CudaCapture::usage                           (FILE *                 fp,
                                 int                    argc,
                                 char **                argv)
{
    fprintf (fp,
            "Usage: %s [options]\n\n"
            "Options:\n"
            "-c | --count N       Frame count (default: %u)\n"
            "-d | --device name   Video device name (default: %s)\n"
            "-f | --format        Capture input pixel format (default: UYVY)\n"
            "-h | --help          Print this message\n"
            "-m | --mmap          Use memory mapped buffers\n"
            "-o | --output        Output file name (default: %s)\n"
            "-s | --size WxH      Frame size (default: %ux%u)\n"
            "-u | --userp         Use application allocated buffers\n"
            "-z | --zcopy         Use zero copy CUDA memory\n"
            "Experimental options:\n"
            "-r | --read          Use read() calls\n"
            "-F | --field         Capture field (default: INTERLACED)\n"
            "",
            argv[0], count, dev_name, file_name, width, height);
}


unsigned int CudaCapture::v4l2_format_code(const char *name)
{
    unsigned int i;

    for (i = 0; i < ARRAY_SIZE(pixel_formats); ++i) {
        if (strcasecmp(pixel_formats[i].name, name) == 0)
            return pixel_formats[i].fourcc;
    }

    return 0;
}


unsigned int CudaCapture::v4l2_field_code(const char *name)
{
    unsigned int i;

    for (i = 0; i < ARRAY_SIZE(fields); ++i) {
        if (strcasecmp(fields[i].name, name) == 0)
            return fields[i].field;
    }

    return -1;
}

void
CudaCapture::run ()
{
    std::cout << "CudaCapture: opening device" << std:: endl;
    open_device ();
    std::cout << "CudaCapture: initializing device" << std:: endl;
    init_device ();
    std::cout << "CudaCapture: initializing cuda" << std:: endl;
    init_cuda ();
    std::cout << "CudaCapture: start capturing" << std:: endl;
    start_capturing ();
    std::cout << "CudaCapture: entering mainloop" << std:: endl;
    mainloop ();

    stop_capturing ();
    printf("stopped capturing\n");
    uninit_device ();
    printf("uninit device\n");
    close_device ();
    printf("closed device\n");

    return;
}

void CudaCapture::CudaCaptureInit(int device)
{
    std::cout << "CudaCaptureInit called" << std::endl;
    io = IO_METHOD_USERPTR;
    cuda_zero_copy = true;
    if (device == 0) strcpy(dev_name, "/dev/video0");
    else strcpy(dev_name, "/dev/video1"); 
    width = 1920;
    height = 1080;
    count = 100;
    n_buffers = 0;
    file_name;
    running = true; 
    std::cout << "Starting Thread" << std::endl;
    tid = std::thread (&CudaCapture::run, this);
    std::cout << "Started Thread: TID = " << tid.get_id() << std::endl;
}

void CudaCapture::cleanup() {
    running = false;
    tid.join();
}

void CudaCapture::lockMutex() {
	mtx.lock();
	//std::cout << " locked" << std::endl;	
}

void CudaCapture::unlockMutex() {
	mtx.unlock();
	//std::cout << " unlocked" << std::endl;	
}