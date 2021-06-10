#pragma once
#include <vector>
#include <thread>
#include <mutex>

#define V4L2_DEVICE_NAME				"/dev/video"
#define V4L2_DEVICE_NAME_SIZE			(sizeof(V4L2_DEVICE_NAME) + 1)	/* + 1 for the number */
#define MAX_IMAGE_SIZE					(1920 * 1080 * sizeof(uint32_t))
#define	USERPTR_ALIGNMENT				256
#define NUM_BUFFERS 2
#ifdef V4LINPUT


#define CAPTURE_BUFFERS_PER_THREAD		2
#define MAX_IMAGE_SIZE_ALIGNED			(MAX_IMAGE_SIZE + USERPTR_ALIGNMENT)
#define BUFFERS_SIZE					(MAX_IMAGE_SIZE_ALIGNED * CAPTURE_BUFFERS_PER_THREAD)

#include <linux/videodev2.h>

#endif

static unsigned char rawImages[NUM_BUFFERS][4 * 1920 * 1080];
static unsigned char heightMap[4][4 * 1920 * 1080];
#ifdef V4LINPUT
    static bool is_running = true;

    struct img_buffer {
	void   *start;
	size_t  length;
    };

    typedef struct {
	uint32_t			data[MAX_IMAGE_SIZE];
} capture_buffer_t;

struct capture_thread_args {
	char v4l2_dev_name[V4L2_DEVICE_NAME_SIZE];	/* Device name + path for open() */
	/* Set by user */
	pthread_t			p_v4l2;			/* PThread handle of the capture thread */
	int					v4l2_id;		/* Defines which /dev/videoX */
	/* Set by device */
	pthread_spinlock_t	lock;			/* Spin lock to access data below */
	unsigned char		img_ctr;		/* Counts the number of successfully captured images */
	capture_buffer_t	buffers[CAPTURE_BUFFERS_PER_THREAD];	/* Buffers for V4L2 capturing */
	void				*img;			/* Most recently captured image */
	struct v4l2_format	fmt;			/* Format detected from V4L2 device (do not change) */
};


#endif
class FrameCapture {


public:
    void start(int input = 0);

    void cleanup();

    std::tuple<int, int, int, unsigned char*> getFrame();

    void lockMutex();

    void unlockMutex();

private:

    struct buffer {
        void* start;
        size_t                  length;
    };



    char dev_name[V4L2_DEVICE_NAME_SIZE];

    std::thread tid;

    std::mutex mtx;

    unsigned char* currentImageInBuffer = nullptr;

    unsigned int n_buffers{ 0 };

    int imageIndex{ 0 };

    int bufferSize{ 8 };

    int texWidth{ 0 };

    int texHeight{ 0 };

    int texChannels{ 0 };

    int inputType{ 0 };

    #ifdef V4LINPUT

    void errno_exit(const char *s);

    int xioctl(int fh, int request, void *arg);

    void close_device(int fd);

    int open_device(char *dev_name);

    void request_v4l2_buffers(int v4l2_fd, struct capture_thread_args *cta, int num_buffers);

    void update_format(int v4l2_fd, struct capture_thread_args *cta);

    void init_device(int v4l2_fd, struct capture_thread_args *cta);

    int read_frame(int fd, struct v4l2_buffer *buf);

    void stop_capturing(int v4l2_fd, struct capture_thread_args *cta);

    void start_capturing(int v4l2_fd, struct img_buffer *bufs, struct capture_thread_args *cta);

    void reset_device(int v4l2_fd, struct capture_thread_args *cta, struct img_buffer *img_bufs);

    void prepare_img_bufs(struct img_buffer *img_bufs, struct capture_thread_args *cta);

    #endif

    void run();

    #ifdef V4LINPUT

    #endif
};