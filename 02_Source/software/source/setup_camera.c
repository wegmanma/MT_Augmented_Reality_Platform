#include "setup_camera.h"

#include "common.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <getopt.h>             /* getopt_long() */

#include <fcntl.h>              /* low-level i/o */
#include <unistd.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/ioctl.h>
#include <errno.h>

#include <linux/videodev2.h>

#include <media/tegra-v4l2-camera.h>

//#define TEGRA_CAMERA_CID_SENSOR_MODE_ID 0x009a2008
//#define DEVICE "/dev/video0"

//static int64_t camera_mode = 4;
static int returned_val;
static int fd = -1;

static int xioctl(int fh, int request, void *arg)
{
        int r;

        do {
            r = ioctl(fh, request, arg);
        } while (-1 == r && EINTR == errno);

        return r;
}

static void get_sensor_mode_boundaries(struct v4l2_query_ext_ctrl * query_ctrl)
{
    memset(query_ctrl, 0, sizeof(query_ctrl));
    query_ctrl->id = TEGRA_CAMERA_CID_SENSOR_MODE_ID;

    returned_val = xioctl(fd, VIDIOC_QUERY_EXT_CTRL, query_ctrl);
    //printf("return: %d\n", returned_val);
    //printf("Error: %s\n", strerror(errno));
}

static void get_sensor_mode_value(struct v4l2_ext_controls * get_ctrls, struct v4l2_ext_control * get_ctrl)
{
    memset(get_ctrls, 0, sizeof(get_ctrls));
    memset(get_ctrl, 0, sizeof(get_ctrl));

    get_ctrls->ctrl_class = V4L2_CTRL_ID2CLASS(TEGRA_CAMERA_CID_SENSOR_MODE_ID);
    get_ctrls->count = 1;
    get_ctrls->controls = get_ctrl;

    get_ctrl->id = TEGRA_CAMERA_CID_SENSOR_MODE_ID;

    returned_val = xioctl(fd, VIDIOC_G_EXT_CTRLS, get_ctrls);
    //printf("return: %d\n", returned_val);
    //printf("Error: %s\n", strerror(errno));
    //printf("Val: %lld\n", get_ctrl->value64);
}

static void set_sensor_mode_value(struct v4l2_ext_controls * get_ctrls, int64_t cam_mode)
{
    get_ctrls->controls->value64 = cam_mode;
    returned_val = xioctl(fd, VIDIOC_S_EXT_CTRLS, get_ctrls);
    //printf("return: %d\n", returned_val);
    //printf("Error: %s\n", strerror(errno));
}

/*
 * Function:  set_camera_mode 
 * --------------------
 * Set a specific camera mode, usually used to set the camera resolution.
 * This should strictly only be used with v4l2 drivers from nvidia.
 *
 * dev: the video device, example: /dev/video0
 * mode: camera mode number
 * 
 * returns: RTN_SUCCESS | RTN_ERROR
 */
int8_t set_camera_mode(const char *dev, const int64_t mode)
{
    struct v4l2_ext_controls get_ctrls;
    struct v4l2_ext_control get_ctrl;

    fd = open(dev, O_RDWR | O_NONBLOCK, 0);
    if(fd == -1){
        // dev could not be opened
        return RTN_ERROR;
    }

    get_sensor_mode_value(&get_ctrls, &get_ctrl);
    set_sensor_mode_value(&get_ctrls, mode);

    close(fd);

    return RTN_SUCCESS;
}

/*
static void setup_camera()
{
    // boundaries
    struct v4l2_query_ext_ctrl query_ctrl;

    // get/set
    struct v4l2_ext_controls get_ctrls;
    struct v4l2_ext_control get_ctrl;

    fd = open(DEVICE, O_RDWR | O_NONBLOCK, 0);

    get_sensor_mode_boundaries(&query_ctrl);
    get_sensor_mode_value(&get_ctrls, &get_ctrl);

    set_sensor_mode_value(&get_ctrls, camera_mode);
    get_sensor_mode_value(&get_ctrls, &get_ctrl);

    close(fd);
}


int main(int argc, char *argv[])
{
    setup_camera();

    return 0;
}

*/