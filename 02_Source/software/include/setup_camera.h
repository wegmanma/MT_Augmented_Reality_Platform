#pragma once
#include <sys/types.h>

/**
 * Set a video input mode for a v4l2 camera
 * 
 * @param[in] dev the device, ex.: /dev/video0
 * @param[in] mode the mode, ex.: 4
 * 
 * @returns RTN_SUCCESS | RTN_ERROR
 * 
 */
int8_t set_camera_mode(const char *dev, const int64_t mode);

