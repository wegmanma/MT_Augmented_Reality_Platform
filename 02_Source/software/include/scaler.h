#ifndef __SCALER_H__
#define __SCALER_H__

#include <stdint.h>
#include <stddef.h>

#include <libavutil/pixfmt.h>

/**
 * Struct for the scaler context
 */
typedef struct {
    int width;
    int height;
    enum AVPixelFormat src_pix_fmt;
    enum AVPixelFormat dst_pix_fmt;
    uint8_t *outputBuffer[4];
    int outputBufferLinesize[4];
    int outputBufferBufsize;
    struct SwsContext *scalerCtx;
} fScalerContextStruct;

/**
 * Allocate and initialize the FFmpeg scaler context
 *
 * @param[out] pContext non allocated fScalerContextStruct pointer, it will be allocated by the function
 * @param[in] src_pix_fmt input pixel format
 * @param[in] dst_pix_fmt putput pixel format
 * @param[in] width image width
 * @param[in] height image height
 * 
 * @return RTN_SUCCESS | RTN_ERROR
 */
int8_t init_scaler_context(void ** pContext, enum AVPixelFormat src_pix_fmt, enum AVPixelFormat dst_pix_fmt, int width, int height);

/**
 * The provided fScalerContextStruct will be properly destroyed and freed
 *
 * @param[in] pContext pointer the fScalerContextStruct
 * 
 */
void destroy_scaler_context(fScalerContextStruct * pContext);

/**
 * Perform the pixel manipulation on the specified data
 *
 * @param[in] pContext pointer the fScalerContextStruct
 * @param[in] pData video frame data, usually located in a video frame structure
 * @param[in] linesize the size of each buffer in pData, usually located in a video frame structure
 * 
 */
void scale(fScalerContextStruct * pContext, uint8_t * pData[], int linesize[]);

#endif