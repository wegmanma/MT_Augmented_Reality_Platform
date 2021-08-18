#ifndef __FRAMEBUFFER_H__
#define __FRAMEBUFFER_H__

#include <stdint.h>

#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/dict.h>
#include <libavutil/frame.h>

/**
 * Struct for the input and output context
 */
typedef struct {
    AVInputFormat *avInputFormat;
    AVOutputFormat *avOutputFormat;
    AVFormatContext *avInputFormatContext;
    AVFormatContext *avOutputFormatContext;
    AVCodecContext *inputCodecContext;
    AVCodecContext *outputCodecContext;
} fContextStruct;

/**
 * Request ffmpeg to search for all the available video device drivers and then tries to open the desired one
 *
 * @param[in] f_context pointer to a pre-allocated fContextStruct
 * @param[in] input_driver name of the driver that should be used for input, example: v4l2
 *
 * @return RTN_SUCCESS | RTN_ERROR
 */
int8_t setup_av_device(fContextStruct *f_context, const char *input_driver);

/**
 * Allocate the format context, set the flags and open the device for input
 *
 * @param[in] f_context pointer to a pre-allocated fContextStruct
 * @param[in] flags a combination of the listed flags in the file 'avformat.h'
 * @param[in] video_device the device, example: /dev/video0
 * @param[in] options options that will be passed to the device context, it can be null
 *
 * @return RTN_SUCCESS | RTN_ERROR
 */
int8_t setup_av_format_context(fContextStruct *f_context, const int flags, const char *video_device, AVDictionary **options);

/**
 * Prepare ffmpeg with all the required information about the video stream
 *
 * @param[in] avFormatContext pointer to the video stream format context
 * @param[in] options this can be null or set if required
 *
 * @return RTN_SUCCESS | RTN_ERROR
 */
int8_t find_streams_on_media(AVFormatContext *avFormatContext, AVDictionary **options);

/**
 * Perform the required steps to set-up and then open the video input codec
 *
 * @param[in] f_context pointer to a pre-allocated fContextStruct
 * @param[in] stream_number usually is 0, any stream available on the device can be selected to be opened by the codec
 * @param[in] options this can be null or set if required
 *
 * @return RTN_SUCCESS | RTN_ERROR
 */
int8_t setup_input_codec(fContextStruct *f_context, const uint8_t stream_number, AVDictionary **options);

/**
 * Allocates memory for the video packets flow
 *
 * @param[out] pPacket a void null pointer that will be allocated
 *
 * @return RTN_SUCCESS | RTN_ERROR
 */
int8_t allocate_packet_flow(void **pPacket);

/**
 * Allocates memory for the video frames flow
 *
 * @param[out] pFrame a void null pointer that will be allocated
 *
 * @return RTN_SUCCESS | RTN_ERROR
 */
int8_t allocate_frame_flow(void **pFrame);

/**
 * The video input stream buffer will be queried and the most recent frame will be returned.
 * If there are no new frames then the most recent frame will be returned continuously.
 *
 * @param[in] f_context: pointer to a pre-allocated fContextStruct
 * @param[in] pPacket: pre-allocated memory to manage packets
 * @param[in] pFrame: pointer to where the last frame will be located
 *
 * @return -1 if no new frame, 0 if exactly 1 new frame, more than 1 is indicating the amount of frames lost
 */
int32_t get_last_frame(fContextStruct *f_context, AVPacket *pPacket, AVFrame *pFrame);

/**
 * Free every allocated resource for the indicated context
 *
 * @param[in] f_context: pointer to a pre-allocated fContextStruct
 * @param[in] pFrame: pointer to where the last frame will be located
 */
void destroy_context(fContextStruct *f_context, AVFrame *pFrame);

/**
 * Allocate, setup and open a codec instance for encoding (output, frames to packets)
 *
 * @param[in] f_context: pointer to a pre-allocated fContextStruct
 * @param[in] codecName: the name of the codec that should be used
 * @param[in] width: frame width
 * @param[in] height: frame height
 * @param[in] fps: at what fps should the encoding by performed
 * @param[in] AVPixelFormat: the format of the frame being encoded
 * @param[in] options: codec specific options, may be null
 * 
 * @return RTN_SUCCESS | RTN_ERROR
 */
int8_t setup_output_codec(fContextStruct *f_context, const char *codecName, const int width, const int height, const int fps, const enum AVPixelFormat AVPixelFormat, AVDictionary **options);

/**
 * Allocate, setup and open a codec instance for encoding (output, frames to packets)
 *
 * @param[in] f_context: pointer to a pre-allocated fContextStruct
 * @param[in] output_frame: frame in which the output buffers will be stored
 * @param[in] aVPictureType: usually a full frame (AV_PICTURE_TYPE_I) is specified, others may be set according to the enum AVPictureType
 * 
 * @return RTN_SUCCESS | RTN_ERROR
 */
int8_t setup_output_frame(fContextStruct *f_context, AVFrame *output_frame, enum AVPictureType aVPictureType);

/**
 * Send frame to the codec and encode it into a packet
 *
 * @param[in] f_context: pointer to a pre-allocated fContextStruct
 * @param[in] frame: frame to be encoded, send NULL to flush the encoder
 * @param[in] pkt: encoded frame
 * 
 * @return RTN_SUCCESS | RTN_ERROR
 */
int8_t output_codec_encode(fContextStruct *f_context, AVFrame *frame, AVPacket *pkt);

/**
 * Setup the format context for the video output
 *
 * @param[in] f_context: pointer to a pre-allocated fContextStruct
 * @param[in] output_file: the file name that will contain the final video on disk
 * @param[in] options: format context options, may be null
 * 
 * @return RTN_SUCCESS | RTN_ERROR
 */
int8_t setup_output_format_context(fContextStruct *f_context, const char *output_file, AVDictionary **options);

/**
 * Terminate gracefully the output context
 *
 * @param[in] f_context: pointer to a pre-allocated fContextStruct
 * 
 * @return RTN_SUCCESS | RTN_ERROR
 */
int8_t close_output_stream(fContextStruct *f_context);

#endif