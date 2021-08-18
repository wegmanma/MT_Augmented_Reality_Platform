#include "framebuffer.h"

#include "common.h"

#include <libavdevice/avdevice.h>
#include <libavutil/error.h>


int8_t setup_av_device(fContextStruct *f_context, const char *input_driver)
{
    avdevice_register_all();

    f_context->avInputFormat = av_find_input_format(input_driver);

    if(!f_context->avInputFormat) {
        av_log(0, AV_LOG_ERROR, "Cannot find input format\n");
        return RTN_ERROR;
    }

    return RTN_SUCCESS;
}


int8_t setup_av_format_context(fContextStruct *f_context, const int flags, const char *video_device, AVDictionary **options)
{
    f_context->avInputFormatContext = avformat_alloc_context();

    if(!f_context->avInputFormatContext) {
        av_log(0, AV_LOG_ERROR, "Cannot allocate memory\n");
        return RTN_ERROR;
    }

    if (flags)
    {
        f_context->avInputFormatContext->flags |= flags;
    }
    
    if(avformat_open_input(&f_context->avInputFormatContext, video_device, f_context->avInputFormat, options) < 0){
        av_log(0, AV_LOG_ERROR, "%s cannot be opened\n", video_device);
        return RTN_ERROR;
    }

    return RTN_SUCCESS;
}


int8_t find_streams_on_media(AVFormatContext *avFormatContext, AVDictionary **options)
{   
    if (avformat_find_stream_info(avFormatContext, options) < 0) {
        av_log(0, AV_LOG_ERROR, "Input streams could not be queried\n");
        return RTN_ERROR;
    }
    
    return RTN_SUCCESS;
}


int8_t setup_input_codec(fContextStruct *f_context, const uint8_t stream_number, AVDictionary **options)
{
    AVCodecParameters *pLocalCodecParameters = f_context->avInputFormatContext->streams[stream_number]->codecpar;

    AVCodec * pLocalCodec = avcodec_find_decoder(pLocalCodecParameters->codec_id);
    if(!pLocalCodec) {
        av_log(0, AV_LOG_ERROR, "No codec could be found for the specified input stream\n");
        return RTN_ERROR;
    }

    f_context->inputCodecContext = avcodec_alloc_context3(pLocalCodec);
    if(!f_context->inputCodecContext) {
        av_log(0, AV_LOG_ERROR, "Cannot allocate memory\n");
        return RTN_ERROR;
    }
    
    if(avcodec_parameters_to_context(f_context->inputCodecContext, pLocalCodecParameters) < 0){
        av_log(0, AV_LOG_ERROR, "Codec parameters could noy be set\n");
        return RTN_ERROR;
    }

    if(avcodec_open2(f_context->inputCodecContext, pLocalCodec, options) < 0){
        av_log(0, AV_LOG_ERROR, "Codec seems to exists but it could not be opened\n");
        return RTN_ERROR;
    }

    return RTN_SUCCESS;
}


int8_t allocate_packet_flow(void **pPacket)
{
    AVPacket *temp_pPacket = av_packet_alloc();

    if(!temp_pPacket){
        *pPacket = NULL;
        av_log(0, AV_LOG_ERROR, "Cannot allocate memory\n");
        return RTN_ERROR;
    }

    *pPacket = temp_pPacket;

    return RTN_SUCCESS;
}


int8_t allocate_frame_flow(void **pFrame)
{
    AVFrame *temp_pFrame = av_frame_alloc();

    if(!temp_pFrame){
        *pFrame = NULL;
        av_log(0, AV_LOG_ERROR, "Cannot allocate memory\n");
        return RTN_ERROR;
    }

    *pFrame = temp_pFrame;

    return RTN_SUCCESS;
}


int32_t get_last_frame(fContextStruct *f_context, AVPacket *pPacket, AVFrame *pFrame)
{
    int32_t lost_frames = -1;
    while (av_read_frame(f_context->avInputFormatContext, pPacket) >= 0)
    {
        lost_frames++;
        avcodec_send_packet(f_context->inputCodecContext, pPacket);  // compressed frame
        avcodec_receive_frame(f_context->inputCodecContext, pFrame); // uncompressed frame
        av_packet_unref(pPacket);
    }

    return lost_frames;
}


void destroy_context(fContextStruct *f_context, AVFrame *pFrame)
{
    av_frame_free(&pFrame);
    avcodec_free_context(&f_context->inputCodecContext);
    avcodec_free_context(&f_context->outputCodecContext);
    avformat_close_input(&f_context->avInputFormatContext);
    avformat_close_input(&f_context->avOutputFormatContext);
    avformat_free_context(f_context->avInputFormatContext);
    avformat_free_context(f_context->avOutputFormatContext);
}


int8_t setup_output_codec(fContextStruct *f_context, const char *codecName, const int width, const int height, const int fps, const enum AVPixelFormat AVPixelFormat, AVDictionary **options)
{
    AVCodec *codec = avcodec_find_encoder_by_name(codecName);
    if (!codec)
    {
        av_log(0, AV_LOG_ERROR, "No codec could be found\n");
        return RTN_ERROR;
    }

    f_context->outputCodecContext = avcodec_alloc_context3(codec);
    if (!f_context->outputCodecContext)
    {
        av_log(0, AV_LOG_ERROR, "Cannot allocate memory\n");
        return RTN_ERROR;
    }
    
    f_context->outputCodecContext->bit_rate = 400000; // sample bit rate
    f_context->outputCodecContext->width = width;
    f_context->outputCodecContext->height = height;
    f_context->outputCodecContext->time_base = (AVRational){1, fps};
    f_context->outputCodecContext->framerate = (AVRational){fps, 1};
    f_context->outputCodecContext->gop_size = 10;
    f_context->outputCodecContext->max_b_frames = 1;
    f_context->outputCodecContext->pix_fmt = AVPixelFormat;

    if(avcodec_open2(f_context->outputCodecContext, codec, options) < 0){
        av_log(0, AV_LOG_ERROR, "Codec seems to exists but it could not be opened\n");
        return RTN_ERROR;
    }
    
    return RTN_SUCCESS;
}


int8_t setup_output_frame(fContextStruct *f_context, AVFrame *output_frame, enum AVPictureType aVPictureType)
{
    output_frame->format = f_context->outputCodecContext->pix_fmt;
    output_frame->width  = f_context->outputCodecContext->width;
    output_frame->height = f_context->outputCodecContext->height;
    output_frame->pict_type = AV_PICTURE_TYPE_I;
    if (av_frame_get_buffer(output_frame, 0) < 0)
    {
        av_log(0, AV_LOG_ERROR, "Output frame buffer cannot be allocated\n");
        return RTN_ERROR;
    }
    
    return RTN_SUCCESS;
}


int8_t output_codec_encode(fContextStruct *f_context, AVFrame *frame, AVPacket *pkt)
{
    if (avcodec_send_frame(f_context->outputCodecContext, frame) < 0)
    {
        av_log(0, AV_LOG_ERROR, "Cannot decode frame\n");
        return RTN_ERROR;
    }

    int ret = 0;
    while (ret >= 0) {
        ret = avcodec_receive_packet(f_context->outputCodecContext, pkt);
        if (ret == AVERROR(EAGAIN))
            return RTN_SUCCESS;
        if (ret == AVERROR_EOF)
            return RTN_EOF;
        else if (ret < 0) {
            av_log(0, AV_LOG_ERROR, "Error during encoding\n");
            return RTN_ERROR;
        }

        if(f_context->avOutputFormatContext)
        {
            av_interleaved_write_frame(f_context->avOutputFormatContext, pkt);
        } else {
            return PKT_READY_NOT_CONSUMED; // signal that packet is ready but was not consumed
        }
    }

    if (f_context->avOutputFormatContext && !frame)
    {
        av_interleaved_write_frame(f_context->avOutputFormatContext, NULL);
    }

    return RTN_SUCCESS;
}


int8_t setup_output_format_context(fContextStruct *f_context, const char *output_file, AVDictionary **options)
{
    if (avformat_alloc_output_context2(&f_context->avOutputFormatContext, NULL, NULL, output_file) < 0)
    {
        av_log(0, AV_LOG_ERROR, "Output context could not be allocated\n");
        return RTN_ERROR;
    }

    AVStream *out_stream = avformat_new_stream(f_context->avOutputFormatContext, f_context->outputCodecContext->codec);
    if (!out_stream)
    {
        av_log(0, AV_LOG_ERROR, "Output stream could not be allocated\n");
        return RTN_ERROR;
    }
    
    f_context->avOutputFormat = f_context->avOutputFormatContext->oformat;
    
    AVCodecParameters *par = avcodec_parameters_alloc();
    if (!par)
    {
        av_log(0, AV_LOG_ERROR, "Could not allocate temp avcodec_params\n");
        return RTN_ERROR;
    }
    
    if(avcodec_parameters_from_context(par, f_context->outputCodecContext) < 0)
    {
        av_log(0, AV_LOG_ERROR, "Could not fetch codec parameters from context\n");
        return RTN_ERROR;
    }

    if(avcodec_parameters_copy(out_stream->codecpar, par) < 0)
    {
        av_log(0, AV_LOG_ERROR, "Could not copy codec paramenters\n");
        return RTN_ERROR;
    }
    avcodec_parameters_free(&par);

    if (f_context->avOutputFormat->flags & AVFMT_GLOBALHEADER)
    {
        f_context->avOutputFormat->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    if (avio_open(&f_context->avOutputFormatContext->pb, output_file, AVIO_FLAG_WRITE) < 0)
    {
        av_log(0, AV_LOG_ERROR, "Could open output file\n");
        return RTN_ERROR;
    }
    
    // dump av format informations
    av_dump_format(f_context->avOutputFormatContext, 0, output_file, 1);

    if (avformat_write_header(f_context->avOutputFormatContext, options) < 0)
    {
        av_log(0, AV_LOG_ERROR, "Output file header could not be written\n");
        return RTN_ERROR;
    }
    
    return RTN_SUCCESS;
}


int8_t close_output_stream(fContextStruct *f_context)
{
    if (av_write_trailer(f_context->avOutputFormatContext) != 0)
    {
        av_log(0, AV_LOG_ERROR, "Trailer could not be written\n");
        return RTN_ERROR;
    }
    
    return RTN_SUCCESS;
}