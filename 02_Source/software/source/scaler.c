#include "scaler.h"

#include "common.h"

#include <stdlib.h>

#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>


int8_t init_scaler_context(void ** pContext, const enum AVPixelFormat src_pix_fmt, const enum AVPixelFormat dst_pix_fmt, const int width, const int height)
{
    *pContext = NULL;
    *pContext = (void *) malloc(1 * sizeof(fScalerContextStruct));

    if(!*pContext){
        return RTN_ERROR;
    }

    fScalerContextStruct *context = (fScalerContextStruct *) *pContext;
    context->outputBufferBufsize = av_image_alloc(context->outputBuffer, context->outputBufferLinesize, width, height, dst_pix_fmt, 1);
    context->src_pix_fmt = src_pix_fmt;
    context->dst_pix_fmt = dst_pix_fmt;
    context->width = width;
    context->height = height;
    context->scalerCtx = sws_getContext(width, height, src_pix_fmt,
                                                    width, height, dst_pix_fmt,
                                                    // here we can change the rescaling algorithm and we can add color filters
                                                    SWS_BILINEAR, NULL, NULL, NULL);
    
    if(!context->scalerCtx){
        free(*pContext);
        *pContext = NULL;
        return RTN_ERROR;
    }
    
    return RTN_SUCCESS;
}


void destroy_scaler_context(fScalerContextStruct * pContext)
{
    av_freep(&pContext->outputBuffer[0]);
    
    sws_freeContext(pContext->scalerCtx);

    free(pContext);
    pContext = NULL;
}


void scale(fScalerContextStruct * pContext, uint8_t * pData[], int linesize[])
{
    sws_scale(pContext->scalerCtx, (const uint8_t * const*)pData, linesize, 0, pContext->height, pContext->outputBuffer, pContext->outputBufferLinesize);
}
