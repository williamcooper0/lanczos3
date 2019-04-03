#include "gpu.h"
#include "opencl.h"
#include "kernel.h"
#include "timer.h"

#include <string.h>

void linearDownscaleGpu(int inWidth, int inHeight, const cl_mem *inPixelsBuffer, int outWidth, int outHeight, cl_mem *outPixelsBuffer, char dx, char dy, const cl_mem *lanczosKernelBuffer, int halfKernelSize)
{
    openCLSetGlobalWorkSize((size_t)inWidth * (size_t)inHeight);
    openCLBeginSettingArgs();


    openCLSetArgI(&inWidth);
    openCLSetArgI(&inHeight);

    openCLSetArgB(inPixelsBuffer);
    openCLSetArgLocal(openCLLocalWorkSize() * sizeof(unsigned char));

    openCLSetArgI(&outWidth);
    openCLSetArgI(&outHeight);

    openCLSetArgB(outPixelsBuffer);

    openCLSetArgC(&dx);
    openCLSetArgC(&dy);

    openCLSetArgB(lanczosKernelBuffer);
    const int kernelSize = 2 * halfKernelSize;
    openCLSetArgLocal((size_t)kernelSize * sizeof(float));

    openCLSetArgI(&halfKernelSize);


    openCLEnqueue();
}

void downscaleGpu(int inSide, const unsigned char *inPixels, int outSide, unsigned char *outPixels)
{
    if(!openCLInitDevice())
        return;

    const char *source0 =
"int indexAt(int x, int y, int width)                                        \n"
"{                                                                           \n"
"    return y * width + x;                                                   \n"
"}                                                                           \n"
"                                                                            \n"
"__kernel void linearDownscale(                                              \n"
"      const int inWidth                                                     \n"
"    , const int inHeight                                                    \n"
"                                                                            \n"
"    , __global const unsigned char *inPixels                                \n"
"    , __local unsigned char *lPixels                                        \n"
"                                                                            \n"
"    , const int outWidth                                                    \n"
"    , const int outHeight                                                   \n"
"                                                                            \n"
"    , __global unsigned char *outPixels                                     \n"
"                                                                            \n"
"    , const char dx                                                         \n"
"    , const char dy                                                         \n"
"                                                                            \n"
"    , __global const float *lanczosKernel                                   \n"
"    , __local float *lKernel                                                \n"
"                                                                            \n"
"    , const int halfKernelSize                                              \n"
")                                                                           \n"
"{                                                                           \n"
        ;
    const char *source1 =
"    const int lid = get_local_id(0);                                        \n"
"    const int gid = get_global_id(0);                                       \n"
"    const int localWorkSize = get_local_size(0);                            \n"
"                                                                            \n"
"    const int x = lid * dx + (gid / localWorkSize) * dy;                    \n"
"    const int y = (gid / localWorkSize) * dx + lid * dy;                    \n"
"                                                                            \n"
"                                                                            \n"
"    lPixels[lid] = inPixels[indexAt(lid * dx + x * dy, y * dx + lid * dy,   \n"
"        inWidth)];                                                          \n"
"                                                                            \n"
"    const int kernelSize = 2 * halfKernelSize;                              \n"
"                                                                            \n"
"    if(lid < kernelSize)                                                    \n"
"        lKernel[lid] = lanczosKernel[lid];                                  \n"
"                                                                            \n"
"    barrier(CLK_LOCAL_MEM_FENCE);                                           \n"
"                                                                            \n"
"                                                                            \n"
"    if(lid >= outWidth * dx + outHeight * dy)                               \n"
"        return;                                                             \n"
"                                                                            \n"
"    const float inX = (x + 0.5) * inWidth / outWidth - 0.5;                 \n"
"    const float inY = (y + 0.5) * inHeight / outHeight - 0.5;               \n"
"                                                                            \n"
"    const float coord = inX * dx + inY * dy;                                \n"
"    const int coordFloor = coord;                                           \n"
"                                                                            \n"
"                                                                            \n"
"    float interpolated = 0;                                                 \n"
"                                                                            \n"
"    for(int i = 0; i < kernelSize; i++) {                                   \n"
"        interpolated += lPixels[clamp(coordFloor - kernelSize / 2 + 1 + i,  \n"
"            0, localWorkSize - 1)] * lKernel[i];                            \n"
"    }                                                                       \n"
"                                                                            \n"
"    outPixels[indexAt(x, y, outWidth)] = clamp(interpolated, 0.5f, 255.5f); \n"
"}                                                                           \n"
        ;

    const size_t len0 = strlen(source0);
    char *source = malloc((len0 + strlen(source1) + 1) * sizeof(char));

    strcpy(source, source0);
    strcpy(source + len0, source1);


    if(!openCLInitKernel(source, "linearDownscale"))
        return;

    openCLSetLocalWorkSize(inSide);

    const float *lanczosKernel;
    const int halfKernelSize = kernelCreate(inSide, outSide, &lanczosKernel);
    const int kernelSize = 2 * halfKernelSize;

    const int tmpWidth = inSide;
    const int tmpHeight = outSide;


    timerStart();

    cl_mem inPixelsBuffer = openCLCreateBuffer((size_t)inSide * (size_t)inSide * sizeof(unsigned char), inPixels);
    cl_mem tmpPixelsBuffer = openCLCreateBuffer((size_t)tmpWidth * (size_t)tmpHeight * sizeof(unsigned char), NULL);
    cl_mem outPixelsBuffer = openCLCreateBuffer((size_t)outSide * (size_t)outSide * sizeof(unsigned char), NULL);

    cl_mem lanczosKernelBuffer = openCLCreateBuffer((size_t)kernelSize * sizeof(float), lanczosKernel);

    linearDownscaleGpu(inSide, inSide, &inPixelsBuffer, tmpWidth, tmpHeight, &tmpPixelsBuffer, 0, 1, &lanczosKernelBuffer, halfKernelSize);
    linearDownscaleGpu(tmpWidth, tmpHeight, &tmpPixelsBuffer, outSide, outSide, &outPixelsBuffer, 1, 0, &lanczosKernelBuffer, halfKernelSize);

    openCLReadBuffer(outPixelsBuffer, (size_t)outSide * (size_t)outSide, outPixels);

    timerEnd("gpu");


    openCLReleaseBuffer(lanczosKernelBuffer);
    free(lanczosKernel);

    openCLReleaseBuffer(outPixelsBuffer);
    openCLReleaseBuffer(tmpPixelsBuffer);
    openCLReleaseBuffer(inPixelsBuffer);

    openCLClear();
}
