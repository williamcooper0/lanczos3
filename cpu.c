#include "cpu.h"
#include "kernel.h"
#include "timer.h"

#include <stdlib.h>

int indexAt(int x, int y, int width)
{
    return y * width + x;
}

#define CLAMP(Z, MIN_VALUE, MAX_VALUE) ((Z) < MIN_VALUE ? MIN_VALUE : ((Z) > MAX_VALUE ? MAX_VALUE : (Z)))

int clamp(int z, int maxValue)
{
    return CLAMP(z, 0, maxValue - 1);
}

void linearDownscaleCpu(int inWidth, int inHeight, const unsigned char *inPixels, int outWidth, int outHeight, unsigned char *outPixels, char dx, char dy, const float *lanczosKernel, int halfKernelSize)
{
    for(int y = 0; y < outHeight; y++) {
        const float inY = (y + 0.5f) * inHeight / outHeight - 0.5f;

        for(int x = 0; x < outWidth; x++) {
            const float inX = (x + 0.5f) * inWidth / outWidth - 0.5f;

            const float coord = inX * dx + inY * dy;
            const int coordFloor = (int)coord;


            float interpolated = 0;

            for(int i = 0; i < 2 * halfKernelSize; i++) {
                const int index = coordFloor - halfKernelSize + 1 + i;
                interpolated += inPixels[indexAt(clamp(index, inWidth) * dx + x * dy, y * dx + clamp(index, inHeight) * dy, inWidth)] * lanczosKernel[i];
            }

            outPixels[indexAt(x, y, outWidth)] = (unsigned char)CLAMP(interpolated, 0.5f, 255.5f);
        }
    }
}

void downscaleCpu(int inSide, const unsigned char *inPixels, int outSide, unsigned char *outPixels)
{
    const int tmpWidth = inSide;
    const int tmpHeight = outSide;

    unsigned char *tmpPixels = malloc((size_t)tmpWidth * (size_t)tmpHeight * sizeof(unsigned char));

    const float *lanczosKernel;
    const int halfKernelSize = kernelCreate(inSide, outSide, &lanczosKernel);


    timerStart();

    linearDownscaleCpu(inSide, inSide, inPixels, tmpWidth, tmpHeight, tmpPixels, 0, 1, lanczosKernel, halfKernelSize);
    linearDownscaleCpu(tmpWidth, tmpHeight, tmpPixels, outSide, outSide, outPixels, 1, 0, lanczosKernel, halfKernelSize);

    timerEnd("cpu");


    free(lanczosKernel);
    free(tmpPixels);
}
