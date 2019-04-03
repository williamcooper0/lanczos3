#include "kernel.h"

#include <stddef.h>
#include <stdlib.h>
#include <math.h>

#define M_PI_F 3.14159265358979323846f

float sinc(float z)
{
    if(z == 0.0f)
        return 1;

    const float t = M_PI_F * z;
    return sinf(t) / t;
}

float lanczos(float z, int a)
{
    return (z < -a || z > a) ? 0 : sinc(z) * sinc(z / a);
}

int kernelCreate(int inSide, int outSide, const float **lanczosKernel)
{
    const char a = 3;
    const int factor = inSide / outSide;
    const int halfSize = a * factor;
    const int size = 2 * halfSize;


    float *kernel = malloc((size_t)size * sizeof(float));
    float sum = 0;

    for(int i = 0; i < size; i++) {
        const float k = i < halfSize ? lanczos((halfSize - 0.5f - i) / factor, a) : kernel[size - i - 1];

        kernel[i] = k;
        sum += k;
    }

    for(int i = 0; i < size; i++)
        kernel[i] /= sum;


    *lanczosKernel = kernel;
    return halfSize;
}
