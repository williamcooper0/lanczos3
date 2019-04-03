extern "C" {
#include "gpu.h"
#include "cpu.h"
}

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

void show(const char *windowName, const Mat &image)
{
    namedWindow(windowName);
    imshow(windowName, image);
}

typedef void (*resize_f)(int, const unsigned char*, int, unsigned char*);

void resize(resize_f function, const Mat &in, int outSide, const char *windowName)
{
    Mat out(outSide, outSide, in.type());
    function(in.rows, in.ptr(), outSide, out.ptr());
    show(windowName, out);
}

int main()
{
    Mat in;
    cvtColor(imread("/home/q/road-sign-1024x1024.png"), in, CV_BGR2GRAY);
    show("in", in);

    const int n = 1;
    const int outSide = in.rows / 2 / n;

    resize(downscaleGpu, in, outSide, "out gpu");
    resize(downscaleCpu, in, outSide, "out cpu");

    for(;;) {
        if(waitKey(1) == 27)
            return 0;
    }
}
