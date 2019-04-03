#ifndef OPENCL_H
#define OPENCL_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

int openCLInitDevice();
int openCLInitKernel(const char *source, const char *function);
cl_mem openCLCreateBuffer(size_t size, const void *pointer);


void openCLSetLocalWorkSize(int size);
size_t openCLLocalWorkSize();

void openCLSetGlobalWorkSize(size_t size);
void openCLBeginSettingArgs();

void openCLSetArgI(const int *pointer);
void openCLSetArgB(const cl_mem *pointer);
void openCLSetArgC(const char *pointer);
void openCLSetArgLocal(size_t size);

void openCLEnqueue();
void openCLReadBuffer(const cl_mem buffer, size_t size, void *pointer);


void openCLReleaseBuffer(const cl_mem buffer);
void openCLClear();

#endif // OPENCL_H
