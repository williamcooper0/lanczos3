#include "opencl.h"

#include <stdio.h>
#include <string.h>

void printError()
{
    fprintf(stderr, "Failed to initialize opencl\n");
}

int checkError(const char *function, cl_int code)
{
    const int success = code == CL_SUCCESS;

    if(!success)
        fprintf(stderr, "%s failed, error code %i\n", function, code);

    return success;
}

static cl_device_id _deviceId;
static cl_context _context;
static cl_command_queue _queue;

void releaseContext()
{
    cl_int res = clReleaseContext(_context);
    checkError("clReleaseContext", res);
}

int openCLInitDevice()
{
    cl_uint platformIdCount = 0;
    cl_int res;

    res = clGetPlatformIDs(0, NULL, &platformIdCount);
    if(!checkError("clGetPlatformIDs", res) || !platformIdCount) {
        printError();
        return 0;
    }

    printf("OpenCL platforms: %i\n", platformIdCount);

    cl_platform_id *platformIds = malloc(platformIdCount * sizeof(cl_platform_id));

    res = clGetPlatformIDs(platformIdCount, platformIds, NULL);
    if(!checkError("clGetPlatformIDs", res)) {
        printError();
        return 0;
    }

    cl_platform_id platformId = platformIds[1];
    free(platformIds);

    cl_uint deviceIdCount = 0;

    res = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, 0, NULL, &deviceIdCount);
    if(!checkError("clGetDeviceIDs", res) || !deviceIdCount) {
        printError();
        return 0;
    }

    printf("OpenCL devices: %i\n", deviceIdCount);

    cl_device_id *deviceIds = malloc(deviceIdCount * sizeof(cl_device_id));

    res = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, deviceIdCount, deviceIds, NULL);
    if(!checkError("clGetDeviceIDs", res)) {
        printError();
        return 0;
    }

    cl_device_id _deviceId = deviceIds[0];
    free(deviceIds);


    char name[1024];

    res = clGetDeviceInfo(_deviceId, CL_DEVICE_NAME, sizeof(name), name, NULL);
    if(!checkError("clGetDeviceInfo", res)) {
        printError();
        return 0;
    }

    printf("%s\n", name);


    const cl_context_properties contextProperties[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platformId,
        0
    };

    _context = clCreateContext(contextProperties, 1, &_deviceId, NULL, NULL, &res);
    if(!checkError("clCreateContext", res) || !_context) {
        printError();
        return 0;
    }

    _queue = clCreateCommandQueue(_context, _deviceId, 0, &res);
    if(!checkError("clCreateCommandQueue", res) || !_queue) {
        releaseContext();
        printError();
        return 0;
    }

    size_t maxWorkGroupSize;
    clGetDeviceInfo(_deviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, 0);
    printf("Max work group size: %i\n", (int)maxWorkGroupSize);

    cl_ulong localMemorySize;
    clGetDeviceInfo(_deviceId, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &localMemorySize, 0);
    printf("Local memory size: %lu\n", localMemorySize);

    return 1;
}

static cl_program _program;
static cl_kernel _kernel;
static size_t _globalWorkSize;

void releaseProgram()
{
    cl_int res = clReleaseProgram(_program);
    checkError("clReleaseProgram", res);
}

void clearDevice()
{
    cl_int res = clReleaseCommandQueue(_queue);
    checkError("clReleaseCommandQueue", res);

    releaseContext();
}

int openCLInitKernel(const char *source, const char *function)
{
    const size_t kernelLength = strlen(source);
    cl_int res;

    _program = clCreateProgramWithSource(_context, 1, &source, &kernelLength, &res);
    if(!checkError("clCreateProgramWithSource", res) || !_program) {
        clearDevice();
        printError();
        return 0;
    }

    res = clBuildProgram(_program, 0, NULL, NULL, NULL, NULL);
    if(!checkError("clBuildProgram", res)) {
        char buildLog[16384];
        clGetProgramBuildInfo(_program, _deviceId, CL_PROGRAM_BUILD_STATUS, sizeof(buildLog), buildLog, NULL);
        fprintf(stderr, "Build log:\n%s\n", buildLog);
        releaseProgram();
        clearDevice();
        printError();
        return 0;
    }

    _kernel = clCreateKernel(_program, function, &res);
    if(!checkError("clCreateKernel", res) || !_kernel) {
        releaseProgram();
        printError();
        return 0;
    }

    return 1;
}

cl_mem openCLCreateBuffer(size_t size, const void *pointer)
{
    static int q = 0;
    q++;
    cl_int res;
    cl_mem buffer = clCreateBuffer(_context, pointer ? (CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR) : CL_MEM_READ_ONLY, size, pointer, &res);
    checkError("clCreateBuffer", res);
    return buffer;
}

static size_t _localWorkSize;

void openCLSetLocalWorkSize(int size)
{
    _localWorkSize = (size_t)size;
}

size_t openCLLocalWorkSize()
{
    return _localWorkSize;
}

void openCLSetGlobalWorkSize(size_t size)
{
    _globalWorkSize = size;
}

static cl_uint _index;

void openCLBeginSettingArgs()
{
    _index = 0;
}

void setArg(size_t size, const void *pointer)
{
    cl_int res = clSetKernelArg(_kernel, _index++, size, pointer);
    checkError("clSetKernelArg", res);
}

void openCLSetArgI(const int *pointer)
{
    setArg(sizeof(int), pointer);
}

void openCLSetArgB(const cl_mem *pointer)
{
    setArg(sizeof(cl_mem*), pointer);
}

void openCLSetArgC(const char *pointer)
{
    setArg(sizeof(char), pointer);
}

void openCLSetArgLocal(size_t size)
{
    setArg(size, NULL);
}

void openCLEnqueue()
{
    cl_int res = clEnqueueNDRangeKernel(_queue, _kernel, 1, NULL, &_globalWorkSize, &_localWorkSize, 0, NULL, NULL);
    checkError("clEnqueueNDRangeKernel", res);
}

void openCLReadBuffer(const cl_mem buffer, size_t size, void *pointer)
{
    cl_int res = clEnqueueReadBuffer(_queue, buffer, CL_TRUE, 0, size, pointer, 0, NULL, NULL);
    checkError("clEnqueueReadBuffer", res);
}

void openCLReleaseBuffer(const cl_mem buffer)
{
    cl_int res = clReleaseMemObject(buffer);
    checkError("clReleaseMemObject", res);
}

void clearKernel()
{
    cl_int res = clReleaseKernel(_kernel);
    checkError("clReleaseKernel", res);

    releaseProgram();
}

void openCLClear()
{
    clearKernel();
    clearDevice();
}
