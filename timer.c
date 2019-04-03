#include "timer.h"

#include <stdio.h>
#include <sys/time.h>

static long _time;

long time()
{
    struct timeval timecheck;
    gettimeofday(&timecheck, NULL);
    return (long)timecheck.tv_sec * 1000 + (long)timecheck.tv_usec / 1000;
}

void timerStart()
{
    struct timeval timecheck;
    gettimeofday(&timecheck, NULL);
    _time = time();
}

void timerEnd(const char *device)
{
    struct timeval timecheck;
    gettimeofday(&timecheck, NULL);
    printf("%s: %ld\n", device, time() - _time);
}
