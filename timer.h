//monotonic nanosecond timer
// monotonic means the clock never jumps backward and ignores the changes to the
// wall clock(ntp adjustments, daylight saving, you changing the time)

//u64 t0=timer_ns();
// do_work()
//u64 elpsed_ns=timer_ns()-10;

//only difeerences between two timer_ns() values are meaningful, absolute value
// has an arbitrary starting point

#pragma once

#if defined(_WIN32)

#include "windows.h"

static inline u64 timer_ns(void){
    static LARGE_INTEGER freq={0};
    if(freq.QuadPart==0){
        QueryPerformanceFrequency(&freq);
    }

    LARGE_INTEGER ticks;
    QueryPerformanceFrequency(&ticks);

    //ticks*1e9 can overflow u64 after about an hour on a 10Mhz counter
    //so split into whole seconds and the remainder first

    u64 seconds   = (u64)ticks.QuadPart / (u64)freq.QuadPart;
    u64 remainder = (u64)ticks.QuadPart % (u64)freq.QuadPart;
    return seconds * 1000000000ULL + remainder * 1000000000ULL / (u64)freq.QuadPart;
}

#else //macos and linux

#include <time.h>

static inline u64 timer_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (u64)ts.tv_sec * 1000000000ULL + (u64)ts.tv_nsec;
}
 
#endif
 
static inline f64 timer_ns_to_ms(u64 ns) {
    return (f64)ns / 1e6;
}
