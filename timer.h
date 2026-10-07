// timer.h
//
// Monotonic nanosecond timer.
//
// "Monotonic" means the clock never jumps backwards and ignores changes to the
// wall-clock (NTP adjustments, daylight saving, you changing the time). That is what you
// want for measuring durations. Never time code with time() or gettimeofday().
//
//   u64 t0 = timer_ns();
//   do_work();
//   u64 elapsed_ns = timer_ns() - t0;
//
// Only DIFFERENCES between two timer_ns() values are meaningful; the absolute value
// has an arbitrary starting point.
//
// Requires base.h (u64) to be included first.

#pragma once

#if defined(_WIN32)

#include <windows.h>

static inline u64 timer_ns(void) {
    static LARGE_INTEGER freq = { 0 };            // ticks per second, constant after boot
    if (freq.QuadPart == 0) {
        QueryPerformanceFrequency(&freq);
    }

    LARGE_INTEGER ticks;
    QueryPerformanceCounter(&ticks);

    // ticks * 1e9 can overflow u64 after about an hour on a 10 MHz counter,
    // so split into whole seconds and the remainder first.
    u64 seconds   = (u64)ticks.QuadPart / (u64)freq.QuadPart;
    u64 remainder = (u64)ticks.QuadPart % (u64)freq.QuadPart;
    return seconds * 1000000000ULL + remainder * 1000000000ULL / (u64)freq.QuadPart;
}

#else  // macOS and Linux

#include <time.h>

// On macOS, CLOCK_MONOTONIC only ticks in whole MICROSECONDS (found the hard way: every
// benchmark time was an exact multiple of 1000 ns). CLOCK_MONOTONIC_RAW is backed by the
// CPU's high-resolution counter instead. Linux's CLOCK_MONOTONIC is already nanosecond-grained.
#if defined(__APPLE__)
    #define TIMER_CLOCK CLOCK_MONOTONIC_RAW
#else
    #define TIMER_CLOCK CLOCK_MONOTONIC
#endif

static inline u64 timer_ns(void) {
    struct timespec ts;
    clock_gettime(TIMER_CLOCK, &ts);
    return (u64)ts.tv_sec * 1000000000ULL + (u64)ts.tv_nsec;
}

#endif

static inline f64 timer_ns_to_ms(u64 ns) {
    return (f64)ns / 1e6;
}

// The clock's practical resolution: the smallest non-zero gap between two back-to-back
// readings. Never trust a measurement shorter than ~1000x this number.
static inline u64 timer_resolution_ns(void) {
    u64 smallest = ~(u64)0;
    for (u32 i = 0; i < 10000; i++) {
        u64 a = timer_ns();
        u64 b = timer_ns();
        if (b > a && b - a < smallest) { smallest = b - a; }
    }
    return smallest;
}
