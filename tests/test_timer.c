// tests/test_timer.c
//
// Before you trust a benchmark, check that the stopwatch itself works.

#include "base.h"
#include "timer.h"

#include "test.h"

#include <time.h>

static void sleep_ms(long ms) {
    struct timespec req = { ms / 1000, (ms % 1000) * 1000000L };
    nanosleep(&req, NULL);
}

static void test_monotonic(void) {
    // 100,000 consecutive readings must never go backwards.
    u64 prev = timer_ns();
    b32 never_backwards = true;
    for (u32 i = 0; i < 100000; i++) {
        u64 now = timer_ns();
        if (now < prev) { never_backwards = false; }
        prev = now;
    }
    CHECK(never_backwards);
}

static void test_measures_sleep(void) {
    u64 t0 = timer_ns();
    sleep_ms(30);
    u64 elapsed = timer_ns() - t0;

    printf("    slept 30 ms, timer says %.3f ms\n", timer_ns_to_ms(elapsed));

    // sleep never returns early (barring signals), and a loaded machine can oversleep,
    // so the upper bound is generous. This catches wrong units (ns vs us vs ms) by a mile.
    CHECK(elapsed >= 29000000ULL);          // at least 29 ms
    CHECK(elapsed <  500000000ULL);         // well under half a second
}

static void test_resolution(void) {
    u64 res = timer_resolution_ns();
    printf("    clock resolution: %llu ns\n", (unsigned long long)res);

    // A clock that ticks in whole microseconds (macOS CLOCK_MONOTONIC) reports exactly
    // 1000 ns here, and every benchmark under ~1 ms would be mostly rounding error.
    // Real high-resolution counters are 1 to ~50 ns (Apple Silicon's is ~42 ns).
    CHECK(res < 500);
}

static void test_ms_conversion(void) {
    CHECK_NEAR(timer_ns_to_ms(1000000ULL), 1.0, 1e-12);
    CHECK_NEAR(timer_ns_to_ms(2500000ULL), 2.5, 1e-12);
    CHECK_NEAR(timer_ns_to_ms(0), 0.0, 1e-12);
}

int main(void) {
    RUN_TEST(test_monotonic);
    RUN_TEST(test_measures_sleep);
    RUN_TEST(test_resolution);
    RUN_TEST(test_ms_conversion);

    return test_summary();
}
