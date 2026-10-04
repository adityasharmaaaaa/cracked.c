#include "base.h"
#include "prng.h"
#include "prng.c"

#include "test.h"

static void test_reference_vector(void) {
    prng_state rng;
    prng_seed_r(&rng, 42u, 54u);

    u32 expected[] = {
        0xa15c02b7u, 0x7b47f409u, 0xba1d3330u,
        0x83d2f293u, 0xbfa4784bu, 0xcbed606eu,
    };

    for (u32 i = 0; i < sizeof(expected) / sizeof(expected[0]); i++) {
        u32 got = prng_rand_r(&rng);
        if (got != expected[i]) {
            printf("    value %u: expected 0x%08x, got 0x%08x\n", i, expected[i], got);
        }
        CHECK(got == expected[i]);
    }
}

static void test_determinism(void) {
    prng_state a, b, c;
    prng_seed_r(&a, 123, 1);
    prng_seed_r(&b, 123, 1);   // same seed as a
    prng_seed_r(&c, 124, 1);   // different seed

    b32 a_equals_b = true;
    b32 a_differs_c = false;
    for (u32 i = 0; i < 1000; i++) {
        u32 x = prng_rand_r(&a);
        u32 y = prng_rand_r(&b);
        u32 z = prng_rand_r(&c);
        if (x != y) a_equals_b = false;
        if (x != z) a_differs_c = true;
    }
    CHECK(a_equals_b);     // same seed -> identical stream (reproducible experiments)
    CHECK(a_differs_c);    // different seed -> different stream

    // The global-state API: re-seeding restarts the stream.
    prng_seed(7, 7);
    u32 first = prng_rand();
    prng_seed(7, 7);
    CHECK(prng_rand() == first);
}

static void test_f32_range_extremes(void) {
    CHECK(prng_f32_from_u32(0) == 0.0f);                 // 0.0 is allowed
    CHECK(prng_f32_from_u32(UINT32_MAX) < 1.0f);         // 1.0 must be impossible
    CHECK(prng_f32_from_u32(0xFFFFFF00u) < 1.0f);        // the largest 24-bit pattern
    CHECK(prng_f32_from_u32(1u << 31) == 0.5f);          // top bit alone is exactly 0.5

    // Monotonic: more bits never gives a smaller number.
    CHECK(prng_f32_from_u32(1000000u) <= prng_f32_from_u32(2000000u));
}

static void test_randf_statistics(void) {
    prng_state rng;
    prng_seed_r(&rng, 42, 54);

    u32 n = 1000000;
    f32 lo = 1.0f, hi = 0.0f;
    f64 sum = 0.0;
    u32 buckets[10] = { 0 };

    for (u32 i = 0; i < n; i++) {
        f32 x = prng_randf_r(&rng);
        CHECK(x >= 0.0f && x < 1.0f);    // every single sample in [0, 1)
        if (x < lo) lo = x;
        if (x > hi) hi = x;
        sum += x;
        buckets[(u32)(x * 10.0f)]++;
    }

    // Mean of U[0,1) is 0.5, standard error of the mean over 1M samples is ~0.0003.
    f64 mean = sum / n;
    CHECK(mean > 0.495 && mean < 0.505);

    // Spread out over the whole range, not stuck in a corner.
    CHECK(lo < 0.001f);
    CHECK(hi > 0.999f);

    // Each tenth of the range should get ~10% of the samples (100000 +- a few hundred).
    for (u32 b = 0; b < 10; b++) {
        CHECK(buckets[b] > 98000 && buckets[b] < 102000);
    }
}

int main(void) {
    RUN_TEST(test_reference_vector);
    RUN_TEST(test_determinism);
    RUN_TEST(test_f32_range_extremes);
    RUN_TEST(test_randf_statistics);

    return test_summary();
}
