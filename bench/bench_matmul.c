// bench/bench_matmul.c
//
// Benchmarks mat_mul (naive, i-k-j loops) for square matrices at several sizes and for all
// four transpose combinations, and optionally compares against a BLAS library
// (Apple Accelerate on macOS) when compiled with -DUSE_BLAS.
//
//   make bench                       builds with -O2 and writes bench/results/matmul_baseline.csv
//   ./build/bench_matmul [max_size]  CSV goes to stdout, progress to stderr (default max 1024)
//
// Method (this is what makes the numbers trustworthy):
//   * one untimed warm-up run first (caches, page tables, CPU clock ramp-up),
//   * each timed sample is a BATCH of back-to-back calls lasting at least ~2 ms, divided by
//     the batch size. A single 3 microsecond call cannot be timed with a clock that ticks every
//     1 microsecond (or even 40 ns) -- a 2 ms batch makes the tick error negligible,
//   * several timed samples, then report the MINIMUM and the MEDIAN.
//     Noise (other processes, thermal throttling) only ever makes a run slower, so the
//     minimum is the best estimate of what the code can do; the median shows how noisy
//     the machine was.
//   * every result is checked against the BLAS answer before it is timed.
//
// GFLOPS: multiplying an m x k by a k x n matrix is 2*m*k*n floating-point operations
// (one multiply + one add per term). flops / nanoseconds = giga-flops per second.

#include "base.h"
#include "arena.h"
#include "prng.h"
#include "matrix.h"
#include "timer.h"

#include "arena.c"
#include "prng.c"
#include "matrix.c"

#include <stdlib.h>

#if defined(USE_BLAS)
    #if defined(__APPLE__)
        #define ACCELERATE_NEW_LAPACK          // use Apple's current CBLAS headers (no deprecation warnings)
        #include <Accelerate/Accelerate.h>
        #define BLAS_NAME "accelerate"
    #else
        #include <cblas.h>
        #define BLAS_NAME "blas"
    #endif
#endif

// Is this build trustworthy for timing?
#if defined(__SANITIZE_ADDRESS__)
    #define BENCH_SANITIZED 1
#elif defined(__has_feature)
    #if __has_feature(address_sanitizer)
        #define BENCH_SANITIZED 1
    #endif
#endif

// ---------------------------------------------------------------------------
// The implementations under test. Same signature so the timing loop is shared.
// ---------------------------------------------------------------------------

typedef void (*matmul_fn)(matrix* out, const matrix* a, const matrix* b, b32 ta, b32 tb);

static void run_naive(matrix* out, const matrix* a, const matrix* b, b32 ta, b32 tb) {
    if (!mat_mul(out, a, b, true, ta, tb)) {
        fprintf(stderr, "mat_mul failed (shape error?)\n");
        exit(1);
    }
}

#if defined(USE_BLAS)
static void run_blas(matrix* out, const matrix* a, const matrix* b, b32 ta, b32 tb) {
    // Row-major C (m x n) = op(A) (m x k) * op(B) (k x n).
    // lda / ldb = the number of columns of the matrix AS STORED (distance between rows).
    int m = (int)(ta ? a->cols : a->rows);
    int k = (int)(ta ? a->rows : a->cols);
    int n = (int)(tb ? b->rows : b->cols);

    cblas_sgemm(CblasRowMajor,
                ta ? CblasTrans : CblasNoTrans,
                tb ? CblasTrans : CblasNoTrans,
                m, n, k,
                1.0f, a->data, (int)a->cols,
                      b->data, (int)b->cols,
                0.0f, out->data, (int)out->cols);   // beta = 0: overwrite C, like zero_out = true
}
#endif

// ---------------------------------------------------------------------------
// Cross-check helper
// ---------------------------------------------------------------------------

#if defined(USE_BLAS)

static f64 max_abs_diff(const matrix* x, const matrix* y) {
    f64 worst = 0.0;
    u64 n = (u64)x->rows * x->cols;
    for (u64 i = 0; i < n; i++) {
        f64 d = fabs((f64)x->data[i] - (f64)y->data[i]);
        if (d > worst) { worst = d; }
    }
    return worst;
}

// A checker that has never been seen failing proves nothing ("max diff 0" on every row
// could mean perfect agreement... or a broken comparison). So: show it can fail.
static void selfcheck_max_abs_diff(mem_arena* arena) {
    mem_arena_temp temp = arena_temp_begin(arena);
    matrix* x = mat_create(arena, 4, 4);
    matrix* y = mat_create(arena, 4, 4);
    mat_fill_rand(x, -1.0f, 1.0f);
    mat_copy(y, x);

    b32 identical_ok = max_abs_diff(x, y) == 0.0;
    y->data[5] += 0.25f;                                  // plant a known difference
    b32 planted_ok = fabs(max_abs_diff(x, y) - 0.25) < 1e-6;

    if (!identical_ok || !planted_ok) {
        fprintf(stderr, "internal error: max_abs_diff is broken\n");
        exit(3);
    }
    arena_temp_end(temp);
}

#endif  // USE_BLAS

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

static int cmp_u64(const void* x, const void* y) {
    u64 a = *(const u64*)x, b = *(const u64*)y;
    return (a > b) - (a < b);
}

typedef struct {
    u64 min_ns;        // per call
    u64 median_ns;     // per call
    u32 inner;         // calls per timed sample (batch size)
} bench_result;

#define BENCH_TARGET_SAMPLE_NS 2000000ULL   // each timed sample should last >= 2 ms

// `reps` must be odd so the median is an actual measurement.
static bench_result bench_run(
    matmul_fn fn, matrix* out, const matrix* a, const matrix* b,
    b32 ta, b32 tb, u32 reps
) {
    u64 times[64];
    if (reps > 63) { reps = 63; }

    fn(out, a, b, ta, tb);                       // warm-up, not recorded

    // Calibrate: how many calls fit in one ~2 ms sample? (Big sizes: 1. Tiny sizes: thousands.)
    u64 c0 = timer_ns();
    fn(out, a, b, ta, tb);
    u64 one_call = timer_ns() - c0;
    if (one_call == 0) { one_call = 1; }

    u64 inner64 = BENCH_TARGET_SAMPLE_NS / one_call;
    if (inner64 < 1) { inner64 = 1; }
    if (inner64 > 1000000) { inner64 = 1000000; }
    u32 inner = (u32)inner64;

    for (u32 r = 0; r < reps; r++) {
        u64 t0 = timer_ns();
        for (u32 i = 0; i < inner; i++) {
            fn(out, a, b, ta, tb);
        }
        times[r] = (timer_ns() - t0) / inner;
    }

    qsort(times, reps, sizeof(u64), cmp_u64);
    return (bench_result){ .min_ns = times[0], .median_ns = times[reps / 2], .inner = inner };
}

static f64 gflops(u32 size, u64 ns) {
    f64 flops = 2.0 * (f64)size * (f64)size * (f64)size;
    return flops / (f64)ns;                      // flops per nanosecond == giga-flops per second
}

static void print_csv_row(const char* impl, u32 size, b32 ta, b32 tb, u32 reps, bench_result r) {
    printf("%s,%u,%d,%d,%u,%u,%.5f,%.5f,%.3f,%.3f\n",
           impl, size, ta ? 1 : 0, tb ? 1 : 0, reps, r.inner,
           timer_ns_to_ms(r.min_ns), timer_ns_to_ms(r.median_ns),
           gflops(size, r.min_ns), gflops(size, r.median_ns));
    fflush(stdout);
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    u32 max_size = 1024;
    if (argc > 1) {
        int requested = atoi(argv[1]);
        if (requested >= 16 && requested <= 4096) { max_size = (u32)requested; }
    }

#if !defined(__OPTIMIZE__)
    fprintf(stderr, "WARNING: built WITHOUT optimization (no -O flag). These numbers are meaningless.\n");
#endif
#if defined(BENCH_SANITIZED)
    fprintf(stderr, "WARNING: built WITH AddressSanitizer. These numbers are meaningless.\n");
#endif
#if !defined(USE_BLAS)
    fprintf(stderr, "note: built without -DUSE_BLAS, so no BLAS comparison and no correctness cross-check.\n");
#endif

    fprintf(stderr, "timer resolution: %llu ns\n", (unsigned long long)timer_resolution_ns());

    // 4 matrices of at most 4096^2 floats (64 MiB each) fit in a 1 GiB reservation.
    mem_arena* arena = arena_create(GiB(1), MiB(64));
    if (!arena) { fprintf(stderr, "arena_create failed\n"); return 1; }

#if defined(USE_BLAS)
    selfcheck_max_abs_diff(arena);
#endif

    printf("impl,size,transpose_a,transpose_b,reps,calls_per_sample,min_ms,median_ms,gflops_best,gflops_median\n");

    u32 sizes[] = { 64, 128, 256, 512, 1024, 2048, 4096 };
    f32 checksum = 0.0f;                         // use the results so nothing can be optimized away

    prng_seed(1234, 5678);

    for (u32 si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++) {
        u32 size = sizes[si];
        if (size > max_size) { break; }

        // Fewer repetitions for big sizes, which take seconds each.
        u32 reps = size <= 256 ? 15 : (size <= 512 ? 7 : 3);

        mem_arena_temp temp = arena_temp_begin(arena);

        matrix* a = mat_create(arena, size, size);
        matrix* b = mat_create(arena, size, size);
        matrix* out = mat_create(arena, size, size);
#if defined(USE_BLAS)
        matrix* out_blas = mat_create(arena, size, size);
        if (!out_blas) { fprintf(stderr, "arena too small\n"); return 1; }
#endif
        if (!a || !b || !out) { fprintf(stderr, "arena too small\n"); return 1; }

        mat_fill_rand(a, -1.0f, 1.0f);
        mat_fill_rand(b, -1.0f, 1.0f);

        for (u32 flags = 0; flags < 4; flags++) {
            b32 ta = (flags & 1) != 0;
            b32 tb = (flags & 2) != 0;
            const char* label = !ta && !tb ? "NN" : (ta && !tb ? "TN" : (!ta && tb ? "NT" : "TT"));

#if defined(USE_BLAS)
            // Cross-check: does our answer match the BLAS answer? A fast wrong answer is worthless.
            run_naive(out, a, b, ta, tb);
            run_blas(out_blas, a, b, ta, tb);
            f64 worst = max_abs_diff(out, out_blas);
            if (worst > 1e-5 * size) {
                fprintf(stderr, "MISMATCH vs BLAS at size %u %s: max abs diff %g\n", size, label, worst);
                return 2;
            }
#endif

            bench_result naive = bench_run(run_naive, out, a, b, ta, tb, reps);
            print_csv_row("naive_ikj", size, ta, tb, reps, naive);
            checksum += mat_sum(out);

#if defined(USE_BLAS)
            bench_result blas = bench_run(run_blas, out_blas, a, b, ta, tb, reps);
            print_csv_row(BLAS_NAME, size, ta, tb, reps, blas);
            checksum += mat_sum(out_blas);

            fprintf(stderr, "size %4u %s | naive %7.2f GFLOPS (median %7.2f) | %s %8.2f GFLOPS | gap %6.1fx | max diff %.2g\n",
                    size, label, gflops(size, naive.min_ns), gflops(size, naive.median_ns),
                    BLAS_NAME, gflops(size, blas.min_ns),
                    (f64)naive.min_ns / (f64)blas.min_ns, worst);
#else
            fprintf(stderr, "size %4u %s | naive %7.2f GFLOPS (median %7.2f)\n",
                    size, label, gflops(size, naive.min_ns), gflops(size, naive.median_ns));
#endif
        }

        arena_temp_end(temp);                    // free all matrices of this size at once
    }

    fprintf(stderr, "checksum (ignore, only prevents dead-code elimination): %g\n", (f64)checksum);

    arena_destroy(arena);
    return 0;
}
