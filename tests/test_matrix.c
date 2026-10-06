#include "base.h"
#include "arena.h"
#include "prng.h"
#include "matrix.h"

#include "arena.c"
#include "prng.c"
#include "matrix.c"

#include "test.h"

// Copies `count` values into a matrix so tests can write matrices as literals.
static void set(matrix* m, const f32* values, u32 count) {
    for (u32 i = 0; i < count; i++) { m->data[i] = values[i]; }
}

static b32 equals(const matrix* m, const f32* values, u32 count) {
    for (u32 i = 0; i < count; i++) {
        if (m->data[i] != values[i]) { return false; }
    }
    return true;
}

// ---------------------------------------------------------------------------
// mat_create
// ---------------------------------------------------------------------------
static void test_create(void) {
    mem_arena* a = arena_create(MiB(1), KiB(64));
    CHECK(a != NULL);
    if (!a) { return; }

    // Dirty the memory first, so "zeroed" proves something (fresh pages are zero anyway).
    u8* dirty = (u8*)arena_push(a, 1024, false);
    memset(dirty, 0xFF, 1024);
    arena_pop(a, 1024);

    u64 pos_before = a->pos;
    matrix* m = mat_create(a, 3, 4);
    CHECK(m != NULL);
    if (!m) { return; }

    CHECK(m->rows == 3 && m->cols == 4);
    CHECK(m->data != NULL);
    CHECK(((uintptr_t)m % ARENA_ALIGN) == 0);
    CHECK(((uintptr_t)m->data % ARENA_ALIGN) == 0);

    b32 zeroed = true;
    for (u32 i = 0; i < 12; i++) { if (m->data[i] != 0.0f) { zeroed = false; } }
    CHECK(zeroed);

    // Exactly one struct + 12 floats were taken from the arena (your exercise-4 prediction).
    CHECK(a->pos == pos_before + sizeof(matrix) + 12 * sizeof(f32));

    // Last element is writable (ASan would catch an undersized allocation here).
    m->data[11] = 1.0f;

    arena_destroy(a);
}

static void test_create_failures(void) {
    mem_arena* a = arena_create(MiB(1), KiB(64));
    CHECK(a != NULL);
    if (!a) { return; }

    u64 pos_before = a->pos;

    // Zero-sized matrices are rejected.
    CHECK(mat_create(a, 0, 5) == NULL);
    CHECK(mat_create(a, 5, 0) == NULL);

    // Dimensions whose byte size overflows u64 must be rejected, not wrapped.
    CHECK(mat_create(a, UINT32_MAX, UINT32_MAX) == NULL);
    CHECK(a->pos == pos_before);

    // The nasty case: 2^31 * 2^31 elements * 4 bytes = 2^64, which wraps to exactly 0 bytes.
    // Without the guard, arena_push(0) "succeeds" and we'd return a matrix claiming
    // 2^62 elements backed by zero bytes. (UINT32_MAX x UINT32_MAX wraps to a HUGE size,
    // which the arena rejects by accident, so it does not test the guard.)
    CHECK(mat_create(a, 1u << 31, 1u << 31) == NULL);
    CHECK(a->pos == pos_before);

    // Too big for the arena.
    CHECK(mat_create(a, 1000, 1000) == NULL);   // 4 MB > 1 MiB
    CHECK(a->pos == pos_before);

    arena_destroy(a);

    // Rollback: the matrix STRUCT fits, but the DATA does not. The struct push must be undone.
    a = arena_create(plat_get_pagesize(), plat_get_pagesize());
    CHECK(a != NULL);
    if (!a) { return; }

    pos_before = a->pos;
    u32 cols = (u32)(a->reserve_size / sizeof(f32));   // data alone fills the whole arena
    CHECK(mat_create(a, 1, cols) == NULL);
    CHECK(a->pos == pos_before);

    arena_destroy(a);
}

// ---------------------------------------------------------------------------
// clear / fill / scale
// ---------------------------------------------------------------------------
static void test_clear_fill_scale(void) {
    mem_arena* a = arena_create(MiB(1), KiB(64));
    if (!a) { CHECK(a != NULL); return; }
    matrix* m = mat_create(a, 2, 3);

    mat_fill(m, 2.5f);
    f32 all_2_5[] = { 2.5f, 2.5f, 2.5f, 2.5f, 2.5f, 2.5f };
    CHECK(equals(m, all_2_5, 6));

    mat_scale(m, 4.0f);
    f32 all_10[] = { 10, 10, 10, 10, 10, 10 };
    CHECK(equals(m, all_10, 6));

    mat_scale(m, -0.5f);
    f32 all_m5[] = { -5, -5, -5, -5, -5, -5 };
    CHECK(equals(m, all_m5, 6));

    mat_clear(m);
    f32 zeros[] = { 0, 0, 0, 0, 0, 0 };
    CHECK(equals(m, zeros, 6));

    // fill must touch the LAST element too (off-by-one guard).
    mat_fill(m, 7.0f);
    CHECK(m->data[5] == 7.0f);

    arena_destroy(a);
}

// ---------------------------------------------------------------------------
// sum
// ---------------------------------------------------------------------------
static void test_sum(void) {
    mem_arena* a = arena_create(MiB(16), MiB(1));
    if (!a) { CHECK(a != NULL); return; }

    matrix* small = mat_create(a, 2, 3);
    f32 vals[] = { 1, 2, 3, 4, 5, 6 };
    set(small, vals, 6);
    CHECK_NEAR(mat_sum(small), 21.0, 1e-6);

    f32 mixed[] = { -1, 1, -2, 2, 10, -10 };
    set(small, mixed, 6);
    CHECK_NEAR(mat_sum(small), 0.0, 1e-6);

    // Precision: 1,000,000 copies of 0.1f. The exact answer is ~100000.
    matrix* big = mat_create(a, 1000, 1000);
    mat_fill(big, 0.1f);

    f32 naive = 0.0f;                             // what you'd get with an f32 accumulator
    for (u32 i = 0; i < 1000000; i++) { naive += big->data[i]; }
    f32 ours = mat_sum(big);
    printf("    sum of 1e6 x 0.1f: naive f32 accumulator = %.2f, mat_sum (f64 accumulator) = %.2f\n",
           naive, ours);

    CHECK_NEAR(ours, 100000.0, 1.0);

    arena_destroy(a);
}

// ---------------------------------------------------------------------------
// argmax
// ---------------------------------------------------------------------------
static void test_argmax(void) {
    mem_arena* a = arena_create(MiB(1), KiB(64));
    if (!a) { CHECK(a != NULL); return; }
    matrix* m = mat_create(a, 1, 5);

    f32 mid[] = { 3, 9, 2, 1, 4 };
    set(m, mid, 5);
    CHECK(mat_argmax(m) == 1);

    f32 first[] = { 9, 1, 2, 3, 4 };
    set(m, first, 5);
    CHECK(mat_argmax(m) == 0);

    f32 last[] = { 1, 2, 3, 4, 9 };
    set(m, last, 5);
    CHECK(mat_argmax(m) == 4);                    // the last element must be examined

    f32 tie[] = { 3, 9, 2, 9, 1 };
    set(m, tie, 5);
    CHECK(mat_argmax(m) == 1);                    // ties: first maximum wins

    f32 negative[] = { -5, -2, -9, -3, -7 };
    set(m, negative, 5);
    CHECK(mat_argmax(m) == 1);                    // all-negative: must not "start from 0.0"

    // Works on a 2D matrix: the index is FLAT (row-major).
    matrix* grid = mat_create(a, 2, 3);
    f32 g[] = { 1, 2, 3,
                4, 9, 5 };
    set(grid, g, 6);
    CHECK(mat_argmax(grid) == 4);                 // row 1, col 1 -> 1 * 3 + 1

    arena_destroy(a);
}

// ---------------------------------------------------------------------------
// copy
// ---------------------------------------------------------------------------
static void test_copy(void) {
    mem_arena* a = arena_create(MiB(1), KiB(64));
    if (!a) { CHECK(a != NULL); return; }

    matrix* src = mat_create(a, 2, 3);
    matrix* dst = mat_create(a, 2, 3);
    f32 vals[] = { 1, 2, 3, 4, 5, 6 };
    set(src, vals, 6);

    CHECK(mat_copy(dst, src));
    CHECK(equals(dst, vals, 6));

    // A real copy: changing the source afterwards must not change the destination.
    src->data[0] = 99.0f;
    CHECK(dst->data[0] == 1.0f);

    // Copy onto itself is harmless.
    CHECK(mat_copy(src, src));
    CHECK(src->data[0] == 99.0f);

    // Shape mismatch -> false, and dst untouched. 2x3 vs 3x2 has the same element count!
    matrix* wrong = mat_create(a, 3, 2);
    mat_fill(wrong, -1.0f);
    CHECK(!mat_copy(wrong, src));
    CHECK(wrong->data[0] == -1.0f && wrong->data[5] == -1.0f);

    matrix* bigger = mat_create(a, 2, 4);
    CHECK(!mat_copy(bigger, src));

    arena_destroy(a);
}

// ---------------------------------------------------------------------------
// add / sub
// ---------------------------------------------------------------------------
static void test_add_sub(void) {
    mem_arena* a = arena_create(MiB(1), KiB(64));
    if (!a) { CHECK(a != NULL); return; }

    matrix* x = mat_create(a, 2, 3);
    matrix* y = mat_create(a, 2, 3);
    matrix* out = mat_create(a, 2, 3);

    f32 xv[] = { 1, 2, 3, 4, 5, 6 };
    f32 yv[] = { 10, 20, 30, 40, 50, 60 };
    set(x, xv, 6);
    set(y, yv, 6);

    CHECK(mat_add(out, x, y));
    f32 sum_expected[] = { 11, 22, 33, 44, 55, 66 };
    CHECK(equals(out, sum_expected, 6));

    CHECK(mat_sub(out, y, x));
    f32 diff_expected[] = { 9, 18, 27, 36, 45, 54 };
    CHECK(equals(out, diff_expected, 6));

    // Subtraction is not commutative: x - y must be the negation of y - x.
    CHECK(mat_sub(out, x, y));
    f32 neg_expected[] = { -9, -18, -27, -36, -45, -54 };
    CHECK(equals(out, neg_expected, 6));

    // In place: out may alias an input.
    CHECK(mat_add(x, x, y));
    CHECK(equals(x, sum_expected, 6));

    // Shape mismatches -> false, and out untouched.
    matrix* transposed_shape = mat_create(a, 3, 2);   // same element count as 2x3!
    matrix* untouched = mat_create(a, 2, 3);
    mat_fill(untouched, -7.0f);
    f32 sentinel[] = { -7, -7, -7, -7, -7, -7 };

    CHECK(!mat_add(untouched, y, transposed_shape));
    CHECK(!mat_add(untouched, transposed_shape, y));
    CHECK(!mat_sub(untouched, y, transposed_shape));
    CHECK(equals(untouched, sentinel, 6));

    matrix* wrong_out = mat_create(a, 3, 2);
    mat_fill(wrong_out, -7.0f);
    CHECK(!mat_add(wrong_out, x, y));                 // output has the wrong shape
    CHECK(wrong_out->data[0] == -7.0f);

    arena_destroy(a);
}

// ---------------------------------------------------------------------------
// fill_rand
// ---------------------------------------------------------------------------
static void test_fill_rand(void) {
    mem_arena* a = arena_create(MiB(16), MiB(1));
    if (!a) { CHECK(a != NULL); return; }

    matrix* m = mat_create(a, 500, 500);              // 250,000 samples

    // The bound is inclusive of `upper` on purpose: lower + r * range can round up to
    // `upper` in f32 when the range is small relative to the numbers involved.
    f32 ranges[][2] = { { 0.0f, 1.0f }, { -1.0f, 1.0f }, { 5.0f, 5.5f }, { -100.0f, -50.0f } };

    for (u32 r = 0; r < 4; r++) {
        f32 lo = ranges[r][0], hi = ranges[r][1];
        prng_seed(42, 54);
        mat_fill_rand(m, lo, hi);

        b32 in_range = true;
        f32 min = m->data[0], max = m->data[0];
        for (u32 i = 0; i < 250000; i++) {
            f32 x = m->data[i];
            if (x < lo || x > hi) { in_range = false; }
            if (x < min) min = x;
            if (x > max) max = x;
        }
        CHECK(in_range);

        // Uniform in [lo, hi] -> the mean sits in the middle, and the whole range is used.
        f32 mid = (lo + hi) / 2.0f;
        f32 width = hi - lo;
        CHECK_NEAR(mat_sum(m) / 250000.0f, mid, width * 0.01);
        CHECK(min < lo + width * 0.01f);
        CHECK(max > hi - width * 0.01f);
    }

    // Reproducible: same seed -> same matrix.
    matrix* m2 = mat_create(a, 500, 500);
    prng_seed(7, 7);
    mat_fill_rand(m, 0.0f, 1.0f);
    prng_seed(7, 7);
    mat_fill_rand(m2, 0.0f, 1.0f);
    CHECK(memcmp(m->data, m2->data, 250000 * sizeof(f32)) == 0);

    arena_destroy(a);
}

static void test_matmul_known(void) {
    mem_arena* a = arena_create(MiB(1), KiB(64));
    if (!a) { CHECK(a != NULL); return; }
 
    // The 2x2 example from the Thursday exercise.
    matrix* A = mat_create(a, 2, 2);
    matrix* B = mat_create(a, 2, 2);
    matrix* C = mat_create(a, 2, 2);
    f32 av[] = { 1, 2, 3, 4 };
    f32 bv[] = { 5, 6, 7, 8 };
    set(A, av, 4);
    set(B, bv, 4);
 
    f32 ab[]   = { 19, 22, 43, 50 };   // A  * B
    f32 atb[]  = { 26, 30, 38, 44 };   // A^T * B
    f32 abt[]  = { 17, 23, 39, 53 };   // A  * B^T
    f32 atbt[] = { 23, 31, 34, 46 };   // A^T * B^T
 
    CHECK(mat_mul(C, A, B, true, false, false));  CHECK(equals(C, ab, 4));
    CHECK(mat_mul(C, A, B, true, true,  false));  CHECK(equals(C, atb, 4));
    CHECK(mat_mul(C, A, B, true, false, true));   CHECK(equals(C, abt, 4));
    CHECK(mat_mul(C, A, B, true, true,  true));   CHECK(equals(C, atbt, 4));
 
    // Identity: A * I = A.
    matrix* I = mat_create(a, 2, 2);
    f32 iv[] = { 1, 0, 0, 1 };
    set(I, iv, 4);
    CHECK(mat_mul(C, A, I, true, false, false));
    CHECK(equals(C, av, 4));
 
    // Non-square: (2x3) * (3x2) = (2x2). The classic textbook example.
    matrix* P = mat_create(a, 2, 3);
    matrix* Q = mat_create(a, 3, 2);
    matrix* R = mat_create(a, 2, 2);
    f32 pv[] = { 1, 2, 3,
                 4, 5, 6 };
    f32 qv[] = { 7,  8,
                 9,  10,
                 11, 12 };
    f32 pq[] = { 58, 64, 139, 154 };
    set(P, pv, 6);
    set(Q, qv, 6);
    CHECK(mat_mul(R, P, Q, true, false, false));
    CHECK(equals(R, pq, 4));
 
    // The same product through the flags: store P^T (3x2) and Q^T (2x3) explicitly
    // and let transpose_a / transpose_b undo them. All four must give the same answer.
    matrix* Pt = mat_create(a, 3, 2);
    matrix* Qt = mat_create(a, 2, 3);
    f32 ptv[] = { 1, 4,
                  2, 5,
                  3, 6 };
    f32 qtv[] = { 7, 9, 11,
                  8, 10, 12 };
    set(Pt, ptv, 6);
    set(Qt, qtv, 6);
 
    CHECK(mat_mul(R, Pt, Q,  true, true,  false)); CHECK(equals(R, pq, 4));
    CHECK(mat_mul(R, P,  Qt, true, false, true));  CHECK(equals(R, pq, 4));
    CHECK(mat_mul(R, Pt, Qt, true, true,  true));  CHECK(equals(R, pq, 4));
 
    // A non-square OUTPUT: (3x2) * (2x3) = (3x3).
    matrix* big = mat_create(a, 3, 3);
    CHECK(mat_mul(big, Q, Qt, true, false, false));
    f32 qqt[] = { 7*7+8*8,   7*9+8*10,   7*11+8*12,
                  9*7+10*8,  9*9+10*10,  9*11+10*12,
                  11*7+12*8, 11*9+12*10, 11*11+12*12 };
    CHECK(equals(big, qqt, 9));
 
    arena_destroy(a);
}


int main(void) {
    RUN_TEST(test_create);
    RUN_TEST(test_create_failures);
    RUN_TEST(test_clear_fill_scale);
    RUN_TEST(test_sum);
    RUN_TEST(test_argmax);
    RUN_TEST(test_copy);
    RUN_TEST(test_add_sub);
    RUN_TEST(test_fill_rand);

    return test_summary();
}