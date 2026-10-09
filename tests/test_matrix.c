// tests/test_matrix.c
//
// Build and run with:  make test

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


// Logical element (i, j) of op(x), straight from the definition of transpose.
static f64 ref_at(const matrix* x, b32 transposed, u32 i, u32 j) {
    return transposed ? x->data[(u64)j * x->cols + i] : x->data[(u64)i * x->cols + j];
}

// out (m x n) = op(a) * op(b), accumulated in f64, with the i-j-k loop order
// (deliberately different from the real implementation's i-k-j and its strides).
static void ref_matmul(f64* out, u32 m, u32 n, u32 k,
                       const matrix* a, const matrix* b, b32 ta, b32 tb) {
    for (u32 i = 0; i < m; i++) {
        for (u32 j = 0; j < n; j++) {
            f64 acc = 0.0;
            for (u32 kk = 0; kk < k; kk++) {
                acc += ref_at(a, ta, i, kk) * ref_at(b, tb, kk, j);
            }
            out[(u64)i * n + j] = acc;
        }
    }
}

static void test_matmul_vs_reference(void) {
    mem_arena* arena = arena_create(MiB(16), MiB(1));
    if (!arena) { CHECK(arena != NULL); return; }

    u32 dims[] = { 1, 2, 3, 5, 8, 17, 33 };
    u32 num_dims = sizeof(dims) / sizeof(dims[0]);
    f64 worst = 0.0;
    u32 cases = 0;

    prng_seed(2024, 1);

    for (u32 mi = 0; mi < num_dims; mi++)
    for (u32 ki = 0; ki < num_dims; ki++)
    for (u32 ni = 0; ni < num_dims; ni++)
    for (u32 flags = 0; flags < 4; flags++) {
        u32 m = dims[mi], k = dims[ki], n = dims[ni];
        b32 ta = (flags & 1) != 0;
        b32 tb = (flags & 2) != 0;

        // Temp scope: everything allocated in this iteration is freed at the end.
        mem_arena_temp scratch = arena_temp_begin(arena);

        // Stored shapes depend on the flags (the table from Tuesday's exercise).
        matrix* a = ta ? mat_create(arena, k, m) : mat_create(arena, m, k);
        matrix* b = tb ? mat_create(arena, n, k) : mat_create(arena, k, n);
        matrix* out = mat_create(arena, m, n);
        f64* expected = PUSH_ARRAY(arena, f64, (u64)m * n);

        mat_fill_rand(a, -1.0f, 1.0f);
        mat_fill_rand(b, -1.0f, 1.0f);
        mat_fill(out, 1234.0f);                        // garbage that zero_out must wipe

        b32 ok = mat_mul(out, a, b, true, ta, tb);
        CHECK(ok);

        ref_matmul(expected, m, n, k, a, b, ta, tb);
        for (u64 i = 0; i < (u64)m * n; i++) {
            f64 err = fabs((f64)out->data[i] - expected[i]);
            if (err > worst) { worst = err; }
        }
        cases++;

        arena_temp_end(scratch);
    }

    printf("    %u shape/flag combinations, worst abs error vs f64 reference = %g\n", cases, worst);
    CHECK(worst < 1e-4);

    arena_destroy(arena);
}

static void test_matmul_zero_out(void) {
    mem_arena* a = arena_create(MiB(1), KiB(64));
    if (!a) { CHECK(a != NULL); return; }

    matrix* P = mat_create(a, 2, 3);
    matrix* Q = mat_create(a, 3, 2);
    matrix* out = mat_create(a, 2, 2);
    f32 pv[] = { 1, 2, 3, 4, 5, 6 };
    f32 qv[] = { 7, 8, 9, 10, 11, 12 };
    set(P, pv, 6);
    set(Q, qv, 6);

    // zero_out = true: previous contents are irrelevant.
    mat_fill(out, 100.0f);
    CHECK(mat_mul(out, P, Q, true, false, false));
    f32 product[] = { 58, 64, 139, 154 };
    CHECK(equals(out, product, 4));

    // zero_out = false: the product is ADDED to what is already there.
    mat_fill(out, 100.0f);
    CHECK(mat_mul(out, P, Q, false, false, false));
    f32 plus_one[] = { 158, 164, 239, 254 };
    CHECK(equals(out, plus_one, 4));

    // Twice: this is exactly what happens when a variable feeds two operations and
    // backprop delivers two gradient contributions that must be summed.
    CHECK(mat_mul(out, P, Q, false, false, false));
    f32 plus_two[] = { 216, 228, 378, 408 };
    CHECK(equals(out, plus_two, 4));

    // First call overwrites, second accumulates -> exactly 2x the product.
    CHECK(mat_mul(out, P, Q, true,  false, false));
    CHECK(mat_mul(out, P, Q, false, false, false));
    f32 doubled[] = { 116, 128, 278, 308 };
    CHECK(equals(out, doubled, 4));

    arena_destroy(a);
}

static void test_matmul_shape_errors(void) {
    mem_arena* a = arena_create(MiB(1), KiB(64));
    if (!a) { CHECK(a != NULL); return; }

    matrix* A = mat_create(a, 2, 3);
    matrix* B = mat_create(a, 3, 2);
    mat_fill(A, 1.0f);
    mat_fill(B, 1.0f);

    f32 sentinel4[] = { -7, -7, -7, -7 };
    matrix* out = mat_create(a, 2, 2);
    mat_fill(out, -7.0f);

    // Inner dimensions disagree: (2x3) * (4x2).
    matrix* B_bad = mat_create(a, 4, 2);
    CHECK(!mat_mul(out, A, B_bad, true, false, false));
    CHECK(equals(out, sentinel4, 4));

    // Output has the wrong shape. 1x4 has the same ELEMENT COUNT as 2x2 but is still wrong.
    matrix* out_1x4 = mat_create(a, 1, 4);
    mat_fill(out_1x4, -7.0f);
    CHECK(!mat_mul(out_1x4, A, B, true, false, false));
    CHECK(out_1x4->data[0] == -7.0f && out_1x4->data[3] == -7.0f);

    matrix* out_3x3 = mat_create(a, 3, 3);
    CHECK(!mat_mul(out_3x3, A, B, true, false, false));

    // The flags change which shapes are valid. X and Y are both stored 2x3:
    matrix* X = mat_create(a, 2, 3);
    matrix* Y = mat_create(a, 2, 3);
    matrix* o22 = mat_create(a, 2, 2);
    matrix* o33 = mat_create(a, 3, 3);
    CHECK(!mat_mul(o22, X, Y, true, false, false));   // (2x3) * (2x3): inner 3 vs 2
    CHECK( mat_mul(o22, X, Y, true, false, true));    // (2x3) * (3x2) -> 2x2
    CHECK( mat_mul(o33, X, Y, true, true,  false));   // (3x2) * (2x3) -> 3x3
    CHECK(!mat_mul(o22, X, Y, true, true,  true));    // (3x2) * (3x2): inner 2 vs 3

    // A failed accumulate (zero_out = false) must not modify out either.
    CHECK(!mat_mul(out, A, B_bad, false, false, false));
    CHECK(equals(out, sentinel4, 4));

    // And a failed zero_out = true must not have cleared it before failing.
    CHECK(!mat_mul(out, A, B_bad, true, false, false));
    CHECK(equals(out, sentinel4, 4));

    arena_destroy(a);
}

static void test_matmul_aliasing(void) {
    mem_arena* a = arena_create(MiB(1), KiB(64));
    if (!a) { CHECK(a != NULL); return; }

    matrix* A = mat_create(a, 2, 2);
    matrix* B = mat_create(a, 2, 2);
    f32 av[] = { 1, 2, 3, 4 };
    f32 bv[] = { 5, 6, 7, 8 };
    set(A, av, 4);
    set(B, bv, 4);

    // Same matrix as output and input: rejected, nothing modified.
    CHECK(!mat_mul(A, A, B, true, false, false));
    CHECK(equals(A, av, 4));
    CHECK(!mat_mul(B, A, B, true, false, false));
    CHECK(equals(B, bv, 4));
    CHECK(!mat_mul(A, A, A, true, false, false));
    CHECK(equals(A, av, 4));

    // Partial overlap: two DIFFERENT matrix structs whose data ranges overlap.
    // Checking only "same pointer" would miss this.
    matrix* backing = mat_create(a, 4, 2);               // 8 floats
    matrix a_view   = { 2, 2, backing->data };           // floats 0..3
    matrix out_view = { 2, 2, backing->data + 2 };       // floats 2..5: overlaps a_view
    CHECK(!mat_mul(&out_view, &a_view, B, true, false, false));

    // Disjoint views are fine.
    matrix out_far = { 2, 2, backing->data + 4 };        // floats 4..7: no overlap with 0..3
    CHECK(mat_mul(&out_far, &a_view, B, true, false, false));

    arena_destroy(a);
}

static void test_relu(void) {
    mem_arena* a = arena_create(MiB(1), KiB(64));
    if (!a) { CHECK(a != NULL); return; }

    matrix* in  = mat_create(a, 2, 3);
    matrix* out = mat_create(a, 2, 3);

    f32 iv[] = { -2.0f, -0.5f, 0.0f,
                  0.5f,  3.0f, -1e30f };
    f32 ev[] = {  0.0f,  0.0f, 0.0f,
                  0.5f,  3.0f,  0.0f };
    set(in, iv, 6);
    CHECK(mat_relu(out, in));
    CHECK(equals(out, ev, 6));
    CHECK(in->data[0] == -2.0f);                      // the input is not modified

    // In place.
    CHECK(mat_relu(in, in));
    CHECK(equals(in, ev, 6));

    // Infinities behave: +inf passes, -inf is clipped.
    matrix* inf_in = mat_create(a, 1, 2);
    matrix* inf_out = mat_create(a, 1, 2);
    inf_in->data[0] = INFINITY;
    inf_in->data[1] = -INFINITY;
    CHECK(mat_relu(inf_out, inf_in));
    CHECK(isinf(inf_out->data[0]) && inf_out->data[0] > 0);
    CHECK(inf_out->data[1] == 0.0f);

    // NaN must come out as NaN. Turning it into 0 would hide whatever bug created it.
    matrix* nan_in = mat_create(a, 1, 1);
    matrix* nan_out = mat_create(a, 1, 1);
    nan_in->data[0] = NAN;
    CHECK(mat_relu(nan_out, nan_in));
    CHECK(isnan(nan_out->data[0]));

    // Shape mismatch (same element count!): false, output untouched.
    matrix* wrong = mat_create(a, 3, 2);
    mat_fill(wrong, -7.0f);
    CHECK(!mat_relu(wrong, in));
    CHECK(wrong->data[0] == -7.0f && wrong->data[5] == -7.0f);

    arena_destroy(a);
}

static b32 row_sums_to_one(const matrix* m) {
    for (u32 r = 0; r < m->rows; r++) {
        f64 sum = 0.0;
        for (u32 c = 0; c < m->cols; c++) { sum += m->data[(u64)r * m->cols + c]; }
        if (fabs(sum - 1.0) > 1e-5) { return false; }
    }
    return true;
}

static void test_softmax(void) {
    mem_arena* a = arena_create(MiB(1), KiB(64));
    if (!a) { CHECK(a != NULL); return; }

    // softmax([1, 2, 3]) = [e^1, e^2, e^3] / (e^1 + e^2 + e^3)
    f32 expected[] = { 0.09003057f, 0.24472847f, 0.66524096f };

    matrix* x = mat_create(a, 1, 3);
    matrix* y = mat_create(a, 1, 3);
    f32 xv[] = { 1, 2, 3 };
    set(x, xv, 3);
    CHECK(mat_softmax(y, x));
    for (u32 i = 0; i < 3; i++) { CHECK_NEAR(y->data[i], expected[i], 1e-6); }
    CHECK(row_sums_to_one(y));

    // Shift invariance: softmax(x + c) == softmax(x). This is also the trick that makes the
    // naive formula's overflow avoidable: with logits near 1000, exp(1000) = inf in f32.
    f32 big[] = { 1001, 1002, 1003 };
    set(x, big, 3);
    CHECK(mat_softmax(y, x));
    for (u32 i = 0; i < 3; i++) {
        CHECK(!isnan(y->data[i]) && !isinf(y->data[i]));
        CHECK_NEAR(y->data[i], expected[i], 1e-6);
    }

    // All-NEGATIVE huge logits: exp(-1000) = 0 for every element unless the max is subtracted
    // (and the max must be found properly, even though every value is below zero).
    f32 neg[] = { -999, -998, -997 };
    set(x, neg, 3);
    CHECK(mat_softmax(y, x));
    for (u32 i = 0; i < 3; i++) {
        CHECK(!isnan(y->data[i]));
        CHECK_NEAR(y->data[i], expected[i], 1e-6);
    }

    // One logit dominates: no NaN, essentially [1, 0, 0].
    f32 dom[] = { 100, 0, 0 };
    set(x, dom, 3);
    CHECK(mat_softmax(y, x));
    CHECK_NEAR(y->data[0], 1.0, 1e-6);
    CHECK(!isnan(y->data[1]) && y->data[1] >= 0.0f && y->data[1] < 1e-30f);

    // The maximum can be ANYWHERE in the row. If the max search misses it, the shift is too
    // small and exp() overflows (exp(200) = inf in f32). Softmax's shift-invariance hides a
    // slightly-wrong max, so the test needs a huge gap to expose a missed element.
    f32 last_big[]  = { 0, 0, 200 };
    f32 mid_big[]   = { 0, 200, 0 };
    f32 first_big[] = { 200, 0, 0 };
    set(x, last_big, 3);
    CHECK(mat_softmax(y, x));
    CHECK(!isnan(y->data[2]) && y->data[2] > 0.999f && y->data[0] < 1e-30f);
    set(x, mid_big, 3);
    CHECK(mat_softmax(y, x));
    CHECK(!isnan(y->data[1]) && y->data[1] > 0.999f);
    set(x, first_big, 3);
    CHECK(mat_softmax(y, x));
    CHECK(!isnan(y->data[0]) && y->data[0] > 0.999f);

    // Uniform logits -> uniform probabilities.
    matrix* u = mat_create(a, 1, 4);
    matrix* uo = mat_create(a, 1, 4);
    mat_fill(u, 5.0f);
    CHECK(mat_softmax(uo, u));
    for (u32 i = 0; i < 4; i++) { CHECK_NEAR(uo->data[i], 0.25, 1e-6); }

    // Rows are independent: a 2x3 matrix is two separate softmaxes, NOT one softmax over 6 numbers.
    matrix* m = mat_create(a, 2, 3);
    matrix* mo = mat_create(a, 2, 3);
    f32 mv[] = { 1, 2, 3,
                 3, 2, 1 };
    set(m, mv, 6);
    CHECK(mat_softmax(mo, m));
    CHECK(row_sums_to_one(mo));                       // each row sums to 1 (not 0.5)
    CHECK_NEAR(mo->data[0], expected[0], 1e-6);
    CHECK_NEAR(mo->data[3], expected[2], 1e-6);       // second row is the first one reversed
    CHECK_NEAR(mo->data[5], expected[0], 1e-6);

    // A single column: each row has one element, so every output is exactly 1.
    matrix* col = mat_create(a, 3, 1);
    matrix* colo = mat_create(a, 3, 1);
    f32 cv[] = { -4, 0, 17 };
    set(col, cv, 3);
    CHECK(mat_softmax(colo, col));
    CHECK(colo->data[0] == 1.0f && colo->data[1] == 1.0f && colo->data[2] == 1.0f);

    // In place gives the same answer as a separate output.
    matrix* ip = mat_create(a, 2, 3);
    mat_copy(ip, m);
    CHECK(mat_softmax(ip, ip));
    for (u32 i = 0; i < 6; i++) { CHECK(ip->data[i] == mo->data[i]); }

    // Shape mismatch: false, output untouched.
    matrix* wrong = mat_create(a, 3, 2);
    mat_fill(wrong, -7.0f);
    CHECK(!mat_softmax(wrong, m));
    CHECK(wrong->data[0] == -7.0f);

    arena_destroy(a);
}

static void test_cross_entropy(void) {
    mem_arena* a = arena_create(MiB(1), KiB(64));
    if (!a) { CHECK(a != NULL); return; }

    matrix* p = mat_create(a, 1, 3);
    matrix* q = mat_create(a, 1, 3);
    matrix* out = mat_create(a, 1, 3);

    // True class is index 1, predicted probability 0.7: loss = -ln(0.7) there, 0 elsewhere.
    f32 pv[] = { 0.0f, 1.0f, 0.0f };
    f32 qv[] = { 0.2f, 0.7f, 0.1f };
    set(p, pv, 3);
    set(q, qv, 3);
    CHECK(mat_cross_entropy(out, p, q));
    CHECK_NEAR(out->data[0], 0.0, 1e-9);
    CHECK_NEAR(out->data[1], 0.35667494, 1e-6);
    CHECK_NEAR(out->data[2], 0.0, 1e-9);

    // Order matters: p is the target, q the prediction. Here p has a zero where q does not.
    matrix* p2 = mat_create(a, 1, 2);
    matrix* q2 = mat_create(a, 1, 2);
    matrix* o2 = mat_create(a, 1, 2);
    f32 p2v[] = { 1.0f, 0.0f };
    f32 q2v[] = { 0.25f, 0.75f };
    set(p2, p2v, 2);
    set(q2, q2v, 2);
    CHECK(mat_cross_entropy(o2, p2, q2));
    CHECK_NEAR(o2->data[0], 1.3862944, 1e-6);         // -ln(0.25)
    CHECK_NEAR(o2->data[1], 0.0, 1e-9);

    // Perfect prediction -> zero loss.
    matrix* one = mat_create(a, 1, 1);
    matrix* oo = mat_create(a, 1, 1);
    mat_fill(one, 1.0f);
    CHECK(mat_cross_entropy(oo, one, one));
    CHECK_NEAR(oo->data[0], 0.0, 1e-9);

    // q = 0 must not blow up. With a target of 0 the result must be exactly 0 (not 0 * -inf = NaN),
    // with a target of 1 it is a large but FINITE number, -ln(MAT_LOG_EPS).
    matrix* pz = mat_create(a, 1, 2);
    matrix* qz = mat_create(a, 1, 2);
    matrix* oz = mat_create(a, 1, 2);
    pz->data[0] = 0.0f; pz->data[1] = 1.0f;
    qz->data[0] = 0.0f; qz->data[1] = 0.0f;
    CHECK(mat_cross_entropy(oz, pz, qz));
    CHECK(oz->data[0] == 0.0f);
    CHECK(!isnan(oz->data[1]) && !isinf(oz->data[1]));
    CHECK_NEAR(oz->data[1], 16.118096, 1e-4);

    // NaN in the prediction must come out as NaN, not be clamped away.
    qz->data[1] = NAN;
    CHECK(mat_cross_entropy(oz, pz, qz));
    CHECK(isnan(oz->data[1]));

    // The sanity check every classifier should pass: with all-zero logits over 10 classes the
    // softmax is uniform (0.1 each), so the loss is -ln(0.1) = ln(10) = 2.3026 per sample,
    // whatever the labels are. A freshly initialised MNIST network should start near this.
    u32 batch = 4, classes = 10;
    matrix* logits = mat_create(a, batch, classes);
    matrix* probs = mat_create(a, batch, classes);
    matrix* labels = mat_create(a, batch, classes);
    matrix* ce = mat_create(a, batch, classes);
    for (u32 r = 0; r < batch; r++) { labels->data[r * classes + (r * 3) % classes] = 1.0f; }

    CHECK(mat_softmax(probs, logits));
    CHECK(mat_cross_entropy(ce, labels, probs));
    f32 mean_loss = mat_sum(ce) / (f32)batch;
    CHECK_NEAR(mean_loss, 2.3025851, 1e-5);

    // Shape mismatches: false, output untouched.
    matrix* wrong = mat_create(a, 3, 1);
    mat_fill(wrong, -7.0f);
    CHECK(!mat_cross_entropy(wrong, p, q));           // out has the wrong shape
    CHECK(wrong->data[0] == -7.0f);
    CHECK(!mat_cross_entropy(out, p, wrong));         // q has the wrong shape

    arena_destroy(a);
}

typedef struct {
    matrix* x;      // the matrix being perturbed (relu / softmax input)
    matrix* out;    // forward output
    matrix* g;      // fixed random upstream gradient: L = sum(g * out)
    matrix* p;      // cross-entropy target
    matrix* q;      // cross-entropy prediction
} gc_ctx;
 
static f64 relu_loss(void* c)    { gc_ctx* t = c; mat_relu(t->out, t->x);              return gc_weighted_sum(t->out, t->g); }
static f64 softmax_loss(void* c) { gc_ctx* t = c; mat_softmax(t->out, t->x);           return gc_weighted_sum(t->out, t->g); }
static f64 ce_loss(void* c)      { gc_ctx* t = c; mat_cross_entropy(t->out, t->p, t->q); return gc_weighted_sum(t->out, t->g); }
 
// ---------------------------------------------------------------------------
// mat_relu_add_grad
// ---------------------------------------------------------------------------
static void test_relu_grad(void) {
    mem_arena* a = arena_create(MiB(4), MiB(1));
    if (!a) { CHECK(a != NULL); return; }
 
    // Hand-computed. in = 0 gets gradient 0 (our convention); negatives are blocked.
    matrix* in  = mat_create(a, 1, 5);
    matrix* g   = mat_create(a, 1, 5);
    matrix* out = mat_create(a, 1, 5);
    f32 iv[] = { -1.0f, 0.0f, 2.0f, 3.0f, -0.001f };
    f32 gv[] = {  5.0f, 6.0f, 7.0f, 8.0f,  9.0f };
    set(in, iv, 5);
    set(g, gv, 5);
 
    CHECK(mat_relu_add_grad(out, in, g));
    f32 expected[] = { 0, 0, 7, 8, 0 };
    CHECK(equals(out, expected, 5));
 
    // ACCUMULATES: a pre-existing gradient is added to, not overwritten.
    mat_fill(out, 1.0f);
    CHECK(mat_relu_add_grad(out, in, g));
    f32 accumulated[] = { 1, 1, 8, 9, 1 };
    CHECK(equals(out, accumulated, 5));
    CHECK(mat_relu_add_grad(out, in, g));             // a second contribution
    f32 twice[] = { 1, 1, 15, 17, 1 };
    CHECK(equals(out, twice, 5));
 
    // Shape mismatch: false, nothing written.
    matrix* wrong = mat_create(a, 5, 1);
    mat_fill(wrong, -7.0f);
    CHECK(!mat_relu_add_grad(wrong, in, g));
    CHECK(wrong->data[0] == -7.0f);
    CHECK(!mat_relu_add_grad(out, in, wrong));
 
    // Finite differences on random data. Keep every |x| well above h, because relu has a
    // kink at 0 and the central difference straddling it would be meaningless.
    prng_seed(11, 1);
    u32 R = 4, C = 7;
    matrix* x  = mat_create(a, R, C);
    matrix* fo = mat_create(a, R, C);
    matrix* fg = mat_create(a, R, C);
    matrix* an = mat_create(a, R, C);
    matrix* nu = mat_create(a, R, C);
    mat_fill_rand(x, 0.05f, 2.0f);
    for (u32 i = 0; i < R * C; i++) { if (prng_randf() < 0.5f) { x->data[i] = -x->data[i]; } }
    mat_fill_rand(fg, -1.0f, 1.0f);
 
    gc_ctx ctx = { .x = x, .out = fo, .g = fg };
    gc_numeric_grad(x, relu_loss, &ctx, GC_H, nu);
    CHECK(mat_relu_add_grad(an, x, fg));
    f64 err = gc_max_error(an, nu);
    printf("    relu: max error vs finite differences = %.2e\n", err);
    CHECK(err < GC_TOLERANCE);
 
    arena_destroy(a);
}

static void test_softmax_grad(void) {
    mem_arena* a = arena_create(MiB(4), MiB(1));
    if (!a) { CHECK(a != NULL); return; }
 
    // Hand-computed (independently, in Python): x = [1, 2, 3], upstream g = [1, 0, 0].
    // out_j = y_j * (g_j - y_0), with y = [0.09003057, 0.24472847, 0.66524096].
    matrix* x = mat_create(a, 1, 3);
    matrix* y = mat_create(a, 1, 3);
    matrix* g = mat_create(a, 1, 3);
    matrix* out = mat_create(a, 1, 3);
    f32 xv[] = { 1, 2, 3 };
    f32 gv[] = { 1, 0, 0 };
    set(x, xv, 3);
    set(g, gv, 3);
    mat_softmax(y, x);
 
    CHECK(mat_softmax_add_grad(out, y, g));
    CHECK_NEAR(out->data[0],  0.08192507, 1e-6);
    CHECK_NEAR(out->data[1], -0.02203304, 1e-6);
    CHECK_NEAR(out->data[2], -0.05989202, 1e-6);
 
    // Shifting every logit by the same amount changes nothing, so the gradient must sum to
    // zero over each row.
    CHECK_NEAR(out->data[0] + out->data[1] + out->data[2], 0.0, 1e-6);
 
    // Accumulates.
    CHECK(mat_softmax_add_grad(out, y, g));
    CHECK_NEAR(out->data[0], 2 * 0.08192507, 2e-6);
 
    // Rows are independent: stack the same row twice, both get the same gradient.
    matrix* x2 = mat_create(a, 2, 3);
    matrix* y2 = mat_create(a, 2, 3);
    matrix* g2 = mat_create(a, 2, 3);
    matrix* o2 = mat_create(a, 2, 3);
    f32 x2v[] = { 1, 2, 3, 1, 2, 3 };
    f32 g2v[] = { 1, 0, 0, 1, 0, 0 };
    set(x2, x2v, 6);
    set(g2, g2v, 6);
    mat_softmax(y2, x2);
    CHECK(mat_softmax_add_grad(o2, y2, g2));
    for (u32 i = 0; i < 3; i++) { CHECK(o2->data[i] == o2->data[3 + i]); }
    CHECK_NEAR(o2->data[0], 0.08192507, 1e-6);
 
    // Shape mismatch.
    matrix* wrong = mat_create(a, 3, 1);
    mat_fill(wrong, -7.0f);
    CHECK(!mat_softmax_add_grad(wrong, y, g));
    CHECK(wrong->data[0] == -7.0f);
 
    // Finite differences on random logits and a random upstream gradient.
    prng_seed(12, 1);
    u32 R = 4, C = 7;
    matrix* rx = mat_create(a, R, C);
    matrix* ry = mat_create(a, R, C);
    matrix* rg = mat_create(a, R, C);
    matrix* an = mat_create(a, R, C);
    matrix* nu = mat_create(a, R, C);
    mat_fill_rand(rx, -5.0f, 5.0f);
    mat_fill_rand(rg, -1.0f, 1.0f);
    mat_softmax(ry, rx);
 
    gc_ctx ctx = { .x = rx, .out = mat_create(a, R, C), .g = rg };
    gc_numeric_grad(rx, softmax_loss, &ctx, GC_H, nu);
    CHECK(mat_softmax_add_grad(an, ry, rg));
    f64 err = gc_max_error(an, nu);
    printf("    softmax: max error vs finite differences = %.2e\n", err);
    CHECK(err < GC_TOLERANCE);
 
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
    RUN_TEST(test_matmul_known);
    RUN_TEST(test_matmul_vs_reference);
    RUN_TEST(test_matmul_zero_out);
    RUN_TEST(test_matmul_shape_errors);
    RUN_TEST(test_matmul_aliasing);
    RUN_TEST(test_relu);
    RUN_TEST(test_softmax);
    RUN_TEST(test_cross_entropy);
    RUN_TEST(test_relu_grad);
    RUN_TEST(test_softmax_grad);
    
    return test_summary();
}
