#include "base.h"
#include "arena.h"
#include "prng.h"
#include "matrix.h"
 
#include "arena.c"
#include "prng.c"
#include "matrix.c"
 
#include "test.h"
 
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
 
    CHECK(a->pos == pos_before + sizeof(matrix) + 12 * sizeof(f32));
 
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