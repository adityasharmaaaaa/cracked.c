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