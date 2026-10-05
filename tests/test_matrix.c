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