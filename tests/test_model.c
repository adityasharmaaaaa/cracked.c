#include "base.h"
#include "arena.h"
#include "prng.h"
#include "matrix.h"
#include "model.h"
 
#include "arena.c"
#include "prng.c"
#include "matrix.c"
#include "model.c"
 
#include "test.h"
 
static void fill_from(matrix* m, const f32* values, u32 count) {
    for (u32 i = 0; i < count; i++) { m->data[i] = values[i]; }
}
 
static b32 all_zero(const matrix* m) {
    u64 n = (u64)m->rows * m->cols;
    for (u64 i = 0; i < n; i++) { if (m->data[i] != 0.0f) { return false; } }
    return true;
}
 
// x -> matmul(x, w1) -> relu -> matmul(., w2) -> softmax -> cross_entropy(labels, .)
typedef struct {
    model_context* m;
    model_var *x, *w1, *h, *a, *w2, *z, *y, *labels, *cost;
} mlp;
 
static b32 build_mlp(mem_arena* arena, mlp* g, u32 batch, u32 in, u32 hid, u32 out) {
    g->m      = model_create(arena);
    if (!g->m) { return false; }
    g->x      = mv_create(arena, g->m, batch, in,  MV_FLAG_INPUT);
    g->w1     = mv_create(arena, g->m, in,    hid, MV_FLAG_PARAMETER);
    g->h      = mv_matmul(arena, g->m, g->x, g->w1, MV_FLAG_NONE);
    g->a      = mv_relu(arena, g->m, g->h, MV_FLAG_NONE);
    g->w2     = mv_create(arena, g->m, hid,   out, MV_FLAG_PARAMETER);
    g->z      = mv_matmul(arena, g->m, g->a, g->w2, MV_FLAG_NONE);
    g->y      = mv_softmax(arena, g->m, g->z, MV_FLAG_OUTPUT);
    g->labels = mv_create(arena, g->m, batch, out, MV_FLAG_DESIRED_OUTPUT);
    g->cost   = mv_cross_entropy(arena, g->m, g->labels, g->y, MV_FLAG_COST);
    return g->x && g->w1 && g->h && g->a && g->w2 && g->z && g->y && g->labels && g->cost;
}
 
// Every variable appears at most once, and only AFTER all of its inputs.
static b32 is_topological(const model_program* p) {
    for (u32 i = 0; i < p->size; i++) {
        model_var* v = p->vars[i];
 
        for (u32 j = 0; j < i; j++) {
            if (p->vars[j] == v) { return false; }                // duplicate
        }
        for (u32 k = 0; k < (u32)MV_NUM_INPUTS(v->op); k++) {
            b32 found_earlier = false;
            for (u32 j = 0; j < i; j++) {
                if (p->vars[j] == v->inputs[k]) { found_earlier = true; }
            }
            if (!found_earlier) { return false; }                 // input not yet computed
        }
    }
    return true;
}
 
// Overwrite the scratch arena's leftover bytes with zeros. Memory that was "freed" by a scratch
// release still holds old data; a bug that reads uninitialised scratch memory can then pass by
// luck (stale 1s looking like "already visited"). Zeroing makes such a bug show up every time.
static void scrub_scratch(void) {
    mem_arena_temp s = arena_scratch_get(NULL, 0);
    if (s.arena) {
        arena_push(s.arena, KiB(256), false);       // push zeroes the bytes it hands out
        arena_scratch_release(s);
    }
}
 
static b32 program_equals(const model_program* p, model_var** expected, u32 count) {
    if (p->size != count) { return false; }
    for (u32 i = 0; i < count; i++) {
        if (p->vars[i] != expected[i]) { return false; }
    }
    return true;
}
 
// ---------------------------------------------------------------------------
// mv_create
// ---------------------------------------------------------------------------
static void test_mv_create(void) {
    mem_arena* a = arena_create(MiB(4), MiB(1));
    if (!a) { CHECK(a != NULL); return; }
 
    model_context* m = model_create(a);
    CHECK(m != NULL);
    if (!m) { return; }
    CHECK(m->num_vars == 0);
    CHECK(m->input == NULL && m->output == NULL && m->desired_output == NULL && m->cost == NULL);
    CHECK(m->forward_prog.size == 0 && m->cost_prog.size == 0);
 
    // A plain input: value, no gradient.
    model_var* x = mv_create(a, m, 4, 3, MV_FLAG_INPUT);
    CHECK(x != NULL);
    if (!x) { return; }
    CHECK(x->val != NULL && x->val->rows == 4 && x->val->cols == 3);
    CHECK(all_zero(x->val));
    CHECK(x->grad == NULL);
    CHECK(x->op == MV_OP_CREATE);
    CHECK(x->index == 0);
    CHECK(m->input == x);
    CHECK(m->num_vars == 1);
 
    // A parameter ALWAYS gets a gradient, even though we did not ask for REQUIRES_GRAD.
    model_var* w = mv_create(a, m, 3, 5, MV_FLAG_PARAMETER);
    CHECK(w != NULL);
    if (!w) { return; }
    CHECK(w->flags & MV_FLAG_REQUIRES_GRAD);
    REQUIRE(w->grad != NULL);
    CHECK(w->grad->rows == 3 && w->grad->cols == 5);
    CHECK(all_zero(w->grad));
    CHECK(w->index == 1);
    CHECK(m->num_vars == 2);
 
    // REQUIRES_GRAD alone also allocates one.
    model_var* r = mv_create(a, m, 2, 2, MV_FLAG_REQUIRES_GRAD);
    CHECK(r != NULL && r->grad != NULL);
 
    // Failure (too big for the arena): NULL, and nothing changed -- no leaked arena space,
    // no phantom variable.
    u64 pos_before = a->pos;
    u32 vars_before = m->num_vars;
    CHECK(mv_create(a, m, 1u << 20, 1u << 10, MV_FLAG_PARAMETER) == NULL);   // 4 GiB: cannot fit
    CHECK(a->pos == pos_before);
    CHECK(m->num_vars == vars_before);
 
    arena_destroy(a);
}
 
// ---------------------------------------------------------------------------
// Roles
// ---------------------------------------------------------------------------
static void test_roles(void) {
    mem_arena* a = arena_create(MiB(4), MiB(1));
    if (!a) { CHECK(a != NULL); return; }
 
    mlp g;
    REQUIRE(build_mlp(a, &g, 4, 6, 5, 3));
 
    CHECK(g.m->input == g.x);
    CHECK(g.m->output == g.y);
    CHECK(g.m->desired_output == g.labels);
    CHECK(g.m->cost == g.cost);
    CHECK(g.m->num_vars == 9);
 
    // Each role belongs to one variable. A second claimant is refused and changes nothing.
    u64 pos_before = a->pos;
    CHECK(mv_create(a, g.m, 2, 2, MV_FLAG_INPUT) == NULL);
    CHECK(mv_softmax(a, g.m, g.z, MV_FLAG_OUTPUT) == NULL);
    CHECK(mv_create(a, g.m, 4, 3, MV_FLAG_DESIRED_OUTPUT) == NULL);
    CHECK(mv_cross_entropy(a, g.m, g.labels, g.y, MV_FLAG_COST) == NULL);
    CHECK(g.m->input == g.x && g.m->output == g.y);
    CHECK(g.m->num_vars == 9);
    CHECK(a->pos == pos_before);                      // the refused variables' memory was given back
 
    arena_destroy(a);
}