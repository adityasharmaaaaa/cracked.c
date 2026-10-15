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

static void test_mv_shapes(void) {
    mem_arena* a = arena_create(MiB(4), MiB(1));
    if (!a) { CHECK(a != NULL); return; }
    model_context* m = model_create(a);
 
    model_var* x43 = mv_create(a, m, 4, 3, MV_FLAG_NONE);
    model_var* y43 = mv_create(a, m, 4, 3, MV_FLAG_NONE);
    model_var* z34 = mv_create(a, m, 3, 4, MV_FLAG_NONE);   // SAME element count as 4x3
    model_var* w35 = mv_create(a, m, 3, 5, MV_FLAG_NONE);
 
    // Unary ops keep the shape and record their input.
    model_var* r = mv_relu(a, m, x43, MV_FLAG_NONE);
    REQUIRE(r != NULL);
    CHECK(r->val->rows == 4 && r->val->cols == 3);
    CHECK(r->op == MV_OP_RELU && r->inputs[0] == x43 && r->inputs[1] == NULL);
    model_var* s = mv_softmax(a, m, x43, MV_FLAG_NONE);
    REQUIRE(s != NULL);
    CHECK(s->val->rows == 4 && s->val->cols == 3 && s->op == MV_OP_SOFTMAX);
 
    // Binary elementwise ops need IDENTICAL shapes (4x3 vs 3x4 must be rejected).
    model_var* sum = mv_add(a, m, x43, y43, MV_FLAG_NONE);
    REQUIRE(sum != NULL);
    CHECK(sum->op == MV_OP_ADD && sum->inputs[0] == x43 && sum->inputs[1] == y43);
    model_var* dif = mv_sub(a, m, x43, y43, MV_FLAG_NONE);
    REQUIRE(dif != NULL);
    CHECK(dif->op == MV_OP_SUB);
 
    // Matmul: (4x3) * (3x5) = (4x5). The inner dimensions must agree.
    model_var* mm = mv_matmul(a, m, x43, w35, MV_FLAG_NONE);
    REQUIRE(mm != NULL);
    CHECK(mm->val->rows == 4 && mm->val->cols == 5 && mm->op == MV_OP_MATMUL);
 
    // Cross-entropy: same shape as its inputs.
    model_var* ce = mv_cross_entropy(a, m, x43, y43, MV_FLAG_NONE);
    REQUIRE(ce != NULL);
    CHECK(ce->val->rows == 4 && ce->val->cols == 3 && ce->op == MV_OP_CROSS_ENTROPY);
    CHECK(ce->inputs[0] == x43 && ce->inputs[1] == y43);    // p first, q second
 
    // Baseline AFTER all the successes; now every failure below must leave it untouched.
    u64 pos_before = a->pos;
    u32 vars_before = m->num_vars;
 
    CHECK(mv_add(a, m, x43, z34, MV_FLAG_NONE) == NULL);
    CHECK(mv_sub(a, m, x43, z34, MV_FLAG_NONE) == NULL);
    CHECK(mv_cross_entropy(a, m, x43, z34, MV_FLAG_NONE) == NULL);
    CHECK(mv_matmul(a, m, x43, y43, MV_FLAG_NONE) == NULL);     // (4x3) * (4x3)
    CHECK(mv_matmul(a, m, w35, x43, MV_FLAG_NONE) == NULL);     // (3x5) * (4x3)
 
    // NULL inputs are rejected, not dereferenced.
    CHECK(mv_relu(a, m, NULL, MV_FLAG_NONE) == NULL);
    CHECK(mv_softmax(a, m, NULL, MV_FLAG_NONE) == NULL);
    CHECK(mv_add(a, m, x43, NULL, MV_FLAG_NONE) == NULL);
    CHECK(mv_matmul(a, m, NULL, w35, MV_FLAG_NONE) == NULL);
 
    // Every failure above left the model and the arena exactly as they were.
    CHECK(m->num_vars == vars_before);
    CHECK(a->pos == pos_before);
 
    arena_destroy(a);
}
 
// ---------------------------------------------------------------------------
// REQUIRES_GRAD propagation
// ---------------------------------------------------------------------------
static void test_requires_grad(void) {
    mem_arena* a = arena_create(MiB(4), MiB(1));
    if (!a) { CHECK(a != NULL); return; }
    model_context* m = model_create(a);
 
    model_var* x      = mv_create(a, m, 4, 3, MV_FLAG_NONE);        // no gradient
    model_var* x2     = mv_create(a, m, 4, 3, MV_FLAG_NONE);        // no gradient
    model_var* w      = mv_create(a, m, 3, 2, MV_FLAG_PARAMETER);   // gradient
 
    // Nothing upstream needs a gradient -> this node needs none, and has no gradient matrix.
    model_var* r = mv_relu(a, m, x, MV_FLAG_NONE);
    REQUIRE(r != NULL);
    CHECK(!(r->flags & MV_FLAG_REQUIRES_GRAD) && r->grad == NULL);
    model_var* s = mv_add(a, m, x, x2, MV_FLAG_NONE);
    REQUIRE(s != NULL);
    CHECK(s->grad == NULL);
 
    // A parameter upstream -> the gradient must flow THROUGH this node, so it needs one.
    model_var* mm = mv_matmul(a, m, x, w, MV_FLAG_NONE);
    REQUIRE(mm != NULL);
    CHECK(mm->flags & MV_FLAG_REQUIRES_GRAD);
    REQUIRE(mm->grad != NULL);
    CHECK(mm->grad->rows == 4 && mm->grad->cols == 2);
 
    // It keeps propagating down the chain.
    model_var* act = mv_relu(a, m, mm, MV_FLAG_NONE);
    CHECK(act != NULL && act->grad != NULL);
 
    // Either input is enough: labels (no gradient) with a prediction that needs one.
    model_var* labels = mv_create(a, m, 4, 2, MV_FLAG_DESIRED_OUTPUT);
    model_var* ce = mv_cross_entropy(a, m, labels, act, MV_FLAG_NONE);
    CHECK(ce != NULL && ce->grad != NULL);
    CHECK(labels->grad == NULL);                      // the labels themselves still have none
 
    // Same, with the requiring input in the first position.
    model_var* sub = mv_sub(a, m, act, labels, MV_FLAG_NONE);
    CHECK(sub != NULL && sub->grad != NULL);
 
    arena_destroy(a);
}

static void test_prog_order(void) {
    mem_arena* a = arena_create(MiB(4), MiB(1));
    if (!a) { CHECK(a != NULL); return; }
 
    mlp g;
    REQUIRE(build_mlp(a, &g, 4, 6, 5, 3));
    model_compile(a, g.m);
 
    // Post-order DFS from the cost, visiting inputs[0] before inputs[1]. For
    // cost = cross_entropy(labels, y), it visits `labels` first, then the whole chain below y.
    model_var* expected_cost[] = { g.labels, g.x, g.w1, g.h, g.a, g.w2, g.z, g.y, g.cost };
    CHECK(program_equals(&g.m->cost_prog, expected_cost, 9));
    CHECK(is_topological(&g.m->cost_prog));
 
    // The forward program needs only what `output` depends on: no labels, no cost.
    model_var* expected_fwd[] = { g.x, g.w1, g.h, g.a, g.w2, g.z, g.y };
    CHECK(program_equals(&g.m->forward_prog, expected_fwd, 7));
    CHECK(is_topological(&g.m->forward_prog));
 
    arena_destroy(a);
}
 
static void test_prog_shapes_of_graphs(void) {
    mem_arena* a = arena_create(MiB(4), MiB(1));
    if (!a) { CHECK(a != NULL); return; }
    model_context* m = model_create(a);
 
    // Fan-out: x is used twice (directly, and through relu). It must appear ONCE.
    model_var* x = mv_create(a, m, 2, 3, MV_FLAG_NONE);
    model_var* r = mv_relu(a, m, x, MV_FLAG_NONE);
    model_var* s = mv_add(a, m, r, x, MV_FLAG_NONE);
 
    model_program p = model_prog_create(a, m, s);
    model_var* expect1[] = { x, r, s };
    CHECK(program_equals(&p, expect1, 3));
    CHECK(is_topological(&p));
 
    // Diamond: one source feeding two branches that meet again.
    model_var* src = mv_create(a, m, 2, 3, MV_FLAG_NONE);
    model_var* b1 = mv_relu(a, m, src, MV_FLAG_NONE);
    model_var* b2 = mv_softmax(a, m, src, MV_FLAG_NONE);
    model_var* join = mv_add(a, m, b1, b2, MV_FLAG_NONE);
    model_program d = model_prog_create(a, m, join);
    model_var* expect2[] = { src, b1, b2, join };
    CHECK(program_equals(&d, expect2, 4));
    CHECK(is_topological(&d));
 
    // Pruning: x, r, s and the whole diamond exist in the model, yet a program for `b1` contains
    // only src and b1.
    model_program only_b1 = model_prog_create(a, m, b1);
    model_var* expect3[] = { src, b1 };
    CHECK(program_equals(&only_b1, expect3, 2));
 
    // A leaf alone is a program of one.
    model_program leaf = model_prog_create(a, m, x);
    model_var* expect4[] = { x };
    CHECK(program_equals(&leaf, expect4, 1));
 
    // Degenerate inputs give an empty program, not a crash.
    model_program none = model_prog_create(a, m, NULL);
    CHECK(none.size == 0 && none.vars == NULL);
 
    // A variable that cannot belong to the model is refused instead of indexing out of bounds.
    // First: a model with no variables at all.
    model_context* empty = model_create(a);
    scrub_scratch();
    model_program from_empty = model_prog_create(a, empty, x);
    CHECK(from_empty.size == 0 && from_empty.vars == NULL);
 
    // Second: a model with one variable, asked about a variable whose index is beyond that
    // (`join` was the 7th variable created in `m`, so its index is 6).
    model_context* tiny = model_create(a);
    mv_create(a, tiny, 1, 1, MV_FLAG_NONE);
    scrub_scratch();
    model_program foreign = model_prog_create(a, tiny, join);
    CHECK(join->index >= tiny->num_vars);
    CHECK(foreign.size == 0 && foreign.vars == NULL);
 
    // A long chain (depth 500): sorted in order, no duplicates, nothing lost.
    model_var* v = mv_create(a, m, 1, 4, MV_FLAG_NONE);
    model_var* first = v;
    for (u32 i = 0; i < 500; i++) {
        v = mv_relu(a, m, v, MV_FLAG_NONE);
    }
    model_program chain = model_prog_create(a, m, v);
    REQUIRE(chain.size == 501);
    CHECK(chain.vars[0] == first && chain.vars[500] == v);
    CHECK(is_topological(&chain));
 
    arena_destroy(a);
}
 
// A program must cost the arena exactly its own array: the sorting work space lives in a
// scratch arena that is handed back afterwards.
static void test_prog_memory(void) {
    mem_arena* a = arena_create(MiB(4), MiB(1));
    if (!a) { CHECK(a != NULL); return; }
 
    mlp g;
    REQUIRE(build_mlp(a, &g, 4, 6, 5, 3));
 
    u64 pos_before = a->pos;
    model_program p = model_prog_create(a, g.m, g.cost);
    CHECK(p.size == 9);
    CHECK(a->pos == ALIGN_UP_POW2(pos_before, ARENA_ALIGN) + 9 * sizeof(model_var*));
 
    // Same for a program smaller than the whole model (7 of 9 variables): the array must be
    // sized by what is reachable, not by how many variables exist.
    pos_before = a->pos;
    model_program fwd = model_prog_create(a, g.m, g.y);
    CHECK(fwd.size == 7);
    CHECK(a->pos == ALIGN_UP_POW2(pos_before, ARENA_ALIGN) + 7 * sizeof(model_var*));
 
    // Nothing was left behind in the scratch arena either.
    mem_arena_temp s = arena_scratch_get(NULL, 0);
    CHECK(s.arena != NULL);
    CHECK(s.start_pos == ARENA_BASE_POS);
    arena_scratch_release(s);
 
    arena_destroy(a);
}