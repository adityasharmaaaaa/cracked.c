// tests/test_model.c
//
// Graph construction, topological sort, forward pass.
// Build and run with:  make test

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
#include "gradcheck.h"

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

static b32 build_mlp_ex(mem_arena* arena, mlp* g, u32 batch, u32 in, u32 hid, u32 out, u32 x_flags) {
    g->m      = model_create(arena);
    if (!g->m) { return false; }
    g->x      = mv_create(arena, g->m, batch, in,  x_flags);
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

static b32 build_mlp(mem_arena* arena, mlp* g, u32 batch, u32 in, u32 hid, u32 out) {
    return build_mlp_ex(arena, g, batch, in, hid, out, MV_FLAG_INPUT);
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

// ---------------------------------------------------------------------------
// Shape checking in the op constructors
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Topological sort: exact order, properties, pruning, fan-out
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------
static void test_forward_known(void) {
    mem_arena* a = arena_create(MiB(4), MiB(1));
    if (!a) { CHECK(a != NULL); return; }

    // The tiny network from the notes, with values worked out independently in Python:
    //   x = [1, 2]    w1 = [[1, 0], [0, -1]]    w2 = [[1, 2], [3, 4]]    labels = [0, 1]
    //   h = [1, -2]   a = relu(h) = [1, 0]      z = [1, 2]    y = softmax(z) = [0.2689414, 0.7310586]
    //   cost = -labels * ln(y) = [0, 0.3132617]
    mlp g;
    REQUIRE(build_mlp(a, &g, 1, 2, 2, 2));
    f32 xv[]  = { 1, 2 };
    f32 w1v[] = { 1, 0, 0, -1 };
    f32 w2v[] = { 1, 2, 3, 4 };
    f32 lv[]  = { 0, 1 };
    fill_from(g.x->val, xv, 2);
    fill_from(g.w1->val, w1v, 4);
    fill_from(g.w2->val, w2v, 4);
    fill_from(g.labels->val, lv, 2);

    model_compile(a, g.m);
    model_feedforward(g.m);

    CHECK_NEAR(g.h->val->data[0],  1.0, 1e-6);
    CHECK_NEAR(g.h->val->data[1], -2.0, 1e-6);
    CHECK_NEAR(g.a->val->data[0],  1.0, 1e-6);
    CHECK_NEAR(g.a->val->data[1],  0.0, 1e-6);       // relu clipped the negative
    CHECK_NEAR(g.z->val->data[0],  1.0, 1e-6);
    CHECK_NEAR(g.z->val->data[1],  2.0, 1e-6);
    CHECK_NEAR(g.y->val->data[0],  0.2689414, 1e-6);
    CHECK_NEAR(g.y->val->data[1],  0.7310586, 1e-6);

    // feedforward runs the FORWARD program only: the cost has not been computed yet.
    CHECK(all_zero(g.cost->val));

    // Running the cost program computes it.
    model_prog_compute(&g.m->cost_prog);
    CHECK_NEAR(g.cost->val->data[0], 0.0,       1e-6);
    CHECK_NEAR(g.cost->val->data[1], 0.3132617, 1e-6);

    // New input, run again: results must REPLACE the old ones, not add to them (a matmul that
    // accumulated instead of overwriting would show stale values here).
    //   x = [2, 1]: h = [2, -1], a = [2, 0], z = [2, 4], y = [0.1192029, 0.8807971], cost[1] = 0.126928
    f32 x2[] = { 2, 1 };
    fill_from(g.x->val, x2, 2);
    model_feedforward(g.m);
    CHECK_NEAR(g.h->val->data[0],  2.0, 1e-6);
    CHECK_NEAR(g.z->val->data[1],  4.0, 1e-6);
    CHECK_NEAR(g.y->val->data[0],  0.1192029, 1e-6);
    CHECK_NEAR(g.y->val->data[1],  0.8807971, 1e-6);
    model_prog_compute(&g.m->cost_prog);
    CHECK_NEAR(g.cost->val->data[1], 0.126928, 1e-6);

    arena_destroy(a);
}

static void test_forward_ops(void) {
    mem_arena* a = arena_create(MiB(4), MiB(1));
    if (!a) { CHECK(a != NULL); return; }
    model_context* m = model_create(a);

    model_var* x = mv_create(a, m, 1, 3, MV_FLAG_NONE);
    model_var* y = mv_create(a, m, 1, 3, MV_FLAG_NONE);
    model_var* sum = mv_add(a, m, x, y, MV_FLAG_NONE);
    model_var* dif = mv_sub(a, m, x, y, MV_FLAG_NONE);
    f32 xv[] = { 5, 6, 7 };
    f32 yv[] = { 1, 2, 3 };
    fill_from(x->val, xv, 3);
    fill_from(y->val, yv, 3);

    // Programs can be built and run for any variable, independently of model_compile.
    model_program ps = model_prog_create(a, m, sum);
    model_prog_compute(&ps);
    CHECK(sum->val->data[0] == 6 && sum->val->data[1] == 8 && sum->val->data[2] == 10);
    CHECK(all_zero(dif->val));                        // not part of ps, so untouched

    model_program pd = model_prog_create(a, m, dif);
    model_prog_compute(&pd);
    CHECK(dif->val->data[0] == 4 && dif->val->data[1] == 4 && dif->val->data[2] == 4);

    arena_destroy(a);
}

// A batch of samples must give, row by row, the same answers as running each sample alone.
static void test_forward_batch(void) {
    mem_arena* a = arena_create(MiB(8), MiB(1));
    if (!a) { CHECK(a != NULL); return; }

    u32 batch = 5, in = 6, hid = 4, out = 3;
    mlp big, one;
    REQUIRE(build_mlp(a, &big, batch, in, hid, out));
    REQUIRE(build_mlp(a, &one, 1, in, hid, out));

    prng_seed(5, 5);
    mat_fill_rand(big.w1->val, -1.0f, 1.0f);
    mat_fill_rand(big.w2->val, -1.0f, 1.0f);
    mat_fill_rand(big.x->val, -1.0f, 1.0f);
    mat_copy(one.w1->val, big.w1->val);
    mat_copy(one.w2->val, big.w2->val);

    model_compile(a, big.m);
    model_compile(a, one.m);
    model_feedforward(big.m);

    b32 rows_match = true;
    for (u32 r = 0; r < batch; r++) {
        memcpy(one.x->val->data, big.x->val->data + (u64)r * in, in * sizeof(f32));
        model_feedforward(one.m);

        for (u32 c = 0; c < out; c++) {
            f32 from_batch  = big.y->val->data[(u64)r * out + c];
            f32 from_single = one.y->val->data[c];
            if (fabsf(from_batch - from_single) > 1e-6f) { rows_match = false; }
        }
    }
    CHECK(rows_match);

    // And every row of the output is a probability distribution.
    for (u32 r = 0; r < batch; r++) {
        f32 sum = 0.0f;
        for (u32 c = 0; c < out; c++) { sum += big.y->val->data[(u64)r * out + c]; }
        CHECK_NEAR(sum, 1.0, 1e-5);
    }

    arena_destroy(a);
}

// ---------------------------------------------------------------------------
// Backward pass
// ---------------------------------------------------------------------------

static f64 sum_f64(const matrix* m) {
    f64 s = 0.0;
    u64 n = (u64)m->rows * m->cols;
    for (u64 i = 0; i < n; i++) { s += m->data[i]; }
    return s;
}

// Loss for finite differences: run the whole cost program, return the SUM of the cost matrix.
// (The backward pass differentiates exactly this: it seeds the cost with 1 everywhere.)
static f64 cost_sum_loss(void* ctx) {
    model_context* m = ctx;
    model_prog_compute(&m->cost_prog);
    return sum_f64(m->cost->val);
}

static void set_one_hot_labels(matrix* labels) {
    mat_clear(labels);
    for (u32 r = 0; r < labels->rows; r++) {
        labels->data[(u64)r * labels->cols + (r * 2 + 1) % labels->cols] = 1.0f;
    }
}

// True if every hidden pre-activation is at least `margin` away from 0. ReLU has a kink at 0,
// and a finite-difference step that straddles it measures nonsense, so we only test networks
// where no unit is near the kink.
static b32 relu_margin_ok(const mlp* g, f32 margin) {
    u64 n = (u64)g->h->val->rows * g->h->val->cols;
    for (u64 i = 0; i < n; i++) {
        if (fabsf(g->h->val->data[i]) < margin) { return false; }
    }
    return true;
}

// Random weights and inputs, one-hot labels; resamples until no ReLU unit sits near its kink.
static b32 randomize_mlp(mlp* g, u64 seed) {
    for (u32 attempt = 0; attempt < 100; attempt++) {
        prng_seed(seed + attempt, 17);
        mat_fill_rand(g->w1->val, -1.0f, 1.0f);
        mat_fill_rand(g->w2->val, -1.0f, 1.0f);
        mat_fill_rand(g->x->val, -1.0f, 1.0f);
        set_one_hot_labels(g->labels->val);
        model_prog_compute(&g->m->cost_prog);
        if (relu_margin_ok(g, 0.05f)) { return true; }
    }
    return false;
}

// Hand-derived (and numpy-checked) gradients for the tiny network from the forward test.
static void test_backward_known(void) {
    mem_arena* a = arena_create(MiB(4), MiB(1));
    if (!a) { CHECK(a != NULL); return; }

    mlp g;
    REQUIRE(build_mlp(a, &g, 1, 2, 2, 2));
    f32 xv[]  = { 1, 2 };
    f32 w1v[] = { 1, 0, 0, -1 };
    f32 w2v[] = { 1, 2, 3, 4 };
    f32 lv[]  = { 0, 1 };
    fill_from(g.x->val, xv, 2);
    fill_from(g.w1->val, w1v, 4);
    fill_from(g.w2->val, w2v, 4);
    fill_from(g.labels->val, lv, 2);
    model_compile(a, g.m);

    model_prog_compute(&g.m->cost_prog);
    model_prog_compute_grads(&g.m->cost_prog);

    // Only the parameters carry gradients; the input and the labels do not.
    CHECK(g.x->grad == NULL);
    CHECK(g.labels->grad == NULL);
    REQUIRE(g.w1->grad != NULL && g.w2->grad != NULL);

    // Final answers (numpy): dL/dw1 = [[-0.2689414, 0], [-0.5378828, 0]], dL/dw2 = [[0.2689414, -0.2689414], [0, 0]].
    CHECK_NEAR(g.w1->grad->data[0], -0.2689414, 1e-6);
    CHECK_NEAR(g.w1->grad->data[1],  0.0,       1e-6);
    CHECK_NEAR(g.w1->grad->data[2], -0.5378828, 1e-6);
    CHECK_NEAR(g.w1->grad->data[3],  0.0,       1e-6);
    CHECK_NEAR(g.w2->grad->data[0],  0.2689414, 1e-6);
    CHECK_NEAR(g.w2->grad->data[1], -0.2689414, 1e-6);
    CHECK_NEAR(g.w2->grad->data[2],  0.0,       1e-6);
    CHECK_NEAR(g.w2->grad->data[3],  0.0,       1e-6);

    // And the gradients at every step along the way (all from the same hand derivation):
    CHECK_NEAR(g.cost->grad->data[0], 1.0, 0.0);              // seed: 1 everywhere
    CHECK_NEAR(g.cost->grad->data[1], 1.0, 0.0);
    CHECK_NEAR(g.y->grad->data[0],  0.0,        1e-6);        // -p / y  = [0, -1/0.7310586]
    CHECK_NEAR(g.y->grad->data[1], -1.3678794,  1e-5);
    CHECK_NEAR(g.z->grad->data[0],  0.2689414,  1e-6);        // y - p
    CHECK_NEAR(g.z->grad->data[1], -0.2689414,  1e-6);
    CHECK_NEAR(g.a->grad->data[0], -0.2689414,  1e-6);        // dz * w2^T
    CHECK_NEAR(g.a->grad->data[1], -0.2689414,  1e-6);
    CHECK_NEAR(g.h->grad->data[0], -0.2689414,  1e-6);        // through relu: h[0] = 1 > 0 passes
    CHECK_NEAR(g.h->grad->data[1],  0.0,        1e-6);        // h[1] = -2 < 0 is blocked

    // The backward pass must not disturb the forward values.
    CHECK_NEAR(g.y->val->data[0], 0.2689414, 1e-6);
    CHECK_NEAR(g.y->val->data[1], 0.7310586, 1e-6);

    arena_destroy(a);
}

// Gradients are zeroed at the start of every backward pass: running it again, or running it
// after the gradients were polluted, must give the same answer, not a sum.
static void test_backward_zeroing(void) {
    mem_arena* a = arena_create(MiB(4), MiB(1));
    if (!a) { CHECK(a != NULL); return; }

    mlp g;
    REQUIRE(build_mlp(a, &g, 4, 6, 5, 3));
    model_compile(a, g.m);
    REQUIRE(randomize_mlp(&g, 100));

    model_prog_compute_grads(&g.m->cost_prog);
    matrix* first = mat_create(a, g.w1->grad->rows, g.w1->grad->cols);
    mat_copy(first, g.w1->grad);

    model_prog_compute_grads(&g.m->cost_prog);                // again: must NOT double
    b32 same = true;
    for (u32 i = 0; i < 6 * 5; i++) { if (g.w1->grad->data[i] != first->data[i]) { same = false; } }
    CHECK(same);

    mat_fill(g.w1->grad, 1000.0f);                            // pollute every gradient
    mat_fill(g.w2->grad, 1000.0f);
    mat_fill(g.h->grad, 1000.0f);
    model_prog_compute_grads(&g.m->cost_prog);
    same = true;
    for (u32 i = 0; i < 6 * 5; i++) { if (g.w1->grad->data[i] != first->data[i]) { same = false; } }
    CHECK(same);

    arena_destroy(a);
}

// Fan-out and shared variables: the cases where gradients must ACCUMULATE.
static void test_backward_fanout(void) {
    mem_arena* a = arena_create(MiB(4), MiB(1));
    if (!a) { CHECK(a != NULL); return; }

    // (a) w is used twice: out = relu(w) + w.   d/dw = relu'(w) + 1 = 2 where w > 0, 1 where w < 0.
    {
        model_context* m = model_create(a);
        model_var* w = mv_create(a, m, 2, 3, MV_FLAG_PARAMETER);
        model_var* r = mv_relu(a, m, w, MV_FLAG_NONE);
        model_var* s = mv_add(a, m, r, w, MV_FLAG_COST);
        REQUIRE(w && r && s);
        f32 wv[] = { 1, -2, 3, -4, 5, -6 };
        fill_from(w->val, wv, 6);
        model_compile(a, m);
        model_prog_compute(&m->cost_prog);
        model_prog_compute_grads(&m->cost_prog);
        f32 expected[] = { 2, 1, 2, 1, 2, 1 };
        b32 ok = true;
        for (u32 i = 0; i < 6; i++) { if (w->grad->data[i] != expected[i]) { ok = false; } }
        CHECK(ok);
    }

    // (b) The SAME variable as both inputs: out = w + w  ->  d/dw = 2.
    {
        model_context* m = model_create(a);
        model_var* w = mv_create(a, m, 2, 2, MV_FLAG_PARAMETER);
        model_var* s = mv_add(a, m, w, w, MV_FLAG_COST);
        REQUIRE(w && s);
        model_compile(a, m);
        model_prog_compute(&m->cost_prog);
        model_prog_compute_grads(&m->cost_prog);
        CHECK(w->grad->data[0] == 2.0f && w->grad->data[3] == 2.0f);
    }

    // (c) Subtraction: out = u - v  ->  d/du = +1, d/dv = -1.
    {
        model_context* m = model_create(a);
        model_var* u = mv_create(a, m, 1, 3, MV_FLAG_PARAMETER);
        model_var* v = mv_create(a, m, 1, 3, MV_FLAG_PARAMETER);
        model_var* s = mv_sub(a, m, u, v, MV_FLAG_COST);
        REQUIRE(u && v && s);
        model_compile(a, m);
        model_prog_compute(&m->cost_prog);
        model_prog_compute_grads(&m->cost_prog);
        CHECK(u->grad->data[0] == 1.0f && u->grad->data[2] == 1.0f);
        CHECK(v->grad->data[0] == -1.0f && v->grad->data[2] == -1.0f);
    }

    // (d) A matrix multiplied by itself: C = W * W, L = sum(C).
    //     dL/dW = dC * W^T + W^T * dC with dC = all ones  =  [[7, 11], [9, 13]]  (numpy).
    //     W feeds BOTH sides of the matmul, so both gradient formulas land in the same matrix.
    {
        model_context* m = model_create(a);
        model_var* w = mv_create(a, m, 2, 2, MV_FLAG_PARAMETER);
        model_var* c = mv_matmul(a, m, w, w, MV_FLAG_COST);
        REQUIRE(w && c);
        f32 wv[] = { 1, 2, 3, 4 };
        fill_from(w->val, wv, 4);
        model_compile(a, m);
        model_prog_compute(&m->cost_prog);
        model_prog_compute_grads(&m->cost_prog);
        f32 expected[] = { 7, 11, 9, 13 };
        b32 ok = true;
        for (u32 i = 0; i < 4; i++) { if (w->grad->data[i] != expected[i]) { ok = false; } }
        CHECK(ok);
    }

    // (f) w is the LEFT operand of a matmul AND is also used by a relu that the backward pass
    //     handles FIRST. The matmul's contribution must be ADDED to what relu already delivered,
    //     not overwrite it.   cost = (w * u) + relu(w),  u = identity, w > 0:
    //     dL/dw = ones * u^T + relu'(w) = ones + ones = 2 everywhere.
    {
        model_context* m = model_create(a);
        model_var* w = mv_create(a, m, 2, 2, MV_FLAG_PARAMETER);
        model_var* u = mv_create(a, m, 2, 2, MV_FLAG_INPUT);
        model_var* mm = mv_matmul(a, m, w, u, MV_FLAG_NONE);
        model_var* r = mv_relu(a, m, w, MV_FLAG_NONE);
        model_var* s = mv_add(a, m, mm, r, MV_FLAG_COST);
        REQUIRE(w && u && mm && r && s);
        f32 wv[] = { 1, 2, 3, 4 };
        f32 uv[] = { 1, 0, 0, 1 };
        fill_from(w->val, wv, 4);
        fill_from(u->val, uv, 4);
        model_compile(a, m);
        model_prog_compute(&m->cost_prog);
        model_prog_compute_grads(&m->cost_prog);
        CHECK(w->grad->data[0] == 2.0f && w->grad->data[1] == 2.0f);
        CHECK(w->grad->data[2] == 2.0f && w->grad->data[3] == 2.0f);
    }

    // (e) A branch that needs no gradient sits inside the program: out = w + relu(x), x is plain
    //     input. relu(x) has no gradient and must be skipped without touching anything.
    {
        model_context* m = model_create(a);
        model_var* w = mv_create(a, m, 1, 3, MV_FLAG_PARAMETER);
        model_var* x = mv_create(a, m, 1, 3, MV_FLAG_INPUT);
        model_var* r = mv_relu(a, m, x, MV_FLAG_NONE);
        model_var* s = mv_add(a, m, w, r, MV_FLAG_COST);
        REQUIRE(w && x && r && s);
        model_compile(a, m);
        model_prog_compute(&m->cost_prog);
        model_prog_compute_grads(&m->cost_prog);
        CHECK(r->grad == NULL && x->grad == NULL);
        CHECK(w->grad->data[0] == 1.0f && w->grad->data[2] == 1.0f);
    }

    arena_destroy(a);
}

// The big one: for whole networks, backprop must agree with measured derivatives of the loss
// for EVERY parameter (and the input). Shapes are deliberately all different and non-square:
// a square network would hide a transposed matmul.
static void test_backward_finite_differences(void) {
    u32 configs[][4] = {            // batch, in, hidden, out
        { 4, 6, 5, 3 },
        { 1, 3, 8, 2 },
        { 7, 5, 2, 4 },
        { 3, 9, 4, 6 },
    };
    f64 worst = 0.0;

    for (u32 c = 0; c < sizeof(configs) / sizeof(configs[0]); c++) {
        u32 batch = configs[c][0], in = configs[c][1], hid = configs[c][2], out = configs[c][3];

        mem_arena* a = arena_create(MiB(8), MiB(1));
        if (!a) { CHECK(a != NULL); return; }

        mlp g;
        // The input also gets a gradient here, so dL/dx is checked as well.
        REQUIRE(build_mlp_ex(a, &g, batch, in, hid, out, MV_FLAG_INPUT | MV_FLAG_REQUIRES_GRAD));
        model_compile(a, g.m);
        REQUIRE(randomize_mlp(&g, 1000 + 10 * c));

        model_prog_compute(&g.m->cost_prog);
        model_prog_compute_grads(&g.m->cost_prog);

        model_var* params[] = { g.w1, g.w2, g.x };
        for (u32 i = 0; i < 3; i++) {
            model_var* p = params[i];
            REQUIRE(p->grad != NULL);

            matrix* numeric = mat_create(a, p->val->rows, p->val->cols);
            gc_numeric_grad(p->val, cost_sum_loss, g.m, GC_H, numeric);
            f64 err = gc_max_error(p->grad, numeric);
            if (err > worst) { worst = err; }
            CHECK(err < GC_TOLERANCE);
        }

        arena_destroy(a);
    }
    printf("    whole-network gradient check (4 shapes, w1 + w2 + input): worst error = %.2e\n", worst);
}

// Both inputs of a cross-entropy node can need gradients (the labels usually do not).
static void test_backward_cross_entropy_both(void) {
    mem_arena* a = arena_create(MiB(4), MiB(1));
    if (!a) { CHECK(a != NULL); return; }

    model_context* m = model_create(a);
    model_var* z    = mv_create(a, m, 3, 4, MV_FLAG_PARAMETER);
    model_var* q    = mv_softmax(a, m, z, MV_FLAG_OUTPUT);
    model_var* p    = mv_create(a, m, 3, 4, MV_FLAG_PARAMETER);       // a "label" that needs a gradient
    model_var* cost = mv_cross_entropy(a, m, p, q, MV_FLAG_COST);
    REQUIRE(z && q && p && cost);
    model_compile(a, m);

    prng_seed(77, 7);
    mat_fill_rand(z->val, -1.0f, 1.0f);
    mat_fill_rand(p->val, 0.1f, 0.9f);

    model_prog_compute(&m->cost_prog);
    model_prog_compute_grads(&m->cost_prog);

    f64 worst = 0.0;
    model_var* both[] = { z, p };
    for (u32 i = 0; i < 2; i++) {
        matrix* numeric = mat_create(a, both[i]->val->rows, both[i]->val->cols);
        gc_numeric_grad(both[i]->val, cost_sum_loss, m, GC_H, numeric);
        f64 err = gc_max_error(both[i]->grad, numeric);
        if (err > worst) { worst = err; }
        CHECK(err < GC_TOLERANCE);
    }
    printf("    cross-entropy with gradients on both p and q: worst error = %.2e\n", worst);

    // Only p needs a gradient; q is a constant (no gradient at all). The p gradient is still
    // computed: dL/dp = -ln(q).
    {
        model_context* m2 = model_create(a);
        model_var* p2 = mv_create(a, m2, 1, 3, MV_FLAG_PARAMETER);
        model_var* q2 = mv_create(a, m2, 1, 3, MV_FLAG_INPUT);
        model_var* c2 = mv_cross_entropy(a, m2, p2, q2, MV_FLAG_COST);
        REQUIRE(p2 && q2 && c2);
        f32 pv[] = { 0.5f, 0.25f, 0.25f };
        f32 qv[] = { 0.2f, 0.7f, 0.1f };
        fill_from(p2->val, pv, 3);
        fill_from(q2->val, qv, 3);
        model_compile(a, m2);
        model_prog_compute(&m2->cost_prog);
        model_prog_compute_grads(&m2->cost_prog);
        CHECK(q2->grad == NULL);
        CHECK_NEAR(p2->grad->data[0], 1.6094379, 1e-6);       // -ln(0.2)
        CHECK_NEAR(p2->grad->data[1], 0.3566749, 1e-6);       // -ln(0.7)
        CHECK_NEAR(p2->grad->data[2], 2.3025851, 1e-6);       // -ln(0.1)
    }

    arena_destroy(a);
}

// Plain SGD: for every parameter, w -= step * grad. `step` is learning_rate / batch_size: the
// backward pass differentiates the SUM of the cost, so dividing by the batch size turns it into
// the gradient of the MEAN loss, and the learning rate no longer depends on the batch size.
static void sgd_step(model_program* prog, f32 step) {
    for (u32 i = 0; i < prog->size; i++) {
        model_var* v = prog->vars[i];
        if (!(v->flags & MV_FLAG_PARAMETER)) { continue; }

        u64 n = (u64)v->val->rows * v->val->cols;
        for (u64 j = 0; j < n; j++) {
            v->val->data[j] -= step * v->grad->data[j];
        }
    }
}

// The classic end-to-end sanity check: a correct network must be able to memorise 8 samples.
// If forward, backward, or the update were wrong anywhere, the loss would not go to ~0.
static void test_overfit_tiny_batch(void) {
    mem_arena* a = arena_create(MiB(8), MiB(1));
    if (!a) { CHECK(a != NULL); return; }

    u32 batch = 8, in = 6, hid = 16, out = 3;
    mlp g;
    REQUIRE(build_mlp(a, &g, batch, in, hid, out));
    model_compile(a, g.m);

    prng_seed(2024, 9);
    mat_fill_rand(g.x->val, -1.0f, 1.0f);
    mat_fill_rand(g.w1->val, -0.5f, 0.5f);
    mat_fill_rand(g.w2->val, -0.5f, 0.5f);
    for (u32 r = 0; r < batch; r++) {
        g.labels->val->data[r * out + (prng_rand() % out)] = 1.0f;     // random class per sample
    }

    f32 loss_at[301];
    for (u32 step = 0; step <= 300; step++) {
        model_prog_compute(&g.m->cost_prog);
        loss_at[step] = mat_sum(g.cost->val) / (f32)batch;             // mean loss over the batch

        model_prog_compute_grads(&g.m->cost_prog);
        sgd_step(&g.m->cost_prog, 0.3f / (f32)batch);
    }

    // An untrained network knows nothing: loss ~ ln(number of classes) = ln 3 = 1.0986.
    CHECK(fabsf(loss_at[0] - logf((f32)out)) < 0.15f);

    // It then improves steadily...
    CHECK(loss_at[10] < loss_at[0]);
    CHECK(loss_at[50] < loss_at[10]);
    CHECK(loss_at[100] < loss_at[50]);
    printf("    loss: step 0 = %.4f, 10 = %.4f, 50 = %.4f, 100 = %.4f, 300 = %.4f\n",
           loss_at[0], loss_at[10], loss_at[50], loss_at[100], loss_at[300]);

    // ...and memorises the batch.
    CHECK(loss_at[300] < 0.01f);

    u32 correct = 0;
    for (u32 r = 0; r < batch; r++) {
        u32 best = 0;
        for (u32 c = 1; c < out; c++) {
            if (g.y->val->data[r * out + c] > g.y->val->data[r * out + best]) { best = c; }
        }
        if (g.labels->val->data[r * out + best] == 1.0f) { correct++; }
    }
    CHECK(correct == batch);

    arena_destroy(a);
}

// ---------------------------------------------------------------------------

int main(void) {
    RUN_TEST(test_mv_create);
    RUN_TEST(test_roles);
    RUN_TEST(test_mv_shapes);
    RUN_TEST(test_requires_grad);
    RUN_TEST(test_prog_order);
    RUN_TEST(test_prog_shapes_of_graphs);
    RUN_TEST(test_prog_memory);
    RUN_TEST(test_forward_known);
    RUN_TEST(test_forward_ops);
    RUN_TEST(test_forward_batch);
    RUN_TEST(test_backward_known);
    RUN_TEST(test_backward_zeroing);
    RUN_TEST(test_backward_fanout);
    RUN_TEST(test_backward_finite_differences);
    RUN_TEST(test_backward_cross_entropy_both);
    RUN_TEST(test_overfit_tiny_batch);

    return test_summary();
}
