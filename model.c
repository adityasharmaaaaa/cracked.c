#include <stdlib.h>
#include "timer.h"

// Internal-error check. The constructors validate every shape up front, so if a mat_* call
// still fails while running a program, that is a bug in this file, not bad user input.
// Stop loudly instead of silently producing garbage.
#define MV_CHECK(expr)                                                                   \
    do {                                                                                 \
        if (!(expr)) {                                                                   \
            fprintf(stderr, "model.c:%d: internal error: %s\n", __LINE__, #expr);        \
            abort();                                                                     \
        }                                                                                \
    } while (0)

#define MV_NOT_IMPLEMENTED()                                                             \
    do {                                                                                 \
        fprintf(stderr, "model.c:%d: not implemented yet\n", __LINE__);                  \
        abort();                                                                         \
    } while (0)

// ---------------------------------------------------------------------------
// Building the graph
// ---------------------------------------------------------------------------

// Allocates the variable, its value and (if it requires a gradient) its gradient.
// Does NOT register it with the model: that happens in mv_finish, after everything succeeded.
static model_var* mv_alloc(mem_arena* arena, model_context* model, u32 rows, u32 cols, u32 flags) {
    model_var* var = PUSH_STRUCT(arena, model_var);   // zeroed: op = MV_OP_NULL, inputs = NULL
    if (var == NULL) { return NULL; }

    var->index = model->num_vars;
    var->flags = flags;

    var->val = mat_create(arena, rows, cols);
    if (var->val == NULL) { return NULL; }

    if (flags & MV_FLAG_REQUIRES_GRAD) {
        var->grad = mat_create(arena, rows, cols);
        if (var->grad == NULL) { return NULL; }
    }

    return var;
}

// Records the roles (input / output / ...) of a freshly built variable in the model.
// Each role can be held by only one variable. All roles are checked BEFORE any is assigned,
// so a refused variable changes nothing.
static b32 mv_claim_roles(model_context* model, model_var* var) {
    if ((var->flags & MV_FLAG_INPUT)          && model->input)          { return false; }
    if ((var->flags & MV_FLAG_OUTPUT)         && model->output)         { return false; }
    if ((var->flags & MV_FLAG_DESIRED_OUTPUT) && model->desired_output) { return false; }
    if ((var->flags & MV_FLAG_COST)           && model->cost)           { return false; }

    if (var->flags & MV_FLAG_INPUT)          { model->input = var; }
    if (var->flags & MV_FLAG_OUTPUT)         { model->output = var; }
    if (var->flags & MV_FLAG_DESIRED_OUTPUT) { model->desired_output = var; }
    if (var->flags & MV_FLAG_COST)           { model->cost = var; }
    return true;
}

// Common ending of every constructor. `var` is NULL if allocation failed. On any failure the
// arena is rolled back to `start_pos` and the model is left exactly as it was.
static model_var* mv_finish(mem_arena* arena, model_context* model, u64 start_pos, model_var* var) {
    if (var == NULL || !mv_claim_roles(model, var)) {
        arena_pop_to(arena, start_pos);
        return NULL;
    }
    model->num_vars++;
    return var;
}

// Shared by every op constructor: allocate, propagate REQUIRES_GRAD, record op and inputs.
static model_var* mv_op(
    mem_arena* arena, model_context* model, model_var_op op,
    u32 rows, u32 cols, u32 flags, model_var* a, model_var* b
) {
    u64 start_pos = arena->pos;

    // The result needs a gradient if any input needs one: the gradient has to flow THROUGH
    // this node to reach that input.
    if ((a && (a->flags & MV_FLAG_REQUIRES_GRAD)) || (b && (b->flags & MV_FLAG_REQUIRES_GRAD))) {
        flags |= MV_FLAG_REQUIRES_GRAD;
    }

    model_var* var = mv_alloc(arena, model, rows, cols, flags);
    if (var) {
        var->op = op;
        var->inputs[0] = a;
        var->inputs[1] = b;
    }
    return mv_finish(arena, model, start_pos, var);
}

model_var* mv_create(
    mem_arena* arena, model_context* model,
    u32 rows, u32 cols, u32 flags
) {
    u64 start_pos = arena->pos;

    // A parameter exists to be learned, so it always needs a gradient.
    if (flags & MV_FLAG_PARAMETER) { flags |= MV_FLAG_REQUIRES_GRAD; }

    model_var* var = mv_alloc(arena, model, rows, cols, flags);
    if (var) { var->op = MV_OP_CREATE; }
    return mv_finish(arena, model, start_pos, var);
}

model_var* mv_relu(
    mem_arena* arena, model_context* model,
    model_var* input, u32 flags
) {
    if (input == NULL) { return NULL; }
    return mv_op(arena, model, MV_OP_RELU, input->val->rows, input->val->cols, flags, input, NULL);
}

model_var* mv_softmax(
    mem_arena* arena, model_context* model,
    model_var* input, u32 flags
) {
    if (input == NULL) { return NULL; }
    return mv_op(arena, model, MV_OP_SOFTMAX, input->val->rows, input->val->cols, flags, input, NULL);
}

// True if the two variables' values have exactly the same rows and cols.
static b32 mv_same_shape(const model_var* a, const model_var* b) {
    return a->val->rows == b->val->rows && a->val->cols == b->val->cols;
}

model_var* mv_add(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
) {
    if (a == NULL || b == NULL || !mv_same_shape(a, b)) { return NULL; }
    return mv_op(arena, model, MV_OP_ADD, a->val->rows, a->val->cols, flags, a, b);
}

model_var* mv_sub(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
) {
    if (a == NULL || b == NULL || !mv_same_shape(a, b)) { return NULL; }
    return mv_op(arena, model, MV_OP_SUB, a->val->rows, a->val->cols, flags, a, b);
}

model_var* mv_matmul(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
) {
    if (a == NULL || b == NULL) { return NULL; }
    if (a->val->cols != b->val->rows) { return NULL; }    // inner dimensions must agree
    return mv_op(arena, model, MV_OP_MATMUL, a->val->rows, b->val->cols, flags, a, b);
}

model_var* mv_cross_entropy(
    mem_arena* arena, model_context* model,
    model_var* p, model_var* q, u32 flags
) {
    if (p == NULL || q == NULL || !mv_same_shape(p, q)) { return NULL; }
    return mv_op(arena, model, MV_OP_CROSS_ENTROPY, p->val->rows, p->val->cols, flags, p, q);
}

// ---------------------------------------------------------------------------
// Programs: topological sort by depth-first search, post-order
// ---------------------------------------------------------------------------

typedef struct {
    model_var** order;      // output: variables in dependency order
    u8* visited;            // indexed by model_var::index
    u32 count;
} topo_state;

static void topo_visit(topo_state* st, model_var* var) {
    if (st->visited[var->index]) { return; }       // already placed (a variable used twice appears once)
    st->visited[var->index] = 1;

    // First everything this variable depends on...
    for (u32 i = 0; i < (u32)MV_NUM_INPUTS(var->op); i++) {
        topo_visit(st, var->inputs[i]);
    }

    // ...then the variable itself: post-order puts every variable AFTER all of its inputs.
    st->order[st->count++] = var;
}

model_program model_prog_create(
    mem_arena* arena, model_context* model, model_var* out_var
) {
    model_program prog = { 0 };
    // `index` is the position of the variable in ITS model. If it is not below this model's
    // variable count, the variable cannot belong here (or the model is empty): refuse, rather than
    // index the `visited` array out of bounds. (An out-of-bounds read inside an arena is invisible
    // to AddressSanitizer: it is all one valid allocation.)
    if (out_var == NULL || out_var->index >= model->num_vars) { return prog; }

    // Work space comes from a scratch arena (not `arena`), so the only thing left in `arena`
    // afterwards is the finished program. Tell the scratch system which arena we are using.
    mem_arena_temp scratch = arena_scratch_get(&arena, 1);
    if (scratch.arena == NULL) { return prog; }

    topo_state st = {
        .order   = PUSH_ARRAY(scratch.arena, model_var*, model->num_vars),
        .visited = PUSH_ARRAY(scratch.arena, u8, model->num_vars),     // zeroed: nothing visited
        .count   = 0,
    };

    if (st.order && st.visited) {
        topo_visit(&st, out_var);

        // Exact-size result: only the reachable variables, nothing else.
        model_var** vars = PUSH_ARRAY(arena, model_var*, st.count);
        if (vars) {
            memcpy(vars, st.order, st.count * sizeof(model_var*));
            prog.vars = vars;
            prog.size = st.count;
        }
    }

    arena_scratch_release(scratch);
    return prog;
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

void model_prog_compute(model_program* prog) {
    for (u32 i = 0; i < prog->size; i++) {
        model_var* v = prog->vars[i];
        model_var* a = v->inputs[0];
        model_var* b = v->inputs[1];

        switch (v->op) {
            case MV_OP_NULL:
            case MV_OP_CREATE:
                break;                                  // leaf: holds data the caller put there

            case MV_OP_RELU:
                MV_CHECK(mat_relu(v->val, a->val));
                break;
            case MV_OP_SOFTMAX:
                MV_CHECK(mat_softmax(v->val, a->val));
                break;

            case MV_OP_ADD:
                MV_CHECK(mat_add(v->val, a->val, b->val));
                break;
            case MV_OP_SUB:
                MV_CHECK(mat_sub(v->val, a->val, b->val));
                break;
            case MV_OP_MATMUL:
                // zero_out = true: overwrite the previous step's result, do not add to it.
                MV_CHECK(mat_mul(v->val, a->val, b->val, true, false, false));
                break;
            case MV_OP_CROSS_ENTROPY:
                MV_CHECK(mat_cross_entropy(v->val, a->val, b->val));
                break;

            case _MV_OP_UNARY_START:
            case _MV_OP_BINARY_START:
            default:
                MV_CHECK(!"invalid op in program");
        }
    }
}

// ---------------------------------------------------------------------------
// Backward pass
// ---------------------------------------------------------------------------

void model_prog_compute_grads(model_program* prog) {
    if (prog->size == 0) { return; }

    // 1. Start from zero. Gradients ACCUMULATE (a variable used twice receives two contributions),
    //    so anything left over from the previous step would be added to the new gradient.
    for (u32 i = 0; i < prog->size; i++) {
        model_var* v = prog->vars[i];
        if (v->grad) { mat_clear(v->grad); }
    }

    // 2. Seed. The last variable of a program is the one it computes (post-order puts the root
    //    last). Its gradient is 1 everywhere: we differentiate the SUM of its elements.
    //    (Averaging over a batch is the optimizer's job: scale the update by 1 / batch.)
    model_var* root = prog->vars[prog->size - 1];
    MV_CHECK(root->grad != NULL);       // fails if nothing upstream is a PARAMETER / REQUIRES_GRAD
    mat_fill(root->grad, 1.0f);

    // 3. Walk the program BACKWARDS. Every consumer of a variable comes after it in the program,
    //    so by the time we reach a variable, all contributions to its gradient have arrived and
    //    it is complete. Then it passes its gradient on to its own inputs.
    for (u32 i = prog->size; i-- > 0; ) {
        model_var* v = prog->vars[i];

        // A variable without a gradient has no input that needs one either: REQUIRES_GRAD
        // propagates forward through every op, so there is nothing to pass on.
        if (v->grad == NULL) { continue; }

        model_var* a = v->inputs[0];
        model_var* b = v->inputs[1];

        switch (v->op) {
            case MV_OP_NULL:
            case MV_OP_CREATE:
                break;                                  // a leaf: its gradient is the final answer

            case MV_OP_RELU:
                if (a->grad) { MV_CHECK(mat_relu_add_grad(a->grad, a->val, v->grad)); }
                break;

            case MV_OP_SOFTMAX:
                // The softmax gradient needs the softmax OUTPUT, which is v's own value.
                if (a->grad) { MV_CHECK(mat_softmax_add_grad(a->grad, v->val, v->grad)); }
                break;

            case MV_OP_ADD:
                // d(a + b) = da + db: the gradient passes through unchanged to both inputs.
                if (a->grad) { MV_CHECK(mat_add(a->grad, a->grad, v->grad)); }
                if (b->grad) { MV_CHECK(mat_add(b->grad, b->grad, v->grad)); }
                break;

            case MV_OP_SUB:
                // d(a - b) = da - db.
                if (a->grad) { MV_CHECK(mat_add(a->grad, a->grad, v->grad)); }
                if (b->grad) { MV_CHECK(mat_sub(b->grad, b->grad, v->grad)); }
                break;

            case MV_OP_MATMUL:
                // C = A * B:   dA += dC * B^T   and   dB += A^T * dC.
                // zero_out = false: ACCUMULATE into the existing gradient.
                if (a->grad) { MV_CHECK(mat_mul(a->grad, v->grad, b->val, false, false, true)); }
                if (b->grad) { MV_CHECK(mat_mul(b->grad, a->val, v->grad, false, true, false)); }
                break;

            case MV_OP_CROSS_ENTROPY:
                // Either gradient may be NULL (the labels need none).
                if (a->grad || b->grad) {
                    MV_CHECK(mat_cross_entropy_add_grad(a->grad, b->grad, a->val, b->val, v->grad));
                }
                break;

            case _MV_OP_UNARY_START:
            case _MV_OP_BINARY_START:
            default:
                MV_CHECK(!"invalid op in program");
        }
    }
}

// ---------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------

model_context* model_create(mem_arena* arena) {
    return PUSH_STRUCT(arena, model_context);          // zeroed
}

void model_compile(mem_arena* arena, model_context* model) {
    if (model->output) {
        model->forward_prog = model_prog_create(arena, model, model->output);
    }
    if (model->cost) {
        model->cost_prog = model_prog_create(arena, model, model->cost);
    }
}

void model_feedforward(model_context* model) {
    MV_CHECK(model->forward_prog.size > 0);            // forgot model_compile(), or no output set
    model_prog_compute(&model->forward_prog);
}

void model_sgd_step(model_program* prog, f32 step) {
    for (u32 i = 0; i < prog->size; i++) {
        model_var* v = prog->vars[i];
        if (!(v->flags & MV_FLAG_PARAMETER)) { continue; }   // only parameters are learned

        MV_CHECK(v->grad != NULL);
        u64 n = (u64)v->val->rows * v->val->cols;
        for (u64 j = 0; j < n; j++) {
            v->val->data[j] -= step * v->grad->data[j];
        }
    }
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

static u32 row_argmax(const f32* row, u32 n) {
    u32 best = 0;
    for (u32 c = 1; c < n; c++) {
        if (row[c] > row[best]) { best = c; }                // ties: the first maximum wins
    }
    return best;
}

f32 model_accuracy(model_context* model, const matrix* images, const matrix* labels) {
    MV_CHECK(model->input && model->output && model->forward_prog.size > 0);
    matrix* in  = model->input->val;
    matrix* out = model->output->val;
    MV_CHECK(images->cols == in->cols);
    MV_CHECK(labels->cols == out->cols);
    MV_CHECK(images->rows == labels->rows);

    if (images->rows == 0) { return 0.0f; }

    u32 batch = in->rows;
    u64 correct = 0;

    for (u32 start = 0; start < images->rows; start += batch) {
        // The last batch may be partial: load only the rows that exist and count only those.
        // (The remaining rows of the input keep old data, which is harmless: rows never mix.)
        u32 count = MIN(batch, images->rows - start);
        memcpy(in->data, images->data + (u64)start * in->cols, (u64)count * in->cols * sizeof(f32));

        model_prog_compute(&model->forward_prog);

        for (u32 r = 0; r < count; r++) {
            u32 predicted = row_argmax(out->data + (u64)r * out->cols, out->cols);
            u32 actual    = row_argmax(labels->data + (u64)(start + r) * labels->cols, labels->cols);
            if (predicted == actual) { correct++; }
        }
    }
    return (f32)((f64)correct / (f64)images->rows);
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

// Fisher-Yates: walk from the end, swap each position with a random one at or before it. Every
// one of the n! orderings is equally likely. (The tempting version, swapping each position with
// a random position ANYWHERE, is subtly biased.)
// `prng_rand() % k` is very slightly biased too, by about k / 2^32: negligible for a dataset size.
static void shuffle_indices(u32* indices, u32 n) {
    if (n < 2) { return; }
    for (u32 i = n - 1; i > 0; i--) {
        u32 j = prng_rand() % (i + 1);
        u32 tmp = indices[i];
        indices[i] = indices[j];
        indices[j] = tmp;
    }
}

void model_train(
    model_context* model,
    const model_training_desc* desc
) {
    MV_CHECK(model->input && model->desired_output && model->output && model->cost);
    MV_CHECK(model->cost_prog.size > 0 && model->forward_prog.size > 0);   // forgot model_compile()
    MV_CHECK(desc->train_images && desc->train_labels);

    matrix* in  = model->input->val;
    matrix* lab = model->desired_output->val;
    u32 batch = in->rows;

    MV_CHECK(desc->batch_size == batch);                       // the graph's shapes are fixed
    MV_CHECK(desc->train_images->cols == in->cols);
    MV_CHECK(desc->train_labels->cols == lab->cols);
    MV_CHECK(desc->train_images->rows == desc->train_labels->rows);
    MV_CHECK(desc->train_images->rows >= batch);
    if (desc->test_images) {
        MV_CHECK(desc->test_labels);
        MV_CHECK(desc->test_images->cols == in->cols && desc->test_labels->cols == lab->cols);
        MV_CHECK(desc->test_images->rows == desc->test_labels->rows);
    }

    u32 n = desc->train_images->rows;
    u32 num_batches = n / batch;
    f32 step = desc->learning_rate / (f32)batch;

    // The shuffled order of the samples lives in a scratch arena: no allocation of our own.
    mem_arena_temp scratch = arena_scratch_get(NULL, 0);
    MV_CHECK(scratch.arena != NULL);
    u32* order = PUSH_ARRAY(scratch.arena, u32, n);
    MV_CHECK(order != NULL);
    for (u32 i = 0; i < n; i++) { order[i] = i; }

    for (u32 epoch = 0; epoch < desc->epochs; epoch++) {
        u64 start_ns = timer_ns();
        shuffle_indices(order, n);

        f64 loss_sum = 0.0;
        for (u32 b = 0; b < num_batches; b++) {
            // Gather this batch's samples (scattered across the dataset) into the graph's inputs.
            for (u32 r = 0; r < batch; r++) {
                u64 src = order[(u64)b * batch + r];
                memcpy(in->data  + (u64)r * in->cols,  desc->train_images->data + src * in->cols,  in->cols  * sizeof(f32));
                memcpy(lab->data + (u64)r * lab->cols, desc->train_labels->data + src * lab->cols, lab->cols * sizeof(f32));
            }

            model_prog_compute(&model->cost_prog);
            loss_sum += (f64)mat_sum(model->cost->val) / (f64)batch;     // mean loss of this batch

            model_prog_compute_grads(&model->cost_prog);
            model_sgd_step(&model->cost_prog, step);
        }

        f64 seconds = (f64)(timer_ns() - start_ns) / 1e9;
        printf("epoch %2u/%u  train loss %.4f", epoch + 1, desc->epochs, loss_sum / (f64)num_batches);
        if (desc->test_images) {
            f32 accuracy = model_accuracy(model, desc->test_images, desc->test_labels);
            printf("  test accuracy %.2f%%", 100.0 * (f64)accuracy);
        }
        printf("  (%.1f s)\n", seconds);
        fflush(stdout);
    }

    arena_scratch_release(scratch);
}