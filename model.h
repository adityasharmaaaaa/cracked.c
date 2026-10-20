// model.h
//
// A static computation graph with automatic differentiation.
//
// You build the graph ONCE with mv_* calls. Each call creates a variable, allocates its value
// (and gradient) matrix from the arena, and records which op produced it and from which inputs.
// Nothing is allocated again while training: the same matrices are overwritten every step.
//
//     model_context* m = model_create(arena);
//     model_var* x  = mv_create(arena, m, batch, 784, MV_FLAG_INPUT);
//     model_var* w1 = mv_create(arena, m, 784, 256, MV_FLAG_PARAMETER);
//     model_var* h  = mv_relu(arena, m, mv_matmul(arena, m, x, w1, MV_FLAG_NONE), MV_FLAG_NONE);
//     ...
//     model_compile(arena, m);        // topologically sort the graph into programs
//     model_feedforward(m);           // run the forward program on whatever is in m->input->val
//
// Rules:
//   * A variable can only use inputs that already exist, so the graph is a DAG by construction.
//   * Every constructor checks shapes. On any failure it returns NULL, leaves the model unchanged
//     (num_vars, roles) and gives back the arena space it had taken.
//   * The result of an op requires a gradient if ANY of its inputs does.
//     MV_FLAG_PARAMETER implies MV_FLAG_REQUIRES_GRAD.
//   * The roles INPUT, OUTPUT, DESIRED_OUTPUT and COST may each be given to at most one variable
//     per model; the model remembers it in model->input / output / desired_output / cost.
//   * All variables in one graph must come from the same model_context and arena.
//
// Layout convention (same as matrix.h): one ROW per sample.
//
// Known limitation: no bias term yet. mv_add needs two matrices of identical shape, so adding a
// 1 x n bias row to a batch x n matrix (broadcasting) is not expressible. A bias-free MLP still
// learns MNIST; broadcasting comes later.

#pragma once

typedef enum {
    MV_FLAG_NONE = 0,
    MV_FLAG_REQUIRES_GRAD = (1 << 0),
    MV_FLAG_PARAMETER = (1 << 1),
    MV_FLAG_INPUT = (1 << 2),
    MV_FLAG_OUTPUT = (1 << 3),
    MV_FLAG_DESIRED_OUTPUT = (1 << 4),
    MV_FLAG_COST = (1 << 5)
} model_var_flags;

typedef enum {
    MV_OP_NULL = 0,
    MV_OP_CREATE,
    _MV_OP_UNARY_START,
    MV_OP_RELU,
    MV_OP_SOFTMAX,
    _MV_OP_BINARY_START,
    MV_OP_ADD,
    MV_OP_SUB,
    MV_OP_MATMUL,
    MV_OP_CROSS_ENTROPY,
} model_var_op;

#define MODEL_VAR_MAX_INPUTS 2
#define MV_NUM_INPUTS(op) ((op) < _MV_OP_UNARY_START ? 0 : ((op) < _MV_OP_BINARY_START ? 1 : 2))

typedef struct model_var {
    u32 index;              // creation order, 0..num_vars-1. Doubles as an array index for bookkeeping.
    u32 flags;

    matrix* val;
    matrix* grad;           // NULL unless the variable requires a gradient

    model_var_op op;
    struct model_var* inputs[MODEL_VAR_MAX_INPUTS];
} model_var;

// A program is a list of variables in an order where every variable comes AFTER all of its inputs.
// Running the list front to back is the forward pass; back to front is the backward pass.
typedef struct {
    model_var** vars;
    u32 size;
} model_program;

typedef struct {
    u32 num_vars;
    model_var* input;
    model_var* output;
    model_var* desired_output;
    model_var* cost;

    model_program forward_prog;     // everything needed to compute `output`
    model_program cost_prog;        // everything needed to compute `cost` (includes the forward part)
} model_context;

typedef struct {
    matrix* train_images;
    matrix* train_labels;
    matrix* test_images;
    matrix* test_labels;

    u32 epochs;
    u32 batch_size;
    f32 learning_rate;
} model_training_desc;

// ---- building the graph ------------------------------------------------------

// A leaf: inputs, labels, or parameters. The value starts zeroed.
model_var* mv_create(
    mem_arena* arena, model_context* model,
    u32 rows, u32 cols, u32 flags
);

// Same shape as the input.
model_var* mv_relu(
    mem_arena* arena, model_context* model,
    model_var* input, u32 flags
);

// Same shape as the input; softmax over each row.
model_var* mv_softmax(
    mem_arena* arena, model_context* model,
    model_var* input, u32 flags
);

// a + b and a - b: both inputs must have identical shapes.
model_var* mv_add(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
);

model_var* mv_sub(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
);

// a * b: a is (m x k), b is (k x n), the result is (m x n).
model_var* mv_matmul(
    mem_arena* arena, model_context* model,
    model_var* a, model_var* b, u32 flags
);

// Elementwise -p * log(q), same shape as p and q (see mat_cross_entropy).
// p is the target (inputs[0]), q the prediction (inputs[1]).
model_var* mv_cross_entropy(
    mem_arena* arena, model_context* model,
    model_var* p, model_var* q, u32 flags
);

// ---- programs --------------------------------------------------------------------

// Topologically sorts everything `out_var` depends on (and nothing else), using a depth-first
// search with post-order output. The result array is allocated from `arena`.
// Returns an empty program (size 0) if out_var is NULL, does not belong to `model` (detected only
// when its index is out of range), or memory runs out.
model_program model_prog_create(
    mem_arena* arena, model_context* model, model_var* out_var
);

// Forward pass: computes every non-leaf variable of the program, front to back.
void model_prog_compute(model_program* prog);

// Backward pass. Call model_prog_compute on the SAME program first (backward reuses the forward
// values). Then:
//   1. every gradient in the program is zeroed,
//   2. the program's last variable (the one it computes, e.g. the cost) is seeded with a gradient
//      of 1.0 in every element, i.e. we differentiate the SUM of its elements,
//   3. the program is walked back to front, each op adding its contribution to its inputs' grads.
// Afterwards every variable that requires a gradient holds d(sum of the root)/d(variable).
//
// Batch averaging is NOT done here: for a mean loss over `rows` samples, the optimizer scales the
// update by 1 / rows (equivalently, learning_rate / batch_size).
//
// The root must require a gradient (some PARAMETER or REQUIRES_GRAD variable upstream), otherwise
// there is nothing to compute and the program aborts with a message.
void model_prog_compute_grads(model_program* prog);

// ---- the model -----------------------------------------------------------------------

model_context* model_create(mem_arena* arena);

// Builds forward_prog (for model->output, if set) and cost_prog (for model->cost, if set).
void model_compile(mem_arena* arena, model_context* model);

// Runs forward_prog. The caller puts data into model->input->val first.
void model_feedforward(model_context* model);

void model_train(
    model_context* model,
    const model_training_desc* training_desc
);
