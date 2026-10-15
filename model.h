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

// Backward pass: back to front, accumulating into the gradients. (Friday.)
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
