#include <stdlib.h>

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