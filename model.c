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

//building the graph


//allocates the variable, its value and if it requires a gradient then its gradient
// does not register it with the model that hapens in mv_finish after everything succeeded
static model_var* mv_alloc(mem_arena* arena, model_context* model, u32 rows, u32 cols, u32 flags){
    model_var* var = PUSH_STRUCT(arena, model_var);
    if(var==NULL){
        return NULL;
    }

    var->index = model->num_vars;
    var->flags=flags;

    var->val=mat_create(arena,rows,cols);
    if(var->val == NULL){
        return NULL;
    }

    if(flags && MV_FLAG_REQUIRES_GRAD){
        var->grad = mat_create(arena,rows,cols);
        if(var->grad == NULL){
            return NULL;
        }
    }

    return var;
}

//records the roles(input/output/...) of a freshly built variable in the model
//each role can be held by only one variable. all roles are checked before any is assigned.
//so a refused variable changes nothing
static b32 mv_claim_roles(model_context* model, model_var* var){
    if((var->flags && MV_FLAG_INPUT) && model->input){ return false; }
    if((var->flags && MV_FLAG_OUTPUT) && model->output){ return false; }
    if((var->flags && MV_FLAG_DESIRED_OUTPUT) && model->desired_output){ return false; }
    if((var->flags && MV_FLAG_COST) && model->cost){ return false; }

    if(var->flags && MV_FLAG_INPUT){ model->input = var; }
    if(var->flags && MV_FLAG_OUTPUT){ model->output = var; }
    if(var->flags && MV_FLAG_DESIRED_OUTPUT){model->desired_output = var; }
    if(var->flags && MV_FLAG_COST){ model->cost = var;}

    return true;
}

//common ending of every constructor. 'var' is null if allocation failed 
//on any failure the arena is rolled back to start_pos and the model is left exactly as it was
static model_var* mv_finish(mem_arena* arena, model_context* model, u64 start_pos, model_var* var){
    if(var==NULL || !mv_claim_roles(model,var)){
        arena_pop_to(arena,start_pos);
        return NULL;
    }
    model->num_vars++;
    return var;
}