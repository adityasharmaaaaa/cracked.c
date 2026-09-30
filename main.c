#define _CRT_SECURE_NO_WARNINGS

#include "base.h"
#include "arena.h"
#include "prng.h"

#include "arena.c"
#include "prng.c"

typedef struct {
    u32 rows, cols;
    f32* data; //row major
} matrix;

matrix* mat_create(mem_arena* arena, u32 rows, u32 cols);
matrix* mat_load(mem_arena* arena, u32 rows, u32 cols, const char* filename);
b32 mat_copy(matrix* dst, matrix* src);
void mat_clear(matrix* mat);
void mat_fill(matrix* mat, f32 x);
void mat_fill_rand(matrix* mat, f32 lower, f32 upper);
void mat_scale(matrix* mat, f32 scale);
f32 mat_sum(matrix* mat);
u64 mat_argmax(matrix* mat);
b32 mat_add(matrix* out, const matrix* a, const matrix* b);
b32 mat_sub(matrix* out, const matrix* a, const matrix* b);
b32 mat_mul(
    matrix* out, const matrix* a, const matrix* b,
    b8 zero_out, b8 transpose_a, b8 transpose_b
);
b32 mat_relu(matrix* out, const matrix* in);
b32 mat_softmax(matrix* out, const matrix* in);
b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q);
b32 mat_relu_add_grad(matrix* out, const matrix* in, const matrix* grad);
b32 mat_softmax_add_grad(matrix* out, const matrix* softmax_out, const matrix* grad);
b32 mat_cross_entropy_add_grad(matrix* p_grad, matrix* q_grad,const matrix* p, const matrix* q, const matrix* grad);

typedef enum{
    MV_FLAG_NONE=0,
    MV_FLAG_REQUIRES_GRAD=(1<<0),
    MV_FLAG_PARAMETER=(1<<1),
    MV_FLAG_INPUT=(1<<2),
    MV_FLAG_OUTPUT=(1<<3),
    MV_FLAG_DESIRED_OUTPUT=(1<<4),
    MV_FLAG_COST=(1<<5)
} model_var_flags;