// matrix.h
//
// A 2D matrix of f32, stored row-major and allocated from an arena.
//
// Conventions used by every mat_* function:
//
//   * Row-major: element (i, j) lives at data[i * cols + j].
//   * Functions that can fail on bad input (shape mismatch, NULL, ...) return b32:
//       true  = success
//       false = failure, and the output matrix has NOT been modified.
//     "Check the shapes, then do the work" -- never half-write an output.
//   * Functions never allocate, except mat_create and mat_load, which take an arena.
//   * Read-only inputs are const. If a function writes to a parameter, it is not const.

#pragma once

typedef struct {
    u32 rows, cols;
    f32* data; // row-major
} matrix;

// ---- creation and basic operations -----------------------------------------

// Allocates a zeroed rows x cols matrix (struct + data) from `arena`.
// Returns NULL if rows or cols is 0, or if the arena cannot hold it. On failure the
// arena is left exactly as it was (a half-built matrix is rolled back).
matrix* mat_create(mem_arena* arena, u32 rows, u32 cols);

// Allocates a matrix and fills it from a file. (File format is decided in a later week.)
matrix* mat_load(mem_arena* arena, u32 rows, u32 cols, const char* filename);

// dst = src. Shapes must match (same rows AND same cols: 2x3 is not 3x2).
b32 mat_copy(matrix* dst, const matrix* src);

void mat_clear(matrix* mat);                              // all elements = 0
void mat_fill(matrix* mat, f32 x);                        // all elements = x
void mat_fill_rand(matrix* mat, f32 lower, f32 upper);    // uniform in [lower, upper]; needs lower <= upper
                                                          // (upper is reachable only via f32 rounding of lower + r*range)
void mat_scale(matrix* mat, f32 scale);                   // mat *= scale
f32  mat_sum(const matrix* mat);                          // sum of all elements (accumulated in f64)
u64  mat_argmax(const matrix* mat);                       // flat index of the largest element; ties -> first

// out = a + b and out = a - b. All three shapes must match. out may alias a or b.
b32 mat_add(matrix* out, const matrix* a, const matrix* b);
b32 mat_sub(matrix* out, const matrix* a, const matrix* b);

// Matrix multiply: out (m x n) = op(a) (m x k) * op(b) (k x n),
// where op(x) is x or x-transposed, chosen by the flags. The transpose is never
// materialized: the loops just read the stored data with swapped indices.
//
//   transpose_a  stored a is   |  transpose_b  stored b is
//   no           m x k         |  no           k x n
//   yes          k x m         |  yes          n x k
//
// zero_out = true : out  = op(a) * op(b)
// zero_out = false: out += op(a) * op(b)   (accumulate: needed when a variable's
//                                           gradient gets contributions from several ops)
//
// Returns false (and writes nothing) if the shapes do not fit, or if out shares ANY memory
// with a or b: unlike mat_add, an in-place multiply would overwrite inputs it still needs.
b32 mat_mul(
    matrix* out, const matrix* a, const matrix* b,
    b8 zero_out, b8 transpose_a, b8 transpose_b
);

// ---- activations and loss ---------------------------------------------------

b32 mat_relu(matrix* out, const matrix* in);                         // out = max(in, 0)
b32 mat_softmax(matrix* out, const matrix* in);                      // numerically stable
b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q);

// ---- gradients (implemented in Week 3, next to autograd) -------------------
//
// "add_grad" means ACCUMULATE into the output gradient (+=), never overwrite.

b32 mat_relu_add_grad(matrix* out, const matrix* in, const matrix* grad);
b32 mat_softmax_add_grad(matrix* out, const matrix* softmax_out, const matrix* grad);
b32 mat_cross_entropy_add_grad(
    matrix* p_grad, matrix* q_grad,
    const matrix* p, const matrix* q, const matrix* grad
);
