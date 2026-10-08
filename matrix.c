// matrix.c
//
// Like arena.c and prng.c, this file is #included from main.c (unity build), after
// base.h, arena.h, prng.h and matrix.h, so it includes nothing itself.

// Same rows AND same cols. (Comparing only rows*cols would wrongly accept 2x3 vs 3x2.)
static b32 mat_same_shape(const matrix* a, const matrix* b) {
    return a->rows == b->rows && a->cols == b->cols;
}

// Number of elements. Cast BEFORE multiplying: u32 * u32 would overflow in 32 bits.
static u64 mat_count(const matrix* mat) {
    return (u64)mat->rows * mat->cols;
}

// True if the two matrices' data ranges share any memory (not just "same pointer":
// a matrix can be a view into the middle of another one's data).
static b32 mat_overlaps(const matrix* x, const matrix* y) {
    uintptr_t x0 = (uintptr_t)x->data;
    uintptr_t x1 = x0 + mat_count(x) * sizeof(f32);
    uintptr_t y0 = (uintptr_t)y->data;
    uintptr_t y1 = y0 + mat_count(y) * sizeof(f32);
    return x0 < y1 && y0 < x1;
}

// ---------------------------------------------------------------------------
// Creation
// ---------------------------------------------------------------------------

matrix* mat_create(mem_arena* arena, u32 rows, u32 cols) {
    if (rows == 0 || cols == 0) { return NULL; }

    // Reject sizes that cannot possibly fit BEFORE computing sizeof(f32) * count:
    // that multiplication could wrap around u64 and produce a tiny (wrong) size.
    u64 count = (u64)rows * cols;
    if (count > arena->reserve_size / sizeof(f32)) { return NULL; }

    // Two pushes (struct, then data). If the second fails we must undo the first,
    // otherwise a failed mat_create would leak arena space.
    u64 start_pos = arena->pos;

    matrix* mat = PUSH_STRUCT(arena, matrix);
    if (mat == NULL) { return NULL; }

    mat->rows = rows;
    mat->cols = cols;
    mat->data = PUSH_ARRAY(arena, f32, count);   // zeroed

    if (mat->data == NULL) {
        arena_pop_to(arena, start_pos);
        return NULL;
    }

    return mat;
}

matrix* mat_load(mem_arena* arena, u32 rows, u32 cols, const char* filename) {
    (void)arena; (void)rows; (void)cols; (void)filename;
    return NULL;
}

// ---------------------------------------------------------------------------
// Basic operations
// ---------------------------------------------------------------------------

b32 mat_copy(matrix* dst, const matrix* src) {
    if (!mat_same_shape(dst, src)) { return false; }

    // memmove, not memcpy: memcpy on overlapping memory (dst == src) is undefined behavior.
    memmove(dst->data, src->data, mat_count(src) * sizeof(f32));
    return true;
}

void mat_clear(matrix* mat) {
    // All-zero bytes is exactly 0.0f in IEEE-754.
    memset(mat->data, 0, mat_count(mat) * sizeof(f32));
}

void mat_fill(matrix* mat, f32 x) {
    u64 n = mat_count(mat);
    for (u64 i = 0; i < n; i++) {
        mat->data[i] = x;
    }
}

void mat_fill_rand(matrix* mat, f32 lower, f32 upper) {
    f32 range = upper - lower;
    u64 n = mat_count(mat);
    for (u64 i = 0; i < n; i++) {
        mat->data[i] = lower + prng_randf() * range;
    }
}

void mat_scale(matrix* mat, f32 scale) {
    u64 n = mat_count(mat);
    for (u64 i = 0; i < n; i++) {
        mat->data[i] *= scale;
    }
}

f32 mat_sum(const matrix* mat) {
    // Accumulate in f64: with f32, once the running total is large, adding a small
    // number loses its low bits and the error grows with every element.
    f64 sum = 0.0;
    u64 n = mat_count(mat);
    for (u64 i = 0; i < n; i++) {
        sum += mat->data[i];
    }
    return (f32)sum;
}

u64 mat_argmax(const matrix* mat) {
    // Start from element 0, NOT from "0.0": a matrix of all-negative numbers must work.
    // Strict > means that on ties the FIRST maximum wins.
    u64 best = 0;
    u64 n = mat_count(mat);
    for (u64 i = 1; i < n; i++) {
        if (mat->data[i] > mat->data[best]) {
            best = i;
        }
    }
    return best;
}

b32 mat_add(matrix* out, const matrix* a, const matrix* b) {
    if (!mat_same_shape(a, b) || !mat_same_shape(a, out)) { return false; }

    u64 n = mat_count(out);
    for (u64 i = 0; i < n; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
    return true;
}

b32 mat_sub(matrix* out, const matrix* a, const matrix* b) {
    if (!mat_same_shape(a, b) || !mat_same_shape(a, out)) { return false; }

    u64 n = mat_count(out);
    for (u64 i = 0; i < n; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
    return true;
}


b32 mat_mul(
    matrix* out, const matrix* a, const matrix* b,
    b8 zero_out, b8 transpose_a, b8 transpose_b
) {
    // Logical shapes: op(a) is m x k and op(b) is k x n. The flags only change how the
    // STORED matrix is laid out: a transposed operand is stored with rows and cols swapped.
    u32 m  = transpose_a ? a->cols : a->rows;
    u32 k  = transpose_a ? a->rows : a->cols;
    u32 kb = transpose_b ? b->cols : b->rows;
    u32 n  = transpose_b ? b->rows : b->cols;

    // Check everything BEFORE writing anything.
    if (k != kb) { return false; }                       // inner dimensions must agree
    if (out->rows != m || out->cols != n) { return false; }

    // Writing out[i][j] while a later iteration still needs the old a or b is wrong.
    // (An output element depends on a whole row of a and a whole column of b.)
    if (mat_overlaps(out, a) || mat_overlaps(out, b)) { return false; }

    if (zero_out) { mat_clear(out); }

    // Strides: the element (r, c) of op(x) lives at x->data[r * row_stride + c * col_stride].
    //
    //                  stored as      (r, c) is at          row_stride  col_stride
    //   not transposed rows x cols    r * cols + c          cols        1
    //   transposed     cols x rows    c * cols + r          1           cols
    //
    // The transpose is never built: we only swap the two strides.
    // u64 strides make `i * stride` a 64-bit multiply (u32 * u32 could overflow).
    u64 a_rs = transpose_a ? 1 : a->cols;
    u64 a_cs = transpose_a ? a->cols : 1;
    u64 b_rs = transpose_b ? 1 : b->cols;
    u64 b_cs = transpose_b ? b->cols : 1;

    // Loop order i-k-j: the innermost loop walks out's row and (when b is not
    // transposed) b's row, both contiguous in memory. a(i, kk) is loop-invariant
    // in the inner loop, so it is read once into a register.
    for (u32 i = 0; i < m; i++) {
        for (u32 kk = 0; kk < k; kk++) {
            f32 a_ik = a->data[i * a_rs + kk * a_cs];

            for (u32 j = 0; j < n; j++) {
                out->data[(u64)i * n + j] += a_ik * b->data[kk * b_rs + j * b_cs];
            }
        }
    }

    return true;
}


b32 mat_relu(matrix* out, const matrix* in) {
    if(!mat_same_shape(out,in)){
        return false;
    }
    u64 n=mat_count(in);
    for(u64 i=0; i<n; i++){
        f32 x=in->data[i];
        out->data[i]=x<0.0f?0.0f:x;
    }
    return true;
}

b32 mat_softmax(matrix* out, const matrix* in) {
    if(!mat_same_shape(out,in)){
        return false;
    }
    
    for(u32 r=0; r<in->rows; r++){
        const f32* x = in->data + (u64)r*in->cols;
        f32* y=out->data + (u64)r*out->cols;

        f32 max=x[0];
        for(u32 c=1; c<in->cols; c++){
            if(x[c]>max){
                max=x[c];
            }
        }

        f32 sum=0.0f;
        for(u32 c=0; c<in->cols; c++){
            y[c]=expf(x[c]-max);
            sum+=y[c];
        }

        for(u32 c=0; c<in->cols; c++){
            y[c]/=sum;
        }
    }
    return true;
}

b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q) {
    (void)out; (void)p; (void)q;
    return false;
}

// mat_relu_add_grad, mat_softmax_add_grad, mat_cross_entropy_add_grad:
// declared in matrix.h, defined in Week 3.

