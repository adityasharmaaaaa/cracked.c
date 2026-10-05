static b32 mat_same_shape(const matrix* a, const matrix* b) {
    return a->rows == b->rows && a->cols == b->cols;
}

// Number of elements. Cast BEFORE multiplying: u32 * u32 would overflow in 32 bits.
static u64 mat_count(const matrix* mat) {
    return (u64)mat->rows * mat->cols;
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

// ---------------------------------------------------------------------------
// Stubs (implemented later)
// ---------------------------------------------------------------------------

b32 mat_mul(
    matrix* out, const matrix* a, const matrix* b,
    b8 zero_out, b8 transpose_a, b8 transpose_b
) {
    (void)out; (void)a; (void)b;
    (void)zero_out; (void)transpose_a; (void)transpose_b;
    return false;
}

b32 mat_relu(matrix* out, const matrix* in) {
    (void)out; (void)in;
    return false;
}

b32 mat_softmax(matrix* out, const matrix* in) {
    (void)out; (void)in;
    return false;
}

b32 mat_cross_entropy(matrix* out, const matrix* p, const matrix* q) {
    (void)out; (void)p; (void)q;
    return false;
}

