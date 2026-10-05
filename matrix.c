static b32 mat_same_shape(const matrix* a, const matrix* b){
    return a->rows==b->rows && a->cols==b->cols;
}

static u64 mat_count(const matrix* mat){
    return (u64)mat->rows*mat->cols;
}

matrix* mat_create(mem_arena* arena, u32 rows, u32 cols) {
    if(rows==0 || cols==0){
        return NULL;
    }
    u64 count = (u64)rows*cols;
    if(count>arena->reserve_size/sizof(f32)){
        return NULL;
    }
    u64 start_pos=arena->pos;
    matrix* mat=PUSH_STRUCT(arena,matrix);
    if(mat==NULL){
        return NULL;
    }
    mat->rows=rows;
    mat->cols=cols;
    mat->data=PUSH_ARRAY(arena,f32,count);

    if(mat->data==NULL){
        arena_pop_to(arena,start_pos)
        return NULL;
    }
    return mat;
}

matrix* mat_load(mem_arena* arena, u32 rows, u32 cols, const char* filename) {
    (void)arena; (void)rows; (void)cols; (void)filename;
    return NULL;
}

b32 mat_copy(matrix* dst, const matrix* src) {
    (void)dst; (void)src;
    return false;
}

void mat_clear(matrix* mat) {
    (void)mat;
}

void mat_fill(matrix* mat, f32 x) {
    (void)mat; (void)x;
}

void mat_fill_rand(matrix* mat, f32 lower, f32 upper) {
    (void)mat; (void)lower; (void)upper;
}

void mat_scale(matrix* mat, f32 scale) {
    (void)mat; (void)scale;
}

f32 mat_sum(const matrix* mat) {
    (void)mat;
    return 0.0f;
}

u64 mat_argmax(const matrix* mat) {
    (void)mat;
    return 0;
}

b32 mat_add(matrix* out, const matrix* a, const matrix* b) {
    (void)out; (void)a; (void)b;
    return false;
}

b32 mat_sub(matrix* out, const matrix* a, const matrix* b) {
    (void)out; (void)a; (void)b;
    return false;
}

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

// mat_relu_add_grad, mat_softmax_add_grad, mat_cross_entropy_add_grad:
// declared in matrix.h, defined in Week 3.
