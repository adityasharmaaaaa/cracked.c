// experiments/overfit_tiny.c
//
// Does the autodiff engine actually learn? Train a small MLP on 8 random samples and watch the
// loss. A correct network must memorise them: the loss starts near ln(3) = 1.0986 (it knows nothing,
// so it guesses uniformly among 3 classes) and falls toward 0.
//
// Build and run from the project root:
//   clang -std=gnu11 -O2 -I. experiments/overfit_tiny.c -o experiments/overfit_tiny -lm
//   ./experiments/overfit_tiny

#include "base.h"
#include "arena.h"
#include "prng.h"
#include "matrix.h"
#include "model.h"

#include "arena.c"
#include "prng.c"
#include "matrix.c"
#include "model.c"

int main(void) {
    u32 batch = 8, in = 6, hidden = 16, classes = 3;
    f32 learning_rate = 0.3f;

    mem_arena* arena = arena_create(MiB(8), MiB(1));
    model_context* m = model_create(arena);

    model_var* x  = mv_create(arena, m, batch, in, MV_FLAG_INPUT);
    model_var* w1 = mv_create(arena, m, in, hidden, MV_FLAG_PARAMETER);
    model_var* h  = mv_relu(arena, m, mv_matmul(arena, m, x, w1, MV_FLAG_NONE), MV_FLAG_NONE);
    model_var* w2 = mv_create(arena, m, hidden, classes, MV_FLAG_PARAMETER);
    model_var* y  = mv_softmax(arena, m, mv_matmul(arena, m, h, w2, MV_FLAG_NONE), MV_FLAG_OUTPUT);
    model_var* labels = mv_create(arena, m, batch, classes, MV_FLAG_DESIRED_OUTPUT);
    model_var* cost   = mv_cross_entropy(arena, m, labels, y, MV_FLAG_COST);
    model_compile(arena, m);

    prng_seed(2024, 9);
    mat_fill_rand(x->val, -1.0f, 1.0f);
    mat_fill_rand(w1->val, -0.5f, 0.5f);
    mat_fill_rand(w2->val, -0.5f, 0.5f);
    for (u32 r = 0; r < batch; r++) {
        labels->val->data[r * classes + (prng_rand() % classes)] = 1.0f;
    }

    printf("ln(%u) = %.4f   (the loss of a network that knows nothing)\n\n", classes, logf((f32)classes));
    printf("step    loss\n");

    for (u32 step = 0; step <= 300; step++) {
        model_prog_compute(&m->cost_prog);
        f32 loss = mat_sum(cost->val) / (f32)batch;
        if (step <= 5 || step % 25 == 0) { printf("%4u    %.4f\n", step, loss); }

        model_prog_compute_grads(&m->cost_prog);
        // The backward pass gives the gradient of the SUM of the cost, so step = learning_rate / batch_size.
        model_sgd_step(&m->cost_prog, learning_rate / (f32)batch);
    }

    u32 correct = 0;
    for (u32 r = 0; r < batch; r++) {
        u32 best = 0;
        for (u32 c = 1; c < classes; c++) {
            if (y->val->data[r * classes + c] > y->val->data[r * classes + best]) { best = c; }
        }
        if (labels->val->data[r * classes + best] == 1.0f) { correct++; }
    }
    printf("\naccuracy on the 8 training samples: %u / %u\n", correct, batch);

    arena_destroy(arena);
    return 0;
}