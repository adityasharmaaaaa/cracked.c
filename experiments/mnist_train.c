#include "base.h"
#include "arena.h"
#include "prng.h"
#include "matrix.h"
#include "idx.h"
#include "model.h"

#include "arena.c"
#include "prng.c"
#include "matrix.c"
#include "idx.c"
#include "model.c"

int main(int argc, char** argv) {
    u32 epochs = argc > 1 ? (u32)atoi(argv[1]) : 10;
    f32 learning_rate = argc > 2 ? (f32)atof(argv[2]) : 0.1f;
    u32 batch = argc > 3 ? (u32)atoi(argv[3]) : 64;

    // Big enough for all the data (about 250 MB as f32) plus the network.
    mem_arena* arena = arena_create(GiB(1), MiB(64));
    if (!arena) { fprintf(stderr, "arena_create failed\n"); return 1; }

    matrix* train_images = idx_load_images(arena, "data/mnist/train-images-idx3-ubyte");
    matrix* train_labels = idx_load_labels(arena, "data/mnist/train-labels-idx1-ubyte", 10);
    matrix* test_images  = idx_load_images(arena, "data/mnist/t10k-images-idx3-ubyte");
    matrix* test_labels  = idx_load_labels(arena, "data/mnist/t10k-labels-idx1-ubyte", 10);
    if (!train_images || !train_labels || !test_images || !test_labels) {
        fprintf(stderr, "could not load MNIST from data/mnist/ (run scripts/get_mnist.sh)\n");
        return 1;
    }
    printf("loaded %u training and %u test images\n", train_images->rows, test_images->rows);

    // The network. One ROW per sample, so a layer is  activations = relu(inputs * weights).
    model_context* m = model_create(arena);
    model_var* x  = mv_create(arena, m, batch, 784, MV_FLAG_INPUT);
    model_var* w1 = mv_create(arena, m, 784, 256, MV_FLAG_PARAMETER);
    model_var* w2 = mv_create(arena, m, 256, 128, MV_FLAG_PARAMETER);
    model_var* w3 = mv_create(arena, m, 128, 10,  MV_FLAG_PARAMETER);

    model_var* h1 = mv_relu(arena, m, mv_matmul(arena, m, x,  w1, MV_FLAG_NONE), MV_FLAG_NONE);
    model_var* h2 = mv_relu(arena, m, mv_matmul(arena, m, h1, w2, MV_FLAG_NONE), MV_FLAG_NONE);
    model_var* y  = mv_softmax(arena, m, mv_matmul(arena, m, h2, w3, MV_FLAG_NONE), MV_FLAG_OUTPUT);

    model_var* labels = mv_create(arena, m, batch, 10, MV_FLAG_DESIRED_OUTPUT);
    model_var* cost   = mv_cross_entropy(arena, m, labels, y, MV_FLAG_COST);
    if (!x || !w1 || !w2 || !w3 || !h1 || !h2 || !y || !labels || !cost) {
        fprintf(stderr, "building the graph failed\n");
        return 1;
    }
    model_compile(arena, m);

    prng_seed(42, 54);
    mat_fill_he_uniform(w1->val);
    mat_fill_he_uniform(w2->val);
    mat_fill_he_uniform(w3->val);

    // An untrained 10-class network guesses: about 10% accuracy.
    printf("before training: test accuracy %.2f%%  (guessing would be 10%%)\n",
           100.0 * (f64)model_accuracy(m, test_images, test_labels));

    model_training_desc desc = {
        .train_images = train_images, .train_labels = train_labels,
        .test_images  = test_images,  .test_labels  = test_labels,
        .epochs = epochs, .batch_size = batch, .learning_rate = learning_rate,
    };
    printf("training: %u epochs, batch %u, learning rate %g\n\n", epochs, batch, (f64)learning_rate);
    model_train(m, &desc);

    printf("\nfinal: train accuracy %.2f%%, test accuracy %.2f%%\n",
           100.0 * (f64)model_accuracy(m, train_images, train_labels),
           100.0 * (f64)model_accuracy(m, test_images, test_labels));

    arena_destroy(arena);
    return 0;
}