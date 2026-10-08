// tests/gradcheck.h
//
// Finite-difference gradient checking: measure a derivative numerically and compare it with
// the formula you implemented. If the two disagree, the formula (or the code) is wrong.
//
// The idea: for a scalar loss L that depends on the matrix x,
//
//     dL/dx[i]  ~=  ( L(x[i] + h) - L(x[i] - h) ) / (2h)         ("central difference")
//
// Usage:
//     f64 my_loss(void* ctx) { ...recompute the forward pass from the CURRENT values, return L... }
//     gc_numeric_grad(x, my_loss, &ctx, 1e-3f, numeric);        // measure
//     // ... compute `analytic` with your add_grad function ...
//     CHECK(gc_max_error(analytic, numeric) < GC_TOLERANCE);    // compare
//
// To get a general upstream gradient (not just all ones), define the loss as
//     L = sum_i  g[i] * out[i]
// with a random fixed g (gc_weighted_sum). Then dL/d(out) is exactly g.
//
// Caveats:
//   * The loss must be smooth around x. For relu, keep every |x[i]| larger than h.
//   * The forward pass runs in f32, so tiny differences of two nearly equal numbers are noisy.
//     The loss is accumulated in f64, and h = 1e-3 balances rounding noise against the
//     truncation error of the formula.


#pragma once

typedef f64 (*gc_loss_fn)(void* ctx);

#define GC_H         1e-3f
#define GC_TOLERANCE 1e-3

// numeric[i] = central-difference estimate of dL/dx[i]. x is restored before returning.
static void gc_numeric_grad(matrix* x, gc_loss_fn loss, void* ctx, f32 h, matrix* numeric) {
    u64 n = (u64)x->rows * x->cols;
    for (u64 i = 0; i < n; i++) {
        f32 original = x->data[i];

        x->data[i] = original + h;
        f64 plus = loss(ctx);

        x->data[i] = original - h;
        f64 minus = loss(ctx);

        x->data[i] = original;
        numeric->data[i] = (f32)((plus - minus) / (2.0 * (f64)h));
    }
}

// Largest mismatch between two gradients, as a RELATIVE error with a floor, so that tiny
// gradients (near 0) are not judged by a ratio of two noise values:
//     |a - n| / (1 + max(|a|, |n|))
static f64 gc_max_error(const matrix* analytic, const matrix* numeric) {
    f64 worst = 0.0;
    u64 n = (u64)analytic->rows * analytic->cols;
    for (u64 i = 0; i < n; i++) {
        f64 a = analytic->data[i];
        f64 b = numeric->data[i];
        f64 scale = 1.0 + fmax(fabs(a), fabs(b));
        f64 err = fabs(a - b) / scale;
        if (!(err <= worst)) { worst = err; }      // `!(<=)` also records NaN
    }
    return worst;
}

// sum_i g[i] * m[i], accumulated in f64.
static f64 gc_weighted_sum(const matrix* m, const matrix* g) {
    f64 sum = 0.0;
    u64 n = (u64)m->rows * m->cols;
    for (u64 i = 0; i < n; i++) {
        sum += (f64)g->data[i] * (f64)m->data[i];
    }
    return sum;
}