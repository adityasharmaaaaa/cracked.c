# cracked.c

**got bored. wrote ML in C.**

A neural network library from scratch: memory arenas, matrix kernels, computation graphs, and eventually CUDA.

No frameworks. No tutorials. Just C, math, and the occasional segfault (character development).

## what's in here

- **arena allocator**: reserves virtual memory up front, commits pages on demand. Linux, macOS and Windows backends (yes i tested all three, no i'm not proud of the Windows part)
- **PCG PRNG**: my own random number generator. it once returned exactly 1.0 when it shouldn't. the test caught it, so now there's a range test.
- **matrix kernels**: matmul, relu, softmax, cross-entropy, plus all their backward pass.
- **autograd engine**: static computation graph, topo sorted, forward and backward. build the graph once, train forever, allocate nothing
- **unity build**: everything `#include`s everything. one translation unit, no linker, zero dependencies.

## does it learn

yeah, surprisingly. `experiments/overfit_tiny.c` trains a small MLP on 8 random samples:

```
step    loss
   0    1.0210     <- knows nothing (ln 3 = 1.0986)
  50    0.0769
 100    0.0244
 300    0.0048

accuracy on the 8 training samples: 8 / 8
```

it memorised 8 random samples. peak machine learning. also the whole network passes a finite-difference gradient check, so the backward pass isn't just vibe

## benchmarks

`make bench` races my matmul against Apple Accelerate on an M1.


| size | cracked.c (naive) | Accelerate |
|---|---|---|
| 1024 x 1024 | ~24 GFLOPS | ~1100 GFLOPS |

yes. i lost by 45x. fixing that is next.

## run it

```
make test           # debug build with ASan + UBSan, runs every suite
make test-release   # -O2, since some bugs only show up optimised
make bench          # matmul baseline, writes a csv
```

## known limitations (a.k.a. things i'm pretending are features)

- no bias term yet. broadcasting is not implemented, so it's a bias-free MLP for now. it still learns, so ¯\\_(ツ)_/¯
- no real training loop or dataset loading yet
- matmul is naive and slow (see above, i'm not over it)
- CUDA is currently a promise

## roadmap

- [x] arena allocator
- [x] matrix ops + gradients
- [x] computation graph + autograd
- [ ] bias + broadcasting
- [ ] training loop
- [ ] MNIST
- [ ] faster matmul (cache blocking, SIMD, threads)
- [ ] CUDA

## status

Work in progress. Things will break. Some commits will just say "fix".

*for plot purposes.*