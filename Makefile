# Makefile for cracked.c
#
#   make test          debug build (AddressSanitizer + UBSan), run ALL tests
#   make test-release  optimized build (-O2), run ALL tests (catches bugs that only appear with -O2)
#   make check         compile main.c with all warnings on (no linking: main.c has no main() yet)
#   make bench         optimized build of bench/bench_matmul.c, writes bench/results/matmul_baseline.csv
#                      (on macOS it also links Apple Accelerate for comparison; try OPT=-O3)
#   make clean
#
# To add a test suite: just create tests/test_foo.c.
#
# NOTE: recipe lines (the indented ones) MUST start with a TAB, not spaces.

CC      ?= clang
CFLAGS  = -std=gnu11 -Wall -Wextra -Wpedantic -I.
DEBUG   = -g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer
RELEASE = -O2

# Every tests/test_*.c file is a test suite. No list to keep in sync (a hand-written list can
# silently miss a suite, and a runner that runs fewer tests than you think still says "green").
TESTS = $(notdir $(basename $(wildcard tests/test_*.c)))

DEPS = base.h arena.h arena.c prng.h prng.c matrix.h matrix.c timer.h tests/test.h

# Benchmark settings. On macOS, compare against Apple's BLAS (Accelerate).
# On Linux with OpenBLAS installed: make bench BLAS_DEFS=-DUSE_BLAS BLAS_LIBS=-lopenblas
OPT ?= -O2
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    BLAS_DEFS ?= -DUSE_BLAS
    BLAS_LIBS ?= -framework Accelerate
endif

.PHONY: test test-release check bench clean

test: $(addprefix build/,$(TESTS))
	@for t in $(TESTS); do ./build/$$t || exit 1; done
	@echo "ran $(words $(TESTS)) test suites: $(TESTS)"

test-release: $(addprefix build/release/,$(TESTS))
	@for t in $(TESTS); do ./build/release/$$t || exit 1; done
	@echo "ran $(words $(TESTS)) test suites: $(TESTS)"

build/%: tests/%.c $(DEPS)
	@mkdir -p build
	$(CC) $(CFLAGS) $(DEBUG) $< -o $@ -lm

build/release/%: tests/%.c $(DEPS)
	@mkdir -p build/release
	$(CC) $(CFLAGS) $(RELEASE) $< -o $@ -lm

check:
	@mkdir -p build
	$(CC) $(CFLAGS) -c main.c -o build/main.o

bench: bench/bench_matmul.c $(DEPS)
	@mkdir -p build bench/results
	$(CC) $(CFLAGS) $(OPT) $(BLAS_DEFS) bench/bench_matmul.c -o build/bench_matmul -lm $(BLAS_LIBS)
	./build/bench_matmul > bench/results/matmul_baseline.csv
	@echo "wrote bench/results/matmul_baseline.csv"

clean:
	rm -rf build