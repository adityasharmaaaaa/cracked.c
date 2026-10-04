# Makefile for cracked.c
#
#   make test          debug build (AddressSanitizer + UBSan), run ALL tests
#   make test-release  optimized build (-O2), run ALL tests (catches bugs that only appear with -O2)
#   make check         compile main.c with all warnings on (no linking: main.c has no main() yet)
#   make clean
#
# To add a test file: create tests/test_foo.c and add `test_foo` to TESTS below.
#
# NOTE: recipe lines (the indented ones) MUST start with a TAB, not spaces.

CC      ?= clang
CFLAGS  = -std=gnu11 -Wall -Wextra -Wpedantic -I.
DEBUG   = -g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer
RELEASE = -O2

TESTS = test_arena test_prng

DEPS = base.h arena.h arena.c prng.h prng.c matrix.h matrix.c tests/test.h

.PHONY: test test-release check clean

test: $(addprefix build/,$(TESTS))
	@for t in $(TESTS); do ./build/$$t || exit 1; done

test-release: $(addprefix build/release/,$(TESTS))
	@for t in $(TESTS); do ./build/release/$$t || exit 1; done

build/%: tests/%.c $(DEPS)
	@mkdir -p build
	$(CC) $(CFLAGS) $(DEBUG) $< -o $@ -lm

build/release/%: tests/%.c $(DEPS)
	@mkdir -p build/release
	$(CC) $(CFLAGS) $(RELEASE) $< -o $@ -lm

check:
	@mkdir -p build
	$(CC) $(CFLAGS) -c main.c -o build/main.o

clean:
	rm -rf build