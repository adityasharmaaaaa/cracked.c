# Makefile for cracked.c
#
#   make test          debug build with AddressSanitizer + UBSan, then run tests
#   make test-release  optimized build, then run tests (catches bugs that only appear with -O2)
#   make check         compile main.c with all warnings on (no linking: main.c has no main() yet)
#   make clean
#
# NOTE: recipe lines (the indented ones) MUST start with a TAB, not spaces.

CC      ?= clang
CFLAGS  = -std=gnu11 -Wall -Wextra -Wpedantic -I.
DEBUG   = -g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer
RELEASE = -O2

DEPS = arena.c arena.h base.h tests/test.h

.PHONY: test test-release check clean

test: build/test_arena
	./build/test_arena

build/test_arena: tests/test_arena.c $(DEPS)
	@mkdir -p build
	$(CC) $(CFLAGS) $(DEBUG) tests/test_arena.c -o $@ -lm

test-release: tests/test_arena.c $(DEPS)
	@mkdir -p build
	$(CC) $(CFLAGS) $(RELEASE) tests/test_arena.c -o build/test_arena_release -lm
	./build/test_arena_release

check:
	@mkdir -p build
	$(CC) $(CFLAGS) -c main.c -o build/main.o

clean:
	rm -rf build