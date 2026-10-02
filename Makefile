CC      ?= clang
CFLAGS  = -std=gnu11 -Wall -Wextra -Wpedantic -I.
DEBUG   = -g -O0 -fsanitize=address,undefined -fno-omit-frame-pointer
RELEASE = -O2

DEPS = arena.c arena.h base.h tests/test.h

.PHONY: test test-release clean

test: build/test_arena
	./build/test_arena

build/test_arena: tests/test_arena.c $(DEPS)
	@mkdir -p build
	$(CC) $(CFLAGS) $(DEBUG) tests/test_arena.c -o $@ -lm

test-release: tests/test_arena.c $(DEPS)
	@mkdir -p build
	$(CC) $(CFLAGS) $(RELEASE) tests/test_arena.c -o build/test_arena_release -lm
	./build/test_arena_release

clean:
	rm -rf build