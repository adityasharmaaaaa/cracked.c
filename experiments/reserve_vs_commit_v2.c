#include "../base.h"
#include "../arena.h"
#include "../arena.c"

#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__)
#include <mach/mach.h>

static u64 resident_now(void) {
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                                 (task_info_t)&info, &count);
    return kr == KERN_SUCCESS ? info.resident_size : 0;
}
#else
static u64 resident_now(void) {
    FILE* f = fopen("/proc/self/statm", "r");
    if (!f) return 0;
    unsigned long long size = 0, resident = 0;
    if (fscanf(f, "%llu %llu", &size, &resident) != 2) resident = 0;
    fclose(f);
    return resident * plat_get_pagesize();
}
#endif

// ---- peak resident set size, in bytes --------------------------------------
static u64 resident_peak(void) {
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0;
#if defined(__APPLE__)
    return (u64)ru.ru_maxrss;          // macOS reports BYTES
#else
    return (u64)ru.ru_maxrss * 1024;   // Linux reports KiB
#endif
}

static void report(const char* label, mem_arena* a) {
    printf("%-44s commit_pos=%4llu MiB | RSS now=%7.2f MiB | RSS peak=%7.2f MiB\n",
           label,
           (unsigned long long)(a->commit_pos / MiB(1)),
           (double)resident_now() / (double)MiB(1),
           (double)resident_peak() / (double)MiB(1));
}

int main(void) {
    printf("page size: %u bytes\n\n", plat_get_pagesize());

    mem_arena* a = arena_create(GiB(1), MiB(1));
    report("1. reserved 1 GiB, committed 1 MiB", a);

    // Commit 200 MiB of address space but DO NOT write to it (non_zero = true skips memset).
    u64* data = (u64*)arena_push(a, MiB(200), true);
    report("2. pushed 200 MiB, never touched", a);

    // Touch every page with non-zero, hard-to-compress data.
    u64 count = MiB(200) / sizeof(u64);
    for (u64 i = 0; i < count; i++) {
        data[i] = (i + 1) * 0x9E3779B97F4A7C15ULL;   // multiplicative hash: looks random
    }
    report("3. wrote non-zero data to all 200 MiB", a);

    // Reproduce your original run: a zeroed push (arena_push does memset(0)).
    void* zeroed = arena_push(a, MiB(200), false);
    report("4. pushed another 200 MiB, zero-filled", a);

    sleep(5);
    report("5. same, 5 seconds later", a);

    // Read one byte back so the compiler cannot discard anything.
    printf("\ncheck values: %llu %d\n",
           (unsigned long long)data[12345], ((u8*)zeroed)[999]);

    arena_destroy(a);
    return 0;
}