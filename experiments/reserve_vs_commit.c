#include "../base.h"
#include "../arena.h"
#include "../arena.c"

int main(void) {
    printf("page size: %u bytes\n", plat_get_pagesize());

    mem_arena* a = arena_create(GiB(1), MiB(1));
    printf("reserved 1 GiB, committed %llu KiB. Check Activity Monitor now.\n",
           (unsigned long long)(a->commit_pos / 1024));
    getchar();

    u8* big = arena_push(a, MiB(200), false);   // zeroed, so every page gets touched
    printf("pushed 200 MiB, commit_pos = %llu MiB. Check again.\n",
           (unsigned long long)(a->commit_pos / MiB(1)));
    getchar();

    arena_destroy(a);
    (void)big;
    return 0;
}