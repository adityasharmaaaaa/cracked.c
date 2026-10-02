#include "base.h"
#include "arena.h"
#include "arena.c"

#include "test.h"

#define ARRAY_LEN(a) (sizeof(a)/sizeof((a)[0]))

static void test_create(void){
    u32 page = plat_get_pagesize();
    mem_arena* a= arena_create(MiB(1), KiB(1));
    CHECK(a!=NULL);
    if(!a){
        return ;
    }
    CHECK(a->pos == ARENA_BASE_POS);
    CHECK(a->reserve_size == MiB(1));
    CHECK(a->reserve_size % page == 0);
    CHECK(a->commit_size % page == 0);
    CHECK(a->commit_pos == a->commit_size);
    arena_destroy(a);

    a=arena_create(1,1);
    CHECK(a != NULL);
    if(a){
        CHECK(a->reserve_size == page);
        CHECK(a->commit_size == page);
        arena_destroy(a);
    }

    a=arena_create(page,MiB(8));
    CHECK(a!=NULL);
    if(a){
        CHECK(a->commit_size==a->reserve_size);
        arena_destroy(a);
    }

    CHECK(arena_create(0,0) == NULL);
}