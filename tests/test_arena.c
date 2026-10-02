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

static void test_alignment(void){
    mem_arena* a = arena_create(MiB(1),KiB(64));
    CHECK(a!=NULL);
    if(!a){
        return ;
    }
    u64 sizes[]={1,3,7,8,13,100,4095,1};
    u8* prev_end=NULL;
    
    for(u32 i=0; i<ARRAY_LEN(sizes); i++){
        u8* p = (u8*)arena_push(a,sizes[i],false);
        CHECK(p!=NULL);
        if(!p){
            break;
        }
        CHECK(((uintptr_t)p%ARENA_ALIGN)==0);
        if(prev_end){
            CHECK(p>=prev_end);
        }
        prev_end=p+sizes[i];
    }
    arena_destroy(a);
}

static void test_zeroing(void){
    mem_arena* a = arena_create(MiB(1),KiB(64));
    CHECK(a!=NULL);
    if(!a){
        return;
    }
    u8* dirty=(u8*)arena_push(a,4096,false);
    CHECK(dirty!=NULL);
    memset(dirty,0xFF,4096);
    arena_pop(a,4096);

    u8* again=(u8*)arena_push(a,4096,true);
    CHECK(again==dirty);
    CHECK(again[0]==0xFF && again[4095]==0xFF);
    arena_pop(a,4096);

    u8* clean=(u8*)arena_push(a,4096,false);
    CHECK(clean==dirty);
    b32 all_zero=true;
    for(u32 i=0; i<4096; i++){
        if(clean[i]!=0){
            all_zero=false;
            break;
        }
    }
    CHECK(all_zero);
    arena_destroy(a);
}