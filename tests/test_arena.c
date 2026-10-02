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

static void test_commit_growth(void){
    u64 commit_size=MAX(KiB(64),(u64)plat_get_pagesize());
    mem_arena* a = arena_create(MiB(1),commit_size);
    CHECK(a!=NULL);
    if(!a){
        return ;
    }
    u64 initial_commit=a->commit_pos;
    CHECK(initial_commit==a->commit_size);

    CHECK(arena_push(a,1000,false)!=NULL);
    CHECK(a->commit_pos==initial_commit);

    u64 size=a->commit_size;
    u8* p=(u8*)arena_push(a,size,false);
    CHECK(p!=NULL);
    printf("    commit_pos: %llu KiB -> %llu KiB (pos = %llu)\n",
           (unsigned long long)(initial_commit / 1024),
           (unsigned long long)(a->commit_pos / 1024),
           (unsigned long long)a->pos);
    CHECK(a->commit_pos>initial_commit);
    CHECK(a->commit_pos%a->commit_size==0);
    CHECK(a->commit_pos>=a->pos);

    if(p){
        for(u64 i=0; i<size; i++){
            p[i]=(u8)(i*31+7);
        }
        b32 intact=true;
        for(u64 i=0; i<size; i++){
            if(p[i]!=(u8)(i*31+7)){
                intact=false;
                break;
            }
        }
        CHECK(intact);
    }
    arena_destroy(a);
}

static void test_out_of_space(void){
    mem_arena* a=arena_create(MiB(1),KiB(64));
    CHECK(a!=NULL);
    if(!a){
        return ;
    }
    u64 pos_before=a->pos;
    CHECK(arena_push(a,MiB(2),false)==NULL);
    CHECK(a->pos==pos_before);

    CHECK(arena_push(a,~(u64)0,false)==NULL);
    CHECK(a->pos == pos_before);

    u64 remaining=a->reserve_size-ARENA_BASE_POS;
    u8* all=(u8*)arena_push(a,remaining,false);
    CHECK(all!=NULL);
    CHECK(a->pos==a->reserve_size);
    CHECK(a->commit_pos==a->reserve_size);
    if(all){
        all[0]=-1;
        all[remaining-1]=2;
    }
    CHECK(arena_push(a,1,false)==NULL);
    arena_destroy(a);
}

static void test_pop(void){
    mem_arena* a=arena_create(MiB(1),KiB(64));
    CHECK(a!=NULL);
    if(!a){
        return ;
    }

    arena_push(a,100,false);
    u64 pos_after_push=a->pos;

    arena_pop(a,40);
    CHECK(a->pos == pos_after_push-40);

    arena_pop(a,MiB(1));
    CHECK(a->pos == ARENA_BASE_POS);

    arena_push(a,500,false);
    u64 mark=a->pos;
    arena_push(a,500,false);
    arena_pop_to(a,mark);
    CHECK(a->pos == mark);

    arena_pop_to(a,mark+12345);
    CHECK(a->pos == mark);

    arena_pop_to(a,0);
    CHECK(a->pos==ARENA_BASE_POS);

    arena_push(a,777,false);
    arena_clear(a);
    CHECK(a->pos == ARENA_BASE_POS);
    arena_destroy(a);
}