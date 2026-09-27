mem_arena* arena_create(u64 reserve_size, u64 commit_size){
    u32 pagesize=plat_get_pagesize();
    reserve_size=ALIGN_UP_POW2(reserve_size, pagesize);
    commit_size=ALIGN_UP_POW2(commit_size, pagesize);
    mem_arena* arena = plat_mem_reserve(reserve_size);

    if(!plat_mem_commit(arena,commit_size)){
        return NULL;
    }

    arena->reserve_size=reserve_size;
    arena->commit_size=commit_size;
    arena->pos=ARENA_BASE_POS;
    arena->commit_pos=commit_size;

    return arena;
}

void arena_destroy(mem_arena* arena){
    plat_mem_release(arena,arena->reserve_size);
}