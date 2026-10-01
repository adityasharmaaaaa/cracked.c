mem_arena* arena_create(u64 reserve_size, u64 commit_size) {
    if (reserve_size == 0) { return NULL; }

    u32 pagesize = plat_get_pagesize();

    reserve_size = ALIGN_UP_POW2(reserve_size, pagesize);
    commit_size  = ALIGN_UP_POW2(commit_size, pagesize);

    // The header lives inside the committed region, so we need at least one page.
    commit_size = MAX(commit_size, (u64)pagesize);
    // Cannot commit more than we reserved.
    commit_size = MIN(commit_size, reserve_size);

    mem_arena* arena = (mem_arena*)plat_mem_reserve(reserve_size);
    if (arena == NULL) { return NULL; }

    if (!plat_mem_commit(arena, commit_size)) {
        // Do not leak the reservation.
        plat_mem_release(arena, reserve_size);
        return NULL;
    }

    arena->reserve_size = reserve_size;
    arena->commit_size  = commit_size;
    arena->pos          = ARENA_BASE_POS;
    arena->commit_pos   = commit_size;

    return arena;
}

void arena_destroy(mem_arena* arena) {
    if (arena == NULL) { return; }

    // Read the size before releasing: the header lives inside the memory we free.
    u64 reserve_size = arena->reserve_size;
    plat_mem_release(arena, reserve_size);
}

void* arena_push(mem_arena* arena, u64 size, b32 non_zero) {
    u64 pos_aligned = ALIGN_UP_POW2(arena->pos, ARENA_ALIGN);

    // Written as a subtraction so a huge `size` cannot overflow u64 and wrap around.
    if (pos_aligned > arena->reserve_size) { return NULL; }
    if (size > arena->reserve_size - pos_aligned) { return NULL; }

    u64 new_pos = pos_aligned + size;

    if (new_pos > arena->commit_pos) {
        // Round the new commit position up to a multiple of commit_size (chunked growth
        // means we make few syscalls), but never past the end of the reservation.
        u64 new_commit_pos = new_pos;
        new_commit_pos += arena->commit_size - 1;
        new_commit_pos -= new_commit_pos % arena->commit_size;
        new_commit_pos  = MIN(new_commit_pos, arena->reserve_size);

        u8* mem = (u8*)arena + arena->commit_pos;
        u64 commit_bytes = new_commit_pos - arena->commit_pos;

        if (!plat_mem_commit(mem, commit_bytes)) {
            return NULL;
        }

        arena->commit_pos = new_commit_pos;
    }

    arena->pos = new_pos;

    u8* out = (u8*)arena + pos_aligned;

    if (!non_zero) {
        memset(out, 0, size);
    }

    return out;
}

void arena_pop(mem_arena* arena, u64 size) {
    // Never pop into the arena header.
    size = MIN(size, arena->pos - ARENA_BASE_POS);
    arena->pos -= size;
}

void arena_pop_to(mem_arena* arena, u64 pos) {
    u64 size = pos < arena->pos ? arena->pos - pos : 0;
    arena_pop(arena, size);
}

void arena_clear(mem_arena* arena) {
    arena_pop_to(arena, ARENA_BASE_POS);
}

// ---------------------------------------------------------------------------
// Temporary regions
// ---------------------------------------------------------------------------

mem_arena_temp arena_temp_begin(mem_arena* arena) {
    return (mem_arena_temp) {
        .arena = arena,
        .start_pos = arena->pos
    };
}

void arena_temp_end(mem_arena_temp temp) {
    // A zeroed temp (from a failed scratch_get) has arena == NULL: ignore it.
    if (temp.arena == NULL) { return; }
    arena_pop_to(temp.arena, temp.start_pos);
}

// ---------------------------------------------------------------------------
// Scratch arenas (two per thread, so a function can get one that does not
// conflict with an arena its caller is already using)
// ---------------------------------------------------------------------------

#if defined(_MSC_VER)
    #define ARENA_THREAD_LOCAL __declspec(thread)
#else
    #define ARENA_THREAD_LOCAL __thread
#endif

static ARENA_THREAD_LOCAL mem_arena* s_scratch_arenas[2] = { NULL, NULL };

mem_arena_temp arena_scratch_get(mem_arena** conflicts, u32 num_conflicts) {
    i32 scratch_index = -1;

    for (i32 i = 0; i < 2; i++) {
        b32 conflict_found = false;

        for (u32 j = 0; j < num_conflicts; j++) {
            if (s_scratch_arenas[i] == conflicts[j]) {
                conflict_found = true;
                break;
            }
        }

        if (!conflict_found) {
            scratch_index = i;
            break;
        }
    }

    if (scratch_index == -1) {
        return (mem_arena_temp){ 0 };
    }

    mem_arena** selected = &s_scratch_arenas[scratch_index];

    if (*selected == NULL) {
        *selected = arena_create(MiB(64), MiB(1));

        // Creation can fail (out of address space). Return an empty temp
        // instead of dereferencing NULL in arena_temp_begin.
        if (*selected == NULL) {
            return (mem_arena_temp){ 0 };
        }
    }

    return arena_temp_begin(*selected);
}

void arena_scratch_release(mem_arena_temp scratch) {
    arena_temp_end(scratch);
}

// ---------------------------------------------------------------------------
// Platform layer
// ---------------------------------------------------------------------------

#if defined(_WIN32)

#include <windows.h>

u32 plat_get_pagesize(void) {
    SYSTEM_INFO sysinfo = { 0 };
    GetSystemInfo(&sysinfo);

    return sysinfo.dwPageSize;
}

void* plat_mem_reserve(u64 size) {
    return VirtualAlloc(NULL, size, MEM_RESERVE, PAGE_READWRITE);
}

b32 plat_mem_commit(void* ptr, u64 size) {
    void* ret = VirtualAlloc(ptr, size, MEM_COMMIT, PAGE_READWRITE);
    return ret != NULL;
}

b32 plat_mem_decommit(void* ptr, u64 size) {
    return VirtualFree(ptr, size, MEM_DECOMMIT);
}

b32 plat_mem_release(void* ptr, u64 size) {
    // With MEM_RELEASE the size MUST be 0 (the whole reservation is freed).
    (void)size;
    return VirtualFree(ptr, 0, MEM_RELEASE);
}

#elif defined(__linux__) || defined(__APPLE__)

#include <unistd.h>
#include <sys/mman.h>

// macOS spells it MAP_ANON.
#ifndef MAP_ANONYMOUS
    #define MAP_ANONYMOUS MAP_ANON
#endif

u32 plat_get_pagesize(void) {
    // 4 KiB on x86-64, 16 KiB on Apple Silicon.
    return (u32)sysconf(_SC_PAGESIZE);
}

void* plat_mem_reserve(u64 size) {
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#if defined(MAP_NORESERVE)
    // Do not count the reservation against the overcommit limit (Linux).
    flags |= MAP_NORESERVE;
#endif

    // PROT_NONE: address space is claimed, but touching it faults.
    void* out = mmap(NULL, size, PROT_NONE, flags, -1, 0);
    if (out == MAP_FAILED) {
        return NULL;
    }
    return out;
}

b32 plat_mem_commit(void* ptr, u64 size) {
    i32 ret = mprotect(ptr, size, PROT_READ | PROT_WRITE);
    return ret == 0;
}

b32 plat_mem_decommit(void* ptr, u64 size) {
    i32 ret = mprotect(ptr, size, PROT_NONE);
    if (ret != 0) { return false; }

#if defined(__APPLE__)
    // On macOS, MADV_DONTNEED is only a hint; MADV_FREE lets the kernel reclaim the pages.
    ret = madvise(ptr, size, MADV_FREE);
#else
    ret = madvise(ptr, size, MADV_DONTNEED);
#endif

    return ret == 0;
}

b32 plat_mem_release(void* ptr, u64 size) {
    i32 ret = munmap(ptr, size);
    return ret == 0;
}

#else
    #error "arena.c: unsupported platform (need _WIN32, __linux__, or __APPLE__)"
#endif