
u32 idx_be32(const u8 bytes[4]) {
    // Build the value from the BYTES, most significant first. This gives the right answer on any
    // CPU. Reading the 4 bytes straight into a u32 would give the right answer only on a
    // big-endian machine; on your little-endian M1 it would turn 60000 (00 00 EA 60) into
    // 1,625,948,160 (0x60EA0000).
    //
    // Each byte is cast to u32 BEFORE shifting: a plain `bytes[0] << 24` promotes to (signed) int,
    // and shifting a byte >= 128 into the sign bit is undefined behavior.
    return ((u32)bytes[0] << 24) | ((u32)bytes[1] << 16) | ((u32)bytes[2] << 8) | (u32)bytes[3];
}

b32 idx_read_header(FILE* f, idx_header* out) {
    u8 magic[4];
    if (fread(magic, 1, 4, f) != 4) { return false; }

    if (magic[0] != 0 || magic[1] != 0) { return false; }      // not an IDX file
    if (magic[2] != 0x08) { return false; }                    // only unsigned bytes are supported

    u32 num_dims = magic[3];
    if (num_dims == 0 || num_dims > IDX_MAX_DIMS) { return false; }

    idx_header header = { 0 };
    header.num_dims = num_dims;
    for (u32 i = 0; i < num_dims; i++) {
        u8 word[4];
        if (fread(word, 1, 4, f) != 4) { return false; }       // header cut short
        header.dims[i] = idx_be32(word);
    }
    header.data_offset = 4 + 4 * (u64)num_dims;

    *out = header;
    return true;
}

// The file must be EXACTLY header + payload. A truncated download or a stray trailing byte means
// something is wrong, and silently training on half a dataset is worse than stopping.
// On success the file position is left at the first payload byte.
static b32 idx_check_size_and_seek(FILE* f, const idx_header* header, u64 payload_bytes) {
    if (fseek(f, 0, SEEK_END) != 0) { return false; }
    long size = ftell(f);
    if (size < 0) { return false; }
    if ((u64)size != header->data_offset + payload_bytes) { return false; }

    return fseek(f, (long)header->data_offset, SEEK_SET) == 0;
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

static matrix* idx_images_from_file(FILE* f, mem_arena* arena) {
    idx_header header;
    if (!idx_read_header(f, &header) || header.num_dims != 3) { return NULL; }

    u64 count = header.dims[0];
    u64 pixels = (u64)header.dims[1] * header.dims[2];         // two u32 factors always fit in u64
    if (count == 0 || pixels == 0) { return NULL; }
    if (count > UINT32_MAX || pixels > UINT32_MAX) { return NULL; }   // must fit matrix's u32 dims

    u64 payload = count * pixels;                              // both < 2^32, so < 2^64: no overflow
    if (!idx_check_size_and_seek(f, &header, payload)) { return NULL; }

    matrix* mat = mat_create(arena, (u32)count, (u32)pixels);
    if (mat == NULL) { return NULL; }

    // Stream through a small buffer instead of loading the whole file (47 MB for MNIST) twice.
    u8 buffer[16384];
    f32* dst = mat->data;
    u64 remaining = payload;
    while (remaining > 0) {
        size_t want = remaining < sizeof(buffer) ? (size_t)remaining : sizeof(buffer);
        if (fread(buffer, 1, want, f) != want) { return NULL; }
        for (size_t i = 0; i < want; i++) {
            *dst++ = (f32)buffer[i] / 255.0f;                  // divide, so 255 -> exactly 1.0f
        }
        remaining -= want;
    }
    return mat;
}

matrix* idx_load_images(mem_arena* arena, const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (f == NULL) { return NULL; }

    u64 start_pos = arena->pos;
    matrix* mat = idx_images_from_file(f, arena);
    fclose(f);                                                 // on EVERY path: no leaked handles

    if (mat == NULL) { arena_pop_to(arena, start_pos); }       // undo any half-built matrix
    return mat;
}

// ---------------------------------------------------------------------------
// Labels
// ---------------------------------------------------------------------------

static matrix* idx_labels_from_file(FILE* f, mem_arena* arena, u32 num_classes) {
    if (num_classes == 0) { return NULL; }

    idx_header header;
    if (!idx_read_header(f, &header) || header.num_dims != 1) { return NULL; }

    u64 count = header.dims[0];
    if (count == 0) { return NULL; }
    if (!idx_check_size_and_seek(f, &header, count)) { return NULL; }

    matrix* mat = mat_create(arena, (u32)count, num_classes);  // zeroed: all 0.0f
    if (mat == NULL) { return NULL; }

    u8 buffer[16384];
    u64 row = 0;
    u64 remaining = count;
    while (remaining > 0) {
        size_t want = remaining < sizeof(buffer) ? (size_t)remaining : sizeof(buffer);
        if (fread(buffer, 1, want, f) != want) { return NULL; }
        for (size_t i = 0; i < want; i++) {
            if (buffer[i] >= num_classes) { return NULL; }     // a label outside 0..num_classes-1
            mat->data[row * num_classes + buffer[i]] = 1.0f;   // one-hot
            row++;
        }
        remaining -= want;
    }
    return mat;
}

matrix* idx_load_labels(mem_arena* arena, const char* filename, u32 num_classes) {
    FILE* f = fopen(filename, "rb");
    if (f == NULL) { return NULL; }

    u64 start_pos = arena->pos;
    matrix* mat = idx_labels_from_file(f, arena, num_classes);
    fclose(f);

    if (mat == NULL) { arena_pop_to(arena, start_pos); }
    return mat;
}

