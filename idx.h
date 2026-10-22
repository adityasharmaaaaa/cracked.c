#pragma once

#define IDX_MAX_DIMS 4

typedef struct {
    u32 num_dims;
    u32 dims[IDX_MAX_DIMS];
    u64 data_offset;          
} idx_header;

// Decodes 4 bytes as a big-endian u32. Works the same on any CPU, whatever its own byte order.
u32 idx_be32(const u8 bytes[4]);

b32 idx_read_header(FILE* f, idx_header* out);

matrix* idx_load_images(mem_arena* arena, const char* filename);

matrix* idx_load_labels(mem_arena* arena, const char* filename, u32 num_classes);