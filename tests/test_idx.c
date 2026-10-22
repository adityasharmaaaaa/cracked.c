#include "base.h"
#include "arena.h"
#include "prng.h"
#include "matrix.h"
#include "idx.h"
 
#include "arena.c"
#include "prng.c"
#include "matrix.c"
#include "idx.c"
 
#include "test.h"
 
#include <stdlib.h>
#include <unistd.h>
 
// ---------------------------------------------------------------------------
// Helpers: build IDX files in memory and write them to a temporary file
// ---------------------------------------------------------------------------
 
static void put_be32(u8* dst, u32 value) {
    dst[0] = (u8)(value >> 24);
    dst[1] = (u8)(value >> 16);
    dst[2] = (u8)(value >> 8);
    dst[3] = (u8)value;
}
 
// Writes `size` bytes to a new temporary file and stores its path in `path` (>= 64 chars).
static b32 write_temp(char* path, const u8* bytes, u64 size) {
    snprintf(path, 64, "/tmp/cracked_idx_XXXXXX");
    int fd = mkstemp(path);
    if (fd < 0) { return false; }
 
    u64 written = 0;
    while (written < size) {
        ssize_t n = write(fd, bytes + written, (size_t)(size - written));
        if (n <= 0) { close(fd); return false; }
        written += (u64)n;
    }
    close(fd);
    return true;
}
 
// Builds a complete image file: header for (count, rows, cols) followed by `payload` bytes
// (zero-padded or cut to the length the header promises, unless `payload_len` says otherwise).
// Returns the total size.
static u64 make_image_file(u8* out, u32 count, u32 rows, u32 cols, const u8* payload, u64 payload_len) {
    out[0] = 0; out[1] = 0; out[2] = 0x08; out[3] = 3;
    put_be32(out + 4, count);
    put_be32(out + 8, rows);
    put_be32(out + 12, cols);
    for (u64 i = 0; i < payload_len; i++) { out[16 + i] = payload ? payload[i] : (u8)(i % 256); }
    return 16 + payload_len;
}
 
static u64 make_label_file(u8* out, u32 count, const u8* labels, u64 payload_len) {
    out[0] = 0; out[1] = 0; out[2] = 0x08; out[3] = 1;
    put_be32(out + 4, count);
    for (u64 i = 0; i < payload_len; i++) { out[8 + i] = labels[i]; }
    return 8 + payload_len;
}
 
static u8 g_buf[1 << 20];       // scratch for building files
 
// ---------------------------------------------------------------------------
// Byte order
// ---------------------------------------------------------------------------
static void test_be32(void) {
    u8 magic[4]  = { 0x00, 0x00, 0x08, 0x03 };
    u8 sixty_k[4] = { 0x00, 0x00, 0xEA, 0x60 };           // 60000: the MNIST training-set size
    u8 pattern[4] = { 0x12, 0x34, 0x56, 0x78 };
    u8 all_ones[4] = { 0xFF, 0xFF, 0xFF, 0xFF };
    u8 high_bit[4] = { 0x80, 0x00, 0x00, 0x00 };
 
    CHECK(idx_be32(magic) == 0x00000803u);
    CHECK(idx_be32(sixty_k) == 60000u);
    CHECK(idx_be32(pattern) == 0x12345678u);
    CHECK(idx_be32(all_ones) == 0xFFFFFFFFu);             // top byte >= 128: must not touch the sign bit
    CHECK(idx_be32(high_bit) == 0x80000000u);
 
    // Show the trap itself: reading the same 4 bytes as a native (little-endian) integer.
    u32 native;
    memcpy(&native, sixty_k, 4);
    printf("    the bytes 00 00 EA 60 read natively on this machine: %u (correct value is 60000)\n", native);
}
 
// ---------------------------------------------------------------------------
// Header parsing
// ---------------------------------------------------------------------------
static void test_read_header(void) {
    char path[64];
 
    // A valid 3-dimensional image header.
    u64 size = make_image_file(g_buf, 3, 2, 2, NULL, 12);
    REQUIRE(write_temp(path, g_buf, size));
    FILE* f = fopen(path, "rb");
    REQUIRE(f != NULL);
    idx_header h;
    CHECK(idx_read_header(f, &h));
    CHECK(h.num_dims == 3);
    CHECK(h.dims[0] == 3 && h.dims[1] == 2 && h.dims[2] == 2);
    CHECK(h.data_offset == 16);
    fclose(f);
    unlink(path);
 
    // A valid 1-dimensional label header.
    u8 labels[4] = { 1, 2, 3, 4 };
    size = make_label_file(g_buf, 4, labels, 4);
    REQUIRE(write_temp(path, g_buf, size));
    f = fopen(path, "rb");
    REQUIRE(f != NULL);
    CHECK(idx_read_header(f, &h));
    CHECK(h.num_dims == 1 && h.dims[0] == 4 && h.data_offset == 8);
    fclose(f);
    unlink(path);
 
    // The largest supported number of dimensions (IDX_MAX_DIMS = 4) is accepted.
    u8 four_d[20] = { 0, 0, 0x08, 4,  0, 0, 0, 2,  0, 0, 0, 3,  0, 0, 0, 4,  0, 0, 1, 0 };
    REQUIRE(write_temp(path, four_d, sizeof(four_d)));
    f = fopen(path, "rb");
    REQUIRE(f != NULL);
    CHECK(idx_read_header(f, &h));
    CHECK(h.num_dims == 4 && h.dims[0] == 2 && h.dims[1] == 3 && h.dims[2] == 4 && h.dims[3] == 256);
    CHECK(h.data_offset == 20);
    fclose(f);
    unlink(path);
 
    // Things that are not valid unsigned-byte IDX files.
    struct { const char* name; u8 bytes[8]; u32 len; } bad[] = {
        { "non-zero first byte",   { 1, 0, 0x08, 1, 0, 0, 0, 1 }, 8 },
        { "non-zero second byte",  { 0, 1, 0x08, 1, 0, 0, 0, 1 }, 8 },
        { "float data type 0x0D",  { 0, 0, 0x0D, 1, 0, 0, 0, 1 }, 8 },
        { "signed byte type 0x09", { 0, 0, 0x09, 1, 0, 0, 0, 1 }, 8 },
        { "zero dimensions",       { 0, 0, 0x08, 0, 0, 0, 0, 0 }, 8 },
        { "five dimensions",       { 0, 0, 0x08, 5, 0, 0, 0, 1 }, 8 },
        { "header cut short",      { 0, 0, 0x08, 2, 0, 0, 0, 1 }, 8 },   // promises 2 dims, has 1
        { "only 3 bytes",          { 0, 0, 0x08, 0, 0, 0, 0, 0 }, 3 },
        { "empty file",            { 0, 0, 0, 0, 0, 0, 0, 0 }, 0 },
    };
    for (u32 i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        REQUIRE(write_temp(path, bad[i].bytes, bad[i].len));
        f = fopen(path, "rb");
        REQUIRE(f != NULL);
        b32 accepted = idx_read_header(f, &h);
        if (accepted) { printf("    wrongly accepted: %s\n", bad[i].name); }
        CHECK(!accepted);
        fclose(f);
        unlink(path);
    }
}

int main(void){
    RUN_TEST(test_be32);
    RUN_TEST(test_read_header);
}