// tests/test_idx.c
//
// The IDX loader. Most tests write tiny IDX files on the fly, so they run anywhere. The last test
// runs against the real MNIST files if you have put them in data/mnist/ (and says SKIPPED if not).
// Build and run with:  make test

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

// ---------------------------------------------------------------------------
// Loading images
// ---------------------------------------------------------------------------
static void test_load_images(void) {
    mem_arena* a = arena_create(MiB(4), MiB(1));
    REQUIRE(a != NULL);
    char path[64];

    // 3 images of 2x2 pixels. Pixel order must be preserved (row-major).
    u8 pixels[12] = {   0, 255, 128,  64,        // image 0
                       10,  20,  30,  40,        // image 1
                      255, 255,   0,   1 };      // image 2
    u64 size = make_image_file(g_buf, 3, 2, 2, pixels, 12);
    REQUIRE(write_temp(path, g_buf, size));

    u64 pos_before = a->pos;
    matrix* m = idx_load_images(a, path);
    unlink(path);
    REQUIRE(m != NULL);

    CHECK(m->rows == 3 && m->cols == 4);                  // count x (rows * cols)
    CHECK(m->data[0] == 0.0f);
    CHECK(m->data[1] == 1.0f);                            // 255 maps to EXACTLY 1.0
    CHECK_NEAR(m->data[2], 128.0 / 255.0, 1e-7);
    CHECK_NEAR(m->data[3], 64.0 / 255.0, 1e-7);
    CHECK_NEAR(m->data[4], 10.0 / 255.0, 1e-7);           // second image starts at column 0 of row 1
    CHECK_NEAR(m->data[7], 40.0 / 255.0, 1e-7);
    CHECK(m->data[8] == 1.0f && m->data[9] == 1.0f && m->data[10] == 0.0f);
    CHECK_NEAR(m->data[11], 1.0 / 255.0, 1e-7);

    // Everything is within [0, 1].
    b32 in_range = true;
    for (u32 i = 0; i < 12; i++) { if (m->data[i] < 0.0f || m->data[i] > 1.0f) { in_range = false; } }
    CHECK(in_range);

    // The matrix and its data are the only things taken from the arena.
    CHECK(a->pos == pos_before + sizeof(matrix) + 12 * sizeof(f32));

    arena_destroy(a);
}

// Values whose BYTES are not symmetric: any byte-order mistake produces absurd sizes.
static void test_load_images_byte_order(void) {
    mem_arena* a = arena_create(MiB(4), MiB(1));
    REQUIRE(a != NULL);
    char path[64];

    // 258 = 0x0102 images of 1x1. Read little-endian this would claim 0x02010000 = 33,619,968 images.
    u64 size = make_image_file(g_buf, 258, 1, 1, NULL, 258);
    REQUIRE(write_temp(path, g_buf, size));
    matrix* m = idx_load_images(a, path);
    unlink(path);
    REQUIRE(m != NULL);
    CHECK(m->rows == 258 && m->cols == 1);
    CHECK_NEAR(m->data[0], 0.0, 1e-9);
    CHECK_NEAR(m->data[255], 1.0, 1e-9);                  // payload byte 255
    CHECK_NEAR(m->data[257], 1.0 / 255.0, 1e-7);          // payload byte 257 % 256 == 1

    // A wide image: 2 images of 1 x 300 (300 = 0x012C). Both dimension words are multi-byte.
    size = make_image_file(g_buf, 2, 1, 300, NULL, 600);
    REQUIRE(write_temp(path, g_buf, size));
    matrix* wide = idx_load_images(a, path);
    unlink(path);
    REQUIRE(wide != NULL);
    CHECK(wide->rows == 2 && wide->cols == 300);

    arena_destroy(a);
}

// ---------------------------------------------------------------------------
// Loading labels
// ---------------------------------------------------------------------------
static void test_load_labels(void) {
    mem_arena* a = arena_create(MiB(4), MiB(1));
    REQUIRE(a != NULL);
    char path[64];

    u8 labels[5] = { 3, 0, 9, 1, 3 };
    u64 size = make_label_file(g_buf, 5, labels, 5);
    REQUIRE(write_temp(path, g_buf, size));
    matrix* m = idx_load_labels(a, path, 10);
    unlink(path);
    REQUIRE(m != NULL);

    CHECK(m->rows == 5 && m->cols == 10);
    for (u32 r = 0; r < 5; r++) {
        f32 sum = 0.0f;
        for (u32 c = 0; c < 10; c++) {
            f32 expected = (c == labels[r]) ? 1.0f : 0.0f;
            CHECK(m->data[r * 10 + c] == expected);       // one-hot: 1.0 only in the label's column
            sum += m->data[r * 10 + c];
        }
        CHECK(sum == 1.0f);
    }

    // The largest legal label (num_classes - 1) works.
    u8 edge[2] = { 9, 0 };
    size = make_label_file(g_buf, 2, edge, 2);
    REQUIRE(write_temp(path, g_buf, size));
    matrix* e = idx_load_labels(a, path, 10);
    unlink(path);
    REQUIRE(e != NULL);
    CHECK(e->data[9] == 1.0f && e->data[10] == 1.0f);

    arena_destroy(a);
}

// ---------------------------------------------------------------------------
// Everything that must be refused, and refused cleanly
// ---------------------------------------------------------------------------
static void test_load_failures(void) {
    mem_arena* a = arena_create(MiB(4), MiB(1));
    REQUIRE(a != NULL);
    char path[64];
    u64 pos_before = a->pos;

    // A file that does not exist.
    CHECK(idx_load_images(a, "/tmp/cracked_no_such_file.idx") == NULL);
    CHECK(idx_load_labels(a, "/tmp/cracked_no_such_file.idx", 10) == NULL);

    // The wrong KIND of file for the loader.
    u8 labels[4] = { 1, 2, 3, 4 };
    u64 label_size = make_label_file(g_buf, 4, labels, 4);
    REQUIRE(write_temp(path, g_buf, label_size));
    CHECK(idx_load_images(a, path) == NULL);              // a 1-D file is not an image file
    unlink(path);

    u64 image_size = make_image_file(g_buf, 3, 2, 2, NULL, 12);
    REQUIRE(write_temp(path, g_buf, image_size));
    CHECK(idx_load_labels(a, path, 10) == NULL);          // a 3-D file is not a label file
    unlink(path);

    // Truncated payload: the header promises 12 bytes, only 10 are there.
    image_size = make_image_file(g_buf, 3, 2, 2, NULL, 10);
    REQUIRE(write_temp(path, g_buf, image_size));
    CHECK(idx_load_images(a, path) == NULL);
    unlink(path);

    // A stray trailing byte: also refused (the file is not what its header says).
    image_size = make_image_file(g_buf, 3, 2, 2, NULL, 13);
    REQUIRE(write_temp(path, g_buf, image_size));
    CHECK(idx_load_images(a, path) == NULL);
    unlink(path);

    // Zero items.
    image_size = make_image_file(g_buf, 0, 2, 2, NULL, 0);
    REQUIRE(write_temp(path, g_buf, image_size));
    CHECK(idx_load_images(a, path) == NULL);
    unlink(path);

    // Corrupt magic number.
    image_size = make_image_file(g_buf, 3, 2, 2, NULL, 12);
    g_buf[2] = 0x0D;
    REQUIRE(write_temp(path, g_buf, image_size));
    CHECK(idx_load_images(a, path) == NULL);
    unlink(path);

    // A label outside 0..num_classes-1 (the LAST label, so the loader has already done real work
    // and allocated when it finds out).
    u8 bad_labels[4] = { 1, 2, 3, 10 };
    label_size = make_label_file(g_buf, 4, bad_labels, 4);
    REQUIRE(write_temp(path, g_buf, label_size));
    CHECK(idx_load_labels(a, path, 10) == NULL);
    CHECK(idx_load_labels(a, path, 11) != NULL);          // same file is fine with 11 classes
    unlink(path);

    u8 ok_labels[4] = { 1, 2, 3, 4 };
    label_size = make_label_file(g_buf, 4, ok_labels, 4);
    REQUIRE(write_temp(path, g_buf, label_size));
    CHECK(idx_load_labels(a, path, 0) == NULL);           // zero classes makes no sense
    unlink(path);

    // After all those failures (plus one deliberate success above), the arena holds nothing from
    // the FAILED loads. Measure by repeating the failures and checking the position is unchanged.
    u64 pos_mid = a->pos;
    label_size = make_label_file(g_buf, 4, bad_labels, 4);
    REQUIRE(write_temp(path, g_buf, label_size));
    CHECK(idx_load_labels(a, path, 10) == NULL);
    unlink(path);
    CHECK(a->pos == pos_mid);

    (void)pos_before;
    arena_destroy(a);

    // A matrix too big for the arena: the file is perfectly valid, the arena is just too small.
    // The loader must fail AND hand back whatever it had allocated.
    mem_arena* tiny = arena_create(plat_get_pagesize(), plat_get_pagesize());
    REQUIRE(tiny != NULL);
    image_size = make_image_file(g_buf, 100, 100, 100, NULL, 1000000);   // 4 MB of floats
    REQUIRE(write_temp(path, g_buf, image_size));
    u64 tiny_before = tiny->pos;
    CHECK(idx_load_images(tiny, path) == NULL);
    CHECK(tiny->pos == tiny_before);
    unlink(path);
    arena_destroy(tiny);
}

// Every path must close the file. If one forgets, the process runs out of file handles (about 256
// on macOS, 1024 on Linux) and perfectly good files then fail to open.
static void test_no_handle_leaks(void) {
    mem_arena* a = arena_create(MiB(4), MiB(1));
    REQUIRE(a != NULL);
    char bad_path[64], good_path[64];

    u64 size = make_image_file(g_buf, 3, 2, 2, NULL, 10);                 // truncated: fails late
    REQUIRE(write_temp(bad_path, g_buf, size));

    for (u32 i = 0; i < 3000; i++) {
        matrix* m = idx_load_images(a, bad_path);
        if (m != NULL) { CHECK(m == NULL); break; }
    }

    size = make_image_file(g_buf, 3, 2, 2, NULL, 12);
    REQUIRE(write_temp(good_path, g_buf, size));
    CHECK(idx_load_images(a, good_path) != NULL);          // still able to open files

    u8 bad_labels[2] = { 5, 200 };                         // fails on the second label
    size = make_label_file(g_buf, 2, bad_labels, 2);
    unlink(bad_path);
    REQUIRE(write_temp(bad_path, g_buf, size));
    for (u32 i = 0; i < 3000; i++) {
        if (idx_load_labels(a, bad_path, 10) != NULL) { CHECK(false); break; }
    }
    CHECK(idx_load_images(a, good_path) != NULL);

    unlink(bad_path);
    unlink(good_path);
    arena_destroy(a);
}



// ---------------------------------------------------------------------------

int main(void) {
    RUN_TEST(test_be32);
    RUN_TEST(test_read_header);
    RUN_TEST(test_load_images);
    RUN_TEST(test_load_images_byte_order);
    RUN_TEST(test_load_labels);
    RUN_TEST(test_load_failures);
    RUN_TEST(test_no_handle_leaks);

    return test_summary();
}