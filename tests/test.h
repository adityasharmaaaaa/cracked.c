#pragma once
 
static int g_checks = 0;
static int g_failed = 0;
 
#define CHECK(cond)                                                         \
    do {                                                                    \
        g_checks++;                                                         \
        if (!(cond)) {                                                      \
            g_failed++;                                                     \
            printf("    FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
        }                                                                   \
    } while (0)
 
#define RUN_TEST(fn)                                                        \
    do {                                                                    \
        int failed_before = g_failed;                                       \
        fn();                                                               \
        printf("[%s] %s\n", g_failed == failed_before ? "PASS" : "FAIL", #fn); \
    } while (0)
 
static inline int test_summary(void) {
    printf("\n%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}

