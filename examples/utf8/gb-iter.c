// @file examples/gb-iter.c
#include "utf8/grapheme.h"
#include <stdio.h>

int main(void) {
    UTF8GraphemeIter it = utf8_gcb_iter(
        "a😀b́👨‍👩‍👧‍👦c👍🏽d🇨🇳e👩🏽‍🚒á👩‍❤️‍💋‍👩á̊🇺🇸f1️⃣g👍🏿123"
    );

    const char* cluster;
    while ((cluster = utf8_gcb_iter_next(&it))) {
        printf("[%s]\n", cluster);
    }

    return 0;
}
