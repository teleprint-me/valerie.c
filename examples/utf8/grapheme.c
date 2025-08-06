#include "utf8/grapheme.h"
#include <stdint.h>
#include <stdio.h>

int main(void) {
    const char* tests[] = {
        // Simple
        "Hello!",
        // CJK Ideographs
        "你好世界！",
        // Greek
        "Γεια σου κόσμο!",
        // Emoji (single)
        "😀",
        // Emoji (skin tone modifier)
        "👍🏿",
        // Emoji (ZWJ sequence: family)
        "👨‍👩‍👧‍👦",
        // Emoji (flag sequence, regional indicators)
        "🇺🇸🇯🇵",
        // Combining diacritics (Latin small letter a with acute)
        "á", // [a][U+0301 combining acute] -- two codepoints, one grapheme
        // Multiple combining marks
        "á̊", // [a][U+0301][U+030A] -- three codepoints, one grapheme
        // Hangul syllables (should not split individual jamos)
        "가나다",
        // Arabic (joined letters, bidi)
        "السلام عليكم",
        // Thai (consonant + vowel + tone)
        "ก้",
        // Cyrillic
        "привет мир",
        // Test: Zero Width Joiner, emoji gender
        "👩🏽‍🚒",
        // Test: Emoji keycap
        "1️⃣", // [1][variation selector][keycap]
        // Test: Emoji sequence with combining and ZWJ
        "👩‍❤️‍💋‍👩",
        // Edge: Control chars
        "A[\x1F]B", // [A][unit separator][B]
        // Test: Hebrew with points
        "שָׁלוֹם",
        // Hindi (Devanagari) complex cluster
        "क्‍ष", // [Ka][virama][ZWJ][Ṣa]
        // Just for fun: long sequence
        "a😀b́c👍🏽d🇨🇳e",
    };

    size_t size = sizeof(tests) / sizeof(char*);

    for (size_t i = 0; i < size; i++) {
        fprintf(stderr, "test[%zu] %s\n", i, tests[i]);

        size_t capacity = 0;
        char** parts = utf8_gcb_split(tests[i], &capacity);
        if (!parts) {
            fprintf(stderr, "[Error] Failed to create grapheme split.\n");
            continue;
        }

        utf8_gcb_split_dump(parts, capacity);
        utf8_gcb_split_free(parts, capacity);
    }

    return 0;
}
