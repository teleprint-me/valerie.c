"""
unicode.grapheme — Unicode Grapheme Cluster Data Generator
==========================================================

Auto-generates C header/source files containing structured Unicode
grapheme cluster break properties for use in dependency-free
text-processing libraries (e.g., for grapheme cluster detection in C).

Features
--------
- Downloads and caches the latest Unicode property data files:
    - GraphemeBreakProperty.txt
    - emoji-data.txt
    - PropList.txt
- Maps codepoint ranges to compact enums for efficient lookup.
- Emits C code with deduplicated, extendable grapheme break categories.

Output
------
- include/utf8/grapheme-data.h    # Grapheme type enum + lookup struct
- src/utf8/grapheme-data.c        # Data array of Unicode codepoint ranges

Usage
-----
Run directly as a module (no arguments):

    python -m unicode.grapheme

On success, generates/updates the above files in-place.
Exit status 0 = success; 1 = failure. No output is emitted on success.

Notes
-----
- Files are cached to `data/` on first run; only re-fetched if absent.
- The C interface is designed for inclusion in portable, dependency-free code.
- Grapheme categories are based on the latest Unicode UCD and emoji data.
- Output files are overwritten on each invocation.

References
----------
- Unicode Standard Annex #29: Unicode Text Segmentation
- https://www.unicode.org/reports/tr29/
- https://www.unicode.org/Public/UCD/latest/ucd/
"""

import os
from dataclasses import dataclass
from datetime import date

import requests


@dataclass
class GraphemeType:
    GCB_UNDEFINED = 0
    # GraphemeBreakProperty.txt
    GCB_PREPEND = 1
    GCB_CR = 2
    GCB_LF = 3
    GCB_CONTROL = 4
    GCB_EXTEND = 5
    GCB_SPACING_MARK = 6
    GCB_L = 7
    GCB_V = 8
    GCB_T = 9
    GCB_LV = 10
    GCB_LVT = 11
    GCB_ZJW = 12
    # emoji-data.txt
    GCB_EMOJI = 13
    GCB_EMOJI_PRESENTATION = 14
    GCB_EMOJI_MODIFIER_BASE = 15
    GCB_EMOJI_COMPONENT = 16
    GCB_EXTENDED_PICTOGRAPHIC = 17
    # PropList.txt
    GCB_BIDI_CONTROL = 18
    GCB_EXTENDER = 19
    GCB_OTHER_GRAPHEME_EXTEND = 20
    GCB_DIACRITIC = 21
    GCB_PREPENDED_CONCATENATION_MARK = 22
    GCB_MODIFIER_COMBINING_MARK = 23


GraphemeMap = {
    "Undefined": GraphemeType.GCB_UNDEFINED,
    # GraphemeBreakProperty.txt
    "Prepend": GraphemeType.GCB_PREPEND,
    "CR": GraphemeType.GCB_CR,
    "LF": GraphemeType.GCB_LF,
    "Control": GraphemeType.GCB_CONTROL,
    "Extend": GraphemeType.GCB_EXTEND,
    "SpacingMark": GraphemeType.GCB_SPACING_MARK,
    "L": GraphemeType.GCB_L,
    "V": GraphemeType.GCB_V,
    "T": GraphemeType.GCB_T,
    "LV": GraphemeType.GCB_LV,
    "LVT": GraphemeType.GCB_LVT,
    "ZWJ": GraphemeType.GCB_ZJW,
    # emoji-data.txt
    "Emoji": GraphemeType.GCB_EMOJI,
    "Emoji_Presentation": GraphemeType.GCB_EMOJI_PRESENTATION,
    "Emoji_Modifier_Base": GraphemeType.GCB_EMOJI_MODIFIER_BASE,
    "Emoji_Component": GraphemeType.GCB_EMOJI_COMPONENT,
    "Extended_Pictographic": GraphemeType.GCB_EXTENDED_PICTOGRAPHIC,
    # PropList.txt
    "Bidi_Control": GraphemeType.GCB_BIDI_CONTROL,
    "Extender": GraphemeType.GCB_EXTENDER,
    "Other_Grapheme_Extend": GraphemeType.GCB_OTHER_GRAPHEME_EXTEND,
    "Diacritic": GraphemeType.GCB_DIACRITIC,
    "Prepended_Concatenation_Mark": GraphemeType.GCB_PREPENDED_CONCATENATION_MARK,
    "Modifier_Combining_Mark": GraphemeType.GCB_MODIFIER_COMBINING_MARK,
}

# List of supported grapheme classes for Unicode segmentation.
#
# Note:
# - Some classes (e.g., Hangul: L, V, T, LV, LVT) are included for completeness, but are not required for most non-Korean text processing.
# - Many categories overlap (e.g., Extend, Extender, Diacritic) and may assign the same codepoint to multiple classes.
# - The Emoji-related classes are required for correct grapheme break behavior in modern Unicode text (e.g., flags, keycap sequences, skin tone modifiers).
# - PropList-only properties (Diacritic, Other_Grapheme_Extend) are rarely used, but included for edge cases.
# - Regional indicator sequences are classified under "Emoji" and "Emoji_Component".
# - This list is processed in order; matching is greedy and first-match-wins.
GRAPHEMES = [
    "Undefined",  # fallback/unclassified codepoints
    # GraphemeBreakProperty.txt
    "CR",  # Carriage Return
    "LF",  # Line Feed
    "Control",
    "Extend",
    "Prepend",
    "SpacingMark",
    # Hangul Syllable Types (rarely used outside Korean)
    "L",
    "V",
    "T",
    "LV",
    "LVT",
    "ZWJ",  # Zero Width Joiner
    # emoji-data.txt (Emoji-related properties)
    "Emoji",  # includes regional indicators
    "Emoji_Presentation",  # emoji by default
    "Emoji_Modifier_Base",  # base for skin tone, hair modifiers
    "Emoji_Component",  # sub-components, includes regionals
    "Extended_Pictographic",  # broad pictographic class
    # PropList.txt (rare; covers obscure edge cases)
    "Bidi_Control",
    "Extender",
    "Other_Grapheme_Extend",
    "Diacritic",
    "Prepended_Concatenation_Mark",
    "Modifier_Combining_Mark",
]


def unicode_data_fetch(url: str, path: str) -> list[str]:
    # Read from local cache
    if os.path.isfile(path):
        with open(path, "r") as file:
            return file.read().splitlines()
    # Write to local cache
    lines = requests.get(url).text.splitlines()
    with open(path, "w") as file:
        for line in lines:
            file.write(line + "\n")
    return lines


def unicode_data_cache() -> list[str]:
    base_dir = "data"
    base_url = "https://www.unicode.org/Public/UCD/latest/ucd"
    os.makedirs(base_dir, exist_ok=True)
    files = {
        "grapheme": (
            f"{base_url}/auxiliary/GraphemeBreakProperty.txt",
            f"{base_dir}/GraphemeBreakProperty.txt",
        ),
        "emoji": (
            f"{base_url}/emoji/emoji-data.txt",
            f"{base_dir}/emoji-data.txt",
        ),
        "prop": (
            f"{base_url}/PropList.txt",
            f"{base_dir}/PropList.txt",
        ),
    }
    lines = []
    for _, (url, path) in files.items():
        lines.extend(unicode_data_fetch(url, path))
    return lines


# C struct is defined as struct { uint32_t lo, hi; }
# lo for the start of the range if it is a tuple or range
# hi for the end of the range if it is a tuple or range
# both lo and hi are equal if it is a singleton
def unicode_data_parse(lines: list[str]) -> list[tuple[int, int, int]]:
    ranges = []
    for line in lines:
        if not line or line[0] == "#" or ";" not in line:
            continue

        lvalue, rvalue = list(value.strip() for value in line.split(";"))

        grapheme = rvalue.split("#")[0].strip()
        if grapheme not in GRAPHEMES:
            continue

        # handle single or range
        if ".." in lvalue:  # range
            lo, hi = list(int(x, 16) for x in lvalue.split(".."))
        elif " " in lvalue:  # tuple
            lo, hi = list(int(x, 16) for x in lvalue.split(" "))
        else:  # singleton
            lo = hi = int(lvalue.strip(), 16)
        ranges.append((lo, hi, GraphemeMap[grapheme]))
    return ranges


def unicode_data_enum_map() -> dict[int, str]:
    # Map grapheme enums to names
    return {v: f"GCB_{k.upper()}" for k, v in GraphemeMap.items()}


def unicode_data_comments() -> list[str]:
    version = 1
    lines = []
    lines.append("/**")
    lines.append(" * @warning This file is auto-generated. Do not edit directly.")
    lines.append(" * @brief Grapheme cluster break property data.")
    lines.append(" * @ref Unicode UCD - Generated by unicode.grapheme.py")
    lines.append(f" * @version {version}")
    lines.append(f" * @date {date.today()}")
    lines.append(" */\n")
    return lines


def unicode_data_include(ranges: list[tuple[int, int, int]]) -> str:
    enum_map = unicode_data_enum_map()

    lines = unicode_data_comments()

    lines.append("#ifndef UTF8_GRAPHEME_DATA_H")
    lines.append("#define UTF8_GRAPHEME_DATA_H\n")

    lines.append("#include <stddef.h>")
    lines.append("#include <stdint.h>\n")

    lines.append("typedef enum UTF8GraphemeClass {")
    for v in GraphemeMap.values():
        lines.append(f"    {enum_map[v]} = {v},")
    lines.append("} UTF8GraphemeClass;\n")

    lines.append("typedef struct UTF8Grapheme {")
    lines.append("    uint32_t lo, hi;")
    lines.append("    UTF8GraphemeClass cls;")
    lines.append("} UTF8Grapheme;\n")

    lines.append("extern const UTF8Grapheme graphemes[];")
    lines.append("extern const size_t UTF8_GRAPHEME_SIZE;\n")

    lines.append("#endif // UTF8_GRAPHEME_DATA_H\n")

    return "\n".join(lines)


def unicode_data_source(ranges: list[tuple[int, int, int]]) -> str:
    enum_map = unicode_data_enum_map()

    lines = unicode_data_comments()

    lines.append('#include "utf8/grapheme-data.h"\n')

    lines.append("const UTF8Grapheme graphemes[] = {")
    for lo, hi, t in ranges:
        lines.append(
            f"    {{0x{lo:06X}, 0x{hi:06X}, {enum_map[t]}}}, // {GRAPHEMES[t]}"
        )
    lines.append("};\n")

    lines.append("const size_t UTF8_GRAPHEME_SIZE = sizeof(graphemes) / sizeof(UTF8Grapheme);\n")

    return "\n".join(lines)


def main():
    cache = unicode_data_cache()
    data = unicode_data_parse(cache)

    header = unicode_data_include(data)
    with open("include/utf8/grapheme-data.h", "w") as file:
        file.write(header)

    source = unicode_data_source(data)
    with open("src/utf8/grapheme-data.c", "w") as file:
        file.write(source)


if __name__ == "__main__":
    main()
