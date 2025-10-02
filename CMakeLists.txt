# Root CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(valerie VERSION 0.1.0 DESCRIPTION "ANSI C library for Language Modeling")

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

# Required for clangd support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)

set(WARN "-Wall -Wextra -Wpedantic -Werror -Wformat-security -Wshadow -fexceptions")
set(EXTRA_WARN "-Wformat -Wnull-dereference -Wdouble-promotion")
set(SANITIZE "-fsanitize=address,undefined -fno-omit-frame-pointer")
set(ANALYSIS "-Wanalyzer-double-free -Wanalyzer-file-leak -Wanalyzer-malloc-leak -Wanalyzer-null-dereference -Wanalyzer-out-of-bounds -Wanalyzer-va-list-leak")
set(COMMON "-D_FILE_OFFSET_BITS=64 ${OpenMP_C_FLAGS} ${WARN}")

set(DEBUG "${COMMON} -fopenmp -g3 ${EXTRA_WARN} ${SANITIZE} ${ANALYSIS}")
set(RELEASE "${COMMON} -fopenmp -O3 -march=native")

find_package(OpenMP REQUIRED)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${DEBUG}")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${RELEASE}")
endif()

# Set the output directory for built binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Core shared library (valerie)
add_library(valerie SHARED
    # Core modules
    "src/core/memory.c"      # POSIX memory allocation and deallocation
    "src/core/logger.c"      # Logging utility
    "src/core/test.c"        # Unit testing utility
    "src/core/regex.c"       # PCRE2 regex compile/free wrapper
    "src/core/strext.c"      # Libc string extensions (transitive helpers)
    "src/core/path.c"        # POSIX-compatible path utilities
    "src/core/hash.c"        # General-purpose hash table (base for map/set)
    "src/core/map.c"         # Hash map with linear probing (key-value store)
    "src/core/set.c"         # Hash set (unique keys, set algebra)
    "src/core/page.c"        # Hash page (table style memory allocator)
    "src/core/lehmer.c"      # Lehmer linear congruential generator
    "src/core/sort.c"        # Heap sort routines
    "src/core/compare.c"     # Numeric and floating-point comparisons
    "src/core/type.c"        # Numeric data types (precision metadata)

    # Tokenizer modules
    "src/tokenizer/vocab.c"  # String-to-int frequency vocab map
    "src/tokenizer/bpe.c"    # Byte Pair Encoding (BPE) algorithms
    "src/tokenizer/model.c"  # BPE tokenizer model interface

    # Model modules
    "src/model/activation.c" # Activation functions and their derivatives
    "src/model/matrix.c"     # Type-generic matrix operations (forward, backward, SGD)
)

target_include_directories(valerie PUBLIC include)
target_link_libraries(valerie PUBLIC m rt pthread pcre2-8)

set(EXAMPLES 
    "examples"
    "examples/core"
    "examples/tokenizer"
    "examples/model"
)

foreach (example IN LISTS EXAMPLES)
    add_subdirectory(${example})
endforeach()

add_custom_target(run_doxy
    COMMAND doxygen ${CMAKE_SOURCE_DIR}/doxy.conf
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen"
)
