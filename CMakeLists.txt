# Root CMakeLists.txt

cmake_minimum_required(VERSION 3.14)
project(valerie VERSION 0.1.0 DESCRIPTION "ANSI C library for Large Language Modeling")

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

set(DEBUG "${COMMON} -g3 ${EXTRA_WARN} ${SANITIZE} ${ANALYSIS}")
set(RELEASE "${COMMON} -Ofast -march=native")

find_package(OpenMP REQUIRED)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${DEBUG}")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${RELEASE}")
endif()

# Set the output directory for built binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Add library
add_library(valerie SHARED
    "src/memory.c" # posix memory allocation and deallocation
    "src/logger.c" # logging utility
    "src/unit.c" # unit testing utility
    "src/page.c" # page table memory allocator
    "src/compare.c" # numeric comparisons, including floating-point
    "src/xorshift.c" # numeric psuedo random number generator
    "src/type.c" # numeric data types, focusing on numeric precision
    "src/utf8/regex.c"
    "src/utf8/byte.c"
    "src/utf8/codepoint.c"
    "src/utf8/grapheme-data.c"
    "src/utf8/grapheme.c"
    "src/utf8/string.c"
    "src/utf8/path.c"
    "src/map.c" # hash map with linear probing
)
target_include_directories(valerie PUBLIC include)
target_link_libraries(valerie PUBLIC m rt pthread pcre2-8)

add_subdirectory(examples)

# enable_testing()
# add_subdirectory(tests)
# add_custom_target(run_tests
#     COMMAND ${CMAKE_CTEST_COMMAND}
#     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
#     COMMENT "Running all tests"
# )

add_custom_target(run_doxy
    COMMAND doxygen ${CMAKE_SOURCE_DIR}/doxy.conf
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen"
)

add_custom_target(run_clean
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}
    COMMENT "Cleaning all generated files"
)
