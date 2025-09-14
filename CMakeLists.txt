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
    "src/type.c" # numeric data types, focusing on numeric precision
    "src/regex.c" # convenience interface to compile and free pcre2 regex
    "src/strext.c" # transitive lib to extend libc string interface
    "src/path.c" # posix compatible pathlib interface
    "src/map.c" # hash map with linear probing
)

target_include_directories(valerie PUBLIC include)
target_link_libraries(valerie PUBLIC m rt pthread pcre2-8)

set(EXAMPLES 
    "examples"
    "examples/tokenizer"
)

foreach (example IN LISTS EXAMPLES)
    add_subdirectory(${example})
endforeach()

add_custom_target(run_doxy
    COMMAND doxygen ${CMAKE_SOURCE_DIR}/doxy.conf
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating API documentation with Doxygen"
)
