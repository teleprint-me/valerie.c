/**
 * @file      core/path.h
 * @author    Austin Berrio
 * @copyright Copyright © 2025
 * @brief     A POSIX C pathlib interface.
 */

#ifndef PATHLIB_H
#define PATHLIB_H

#include <stdbool.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef PATH_MAX
    #define PATH_MAX 4096
#endif  // PATH_MAX

// Checks if path is valid input
bool path_is_valid(const char* path);

// Checks if path exists
bool path_exists(const char* path);

// Checks if path is a directory
bool path_is_dir(const char* path);

// Checks if path is a regular file
bool path_is_file(const char* path);

// Saner mkdir wrapper (true on success, false on failure)
bool path_mkdir(const char* path);

// Returns the directory path
char* path_dirname(const char* path);

// Returns the file name
char* path_basename(const char* path);

// Concatenate two path components, inserting a '/' if needed
char* path_cat(const char* root, const char* sub);

// Splits a path into components
char** path_split(const char* path, size_t* count);

// Read directory paths into memory (dirs only)
char** path_list_dirs(const char* dirname, size_t* count);

// Read file paths into memory (files only)
char** path_list_files(const char* dirname, size_t* count);

// Free all allocated path components
void path_free_parts(char** parts, size_t count);

// Free an allocated path component
void path_free(char* path);

#ifdef __cplusplus
}
#endif

#endif  // PATHLIB_H
