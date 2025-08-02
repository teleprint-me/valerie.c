/**
 * Copyright © 2023 Austin Berrio
 *
 * @file include/interface/path.h
 *
 * @brief Path manipulation interface for C.
 */

#ifndef UTF8_PATH_H
#define UTF8_PATH_H

#include <asm/unistd.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define PATH_ACCESS_READ 0x01 // Read permission
#define PATH_ACCESS_WRITE 0x02 // Write permission
#define PATH_ACCESS_EXEC 0x04 // Execute permission

#define PATH_SEPARATOR_CHR '/'
#define PATH_SEPARATOR_STR "/"

typedef enum PathState {
    PATH_SUCCESS,
    PATH_ERROR,
    PATH_INVALID_ARGUMENT,
    PATH_PERMISSION_DENIED,
    PATH_NOT_FOUND,
    PATH_NOT_A_DIRECTORY,
    PATH_SYMLINK_LOOP,
    PATH_MEMORY_ALLOCATION,
    PATH_UNKNOWN
} PathState;

typedef enum PathNormalize {
    PATH_NORMALIZE_NONE = 0,
    PATH_NORMALIZE_ADD_LEADING_SLASH = 1 << 0,
    PATH_NORMALIZE_REMOVE_LEADING_SLASH = 1 << 1,
    PATH_NORMALIZE_ADD_TRAILING_SLASH = 1 << 2,
    PATH_NORMALIZE_REMOVE_TRAILING_SLASH = 1 << 3
} PathNormalize;

typedef enum {
    FILE_TYPE_UNKNOWN, // Unknown file type
    FILE_TYPE_REGULAR, // Regular file
    FILE_TYPE_DIRECTORY, // Directory
    FILE_TYPE_SYMLINK, // Symbolic link
    FILE_TYPE_BLOCK_DEVICE, // Block device
    FILE_TYPE_CHAR_DEVICE, // Character device
    FILE_TYPE_PIPE, // Named pipe (FIFO)
    FILE_TYPE_SOCKET // Socket
} PathType;

typedef struct {
    char* path; // Full path
    char* name; // Entry name (basename)
    char* parent; // Parent directory (dirname)
    PathType type; // File type (enum)
    off_t size; // File size
    ino_t inode; // Inode number
    uid_t uid; // Owner user ID
    gid_t gid; // Owner group ID
    time_t atime; // Last access time
    time_t mtime; // Last modification time
    time_t ctime; // Creation (or metadata change) time
    mode_t permissions; // File permissions (POSIX)
    uint8_t access; // Access flags (PATH_ACCESS_READ, WRITE, EXEC)
} PathInfo;

typedef struct PathEntry {
    PathInfo** info; // Array of PathInfo pointers
    uint32_t length; // Number of entries
} PathEntry;

typedef struct PathSplit {
    char** parts; // Array of strings for path components
    uint32_t length; // Number of components
} PathSplit;

// Lifecycle management

PathInfo* path_create_info(const char* path); // Retrieves metadata (caller must free)
void path_free_info(PathInfo* info); // Frees a PathInfo object
void path_print_info(const PathInfo* info); // Prints a PathInfo object to stdout

PathEntry* path_create_entry(
    const char* path, int current_depth, int max_depth
); // Allocates directory entries
void path_free_entry(PathEntry* entry); // Frees a PathEntry structure

PathSplit* path_split(const char* path); // Splits a path into components
void path_free_split(PathSplit* split); // Frees a PathSplit object

void path_free_string(char* path); // Frees a string returned by path functions

// Path existence and checks
bool path_is_valid(const char* path); // Checks if a path is valid
bool path_exists(const char* path); // Checks if a path exists
bool path_is_directory(const char* path); // Checks if a path is a directory
bool path_is_file(const char* path); // Checks if a path is a regular file
bool path_is_symlink(const char* path); // Checks if a path is a symbolic link

// Path normalization
bool path_has_leading_slash(const char* path); // Checks if a path has a leading slash
bool path_has_trailing_slash(const char* path); // Checks if a path has a trailing slash
char* path_normalize(const char* path, PathNormalize flags); // Normalizes a path

// Path manipulation
char* path_dirname(const char* path); // Gets the directory part of a path
char* path_basename(const char* path); // Gets the basename of a path
char* path_join(const char* base, const char* sub); // Joins two paths

#endif // UTF8_PATH_H
