/**
 * @file      path.c
 * @author    Austin Berrio
 * @copyright Copyright © 2025
 * @brief     A POSIX C pathlib interface.
 */

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <unistd.h>

#include "core/path.h"

// Checks if path is valid input
bool path_is_valid(const char* path) {
    return path && *path != '\0';
}

// Checks if a path exists
bool path_exists(const char* path) {
    struct stat buffer;
    return path_is_valid(path) && stat(path, &buffer) == 0;
}

// Checks if path is a directory
bool path_is_dir(const char* path) {
    if (!path_is_valid(path)) {
        return false;
    }

    struct stat buffer;
    if (stat(path, &buffer) != 0) {
        return false;
    }
    return S_ISDIR(buffer.st_mode);
}

// Checks if path is a regular file
bool path_is_file(const char* path) {
    if (!path_is_valid(path)) {
        return false;
    }

    struct stat buffer;
    if (stat(path, &buffer) != 0) {
        return false;
    }
    return S_ISREG(buffer.st_mode);
}

// Saner mkdir wrapper
bool path_mkdir(const char* path) {
    if (mkdir(path, 0755) == -1 && errno != EEXIST) {
        return false;
    }
    return true;
}

// Returns the directory path
char* path_dirname(const char* path) {
    if (!path_is_valid(path)) {
        return strdup("");  // Invalid input -> empty string
    }

    // Find the last slash
    const char* last_slash = strrchr(path, '/');
    if (!last_slash) {
        return strdup(".");  // No slash -> current directory
    }

    // Handle root case (e.g., "/")
    if (last_slash == path) {
        return strdup("/");
    }

    // Extract the directory part
    size_t length = last_slash - path;
    char* dir = (char*) malloc(length + 1);
    if (!dir) {
        return strdup("");  // Fallback on allocation failure
    }

    strncpy(dir, path, length);
    dir[length] = '\0';
    return dir;
}

// Returns the file name
char* path_basename(const char* path) {
    if (!path_is_valid(path)) {
        return strdup("");  // Invalid input -> empty string
    }

    // Find the last slash
    const char* last_slash = strrchr(path, '/');
    if (!last_slash) {
        return strdup(path);  // No slash -> whole path is basename
    }

    // Return the part after the last slash
    return strdup(last_slash + 1);
}

// Concatenate two path components, inserting a '/' if needed
char* path_join(const char* root, const char* sub) {
    if (!path_is_valid(root) || !path_is_valid(sub)) {
        return NULL;
    }

    size_t len_dir = strlen(root);
    size_t len_file = strlen(sub);
    int needs_slash = (len_dir > 0 && root[len_dir - 1] != '/');
    size_t total_len = len_dir + needs_slash + len_file + 1;

    char* path = (char*) malloc(total_len);
    if (!path) {
        return NULL;
    }

    strcpy(path, root);
    if (needs_slash) {
        strcat(path, "/");
    }
    strcat(path, sub);

    return path;
}

// Splits a path into components
char** path_split(const char* path, size_t* count) {
    if (!path_is_valid(path)) {
        return NULL;
    }

    *count = 0;
    char** parts = NULL;

    // Estimate components length and allocate memory
    char* temp = strdup(path);
    char* token = strtok(temp, "/");
    while (token) {
        parts = (char**) realloc(parts, (*count + 1) * sizeof(char*));
        parts[*count] = strdup(token);
        *count += 1;
        token = strtok(NULL, "/");
    }

    free(temp);
    return parts;
}

// Read directory paths into memory (dirs only)
char** path_list_dirs(const char* dirname, size_t* count) {
    if (!path_is_dir(dirname)) {
        return NULL;
    }

    DIR* dir = opendir(dirname);
    if (!dir) {
        return NULL;
    }

    *count = 0;
    char** dirs = NULL;

    struct dirent* entry;
    while ((entry = readdir(dir))) {
        int current = strcmp(entry->d_name, ".");
        int previous = strcmp(entry->d_name, "..");
        if (0 == current || 0 == previous) {
            continue;
        }

        char* fullpath = path_join(dirname, entry->d_name);
        if (!path_is_dir(fullpath)) {
            free(fullpath);
            continue;
        }

        dirs = (char**) realloc(dirs, (*count + 1) * sizeof(char*));
        dirs[*count] = fullpath;
        (*count)++;
    }

    closedir(dir);
    return dirs;
}

// Read file paths into memory (files only)
char** path_list_files(const char* dirname, size_t* count) {
    if (!path_is_dir(dirname)) {
        return NULL;
    }

    DIR* dir = opendir(dirname);
    if (!dir) {
        return NULL;
    }

    *count = 0;
    char** files = NULL;

    struct dirent* entry;
    while ((entry = readdir(dir))) {
        int current = strcmp(entry->d_name, ".");
        int previous = strcmp(entry->d_name, "..");
        if (0 == current || 0 == previous) {
            continue;
        }

        char* fullpath = path_join(dirname, entry->d_name);
        if (!path_is_file(fullpath)) {
            free(fullpath);
            continue;
        }

        files = (char**) realloc(files, (*count + 1) * sizeof(char*));
        files[*count] = fullpath;
        (*count)++;
    }

    closedir(dir);
    return files;
}

// Free all allocated path components
void path_free_parts(char** parts, size_t count) {
    if (parts) {
        for (size_t i = 0; i < count; i++) {
            if (path_is_valid(parts[i])) {
                free(parts[i]);
            }
        }
        free(parts);
    }
}

// Free an allocated path component
void path_free(char* path) {
    if (path_is_valid(path)) {
        free(path);
    }
}
