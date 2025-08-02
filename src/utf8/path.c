/**
 * Copyright © 2023 Austin Berrio
 *
 * @file src/interface/path.c
 *
 * @brief
 */

#include "logger.h"
#include "utf8/path.h"

// PathInfo lifecycle

// Retrieves metadata (caller must free)
PathInfo* path_create_info(const char* path) {
    if (!path) {
        LOG_ERROR("Path is NULL!\n");
        return NULL;
    }

    struct stat statbuf;
    if (stat(path, &statbuf) != 0) {
        LOG_ERROR("Failed to stat path '%s': %s\n", path, strerror(errno));
        return NULL;
    }

    PathInfo* info = (PathInfo*) malloc(sizeof(PathInfo));
    if (!info) {
        LOG_ERROR("Failed to allocate memory for PathInfo.\n");
        return NULL;
    }

    memset(info, 0, sizeof(PathInfo));

    // Populate PathInfo fields
    info->path = strdup(path);
    info->type = S_ISREG(statbuf.st_mode)    ? FILE_TYPE_REGULAR
                 : S_ISDIR(statbuf.st_mode)  ? FILE_TYPE_DIRECTORY
                 : S_ISLNK(statbuf.st_mode)  ? FILE_TYPE_SYMLINK
                 : S_ISBLK(statbuf.st_mode)  ? FILE_TYPE_BLOCK_DEVICE
                 : S_ISCHR(statbuf.st_mode)  ? FILE_TYPE_CHAR_DEVICE
                 : S_ISFIFO(statbuf.st_mode) ? FILE_TYPE_PIPE
                 : S_ISSOCK(statbuf.st_mode) ? FILE_TYPE_SOCKET
                                             : FILE_TYPE_UNKNOWN;
    info->size = statbuf.st_size;
    info->inode = statbuf.st_ino;
    info->uid = statbuf.st_uid;
    info->gid = statbuf.st_gid;
    info->atime = statbuf.st_atime;
    info->mtime = statbuf.st_mtime;
    info->ctime = statbuf.st_ctime;
    info->permissions = statbuf.st_mode;

    // Set access flags
    info->access = 0;
    if (access(path, R_OK) == 0) {
        info->access |= PATH_ACCESS_READ;
    }
    if (access(path, W_OK) == 0) {
        info->access |= PATH_ACCESS_WRITE;
    }
    if (access(path, X_OK) == 0) {
        info->access |= PATH_ACCESS_EXEC;
    }

    return info;
}

// Frees a PathInfo object
void path_free_info(PathInfo* info) {
    if (!info) {
        return;
    }

    if (info->path) {
        free(info->path);
    }
    if (info->name) {
        free(info->name);
    }
    if (info->parent) {
        free(info->parent);
    }
    free(info);
}

// Helper to print PathInfo
void path_print_info(const PathInfo* info) {
    if (!info) {
        return;
    }

    printf("Path: %s\n", info->path);
    printf("Type: %d\n", info->type);
    printf("Size: %ld bytes\n", info->size);
    printf("Inode: %ld\n", info->inode);
    printf("Owner: UID=%d, GID=%d\n", info->uid, info->gid);
    printf("Access Time: %ld\n", info->atime);
    printf("Modification Time: %ld\n", info->mtime);
    printf("Change Time: %ld\n", info->ctime);
    printf("Permissions: %o\n", info->permissions);

    printf("Access: ");
    if (info->access & PATH_ACCESS_READ) {
        printf("Read ");
    }
    if (info->access & PATH_ACCESS_WRITE) {
        printf("Write ");
    }
    if (info->access & PATH_ACCESS_EXEC) {
        printf("Execute");
    }
    printf("\n");
}

// PathEntry lifecycle

// Allocates directory entries
PathEntry* path_create_entry(const char* path, int current_depth, int max_depth) {
    if (!path_is_valid(path) || !path_is_directory(path) || current_depth > max_depth) {
        return NULL;
    }

    DIR* dir = opendir(path);
    if (!dir) {
        LOG_ERROR("Failed to open directory '%s': %s\n", path, strerror(errno));
        return NULL;
    }

    PathEntry* entry = malloc(sizeof(PathEntry));
    if (!entry) {
        closedir(dir);
        return NULL;
    }

    entry->info = NULL;
    entry->length = 0;

    struct dirent* dir_entry;
    while ((dir_entry = readdir(dir)) != NULL) {
        if (strcmp(dir_entry->d_name, ".") == 0 || strcmp(dir_entry->d_name, "..") == 0) {
            continue;
        }

        char* entry_path = path_join(path, dir_entry->d_name);
        if (!entry_path) {
            LOG_ERROR("Failed to join path for '%s'.\n", dir_entry->d_name);
            continue;
        }

        PathInfo* info = path_create_info(entry_path);
        if (!info) {
            LOG_ERROR("Failed to retrieve metadata for '%s'.\n", entry_path);
            path_free_string(entry_path);
            continue;
        }

        // If it's a directory and within depth limit, recursively fetch sub-entries
        if (info->type == FILE_TYPE_DIRECTORY && current_depth < max_depth) {
            PathEntry* sub_entry = path_create_entry(entry_path, current_depth + 1, max_depth);
            if (sub_entry) {
                // Append sub-entry items to the current entry
                for (uint32_t i = 0; i < sub_entry->length; i++) {
                    PathInfo** new_info
                        = realloc(entry->info, sizeof(PathInfo*) * (entry->length + 1));
                    if (!new_info) {
                        path_free_info(sub_entry->info[i]);
                        continue;
                    }
                    entry->info = new_info;
                    entry->info[entry->length++] = sub_entry->info[i];
                }
                free(sub_entry->info);
                free(sub_entry);
            }
        }

        // Append current `info` to `entry->info`
        PathInfo** new_info = realloc(entry->info, sizeof(PathInfo*) * (entry->length + 1));
        if (!new_info) {
            path_free_info(info);
            path_free_string(entry_path);
            continue;
        }
        entry->info = new_info;
        entry->info[entry->length++] = info;

        path_free_string(entry_path);
    }

    closedir(dir);
    return entry;
}

// Frees a PathEntry structure
void path_free_entry(PathEntry* entry) {
    if (entry) {
        for (uint32_t i = 0; i < entry->length; i++) {
            path_free_info(entry->info[i]);
        }
        if (entry->info) {
            free(entry->info);
        }
        free(entry);
    }
}

// PathSplit lifecycle

// Splits a path into components
PathSplit* path_split(const char* path) {
    if (!path || *path == '\0') {
        return NULL;
    }

    PathSplit* split = (PathSplit*) malloc(sizeof(PathSplit));
    if (!split) {
        return NULL;
    }
    split->length = 0;
    split->parts = NULL;

    // Estimate components length and allocate memory
    char* temp = strdup(path);
    char* token = strtok(temp, "/");
    while (token) {
        split->parts = realloc(split->parts, (split->length + 1) * sizeof(char*));
        split->parts[split->length] = strdup(token);
        split->length += 1;
        token = strtok(NULL, "/");
    }

    free(temp);
    return split;
}

// Frees a PathSplit object
void path_free_split(PathSplit* split) {
    if (split) {
        if (split->parts) {
            for (uint32_t i = 0; i < split->length; i++) {
                free(split->parts[i]);
            }
            free(split->parts);
        }
        free(split);
    }
}

// String lifecycle

// Frees a string returned by path functions
void path_free_string(char* path) {
    if (path) {
        free(path);
    }
}

// Path existence and checks

bool path_is_valid(const char* path) {
    return path && *path != '\0';
}

// Checks if a path exists
bool path_exists(const char* path) {
    return path_is_valid(path) && access(path, F_OK) == 0;
}

// Checks if a path is a directory
bool path_is_directory(const char* path) {
    if (!path_is_valid(path)) {
        return false;
    }

    struct stat buffer;
    if (stat(path, &buffer) != 0) {
        LOG_ERROR("Failed to stat path '%s': %s\n", path, strerror(errno));
        return false;
    }
    return S_ISDIR(buffer.st_mode);
}

// Checks if a path is a regular file
bool path_is_file(const char* path) {
    if (!path_is_valid(path)) {
        return false;
    }

    struct stat buffer;
    if (stat(path, &buffer) != 0) {
        LOG_ERROR("Failed to stat path '%s': %s\n", path, strerror(errno));
        return false;
    }
    return S_ISREG(buffer.st_mode);
}

// Checks if a path is a symbolic link
bool path_is_symlink(const char* path) {
    if (!path_is_valid(path)) {
        return false;
    }

    struct stat buffer;
    if (lstat(path, &buffer) != 0) {
        LOG_ERROR("Failed to lstat path '%s': %s\n", path, strerror(errno));
        return false;
    }
    return S_ISLNK(buffer.st_mode);
}

// Path normalization

bool path_has_leading_slash(const char* path) {
    return path_is_valid(path) && path[0] == '/';
}

bool path_has_trailing_slash(const char* path) {
    return path_is_valid(path) && path[strlen(path) - 1] == '/';
}

char* path_normalize(const char* path, PathNormalize flags) {
    if (!path || *path == '\0') {
        return NULL; // Invalid input
    }

    size_t length = strlen(path);
    bool add_leading = flags & PATH_NORMALIZE_ADD_LEADING_SLASH;
    bool remove_leading = flags & PATH_NORMALIZE_REMOVE_LEADING_SLASH;
    bool add_trailing = flags & PATH_NORMALIZE_ADD_TRAILING_SLASH;
    bool remove_trailing = flags & PATH_NORMALIZE_REMOVE_TRAILING_SLASH;

    // Calculate new length based on flags
    size_t new_length = length;
    if (add_leading && !path_has_leading_slash(path)) {
        new_length++;
    }
    if (remove_leading && path_has_leading_slash(path)) {
        new_length--;
    }
    if (add_trailing && !path_has_trailing_slash(path)) {
        new_length++;
    }
    if (remove_trailing && path_has_trailing_slash(path)) {
        new_length--;
    }

    char* normalized = malloc(new_length + 1); // Space for null terminator
    if (!normalized) {
        return NULL; // Memory allocation failed
    }

    char* cursor = normalized;

    // Add leading slash if requested
    if (add_leading && !path_has_leading_slash(path)) {
        *cursor++ = '/';
    }

    // Skip leading slash if removing it
    const char* start = path;
    if (remove_leading && path_has_leading_slash(path)) {
        start++;
    }

    // Copy main path content, excluding trailing slash if removing it
    size_t copy_length = strlen(start);
    if (remove_trailing && path_has_trailing_slash(start)) {
        copy_length--; // Exclude trailing slash
    }
    strncpy(cursor, start, copy_length);
    cursor += copy_length;

    // Add trailing slash if requested
    if (add_trailing && !path_has_trailing_slash(path)) {
        *cursor++ = '/';
    }

    *cursor = '\0'; // Null-terminate the string
    return normalized;
}

// Path manipulation

char* path_dirname(const char* path) {
    if (!path_is_valid(path)) {
        return strdup(""); // Invalid input -> empty string
    }

    // Find the last slash
    const char* last_slash = strrchr(path, '/');
    if (!last_slash) {
        return strdup("."); // No slash -> current directory
    }

    // Handle root case (e.g., "/")
    if (last_slash == path) {
        return strdup("/");
    }

    // Extract the directory part
    size_t length = last_slash - path;
    char* dir = malloc(length + 1);
    if (!dir) {
        return strdup(""); // Fallback on allocation failure
    }

    strncpy(dir, path, length);
    dir[length] = '\0';
    return dir;
}

char* path_basename(const char* path) {
    if (!path_is_valid(path)) {
        return strdup(""); // Invalid input -> empty string
    }

    // Find the last slash
    const char* last_slash = strrchr(path, '/');
    if (!last_slash) {
        return strdup(path); // No slash -> whole path is basename
    }

    // Return the part after the last slash
    return strdup(last_slash + 1);
}

char* path_join(const char* root_path, const char* sub_path) {
    if (!path_is_valid(root_path) || !path_is_valid(sub_path)) {
        return NULL; // Invalid inputs
    }

    // Normalize root and sub paths
    char* normalized_root = path_normalize(root_path, PATH_NORMALIZE_ADD_TRAILING_SLASH);
    if (!normalized_root) {
        return NULL;
    }

    char* normalized_sub = path_normalize(sub_path, PATH_NORMALIZE_REMOVE_LEADING_SLASH);
    if (!normalized_sub) {
        free(normalized_root);
        return NULL;
    }

    // Allocate and concatenate
    size_t new_length = strlen(normalized_root) + strlen(normalized_sub) + 1;
    char* joined_path = malloc(new_length);
    if (!joined_path) {
        free(normalized_root);
        free(normalized_sub);
        return NULL;
    }

    strcpy(joined_path, normalized_root);
    strcat(joined_path, normalized_sub);

    free(normalized_root);
    free(normalized_sub);
    return joined_path;
}