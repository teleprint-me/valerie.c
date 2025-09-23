/**
 * Copyright © 2023 Austin Berrio
 *
 * @file core/logger.h
 * @brief A simple and lightweight thread-safe logger in pure C.
 *
 * Provides multi-level logging (DEBUG, INFO, WARN, ERROR) to console or files.
 * Designed for easy integration into C projects with thread safety via mutexes.
 */

#ifndef LOGGER_H
#define LOGGER_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdlib.h>  // IWYU pragma: keep
#include <string.h>
#include <stdio.h>

#include <errno.h>  // IWYU pragma: keep
#include <string.h>  // IWYU pragma: keep
#include <pthread.h>

/**
 * @brief Logging severity levels.
 */
typedef enum LogLevel {
    LOG_LEVEL_DEBUG, /**< Detailed debug information. */
    LOG_LEVEL_INFO, /**< General informational messages. */
    LOG_LEVEL_WARN, /**< Warning conditions. */
    LOG_LEVEL_ERROR /**< Error conditions. */
} LogLevel;

/**
 * @brief Types of logging output.
 */
typedef enum LogType {
    LOG_TYPE_UNKNOWN, /**< Unknown or uninitialized logger type. */
    LOG_TYPE_STREAM, /**< Logging to a stream (e.g., stdout, stderr). */
    LOG_TYPE_FILE /**< Logging to a file. */
} LogType;

/**
 * @brief Logger object encapsulating log state.
 */
typedef struct Logger {
    LogLevel log_level; /**< Current log level threshold. */
    LogType log_type; /**< Type of logging output. */
    const char* log_type_name; /**< String name of logger type. */
    FILE* file_stream; /**< File or stream pointer for logging. */
    const char* file_path; /**< Path to log file if applicable. */
    pthread_mutex_t thread_lock; /**< Mutex for thread-safe logging. */
} Logger;

/**
 * @name Logger Lifecycle
 * @{
 */

/**
 * @brief Sets logger type and corresponding name.
 *
 * @param logger Logger instance to update.
 * @param log_type Desired logger type.
 * @return true if successful, false otherwise.
 */
bool logger_set_type_and_name(Logger* logger, LogType log_type);

/**
 * @brief Sets file path and opens file stream for logging.
 *
 * If file_path is NULL or cannot be opened, defaults to stderr.
 *
 * @param logger Logger instance to update.
 * @param file_path Path to log file or NULL for stderr.
 * @return true if file stream set successfully, false otherwise.
 */
bool logger_set_file_path_and_stream(Logger* logger, const char* file_path);

/**
 * @brief Creates a new logger with default settings.
 *
 * Allocates and initializes a Logger instance with given type.
 *
 * @param log_type Desired logger type.
 * @return Pointer to newly created Logger or NULL on failure.
 */
Logger* logger_new(LogType log_type);

/**
 * @brief Creates a new logger with specified log level, type, and optional file.
 *
 * @param log_level Log level threshold.
 * @param log_type Logger type.
 * @param file_path Optional log file path (NULL logs to stderr).
 * @return Pointer to newly created Logger or NULL on failure.
 */
Logger* logger_create(LogLevel log_level, LogType log_type, const char* file_path);

/**
 * @brief Frees a Logger instance and associated resources.
 *
 * Closes open file streams and destroys mutex.
 *
 * @param logger Pointer to Logger to free.
 * @return true if successfully freed, false otherwise.
 */
bool logger_free(Logger* logger);

/** @} */

/**
 * @name Logging Functions
 * @{
 */

/**
 * @brief Logs a formatted message if log_level is sufficient.
 *
 * Thread-safe: acquires mutex during logging.
 *
 * @param logger Logger instance to use.
 * @param log_level Severity level of this message.
 * @param format printf-style format string.
 * @param ... Variadic arguments for format.
 * @return true if message logged successfully, false otherwise.
 */
bool logger_message(Logger* logger, LogLevel log_level, const char* format, ...);

/** @} */

/**
 * @name Logging Macros
 * @{
 */

// monkey patch to print only the basename of the filepath.
// this needs to be revisited in the future. this is fine for now.
#define LOG_BASENAME(path) (strrchr((path), '/') ? strrchr((path), '/') + 1 : (path))

/**
 * @brief Macro to log messages with file, function, and line info.
 *
 * Usage:
 * @code
 * LOG(logger, LOG_LEVEL_DEBUG, "Value: %d", val);
 * @endcode
 */
#define LOG(logger, level, format, ...) \
    logger_message( \
        (logger), \
        (level), \
        "[%s:%s:%d] " format "\n", \
        LOG_BASENAME(__FILE__), \
        __func__, \
        __LINE__, \
        ##__VA_ARGS__ \
    )

/**
 * @brief Convenience macros for logger_global usage.
 */
#define LOG_DEBUG(format, ...) LOG(&logger_global, LOG_LEVEL_DEBUG, format, ##__VA_ARGS__)
#define LOG_INFO(format, ...) LOG(&logger_global, LOG_LEVEL_INFO, format, ##__VA_ARGS__)
#define LOG_WARN(format, ...) LOG(&logger_global, LOG_LEVEL_WARN, format, ##__VA_ARGS__)
#define LOG_WARNING LOG_WARN  // Keep it simple, smart-ass
#define LOG_ERROR(format, ...) LOG(&logger_global, LOG_LEVEL_ERROR, format, ##__VA_ARGS__)

/** @} */

/**
 * @brief Global logger instance for application-wide logging.
 *
 * Initialized with sane defaults. Thread-safe.
 */
extern Logger logger_global;

/**
 * @brief Mutates global logger with specified parameters.
 *
 * Should be called once before logger_global usage.
 *
 * @param log_level Desired log level.
 * @param log_type Logger type.
 * @param log_type_name Name string of logger type.
 * @param file_stream FILE pointer for logging (NULL for default).
 * @param file_path File path if applicable (NULL otherwise).
 */
void logger_set_global(
    LogLevel log_level,
    LogType log_type,
    const char* log_type_name,
    FILE* file_stream,
    const char* file_path
);

#endif  // LOGGER_H
