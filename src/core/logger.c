/**
 * Copyright © 2023 Austin Berrio
 *
 * @file core/logger.c
 * @brief A simple and lightweight logger written in pure C
 *
 * `logger.c` provides a minimal logging library that allows you to log
 * messages with different severity levels (e.g., DEBUG, INFO, WARN, ERROR) to
 * various outputs like console or file. It's designed to be lightweight and
 * easily integrated into existing C projects.
 */

#include "core/memory.h"
#include "core/logger.h"

const char* LOG_TYPE_NAME[] = {"unknown", "stream", "file"};

/**
 * @brief Sets the logger type and name.
 *
 * This function validates and sets the logger type and name
 * based on the provided logger type.
 *
 * @param logger The logger structure to update.
 * @param log_type The desired logger type.
 *
 * @return True if the type and name were set successfully, false otherwise.
 */
bool logger_set_type_and_name(Logger* logger, LogType log_type) {
    // Set logger type based on provided logger type
    switch (log_type) {
        case LOG_TYPE_UNKNOWN:
        case LOG_TYPE_STREAM:
            logger->log_type = LOG_TYPE_STREAM;
            logger->log_type_name = LOG_TYPE_NAME[LOG_TYPE_STREAM];
            return true;
        case LOG_TYPE_FILE:
            logger->log_type = LOG_TYPE_FILE;
            logger->log_type_name = LOG_TYPE_NAME[LOG_TYPE_FILE];
            return true;
        default:
            fprintf(stderr, "Invalid logger type\n");
            return false;
    }
}

/**
 * @brief Sets the file path and stream for the logger.
 *
 * This function sets the file path and stream for the logger based on the
 * provided file path. If the file path is NULL, the logger stream is set to
 * stderr, and false is returned to indicate a user error. If the file path
 * is not NULL, an attempt is made to open the file for writing. If the file
 * opening fails, the logger stream is set to stderr, and false is returned
 * to indicate an unexpected condition. If the file path is set successfully,
 * true is returned.
 *
 * @param logger The logger structure to update.
 * @param file_path The path to the log file. Pass NULL to log messages to
 * stderr.
 *
 * @return True if the file path was set successfully, false otherwise.
 */
bool logger_set_file_path_and_stream(Logger* logger, const char* file_path) {
    if (file_path == NULL) {
        // set the logger type to stream upon failure.
        logger_set_type_and_name(logger, LOG_TYPE_STREAM);
        logger->file_stream = stderr;
        return false; // File path is NULL (user error)
    }

    logger->file_stream = fopen(file_path, "w");
    if (logger->file_stream == NULL) {
        // set the logger type to stream upon failure.
        logger_set_type_and_name(logger, LOG_TYPE_STREAM);
        logger->file_stream = stderr;
        return false; // Failed to open file (unexpected condition)
    }

    // File path set successfully
    logger->file_path = file_path;
    return true;
}

/**
 * @brief Creates a new logger instance.
 *
 * This function dynamically allocates memory for a new logger instance
 * and initializes it with sane default values.
 *
 * @param log_type The desired logger type.
 *
 * @return A pointer to the newly created logger instance, or NULL if memory
 * allocation fails or if the logger type is invalid.
 */
Logger* logger_new(LogType log_type) {
    // Allocate memory for the logger instance
    Logger* logger = (Logger*) memory_alloc(sizeof(Logger), alignof(Logger));

    // Check if memory allocation was successful
    if (NULL == logger) {
        fprintf(stderr, "Failed to allocate memory for logger\n");
        return NULL;
    }

    // Set default values for the logger
    logger->log_level = LOG_LEVEL_DEBUG;

    // Set logger type and name
    if (!logger_set_type_and_name(logger, log_type)) {
        fprintf(stderr, "Failed to initialize logger with type: %d\n", log_type);
        memory_free(logger); // Clean up allocated memory to avoid memory leaks
        return NULL;
    }

    logger->file_path = NULL;
    logger->file_stream = NULL;

    // Initialize the mutex for thread safety
    int error_code = pthread_mutex_init(&logger->thread_lock, NULL);
    if (0 != error_code) {
        fprintf(stderr, "Failed to initialize mutex with error: %d\n", error_code);
        memory_free(logger); // Clean up allocated memory to avoid memory leaks
        return NULL;
    }

    return logger; // Return the created logger instance
}

/**
 * @brief Creates a new logger instance with the specified log file path and
 * log level.
 *
 * This function creates a new logger instance and initializes it with the
 * specified log level. If a log file path is provided, the logger will attempt
 * to open the file for writing. If the file opening fails, the logger will
 * fall back to writing log messages to stderr.
 *
 * @param log_level The desired log level for the logger.
 * @param log_type The desired log type for the logger.
 * @param file_path The path to the log file. Pass NULL to log messages to
 * stderr.
 *
 * @return A pointer to the newly created logger instance, or NULL if memory
 * allocation fails or if the specified log file cannot be opened.
 */
Logger* logger_create(LogLevel log_level, LogType log_type, const char* file_path) {
    // Create a new logger instance
    Logger* logger = logger_new(log_type);
    if (logger == NULL) {
        return NULL;
    }

    // Set the log level
    logger->log_level = log_level;

    // Set the file path and stream based on the logger type
    switch (log_type) {
        case LOG_TYPE_UNKNOWN:
        case LOG_TYPE_STREAM:
            // Use stdout or stderr directly
            logger->file_stream = stderr;
            break;
        case LOG_TYPE_FILE:
            if (!logger_set_file_path_and_stream(logger, file_path)) {
                fprintf(stderr, "Failed to set log file path. Fallback to stderr.\n");
            }
            break;
        default:
            // Unknown logger type; fallback to stderr
            fprintf(stderr, "Unknown logger type. Fallback to stderr.\n");
            logger->file_stream = stderr;
            break;
    }

    return logger;
}

/**
 * @brief Destroys a logger instance and releases associated resources.
 *
 * This function closes the log file associated with the logger, if any,
 * and frees the memory allocated for the logger instance.
 *
 * @param logger A pointer to the logger instance to be destroyed.
 * @return True if the logger was successfully destroyed, false otherwise.
 */
bool logger_free(Logger* logger) {
    if (NULL == logger) {
        return false;
    }

    // Close the log file if it's a file logger
    if (logger->file_stream && LOG_TYPE_FILE == logger->log_type) {
        if (fclose(logger->file_stream) != 0) {
            fprintf(stderr, "Failed to close log file: %s\n", logger->file_path);
            return false; // return false to prevent dangling pointers
        }
    }

    // Destroy the mutex
    int mutex_error = pthread_mutex_destroy(&logger->thread_lock);
    if (0 != mutex_error) {
        fprintf(stderr, "Failed to destroy mutex with error: %d\n", mutex_error);
        return false; // return false to prevent dangling pointers
    }

    // Free memory for the logger instance
    memory_free(logger);
    return true;
}

/**
 * @brief Logs a message with the specified log level to the logger's file.
 *
 * This function logs a message with the specified log level to the logger's
 * file. If the logger's log level is lower than the specified log level, the
 * message will not be logged.
 *
 * @param logger A pointer to the logger instance to use for logging.
 * @param log_level The log level of the message to be logged.
 * @param format The format string of the message to be logged.
 * @param ... Additional arguments for formatting the message (optional).
 *
 * @return true if the message was successfully logged, false otherwise.
 */
bool logger_message(Logger* logger, LogLevel log_level, const char* format, ...) {
    // block if and only if the LogLevel is less than the logger->LogLevel
    if (log_level < logger->log_level) {
        return false; // Do not log messages below the current
                      // logger->LogLevel
    }

    // Only lock the thread if LogLevel is valid!
    pthread_mutex_lock(&logger->thread_lock);

    int err = errno; // Capture errno at the start of the function to avoid changes

    // Apply lazy initialization for global logger
    if (NULL == logger->file_stream) {
        logger->file_stream = stderr;
        // WARN: DO NOT REINITIALIZE THE MUTEX
    }

    // Prefix log messages based on the level
    switch (log_level) {
        case LOG_LEVEL_DEBUG:
            fprintf(logger->file_stream, "[DEBUG] ");
            break;
        case LOG_LEVEL_INFO:
            fprintf(logger->file_stream, "[INFO] ");
            break;
        case LOG_LEVEL_WARN:
            if (err != 0) {
                fprintf(logger->file_stream, "[WARN:%s] ", strerror(err));
            } else {
                fprintf(logger->file_stream, "[WARN] ");
            }
            break;
        case LOG_LEVEL_ERROR:
            if (err != 0) {
                fprintf(logger->file_stream, "[ERROR:%s] ", strerror(err));
            } else {
                fprintf(logger->file_stream, "[ERROR] ");
            }
            break;
    }

    va_list args;
    va_start(args, format);
    vfprintf(logger->file_stream, format, args);
    va_end(args);

    fflush(logger->file_stream); // Ensure the message is written immediately

    pthread_mutex_unlock(&logger->thread_lock);

    return true;
}

/**
 * @brief Global Logger Object
 *
 * The global logger object provides a centralized logging mechanism for the
 * program. It is statically initialized with default values and can be
 * accessed from any part of the program to log messages at various log levels.
 * The logger ensures thread-safe logging through the use of a mutex.
 *
 * @note The global logger object should not be reinitialized or modified after
 * its creation, as this could lead to undefined behavior. Additionally,
 * explicitly destroying the mutex associated with the logger is not strictly
 * necessary but can be considered good practice, especially in environments
 * where resources are checked meticulously at program exit.
 *
 * @var logger_global
 * The global logger object has the following attributes:
 * - log_level: The logging level of the logger.
 * - log_type: The type of logger.
 * - log_type_name: The name associated with the logger type.
 * - file_stream: The file stream for writing log messages.
 * - file_path: The path to the log file.
 * - thread_lock: Mutex to ensure thread-safe logging.
 *
 * @warning Modifying the global logger object or attempting to reinitialize
 * the mutex after initialization can lead to undefined behavior.
 */
Logger logger_global = {
    LOG_LEVEL_DEBUG, /**< Logging level */
    LOG_TYPE_STREAM, /**< Logger type */
    "stream", /**< Logger type name */
    NULL, /**< File stream */
    NULL, /**< File path */
    PTHREAD_MUTEX_INITIALIZER /**< Mutex for thread safety */
};

/**
 * @brief Initialize Global Logger
 *
 * Initializes the global logger object with the specified attributes.
 * This function allows customization of the logger's properties such as
 * log level, log type, file stream, and file path. It should be called
 * before using the global logger to ensure proper logging behavior.
 *
 * @param log_level The desired logging level for the logger.
 * @param log_type The type of logger to be used (e.g., stream, file).
 * @param log_type_name The name associated with the logger type.
 * @param file_stream The file stream for writing log messages (NULL if not
 * applicable).
 * @param file_path The path to the log file (NULL if not applicable).
 *
 * @warning Calling this function may alter the behavior of the global logger
 * and should be used with caution. Avoid calling this function after the
 * global logger has been initialized to prevent unintended side effects.
 */
void logger_set_global(
    LogLevel log_level,
    LogType log_type,
    const char* log_type_name,
    FILE* file_stream,
    const char* file_path
) {
    logger_global.log_level = log_level;
    logger_global.log_type = log_type;
    logger_global.log_type_name = log_type_name;
    logger_global.file_stream = file_stream;
    logger_global.file_path = file_path;
    /// @warning REINITIALIZING THE MUTEX IS UNDEFINED BEHAVIOR!
}
