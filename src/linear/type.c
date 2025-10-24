/**
 * @file type.c
 * @brief Core numeric type definitions and quantization interface.
 * @copyright Copyright © 2023 Austin Berrio
 */

#include <stdint.h>
#include <stddef.h>
#include "linear/type.h"

/**
 * Metadata Accessors
 * @{
 */

const Type* type_data(TypeId id) {
    return id < TYPE_COUNT ? &TYPE_DATA[id] : NULL;
}

const char* type_name(TypeId id) {
    const Type* type = type_data(id);
    return type ? type->name : "unknown";
}

uint32_t type_size(TypeId id) {
    const Type* type = type_data(id);
    return type ? type->size : 0;
}

/** @} */
