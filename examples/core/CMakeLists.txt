# examples/tokenizer

set(EXAMPLES
    "lehmer"
    "alias"
    "map"
    "set"
    "q8"
)

set(INPUT_DIR ${PROJECT_SOURCE_DIR}/examples/core)
set(OUTPUT_DIR ${PROJECT_SOURCE_DIR}/build/examples/core)

foreach(example IN LISTS EXAMPLES)
    add_executable(${example} ${INPUT_DIR}/${example}.c)
    target_link_libraries(${example} "valerie")
    target_include_directories(${example} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    set_target_properties(${example} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
endforeach()
