# examples/tokenizer

set(EXAMPLES
    "vocab"
    "bpe"
    "model"
)

set(INPUT_DIR ${PROJECT_SOURCE_DIR}/examples/tokenizer)
set(OUTPUT_DIR ${PROJECT_SOURCE_DIR}/build/examples/tokenizer)

foreach(example IN LISTS EXAMPLES)
    add_executable(${example} ${INPUT_DIR}/${example}.c)
    target_link_libraries(${example} "valerie")
    target_include_directories(${example} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    set_target_properties(${example} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
endforeach()
