# examples/model

set(LLM
    "embedding"
    "rotary"
    "cache"
    "forward"
    "backward"
)

set(INPUT_DIR ${PROJECT_SOURCE_DIR}/examples/model)
set(OUTPUT_DIR ${PROJECT_SOURCE_DIR}/build/examples/model)

foreach(example IN LISTS LLM)
    add_executable(${example} ${INPUT_DIR}/${example}.c)
    target_link_libraries(${example} "valerie")
    target_include_directories(${example} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    set_target_properties(${example} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${OUTPUT_DIR})
endforeach()
