list(APPEND CMAKE_PREFIX_PATH ${CMAKE_SOURCE_DIR}/external_libs/lib/cmake/eigen)
#[[ Create external library ]]
add_library(external_header_libs ${CMAKE_CURRENT_SOURCE_DIR}/inih/cpp/INIReader.cpp ${CMAKE_CURRENT_SOURCE_DIR}/inih/ini.c)
target_include_directories(external_header_libs
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/pcg-cpp/include
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inih/cpp
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/inih
    INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/argparse/include/argparse
    )
