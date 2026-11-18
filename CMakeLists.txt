cmake_minimum_required(VERSION 3.23)

if("${CMAKE_BINARY_DIR}" STREQUAL "${CMAKE_SOURCE_DIR}")
    message(FATAL_ERROR "In-source builds are not allowed. Use a separate build directory.")
endif()

project(abm VERSION 9.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -funroll-loops")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    # AVX2/FMA baseline, matching the cluster's compute nodes.
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=x86-64-v3")
endif()
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer")

find_package(OpenMP REQUIRED)
find_package(MPI COMPONENTS CXX)
if(MPI_CXX_FOUND)
    message(STATUS "MPI found: multi-rank runs enabled")
else()
    message(STATUS "MPI not found: building single-process only")
endif()

add_subdirectory(external_libs)

set(CORE_SOURCES
    src/abm.cpp
    src/agent_factory.cpp
    src/citation_engine.cpp
    src/config.cpp
    src/neighborhood_search.cpp
    src/signal_supervisor.cpp
    src/utils.cpp
    src/vectorized_scoring.cpp
    src/vose_alias.cpp
    src/ladybug/ladybug_authors.cpp
    src/ladybug/ladybug_csr.cpp
    src/ladybug/ladybug_db.cpp
    src/ladybug/ladybug_export.cpp
    src/ladybug/ladybug_ingest.cpp
    src/ladybug/ladybug_snapshot.cpp
    src/distributed/cluster_coordinator.cpp
    src/distributed/distributed_partitioner.cpp
)

add_library(cantatas_core STATIC ${CORE_SOURCES})
target_include_directories(cantatas_core PUBLIC ${CMAKE_SOURCE_DIR}/includes)
target_link_libraries(cantatas_core PUBLIC OpenMP::OpenMP_CXX external_header_libs)
if(MPI_CXX_FOUND)
    target_link_libraries(cantatas_core PUBLIC MPI::MPI_CXX)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_executable(abm src/main.cpp)
target_link_libraries(abm PRIVATE cantatas_core)

# Reduces a run's CSV output to summary statistics (see tools/compare_runs.py).
add_executable(run_summary tools/run_summary.cpp)

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
