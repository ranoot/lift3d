cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(lift3d LANGUAGES C CXX)  # C needed by VTK/MPI via PCL

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to the CPU back end so the mapping runs without a GPU. Flip on with
# -DLIFT3D_USE_CUDA=ON when a usable CUDA device/driver is available.
option(LIFT3D_USE_CUDA "Build the CUDA back end instead of the CPU one" OFF)

if(LIFT3D_USE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # GPU not detectable at configure time here (WSL), so default to a broad set of
  # architectures. Override with -DCMAKE_CUDA_ARCHITECTURES=<sm> for your card.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)
  endif()
endif()

find_package(PCL 1.3 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)  # pose math in dog_log_adaptor

# Rerun C++ SDK for the visualizer (viz_universe). FetchContent downloads the
# prebuilt SDK zip and builds the bundled Apache Arrow from source -- that Arrow
# build is heavy, so throttle the first build (e.g. `cmake --build build -j1`) on
# RAM-limited machines. To skip the Arrow build, install libarrow and configure
# with -DRERUN_DOWNLOAD_AND_BUILD_ARROW=OFF.
include(FetchContent)
FetchContent_Declare(rerun_sdk URL
  https://github.com/rerun-io/rerun/releases/download/0.33.1/rerun_cpp_sdk.zip)
FetchContent_MakeAvailable(rerun_sdk)

# yaml-cpp for the whole-pipeline YAML config (run_config). FetchContent so no system
# -dev package is needed; skip its tests/tools to keep the build lean.
set(YAML_CPP_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS   OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL       OFF CACHE BOOL "" FORCE)
# yaml-cpp 0.8.0 still declares cmake_minimum_required(VERSION 3.4); newer CMake has
# removed <3.5 compatibility, so raise the policy floor just for its configure.
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_Declare(yaml_cpp URL
  https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.8.0.tar.gz)
FetchContent_MakeAvailable(yaml_cpp)
unset(CMAKE_POLICY_VERSION_MINIMUM)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# pcd_write_test.cpp no longer present in the tree; re-add if you restore it.
# add_executable(pcd_write_test pcd_write_test.cpp)
# target_link_libraries(pcd_write_test ${PCL_LIBRARIES})

# find_package(cppzmq)
# target_link_libraries(lift3d cppzmq)

# SGS-3D point-to-pixel mapping: same API, CPU or CUDA back end.
if(LIFT3D_USE_CUDA)
  add_library(point_pixel_mapping STATIC point_pixel_mapping/point_pixel_mapping.cu)
  # RESOLVE_DEVICE_SYMBOLS performs the CUDA device link when the static lib is built,
  # so plain C++ executables (run_/viz_ etc.) can link it without a
  # "__cudaRegisterLinkedBinary undefined reference" at their own link step.
  set_target_properties(point_pixel_mapping PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON)
else()
  add_library(point_pixel_mapping STATIC point_pixel_mapping/point_pixel_mapping_cpu.cpp)
  set_target_properties(point_pixel_mapping PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()
target_include_directories(point_pixel_mapping PUBLIC point_pixel_mapping)

# demo_mapping.cpp no longer present in the tree; re-add if you restore it.
# add_executable(demo_mapping demo_mapping.cpp)
# target_link_libraries(demo_mapping point_pixel_mapping)

# Adaptor for ~/dog_logs (Raibo EAI logs): file decode + the transport-agnostic
# online sync layer (PoseInterpolator + DogStream) + the offline simulated ingestor.
add_library(dog_log_adaptor STATIC
  dog_log_adaptor/dog_log_adaptor.cpp
  dog_log_adaptor/pose_interp.cpp
  dog_log_adaptor/dog_stream.cpp
  dog_log_adaptor/dog_log_ingestor.cpp)
target_include_directories(dog_log_adaptor PUBLIC dog_log_adaptor)
target_link_libraries(dog_log_adaptor PRIVATE Eigen3::Eigen)
set_target_properties(dog_log_adaptor PROPERTIES POSITION_INDEPENDENT_CODE ON)

# run_dog_mapping.cpp no longer present in the tree; re-add if you restore it.
# add_executable(run_dog_mapping run_dog_mapping.cpp)
# target_link_libraries(run_dog_mapping dog_log_adaptor point_pixel_mapping)

# Superpoint segmentation via L0 cut-pursuit (superpoint_graph submodule).
# cut-pursuit is header-only; its core uses std::binary_function, which is
# deprecated but still provided by libstdc++ through C++20, so it builds at the
# project standard. API.h is included only by spg_segment.cpp, so Boost stays
# private and does not leak through spg_segment.h.
set(CUTPURSUIT_INCLUDE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/superpoint_graph/partition/cut-pursuit/include)
find_package(OpenMP REQUIRED)

# Guarded: the spg_segment sources are a work-in-progress and may not exist yet.
# Without this, a reconfigure fails outright ("Cannot find source file") and blocks
# every other target. Drop the guard once spg_segment/spg_segment.cpp lands.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/spg_segment/spg_segment.cpp)
  add_library(spg_segment STATIC spg_segment/spg_segment.cpp)
  target_include_directories(spg_segment
    PUBLIC spg_segment
    PRIVATE ${CUTPURSUIT_INCLUDE_DIR})
  target_link_libraries(spg_segment
    PUBLIC ${PCL_LIBRARIES}
    PRIVATE OpenMP::OpenMP_CXX)
  set_target_properties(spg_segment PROPERTIES POSITION_INDEPENDENT_CODE ON)

  add_executable(spg_segment_cli spg_segment/spg_segment_cli.cpp)
  target_link_libraries(spg_segment_cli spg_segment ${PCL_LIBRARIES})
endif()

# inf_server IPC client: cppzmq (vendored) + a hand-rolled msgpack subset. Uses the
# vendored zmq.h headers against the system libzmq runtime (no -dev package needed).
find_library(ZMQ_LIBRARY NAMES zmq)
if(NOT ZMQ_LIBRARY)
  file(GLOB _zmq_cand /usr/lib/x86_64-linux-gnu/libzmq.so.5* /usr/lib/libzmq.so.5*
                      /usr/lib/x86_64-linux-gnu/libzmq.so /usr/lib/libzmq.so)
  if(_zmq_cand)
    list(GET _zmq_cand 0 ZMQ_LIBRARY)
  endif()
endif()
if(NOT ZMQ_LIBRARY)
  message(FATAL_ERROR "libzmq runtime not found (install libzmq3 or point ZMQ_LIBRARY at it)")
endif()
message(STATUS "inf_client using libzmq: ${ZMQ_LIBRARY}")

add_library(inf_client STATIC inf_client/inf_client.cpp)
target_include_directories(inf_client
  PUBLIC inf_client
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cppzmq ${CMAKE_CURRENT_SOURCE_DIR}/libzmq/include)
target_link_libraries(inf_client PUBLIC ${ZMQ_LIBRARY})
set_target_properties(inf_client PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Universe: PCL-backed world-space point accumulator + 2D<->3D projection bridge.
# superpoints.cpp adds VCCS oversegmentation (pcl_segmentation, already in PCL_LIBRARIES).
add_library(universe STATIC
  universe/universe.cpp universe/superpoints.cpp
  universe/instance_ids.cpp)
target_include_directories(universe
  PUBLIC universe point_pixel_mapping)
target_link_libraries(universe
  PUBLIC point_pixel_mapping ${PCL_LIBRARIES} inf_client)
set_target_properties(universe PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Parallel HDBSCAN* (wangyiqiu/hdbscan): compile the library TUs (all dims 2-20 are
# instantiated -- one-time cost) against the vendored, header-only parlaylib. The
# HOMEGROWN define selects parlay's built-in scheduler; pthread is the runtime.
find_package(Threads REQUIRED)
add_library(pargeo_hdbscan STATIC hdbscan/src/hdbscan.cpp hdbscan/src/dendrogram.cpp)
target_include_directories(pargeo_hdbscan PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/hdbscan/include
  ${CMAKE_CURRENT_SOURCE_DIR}/hdbscan/external/parlaylib/include)
target_compile_definitions(pargeo_hdbscan PUBLIC HOMEGROWN)
target_compile_options(pargeo_hdbscan PRIVATE -O3)
target_link_libraries(pargeo_hdbscan PUBLIC Threads::Threads)
set_target_properties(pargeo_hdbscan PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Object store: per-class LOCAL-window HDBSCAN* seeding + the (parlay-free) flat-cluster
# extraction, feature-guided growing (superpoints->seeds), and seeds->objects
# consolidation. object_seeds.cpp is the ONLY TU that pulls in parlay, so pargeo_hdbscan
# is linked PRIVATE -- its heavy includes/defs build object_seeds.cpp but never leak to
# the drivers (which still transitively link the static lib). object_grow.cpp (growing)
# and object_consolidate.cpp (consolidation) stay parlay-free and read only the Universe/
# Superpoints headers (shared voxel math in voxel_util.h).
add_library(object_seeds STATIC
  universe/object_seeds.cpp universe/object_grow.cpp universe/object_consolidate.cpp
  universe/hdbscan_extract.cpp)
target_include_directories(object_seeds PUBLIC universe)
target_link_libraries(object_seeds PUBLIC universe PRIVATE pargeo_hdbscan)
set_target_properties(object_seeds PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Whole-pipeline YAML config loader (shared by both semantic drivers).
add_library(run_config STATIC universe/run_config.cpp)
target_include_directories(run_config PUBLIC universe)
target_link_libraries(run_config
  PUBLIC universe dog_log_adaptor inf_client
  PRIVATE yaml-cpp)
set_target_properties(run_config PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(run_universe universe/run_universe.cpp)
target_include_directories(run_universe PRIVATE dog_log_adaptor)
target_link_libraries(run_universe universe dog_log_adaptor ${PCL_LIBRARIES})

# Rerun visualizer: replays a dog log, builds up the map, logs to a .rrd.
add_executable(viz_universe universe/viz_universe.cpp)
target_include_directories(viz_universe PRIVATE dog_log_adaptor)
target_link_libraries(viz_universe universe dog_log_adaptor ${PCL_LIBRARIES} rerun_sdk)

# Semantic Universe: accumulate + lift inf_server 2D labels onto every world point.
add_executable(run_semantic_universe universe/run_semantic_universe.cpp)
target_link_libraries(run_semantic_universe
  universe dog_log_adaptor inf_client object_seeds run_config gmd_adaptor gmd_types
  ${PCL_LIBRARIES})

# Rerun visualizer for the semantic map (world coloured by class + RGB frames).
add_executable(viz_semantic_universe universe/viz_semantic_universe.cpp)
target_link_libraries(viz_semantic_universe
  universe dog_log_adaptor inf_client object_seeds run_config ${PCL_LIBRARIES} rerun_sdk)

# Per-view cost + DVIS id-overlap experiment (throwaway measurement; same link set as
# run_semantic_universe -- parlay stays confined to object_seeds via hdbscanTimeOnly).
add_executable(exp_per_view universe/exp_per_view.cpp)
target_link_libraries(exp_per_view
  universe dog_log_adaptor inf_client object_seeds run_config ${PCL_LIBRARIES})

# ---------------------------------------------------------------------------------------
# GMD deployment interface translation layer (real-robot RA message types -> DogStream).
# Additive: nothing above is modified. gmd_types compiles only the OpenCV-free GMD entity
# TUs (skips ImageUtils.cpp / PoseUtils.cpp, which need OpenCV); the adaptor + its test are
# PCL/CUDA/server-free.
# ---------------------------------------------------------------------------------------
add_library(gmd_types STATIC
  gmdDataTypesForInterns2/GMDBase/Entity/TimeStamp.cpp
  gmdDataTypesForInterns2/Common/Entity/Transformation.cpp)
target_include_directories(gmd_types PUBLIC gmdDataTypesForInterns2)
target_link_libraries(gmd_types PUBLIC Eigen3::Eigen)
set_target_properties(gmd_types PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(gmd_adaptor STATIC
  gmd_adaptor/gmd_stream_adaptor.cpp
  gmd_adaptor/object_publisher.cpp)
target_include_directories(gmd_adaptor PUBLIC gmd_adaptor gmdDataTypesForInterns2)
target_link_libraries(gmd_adaptor PUBLIC gmd_types dog_log_adaptor Eigen3::Eigen)
set_target_properties(gmd_adaptor PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Standalone translation-layer check (no PCL / GPU / servers).
add_executable(gmd_stream_check tests/gmd_stream_check.cpp)
target_link_libraries(gmd_stream_check gmd_adaptor gmd_types dog_log_adaptor Eigen3::Eigen)

# Object egress check: EAIRoomObject 2D-hull publisher + keyed topic (no PCL / GPU / servers).
add_executable(object_publisher_check tests/object_publisher_check.cpp)
target_link_libraries(object_publisher_check gmd_adaptor gmd_types)
