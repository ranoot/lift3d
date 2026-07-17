cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(lift3d LANGUAGES C CXX)  # C needed by VTK/MPI via PCL

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Default to the CPU back end so the mapping runs without a GPU. Flip on with
# -DLIFT3D_USE_CUDA=ON when a usable CUDA device/driver is available.
option(LIFT3D_USE_CUDA "Build the CUDA back end instead of the CPU one" OFF)

# Async sign-text egress: compile the teammate's SignUnderstanding module (sign-unds/) +
# our sign_adaptor bridge into run_semantic_universe. OFF by default so the normal build
# needs no OpenCV/CURL/VLM deps. Use a separate build dir (e.g. build-sign) when ON.
option(LIFT3D_USE_SIGN "Build the async sign-text egress (SignUnderstanding + bridge)" OFF)

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

# FetchContent is used below for yaml-cpp. The rerun C++ SDK (and its bundled Apache Arrow)
# is intentionally GONE: all visualization moved to the Python viz_server/ (rerun + a live
# viewer), which run_semantic_universe feeds over ZMQ. This build therefore carries no
# rerun/Arrow dependency, so it configures on locked-down systems that cannot build Arrow.
include(FetchContent)

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

# Viz egress: serialize per-frame + final pipeline state onto a ZMQ PUSH socket (msgpack,
# via inf_client's msgpack_lite.h) for the Python viz_server. Same cppzmq/libzmq include +
# ndarray-envelope pattern as inf_client; reads the Universe/Superpoints/... types (defined
# after this, so linked below where universe exists).

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

# Viz egress lib (declared here, after `universe` exists): PUSH-socket msgpack publisher.
add_library(viz_publisher STATIC universe/viz_publisher.cpp)
target_include_directories(viz_publisher
  PUBLIC universe
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cppzmq ${CMAKE_CURRENT_SOURCE_DIR}/libzmq/include)
target_link_libraries(viz_publisher
  PUBLIC universe dog_log_adaptor inf_client ${ZMQ_LIBRARY} ${PCL_LIBRARIES})
set_target_properties(viz_publisher PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Semantic Universe: accumulate + lift inf_server 2D labels onto every world point, and
# (when --viz / viz_endpoint is set) stream per-frame + final state to the Python viz_server.
# Carries NO rerun/arrow dependency -- all rendering lives in viz_server/.
add_executable(run_semantic_universe universe/run_semantic_universe.cpp)
target_link_libraries(run_semantic_universe
  universe dog_log_adaptor inf_client object_seeds run_config viz_publisher
  ${PCL_LIBRARIES})

# GMD object egress helper (gmd::GmdEgress), extracted out of run_semantic_universe and
# UNWIRED -- built here so it keeps compiling (the seed of a future all-in-one entry), but
# no executable links it. Reads the Universe, so it links `universe` (unlike the PCL-free
# gmd_adaptor lib, which the smoke tests use).
add_library(gmd_egress STATIC gmd_adaptor/gmd_egress.cpp)
target_include_directories(gmd_egress PUBLIC gmd_adaptor)
target_link_libraries(gmd_egress
  PUBLIC universe dog_log_adaptor inf_client gmd_adaptor gmd_types ${PCL_LIBRARIES})
set_target_properties(gmd_egress PROPERTIES POSITION_INDEPENDENT_CODE ON)

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
  gmd_adaptor/gmd_frame_source.cpp
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

# ---------------------------------------------------------------------------------------
# Sign-text egress (LIFT3D_USE_SIGN). Additive + optional: compiles the teammate's
# SignUnderstanding detector chain (sign-unds/) plus our sign_adaptor bridge, and folds
# resolved sign text into gmd_egress's EAIRoomObject::directionContent. Nothing above is
# touched when the option is OFF. (Wires into the unwired gmd_egress helper now that the GMD
# path has moved out of run_semantic_universe; dormant until a driver links gmd_egress.)
# ---------------------------------------------------------------------------------------
if(LIFT3D_USE_SIGN)
  find_package(CURL REQUIRED)
  find_package(OpenCV REQUIRED)
  find_package(JPEG REQUIRED)   # VlmClient uses libjpeg directly
  find_package(PNG REQUIRED)    # VlmClient uses libpng directly

  # nlohmann/json: prefer a system package, else fetch it (mirrors sign-unds/CMakeLists.txt).
  find_package(nlohmann_json QUIET)
  if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann/json not found; fetching v3.11.3")
    FetchContent_Declare(nlohmann_json
      GIT_REPOSITORY https://github.com/nlohmann/json.git GIT_TAG v3.11.3)
    FetchContent_MakeAvailable(nlohmann_json)
  endif()

  set(SIGN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/sign-unds)
  # NOTE: GmdTimeStubs.cpp is intentionally excluded -- its Common::Time symbols are already
  # defined by gmd_types (real TimeStamp.cpp), so compiling it here would duplicate them.
  add_library(sign_adaptor STATIC
    sign_adaptor/sign_bridge.cpp
    ${SIGN_SRC}/src/ConcurrentDetectorWrapper.cpp
    ${SIGN_SRC}/src/DetectorWrapper.cpp
    ${SIGN_SRC}/src/VlmClient.cpp
    ${SIGN_SRC}/src/CroppingEngine.cpp
    ${SIGN_SRC}/src/ResponseProcessor.cpp)
  target_include_directories(sign_adaptor PUBLIC
    sign_adaptor ${SIGN_SRC}/include gmdDataTypesForInterns2)
  target_link_libraries(sign_adaptor
    PUBLIC  universe dog_log_adaptor gmd_types
    PRIVATE CURL::libcurl ${OpenCV_LIBS} nlohmann_json::nlohmann_json
            ${JPEG_LIBRARIES} ${PNG_LIBRARIES} Eigen3::Eigen Threads::Threads)
  set_target_properties(sign_adaptor PROPERTIES POSITION_INDEPENDENT_CODE ON)

  # Turn on the #ifdef'd sign wiring in the gmd_egress helper and link the detector in.
  # PUBLIC so the LIFT3D_HAVE_SIGN define (which changes GmdEgress's ABI -- adds the
  # SignBridge member) is seen identically by anything that later links gmd_egress.
  target_compile_definitions(gmd_egress PUBLIC LIFT3D_HAVE_SIGN)
  target_link_libraries(gmd_egress PUBLIC sign_adaptor)
  message(STATUS "LIFT3D_USE_SIGN: sign-text egress enabled (via gmd_egress)")
endif()
