# Existing find_package and setup commands
find_package(Popt REQUIRED)
find_package(GCrypt REQUIRED)
# Assuming pkg-config is available, otherwise, you'll need to find these packages manually
find_package(PkgConfig REQUIRED)

pkg_check_modules(OPUS REQUIRED opus)
#pkg_check_modules(SAMPLERATE REQUIRED samplerate)
pkg_check_modules(OGG REQUIRED ogg)

# Use an environment variable for ONNXRUNTIME_ROOT_DIR
set(ONNXRUNTIME_ROOT_DIR $ENV{ONNXRUNTIME_ROOT_DIR})

# Fallback in case the environment variable is not set
if(NOT ONNXRUNTIME_ROOT_DIR)
    message(FATAL_ERROR "Environment variable ONNXRUNTIME_ROOT_DIR is not set")
endif()

find_library(ONNX_RUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_ROOT_DIR}/lib)
if(NOT ONNX_RUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime library not found")
endif()

# Assuming jsoncpp setup remains unchanged
pkg_check_modules(JSONCPP REQUIRED jsoncpp)

# Include directories for all found packages
include_directories(
        ${POPT_INCLUDE_DIRS}
        ${GCRYPT_INCLUDE_DIRS}
        ${ONNXRUNTIME_ROOT_DIR}/include
        ${JSONCPP_INCLUDE_DIRS}
        ${OPUS_INCLUDE_DIRS}
        ${SAMPLERATE_INCLUDE_DIRS}
        ${OGG_INCLUDE_DIRS}
)

# Add definitions if any provided by the packages
add_definitions(
        ${POPT_DEFINITIONS}
        ${GCRYPT_DEFINITIONS}
)

# Libraries to link against
set(LIBS
        ${LIBS}
        ${POPT_LIBRARIES}
        ${GCRYPT_LIBRARIES}
        ${JSONCPP_LIBRARIES}
        ${OPUS_LIBRARIES}
        ${SAMPLERATE_LIBRARIES}
        ${OGG_LIBRARIES}
        asynccpp asyncaudio asynccore svxmisc
        ${ONNX_RUNTIME_LIB}
)

# Build the executable
add_executable(svxreflector
        svxreflector.cpp
        Reflector.cpp
        ReflectorClient.cpp
        TGHandler.cpp
        opus_wrapper.cpp
)

target_link_libraries(svxreflector ${LIBS})

set_target_properties(svxreflector PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${RUNTIME_OUTPUT_DIRECTORY}
        INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)

# Install targets
install(TARGETS svxreflector DESTINATION ${BIN_INSTALL_DIR})
install_if_not_exists(svxreflector.conf ${SVX_SYSCONF_INSTALL_DIR})
install(PROGRAMS svxreflector-status DESTINATION ${BIN_INSTALL_DIR})
