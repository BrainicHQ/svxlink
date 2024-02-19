# Find the popt library
find_package(Popt REQUIRED)
set(LIBS ${LIBS} ${POPT_LIBRARIES})
include_directories(${POPT_INCLUDE_DIRS})
add_definitions(${POPT_DEFINITIONS})

# ONNX Runtime
set(ONNXRUNTIME_ROOT_DIR "/home/silviu/onnxruntime")
include_directories(${ONNXRUNTIME_ROOT_DIR}/include)
find_library(ONNX_RUNTIME_LIB onnxruntime HINTS ${ONNXRUNTIME_ROOT_DIR}/lib)

if(NOT ONNX_RUNTIME_LIB)
    message(FATAL_ERROR "ONNX Runtime library not found")
endif()

# Find the GCrypt library
find_package(GCrypt REQUIRED)
set(LIBS ${LIBS} ${GCRYPT_LIBRARIES})
include_directories(${GCRYPT_INCLUDE_DIRS})
add_definitions(${GCRYPT_DEFINITIONS})

# Find jsoncpp library
#find_package(jsoncpp REQUIRED)
#get_target_property(JSONCPP_INC_PATH jsoncpp_lib INTERFACE_INCLUDE_DIRECTORIES)
#include_directories(${JSONCPP_INC_PATH})
#set(LIBS ${LIBS} jsoncpp_lib)
pkg_check_modules (JSONCPP REQUIRED jsoncpp)
include_directories(${JSONCPP_INCLUDE_DIRS})
set(LIBS ${LIBS} ${JSONCPP_LIBRARIES})

# Add project libraries
set(LIBS asynccpp asyncaudio asynccore svxmisc ${LIBS})

# Build the executable
add_executable(svxreflector
        svxreflector.cpp Reflector.cpp ReflectorClient.cpp TGHandler.cpp
)
target_link_libraries(svxreflector ${LIBS})
set_target_properties(svxreflector PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${RUNTIME_OUTPUT_DIRECTORY}
        INSTALL_RPATH "${ONNXRUNTIME_ROOT_DIR}/lib"
)

# Install targets
install(TARGETS svxreflector DESTINATION ${BIN_INSTALL_DIR})
install_if_not_exists(svxreflector.conf ${SVX_SYSCONF_INSTALL_DIR})
install(PROGRAMS svxreflector-status DESTINATION ${BIN_INSTALL_DIR})