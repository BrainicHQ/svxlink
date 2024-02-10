# Find the popt library
find_package(Popt REQUIRED)
set(LIBS ${LIBS} ${POPT_LIBRARIES})
include_directories(${POPT_INCLUDE_DIRS})
add_definitions(${POPT_DEFINITIONS})

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

# Explicitly set and use the paths for the fvad library
set(fvad_INCLUDE_DIR "/usr/local/include") # Path to the fvad header files
set(fvad_LIBRARY "/usr/local/lib/libfvad.so") # Path to the fvad library file

# Include the fvad header files
include_directories(${fvad_INCLUDE_DIR})

# Find the fvad library to link against
find_library(fvad_LIB NAMES fvad PATHS "/usr/local/lib" NO_DEFAULT_PATH)
if(NOT fvad_LIB)
    message(FATAL_ERROR "fvad library not found")
endif()

# Add the fvad library to the list of libraries to link against
set(LIBS ${LIBS} ${fvad_LIB})

# Add project libraries
set(LIBS asynccpp asyncaudio asynccore svxmisc ${LIBS})

# Build the executable
add_executable(svxreflector
  svxreflector.cpp Reflector.cpp ReflectorClient.cpp TGHandler.cpp
)
target_link_libraries(svxreflector ${LIBS})
set_target_properties(svxreflector PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${RUNTIME_OUTPUT_DIRECTORY}
)

# Install targets
install(TARGETS svxreflector DESTINATION ${BIN_INSTALL_DIR})
install_if_not_exists(svxreflector.conf ${SVX_SYSCONF_INSTALL_DIR})
install(PROGRAMS svxreflector-status DESTINATION ${BIN_INSTALL_DIR})
