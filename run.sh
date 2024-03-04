#!/bin/bash

# Exit script on any error and enable debugging
set -eo pipefail
# set -x  # Uncomment this line to enable debugging

# Fetch the latest version of the code
git pull

# Set the full path to the Silero model to current directory
SILERO_MODEL_PATH=$(pwd)/silero_vad.onnx

# Set the ONNX variable path
# Download for your architecture from https://github.com/microsoft/onnxruntime/releases and extract it to the current directory and rename it to onnxruntime
# The existing one is for arm64

ONNXRUNTIME_ROOT_DIR=$(pwd)/onnxruntime

# Export SILERO_MODEL_PATH for immediate use in this script
export SILERO_MODEL_PATH

# Export ONNXRUNTIME_ROOT_DIR for immediate use in this script
export ONNXRUNTIME_ROOT_DIR

# Function to add or update an environment variable in /etc/environment
set_env_variable() {
    local var_name="$1"
    local var_value="$2"
    if grep -q "^${var_name}=" /etc/environment; then
        # Variable exists, replace it
        sudo sed -i "s|^${var_name}=.*|${var_name}=\"${var_value}\"|" /etc/environment
    else
        # Variable does not exist, append it
        echo "${var_name}=\"${var_value}\"" | sudo tee -a /etc/environment > /dev/null
    fi
}

# Set or update the environment variables in /etc/environment for all users
set_env_variable "SILERO_MODEL_PATH" "$SILERO_MODEL_PATH"
set_env_variable "ONNXRUNTIME_ROOT_DIR" "$ONNXRUNTIME_ROOT_DIR"

# Clean the build directory
rm -rf src/build

# Ensure the build directory exists and enter it
mkdir -p src/build && cd src/build || exit 1

# Configure the build
cmake -DUSE_QT=OFF \
      -DCMAKE_INSTALL_PREFIX=/opt/rolink \
      -DSYSCONF_INSTALL_DIR=/opt/rolink/etc \
      -DSVX_SYSCONF_INSTALL_DIR=/opt/rolink/conf \
      -DSHARE_INSTALL_PREFIX=/opt/rolink/share \
      -DSVX_SHARE_INSTALL_DIR=/opt/rolink/share \
      -DSVX_MODULE_INSTALL_DIR=/opt/rolink/lib/modules \
      -DLOCAL_STATE_DIR=/opt/rolink/var \
      -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_C_FLAGS="-march=native" \
      -DCMAKE_CXX_FLAGS="-march=native" ..

# Build and install
sudo make install

# Start the node and the reflector as a regular user
/opt/rolink/bin/svxlink &
/opt/rolink/bin/svxreflector &

echo "Script completed successfully."