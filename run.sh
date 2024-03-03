#!/bin/bash

# Improved Bash Script

# Exit script on any error and enable debugging
set -eo pipefail
# set -x  # Uncomment this line to enable debugging

# Fetch the latest version of the code
git pull

# Set the full path to the Silero model to current directory
SILERO_MODEL_PATH=$(pwd)/silero_vad.onnx

# Export SILERO_MODEL_PATH for immediate use in this script
export SILERO_MODEL_PATH

# Set the SILERO_MODEL_PATH environment variable for all users
echo "export SILERO_MODEL_PATH=${SILERO_MODEL_PATH}" | sudo tee /etc/profile.d/silero_model.sh > /dev/null

# Reload the system-wide environment variables to make SILERO_MODEL_PATH available in the current session
source /etc/profile

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