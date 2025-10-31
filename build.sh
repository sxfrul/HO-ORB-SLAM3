#!/bin/bash

# Set CUDA environment variables for your specific installation
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Suppress ALL warnings including unused variable warnings
export CXXFLAGS="-w -Wno-unused-but-set-variable -Wno-unused-variable -Wno-unused-parameter -Wno-reorder"
export CFLAGS="-w"

echo "Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    echo "CUDA compiler found: $(nvcc --version | grep release)"
    echo "NVCC path: $(which nvcc)"
    echo "CUDA_HOME: $CUDA_HOME"
else
    echo "WARNING: CUDA compiler not found. Building without CUDA support."
fi

echo "Configuring and building Thirdparty/DBoW2 ..."

cd Thirdparty/DBoW2
# Clean previous build
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j2

cd ../../g2o

echo "Configuring and building Thirdparty/g2o ..."

# Clean previous build
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j2

cd ../../Sophus

echo "Configuring and building Thirdparty/Sophus ..."

# Clean previous build
rm -rf build
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j2

cd ../../../

echo "Uncompress vocabulary ..."

cd Vocabulary
tar -xf ORBvoc.txt.tar.gz
cd ..

echo "Configuring and building ORB_SLAM3 ..."

# Clean previous build
rm -rf build
mkdir build
cd build

# Build with CUDA support if available
if command -v nvcc &> /dev/null; then
    echo "Building with CUDA support..."
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc
else
    echo "Building without CUDA support..."
    cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_CUDA=OFF
fi

make -j2

echo "Build complete!"