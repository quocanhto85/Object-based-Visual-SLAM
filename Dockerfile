FROM arm64v8/ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC

# Install all system dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake git wget curl ca-certificates pkg-config \
    libgtk2.0-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    python3-dev python3-numpy \
    libtbb2 libtbb-dev \
    libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev \
    libboost-all-dev libeigen3-dev \
    libglew-dev libglfw3-dev libssl-dev \
    libx11-dev libxext-dev libxrender-dev \
    libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev \
    libxkbcommon-x11-0 libxkbcommon-dev \
    libwayland-dev wayland-protocols \
    libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libxxf86vm-dev \
    libepoxy-dev libepoxy0 \
    libegl1-mesa-dev libgles2-mesa-dev \
    mesa-utils x11-apps unzip \
    libwayland-egl-backend-dev \
 && rm -rf /var/lib/apt/lists/*

# Build and install OpenCV (use -j1 to reduce memory usage)
RUN git clone --depth 1 -b 4.4.0 https://github.com/opencv/opencv.git /tmp/opencv && \
    git clone --depth 1 -b 4.4.0 https://github.com/opencv/opencv_contrib.git /tmp/opencv_contrib && \
    mkdir -p /tmp/opencv/build && cd /tmp/opencv/build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
          -D BUILD_EXAMPLES=OFF \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D WITH_TBB=ON \
          -D WITH_OPENGL=ON \
          -D WITH_IPP=OFF \
          .. && \
    make -j1 && \
    make install && \
    ldconfig && \
    rm -rf /tmp/opencv /tmp/opencv_contrib

# Build and install Pangolin (use -j1)
ARG PANGOLIN_TAG=v0.6
RUN git clone --depth 1 -b ${PANGOLIN_TAG} https://github.com/stevenlovegrove/Pangolin.git /tmp/Pangolin && \
    cmake -S /tmp/Pangolin -B /tmp/Pangolin/build \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D BUILD_EXAMPLES=OFF -D BUILD_PANGOLIN_PYTHON=OFF -D BUILD_TOOLS=OFF && \
    cmake --build /tmp/Pangolin/build -- -j1 && \
    cmake --install /tmp/Pangolin/build && ldconfig && \
    rm -rf /tmp/Pangolin

# Install nlohmann/json
RUN git clone --depth 1 https://github.com/nlohmann/json.git /tmp/json && \
    mkdir -p /usr/local/include/nlohmann && \
    cp /tmp/json/single_include/nlohmann/json.hpp /usr/local/include/nlohmann/json.hpp && \
    rm -rf /tmp/json

# Create workspace directory (where we'll mount ORB_SLAM3)
RUN mkdir -p /workspace/ORB_SLAM3

# Download ORB vocabulary (cache this in the image)
RUN mkdir -p /workspace/vocabulary && \
    cd /workspace/vocabulary && \
    wget https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/master/Vocabulary/ORBvoc.txt.tar.gz && \
    tar -xzvf ORBvoc.txt.tar.gz && \
    rm ORBvoc.txt.tar.gz

# Create build script with memory-efficient compilation
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "=== Building ORB-SLAM3 ==="\n\
cd /workspace/ORB_SLAM3\n\
\n\
# Clean previous builds\n\
rm -rf build lib\n\
rm -rf Thirdparty/DBoW2/build Thirdparty/DBoW2/lib\n\
rm -rf Thirdparty/g2o/build Thirdparty/g2o/lib\n\
\n\
# Copy vocabulary if not present\n\
if [ ! -f Vocabulary/ORBvoc.txt ]; then\n\
    echo "Copying vocabulary..."\n\
    mkdir -p Vocabulary\n\
    cp /workspace/vocabulary/ORBvoc.txt Vocabulary/\n\
fi\n\
\n\
# Fix Homebrew paths\n\
sed -i "s|set(OpenCV_DIR \\"/opt/homebrew/opt/opencv/lib/cmake/opencv4\\")|# set(OpenCV_DIR \\"/opt/homebrew/opt/opencv/lib/cmake/opencv4\\")|g" CMakeLists.txt || true\n\
\n\
# Fix tr1 headers\n\
echo "Fixing deprecated tr1 headers..."\n\
find include -type f \\( -name "*.h" -o -name "*.hpp" \\) -exec sed -i \\\n\
    -e "s|#include <tr1/unordered_map>|#include <unordered_map>|g" \\\n\
    -e "s|std::tr1::unordered_map|std::unordered_map|g" \\\n\
    -e "s|std::tr1::|std::|g" {} + 2>/dev/null || true\n\
find src -type f \\( -name "*.cc" -o -name "*.cpp" \\) -exec sed -i \\\n\
    -e "s|#include <tr1/unordered_map>|#include <unordered_map>|g" \\\n\
    -e "s|std::tr1::unordered_map|std::unordered_map|g" \\\n\
    -e "s|std::tr1::|std::|g" {} + 2>/dev/null || true\n\
find Thirdparty -type f \\( -name "*.h" -o -name "*.hpp" -o -name "*.cc" -o -name "*.cpp" \\) -exec sed -i \\\n\
    -e "s|#include <tr1/unordered_map>|#include <unordered_map>|g" \\\n\
    -e "s|std::tr1::unordered_map|std::unordered_map|g" \\\n\
    -e "s|std::tr1::|std::|g" {} + 2>/dev/null || true\n\
find . -type f \\( -name "*.h" -o -name "*.hpp" -o -name "*.cc" -o -name "*.cpp" \\) -exec sed -i "s|#include <tr1/|#include <|g" {} + 2>/dev/null || true\n\
\n\
# Fix C++ standard\n\
sed -i "s/-std=c++11/-std=c++14/g" CMakeLists.txt\n\
\n\
# Add memory-efficient compiler flags\n\
# These flags reduce memory usage during compilation:\n\
# -Os: Optimize for size (uses less memory than -O3)\n\
# -fno-var-tracking: Disable variable tracking (reduces memory)\n\
# -fno-var-tracking-assignments: Further reduce memory\n\
# --param ggc-min-expand=20: Reduce garbage collector memory expansion\n\
# --param ggc-min-heapsize=32768: Set minimum GC heap size\n\
sed -i "s|set(CMAKE_CXX_FLAGS \\"\\${CMAKE_CXX_FLAGS} -Wall   -O3\\")|set(CMAKE_CXX_FLAGS \\"\\${CMAKE_CXX_FLAGS} -Wall -Wno-deprecated-declarations -Wno-reorder -Wno-sign-compare -Wno-maybe-uninitialized -Os -fno-var-tracking -fno-var-tracking-assignments --param ggc-min-expand=20 --param ggc-min-heapsize=32768\\")|g" CMakeLists.txt\n\
sed -i "s|set(CMAKE_C_FLAGS \\"\\${CMAKE_C_FLAGS}  -Wall   -O3\\")|set(CMAKE_C_FLAGS \\"\\${CMAKE_C_FLAGS} -Wall -Wno-deprecated-declarations -Os -fno-var-tracking --param ggc-min-expand=20 --param ggc-min-heapsize=32768\\")|g" CMakeLists.txt\n\
\n\
# Build with single job to prevent OOM\n\
chmod +x build.sh\n\
sed -i "s/make -j/make -j1/g" build.sh\n\
\n\
# Export compiler flags for all sub-builds\n\
export CXXFLAGS="-Os -fno-var-tracking -fno-var-tracking-assignments --param ggc-min-expand=20 --param ggc-min-heapsize=32768"\n\
export CFLAGS="-Os -fno-var-tracking --param ggc-min-expand=20 --param ggc-min-heapsize=32768"\n\
\n\
echo "Building with memory-efficient settings..."\n\
./build.sh\n\
\n\
# Copy KITTI config if needed\n\
cd Examples/Monocular\n\
if [ -f KITTI04-12.yaml ] && [ ! -f KITTI08.yaml ]; then\n\
    cp KITTI04-12.yaml KITTI08.yaml\n\
fi\n\
\n\
echo "=== Build complete ==="\n\
echo "Library:"\n\
ls -lh /workspace/ORB_SLAM3/lib/ || echo "lib/ not found"\n\
echo "Executables:"\n\
ls -lh /workspace/ORB_SLAM3/Examples/Monocular/mono_kitti || echo "mono_kitti not found"' > /usr/local/bin/build_orbslam3.sh && \
    chmod +x /usr/local/bin/build_orbslam3.sh

# Create run script
RUN echo '#!/bin/bash\n\
cd /workspace/ORB_SLAM3/Examples/Monocular\n\
exec ./mono_kitti "$@"' > /usr/local/bin/run_mono_kitti && \
    chmod +x /usr/local/bin/run_mono_kitti

ENV LD_LIBRARY_PATH=/workspace/ORB_SLAM3/lib:/workspace/ORB_SLAM3/Thirdparty/DBoW2/lib:/workspace/ORB_SLAM3/Thirdparty/g2o/lib
ENV LIBGL_ALWAYS_INDIRECT=1
ENV QT_X11_NO_MITSHM=1

WORKDIR /workspace/ORB_SLAM3

CMD ["/bin/bash"]