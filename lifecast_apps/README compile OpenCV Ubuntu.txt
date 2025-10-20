# Install dependencies
sudo apt install build-essential cmake git pkg-config libgtk-3-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    gfortran openexr libatlas-base-dev python3-dev python3-numpy \
    libtbbmalloc2 libtbb-dev libdc1394-dev

# Clone the repositories
mkdir -p ~/dev/opencv_build && cd ~/dev/opencv_build

git clone --depth 1 --branch 4.12.0 https://github.com/opencv/opencv.git
git clone --depth 1 --branch 4.12.0 https://github.com/opencv/opencv_contrib.git

# Create build directory
cd opencv
mkdir -p build && cd build

# Configure
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D BUILD_EXAMPLES=OFF \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_opencv_python3=OFF \
      -D WITH_CUDA=OFF \
      -D WITH_QT=OFF \
      -D WITH_GTK=ON \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D WITH_TIFF=ON \
      -D BUILD_TIFF=ON \
      -D WITH_JPEG=ON \
      -D WITH_PNG=ON \
      -D WITH_WEBP=ON \
      -D WITH_JASPER=ON \
      -D WITH_OPENEXR=ON \
      -D WITH_EIGEN=ON \
      -D BUILD_opencv_flann=ON \
      -D BUILD_opencv_features2d=ON \
      -D BUILD_opencv_calib3d=ON \
      -D BUILD_opencv_ml=ON \
      -D BUILD_opencv_video=ON \
      -D BUILD_opencv_videoio=ON \
      -D BUILD_opencv_highgui=ON \
      -D BUILD_opencv_objdetect=ON \
      -D BUILD_opencv_dnn=ON \
      -D BUILD_opencv_xfeatures2d=ON \
      -D WITH_FFMPEG=ON \
      -D WITH_V4L=ON \
      -D WITH_GSTREAMER=ON \
      ..

# Build and install
make -j$(nproc)
sudo make install
sudo ldconfig
