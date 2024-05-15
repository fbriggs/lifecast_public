## lifecast.ai NeRF video engine ##

This part of the repo provides minimal implementation of our NeRF video engine, and the accompanying algorithms for baking NeRFs into layered depth images for real time video streaming.
For more info and demos, visit https://lifecast.ai

## Example Use (DeepView video dataset) ##

* Download the video dataset 01_Welder from https://github.com/augmentedperception/deepview_video_dataset and extract 01_Welder.zip to ~/Downloads/01_Welder

* Run a script to convert the camera model to our convention:
```
python3 deepview_to_lifecast.py \
--deepview_json_path ~/Downloads/01_Welder/models.json \
--output_json_path ~/Downloads/01_Welder/dataset.json
```

* View the camera poses
```
bazel run -- //source:point_cloud_viz \
--cam_json ~/Downloads/01_Welder/dataset.json
```

* Make a directory to store output files:
```
mkdir ~/Desktop/nerf
```

* Run the video nerf to LDI3 pipeline:
```
bazel run -- //source:lifecast_nerf \
--vid_dir ~/Downloads/01_Welder \
--output_dir ~/Desktop/nerf \
--floater_min_dist 0.3
```

* Compress the LDI3 frames into h264 or h265 flavors optimized for different platforms:
```
ffmpeg -y -framerate 30 \
-i ~/Desktop/nerf/ldi3_%06d.png \
-c:v libx265 -preset medium -crf 29 -pix_fmt yuv420p -movflags faststart \
-profile:v main -tag:v hvc1 \
~/Desktop/nerf/ldi3_h265_hvc1.mp4 \
&& \
ffmpeg -y -framerate 30 \
-i ~/Desktop/nerf/ldi3_%06d.png \
-c:v libx264 -preset fast -crf 29 -pix_fmt yuv420p -movflags faststart \
~/Desktop/nerf/ldi3_h264.mp4 \
&& \
ffmpeg -y \
-framerate 30 \
-i ~/Desktop/nerf/ldi3_%06d.png \
-vf scale=1920:1920 \
-c:v libx264 -crf 29 -movflags faststart -pix_fmt yuv420p \
~/Desktop/nerf/ldi3_h264_1920x1920.mp4
```

* Play the LDI3 video files using the player example projects here:
https://github.com/fbriggs/lifecast_public/tree/main/web

## Example Use (iPhone static scenes) ##

* Record a short (5-15 second) video with an iPhone, or get the example dataset here:
https://lifecastvideocdn.s3.us-west-1.amazonaws.com/public/crystal_orb.mov

Put it at ~/Downloads/crystal_orb.mov

* Run our structure from motion solver to estimate camera poses and focal lengths:
```
bazel run -- //source:rectilinear_sfm \
--src_vid ~/Downloads/crystal_orb.mov \
--dest_dir ~/Desktop/nerf \
```

* Visualize the estimated camera poses and SFM point cloud:
```
bazel run -- //source:point_cloud_viz \
--point_size 2 \
--point_cloud ~/Desktop/nerf/pointcloud_sfm.bin \
--cam_json ~/Desktop/nerf/dataset_all.json
```

* Train a NeRF and save the model files:
```
bazel run -- //source:lifecast_nerf \
--train_images_dir ~/Desktop/nerf \
--train_json ~/Desktop/nerf/dataset_train.json \
--output_dir ~/Desktop/nerf
```

* Bake the NeRF to an LDI:
```
bazel run -- //source:lifecast_nerf \
--distill_ldi3 \
--distill_model_dir ~/Desktop/nerf \
--output_dir ~/Desktop/nerf
```

## Setup (OS X) ##
```
brew tap bazelbuild/tap
brew install bazelbuild/tap/bazel
brew upgrade bazelbuild/tap/bazel
bazel --version

brew install clang-format

brew install glfw
brew install opencv@4
Run 'brew doctor' and fix all errors
brew install ceres-solver
brew install libusb
brew install jpeg-turbo
brew install sox
brew install cpprestsdk
brew install openssl
brew install libtorch
```
IF YOU ARE ON AN ARM MAC (M1 etc), homebrew is installed in a different location. To work around this, run:
```
sudo ln -s /opt/homebrew/Cellar /usr/local/Cellar
```
Also on ARM MAC, make sure to add this before commands:

PYTORCH_ENABLE_MPS_FALLBACK=1

e.g.
```
PYTORCH_ENABLE_MPS_FALLBACK=1 bazel run -- //source:lifecast_nerf ...
```

## Setup (Ubuntu 22) ##

Install bazel:
https://docs.bazel.build/versions/main/install-ubuntu.html

You can use bazelisk instead of bazel depending on the installation option you chose.

Try compiling hello world:
bazel build //examples:hello_world

If there is an error about not being able to find CC, add the following to ~/.bash_profile:
export CC=/usr/bin/cc
Then run source ~/.bash_profile
```
sudo apt install libeigen3-dev
sudo apt install libglfw3-dev libglfw3
sudo apt install libceres-dev
sudo apt install kdialog
sudo apt install libopencv-dev python3-opencv
sudo apt install libomp-dev
sudo apt install libturbojpeg libturbojpeg-dev
sudo apt install libcpprest-dev
sudo apt install libssl-dev
```
To make tinyfiledialogs work, add to ~/.bash_profile:
export DISPLAY=:0.0

install XIMEA libs:
https://www.ximea.com/support/wiki/apis/ximea_linux_software_package


Install torch:

https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.12.0%2Bcu116.zip

Extract the libtorch folder, then run:

sudo mv libtorch /usr/local/torch

Next, we need to make a few symlinks to get everything linking OK:
```
ln -s /usr/local/torch/lib/libcudart-45da57e3.so.11.0 /usr/local/torch/lib/libcudart.so.11.0
ln -s /usr/local/torch/lib/libnvToolsExt-847d78f2.so.1 /usr/local/torch/lib/libnvToolsExt.so.1
ln -s /usr/local/torch/lib/libnvrtc-280a23f6.so.11.2 /usr/local/torch/lib/libnvrtc.so.11.2
ln -s /usr/local/torch/lib/libnvrtc-builtins-3bf976fe.so.11.6 /usr/local/torch/lib/libnvrtc-builtins.so.11.6
ln -s /usr/local/torch/lib/libgomp-52f2fd74.so.1 /usr/local/torch/lib/libgomp.so.1
```
You also need to add something to $LD_LIBRARY_PATH. To do so, run:
```
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/torch/lib/' >> ~/.bashrc
```
To use the latest version of libtorch, go to https://pytorch.org/
Select: Stable, LibTorch, C++/Java, CUDA 11.6
Choose the cxx11 ABI link.
The library hashes and version numbers will likely be different

## Bazel on Windows ##

https://docs.bazel.build/versions/main/install-windows.html
https://bazel.build/docs/windows

Installing libtorch:

    Download this zip and extracting it to ~/dev

    https://download.pytorch.org/libtorch/cu116/libtorch-win-shared-with-deps-1.12.0%2Bcu116.zip

    Torch puts its DLLs in the lib folder instead of bin folder, which confuses bazel.
    So we'll hack around this with a symlink.
    Update the paths here with your username, and run these commands:

    mv ~\dev\libtorch-win-shared-with-deps-1.12.0+cu116\libtorch\bin ~\dev\libtorch-win-shared-with-deps-1.12.0+cu116\libtorch\bin_old 

    From an admin termainal:

    New-Item -ItemType SymbolicLink -Path "C:\Users\Forrest\dev\libtorch-win-shared-with-deps-1.12.0+cu116\libtorch\bin" -Target "C:\Users\Forrest\dev\libtorch-win-shared-with-deps-1.12.0+cu116\libtorch\lib"

Install vcpkg following instructions below. Make sure to install to ~/dev/vcpkg or later it wont compile:
https://vcpkg.io/en/getting-started.html
```
cd ~/dev/vcpkg
.\vcpkg.exe install glfw3:x64-windows
.\vcpkg.exe install gflags:x64-windows
.\vcpkg.exe install opencv[contrib,ffmpeg]:x64-windows --recurse
.\vcpkg.exe install glog:x64-windows
.\vcpkg install ceres[eigensparse]:x64-windows --recurse
.\vcpkg.exe install cpprestsdk:x64-windows
.\vcpkg.exe install glew:x64-windows
.\vcpkg.exe install protobuf:x64-windows
```
NOTE installing opencv with vcpkg will bring in a lot of other dependencies, e.g., zlib. The WORKSPACE file assumes these are present.

Note: if you get a warning like: "cl : Command line warning D9025 : overriding '/Od' with '/O2'"
Try adding "-c opt" to your bazel build command.

To make ~ expansion work with PowerShell, run:
```
Enable-ExperimentalFeature PSNativePSPathResolution
```
## A note about coordinate systems ##

Unless otherwise noted, *everything* is in a coordinate system where:
```
+X = right
+Y = up
+Z = forward
```
