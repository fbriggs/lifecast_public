## lifecast.ai NeRF video engine ##

This part of the repo provides minimal implementation of our NeRF video engine, and the accompanying algorithms for baking NeRFs into layered depth images for real time video streaming.
For more info and demos, visit https://lifecast.ai

## Example Use (DeepView video dataset) ##

* Download the video dataset 01_Welder from https://github.com/augmentedperception/deepview_video_dataset and extract 01_Welder.zip to ~/Downloads/01_Welder

* Run a script to convert the camera model to our convention:
```sh
python3 deepview_to_lifecast.py \
--deepview_json_path ~/Downloads/01_Welder/models.json \
--output_json_path ~/Downloads/01_Welder/dataset.json
```

* View the camera poses
```sh
bazel run -- //source:point_cloud_viz \
--cam_json ~/Downloads/01_Welder/dataset.json
```

* Make a directory to store output files:
```sh
mkdir ~/Desktop/nerf
```

* Run the video nerf to LDI3 pipeline:
```sh
bazel run -- //source:lifecast_nerf \
--vid_dir ~/Downloads/01_Welder \
--output_dir ~/Desktop/nerf \
--floater_min_dist 0.3
```

* Compress the LDI3 frames into h264 or h265 flavors optimized for different platforms:
```sh
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
```sh
bazel run -- //source:rectilinear_sfm \
--src_vid ~/Downloads/crystal_orb.mov \
--dest_dir ~/Desktop/nerf \
```

* Visualize the estimated camera poses and SFM point cloud:
```sh
bazel run -- //source:point_cloud_viz \
--point_size 2 \
--point_cloud ~/Desktop/nerf/pointcloud_sfm.bin \
--cam_json ~/Desktop/nerf/dataset_all.json
```

* Train a NeRF and save the model files:
```sh
bazel run -- //source:lifecast_nerf \
--train_images_dir ~/Desktop/nerf \
--train_json ~/Desktop/nerf/dataset_train.json \
--output_dir ~/Desktop/nerf
```

* Bake the NeRF to an LDI:
```sh
bazel run -- //source:lifecast_nerf \
--distill_ldi3 \
--distill_model_dir ~/Desktop/nerf \
--output_dir ~/Desktop/nerf
```

## General CUDA tips ##

By default on Windows and Linux, CUDA code is generated for all architectures from 75 to 90 (2000-series and upward). This ensures maximum compatibility with consumer GPUs as well as workstation & cloud hardware.

To speed up clean build time, you can override this using --cuda_archs to match your particular GPU. E.g. for a 3090 you would do:

```sh
bazel run --cuda_archs=compute_86 <rest of options>
```

## Setup (OS X) ##
```sh
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
```sh
sudo ln -s /opt/homebrew/Cellar /usr/local/Cellar
```
Also on ARM MAC, make sure to add this before commands:

PYTORCH_ENABLE_MPS_FALLBACK=1

e.g.
```sh
PYTORCH_ENABLE_MPS_FALLBACK=1 bazel run -- //source:lifecast_nerf ...
```

## Setup (Ubuntu 22) ##

### Install bazel: ###
https://docs.bazel.build/versions/main/install-ubuntu.html

You can use bazelisk instead of bazel depending on the installation option you chose.

Try compiling hello world:
```sh
bazel build -c opt //source:lifecast_nerf
```

If there is an error about not being able to find CC, add the following to ~/.bash_profile:
export CC=/usr/bin/cc
Then run source ~/.bash_profile
```sh
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
```sh
export DISPLAY=:0.0
```

### Install XIMEA libs: ###

https://www.ximea.com/support/wiki/apis/ximea_linux_software_package


### Install libtorch ###

https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.2.1%2Bcu121.zip

Extract the libtorch folder, then run:

```sh
sudo mv libtorch /usr/local/torch
```

Next, we need to make a few symlinks to get everything linking OK:
```sh
ln -s /usr/local/torch/lib/libcudart-9335f6a2.so.12 /usr/local/torch/lib/libcudart.so.12
ln -s /usr/local/torch/lib/libnvToolsExt-847d78f2.so.1 /usr/local/torch/lib/libnvToolsExt.so.1
ln -s /usr/local/torch/lib/libnvrtc-b51b459d.so.12 /usr/local/torch/lib/libnvrtc.so.12
ln -s /usr/local/torch/lib/libnvrtc-builtins-6c5639ce.so.12.1 /usr/local/torch/lib/libnvrtc-builtins.so.12.1
ln -s /usr/local/torch/lib/libgomp-98b21ff3.so.1 /usr/local/torch/lib/libgomp.so.1
```
You also need to add something to $LD_LIBRARY_PATH. To do so, run:
```sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/torch/lib/' >> ~/.bashrc
```
To use the latest version of libtorch, go to https://pytorch.org/
Select: Stable, LibTorch, C++/Java, CUDA 12.1
Choose the cxx11 ABI link.
The library hashes and version numbers will likely be different

### Install CUDA toolkit ###

Update to the latest Nvidia drivers, then download the cuda toolkit:

https://developer.nvidia.com/cuda-12-1-0-download-archive

Select Linux -> x86_64 -> Ubuntu -> 22.04. You can use any of the installer types, but you must ensure that version 12.1 is installed. In the default network install instructions it says to `sudo apt-get -y install cuda`, but this will install 12.4 in our testing.

As of this writing, the following commands will run the network installer for CUDA 12.1 on native linux (*not* WSL. Select WSL in the chooser above for the correct link).

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-1
```

## Bazel on Windows ##

**TODO** these instructions should not say ~\dev. They should either be a directory above lifecast_public (i.e. ..\..) or we should let the user specify where they're installed.

https://docs.bazel.build/versions/main/install-windows.html
https://bazel.build/docs/windows

### Install libtorch ###

Download this zip and extract it alongside the lifecast_public directory. E.g. If this repo is in ~\dev\lifecast_public, then extract libtorch to `~\dev\libtorch-win-shared-with-deps+cu121\` (there should be a `libtorch` directory inside there).

https://download.pytorch.org/libtorch/cu121/libtorch-win-shared-with-deps-2.2.1%2Bcu121.zip

Torch puts its DLLs in the lib folder instead of bin folder, which confuses bazel.
So we'll hack around this with a symlink.
Run these commands with ~\dev paths changed to suit your environment:

```powershell
mv "~\dev\libtorch-win-shared-with-deps-2.2.1+cu121\libtorch\bin" "~\dev\libtorch-win-shared-with-deps-2.2.1+cu121\libtorch\bin_old"
```

From an _admin_ powershell:

```powershell
New-Item -ItemType SymbolicLink -Path "~\dev\libtorch-win-shared-with-deps-2.2.1+cu121\libtorch\bin" -Target "~\dev\libtorch-win-shared-with-deps-2.2.1+cu121\libtorch\lib"
```

### Install vcpkg packages ###

Install vcpkg following instructions below. Make sure to install alongside `lifecast_public` (e.g. ~/dev/vcpkg) or later it wont compile:
https://vcpkg.io/en/getting-started.html

Then run the following commands:
```powershell
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

### Install CUDA toolkit ###

Install Visual Studio 2022 - NOTE: you must install a version _older_ than 17.10, which is incompatible with CUDA 12.1. See https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-history#fixed-version-bootstrappers for links to specific-version installers.

Download and install the CUDA toolkit version 12.1. NOTE: this must match the cuda version used by libtorch above. https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64

You may need to log out and back in for the environment variables to take effect.

### Windows Tips ###

Note: if you get a warning like: "cl : Command line warning D9025 : overriding '/Od' with '/O2'"
Try adding "-c opt" to your bazel build command.

To make ~ expansion work with PowerShell, run:
```powershell
Enable-ExperimentalFeature PSNativePSPathResolution
```
## A note about coordinate systems ##

Unless otherwise noted, *everything* is in a coordinate system where:
```
+X = right
+Y = up
+Z = forward
```
