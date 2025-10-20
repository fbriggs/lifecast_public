# Instructions for building Lifecast apps

## All platforms

Git LFS is required. https://github.com/git-lfs/git-lfs

Make sure you've run `git lfs install` once for your user account before cloning.

## MacOS

```
brew install bazel@7
```

NOTE: bazel 7 is required

```
brew install clang-format
brew install glfw
brew install opencv@4
```

Run 'brew doctor' and fix all errors

```
brew install ceres-solver
brew install libusb
brew install jpeg-turbo
brew install sox
brew install cpprestsdk
brew install openssl
brew install libtorch
brew install ffmpeg@6
```

IF YOU ARE ON AN ARM MAC (M1 etc), homebrew is installed in a different location. To work around this, run:

```
sudo ln -s /opt/homebrew/Cellar /usr/local/Cellar
```


## Ubuntu 22.04 or 24.04

### Install bazel

https://docs.bazel.build/versions/main/install-ubuntu.html

You can use bazelisk instead of bazel depending on the installation option you chose.

Installing bazel 7.2 specifically:

```
sudo apt update && sudo apt install apt-transport-https curl gnupg
curl -fsSL https://bazel.build/bazel-release.pub.gpg | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/bazel.gpg
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt update
apt-cache policy bazel
sudo apt install bazel=7.2.0
```

### Try compiling hello world

```
bazel build //examples:hello_world
```

If there is an error about not being able to find CC, add the following to ~/.bash_profile:

```
export CC=/usr/bin/cc
```

Then run `source ~/.bash_profile`

### Install prereqs

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
sudo apt install ffmpeg libavcodec-dev libavformat-dev libavutil-dev libx264-dev libx265-dev libavcodec-extra
```

If you're on Ubuntu 22.04 you'll need to create the following symlink:

```
ln -s /usr/lib/libceres.so /usr/lib/x86_64-linux-gnu/libceres.so
```

To make tinyfiledialogs work, add one of the following to ~/.bash_profile:

Ubuntu 24.04:

```
export DISPLAY=:1
```

Ubuntu 22.04

```
export DISPLAY=0:0
```

If those values don't work, use `who` to find out which display your user is on

### Install torch

Download libtorch from https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip

Extract the libtorch folder, then run:

```
sudo mv libtorch /usr/local/torch
```

Next, we need to make a few symlinks to get everything linking OK:

```
ln -s /usr/local/torch/lib/libcudart-8774224f.so.12 /usr/local/torch/lib/libcudart.so.12
ln -s /usr/local/torch/lib/libnvToolsExt-847d78f2.so.1 /usr/local/torch/lib/libnvToolsExt.so.1
ln -s /usr/local/torch/lib/libnvrtc-6d168ef8.so.12 /usr/local/torch/lib/libnvrtc.so.12
ln -s /usr/local/torch/lib/libnvrtc-builtins.so /usr/local/torch/lib/libnvrtc-builtins.so.12
ln -s /usr/local/torch/lib/libnvrtc-builtins.so /usr/local/torch/lib/libnvrtc-builtins.so.12.4
ln -s /usr/local/torch/lib/libgomp-98b21ff3.so.1 /usr/local/torch/lib/libgomp.so.1
```

You also need to add something to `$LD_LIBRARY_PATH`. To do so, run:

```
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/torch/lib/' >> ~/.bashrc
```

To use the latest version of libtorch:

* go to https://pytorch.org/
* Select: Stable, LibTorch, C++/Java, CUDA 12.6
* Choose the cxx11 ABI link.
  
You will likely need to update the version numbers and hashes of the symlinks above. Sometimes there will be new or removed libraries as well.


### Install CUDA toolkit

Download cuda 12.6 from https://developer.nvidia.com/cuda-12-6-0-download-archive

Select Linux -> x86_64 -> Ubuntu -> 24.04 (you can also use 22.04, replace as necessary below)
You can use any of the installer types.

As of this writing, the following commands will run the network installer for CUDA 12.6 on native linux (*not* WSL)

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6
```

## Windows

Tip: To make ~ expansion work with PowerShell, run:

```
Enable-ExperimentalFeature PSNativePSPathResolution
```

### Install Bazel

Download from https://docs.bazel.build/versions/main/install-windows.html

Docs: https://bazel.build/docs/windows

### Install libtorch

Download this zip and extracting it to the same directory where `lifecast_public` is cloned (e.g. `~/dev`)

https://download.pytorch.org/libtorch/cu121/libtorch-win-shared-with-deps-2.2.1%2Bcu121.zip

Torch puts its DLLs in the lib folder instead of bin folder, which confuses bazel.
So we'll hack around this with a symlink.
Update the paths here with your username, and run these commands:

```
mv ~\dev\libtorch-win-shared-with-deps-2.2.1+cu121\libtorch\bin ~\dev\libtorch-win-shared-with-deps-2.2.1+cu121\libtorch\bin_old
```

Windows has all its bin files in the lib dir so the following symlink simplifies bazel config. From an admin termainal:

```
New-Item -ItemType SymbolicLink -Path "~\dev\libtorch-win-shared-with-deps-2.2.1+cu121\libtorch\bin" -Target "~\dev\libtorch-win-shared-with-deps-2.2.1+cu121\libtorch\lib"
```

### vcpkg packages

Install vcpkg following https://vcpkg.io/en/getting-started.html, to `~/devel/vcpkg` where `~/devel` is the same directory that contains lifecast_public.

Then from within `~/devel/vcpkg` run:

```
.\vcpkg.exe install glfw3:x64-windows
.\vcpkg.exe install gflags:x64-windows
.\vcpkg.exe install x264:x64-windows
.\vcpkg.exe install x265:x64-windows
.\vcpkg.exe install ffmpeg[x264,x265]:x64-windows --recurse
.\vcpkg.exe install openexr:x64-windows
.\vcpkg.exe install imath:x64-windows
.\vcpkg.exe install libdeflate:x64-windows
.\vcpkg.exe install opencv[contrib,ffmpeg,openexr]:x64-windows --recurse
.\vcpkg.exe install glog:x64-windows
.\vcpkg install ceres[eigensparse]:x64-windows --recurse
.\vcpkg.exe install cpprestsdk:x64-windows
.\vcpkg.exe install glew:x64-windows
.\vcpkg.exe install protobuf:x64-windows
```

NOTE installing opencv with vcpkg will bring in a lot of other dependencies, e.g., zlib. The WORKSPACE file assumes these are present.

### Test build

```
bazel run -c opt //examples:hello_glfw
```

Note: if you get a warning like: "cl : Command line warning D9025 : overriding '/Od' with '/O2'"
Try adding "-c opt" to your bazel build command.

### CUDA

Install Visual Studio 2022 - NOTE: you must install a version older than 17.10, which is incompatible with CUDA 12.1. See https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-history#fixed-version-bootstrappers for links to specific-version installers. TODO: Upgrade to cuda 12.6 like on linux

Download and install the CUDA toolkit version 12.1. NOTE: this must match the cuda version used by libtorch above. https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Windows&target_arch=x86_64

You may need to log out and back in for the environment variables to take effect. Then you should be able to do:

```
bazel run -c opt //examples:hello_cuda
```

### General CUDA tips

By default, cuda code is generated for all architectures from 75 to 90 (2000-series and upward). This ensures maximum compatibility with consumer GPUs as well as workstation & cloud hardware.

To speed up clean build time, you can override --cuda_archs to match your particular GPU. E.g. for a 3090 you would do:

    bazel build --cuda_archs=compute_86 //examples:hello_cuda

See https://developer.nvidia.com/cuda-gpus to find out your architecture (8.6 => 86)

Code has been run on 2000-, 3000-, and 4000- series.

## Run Unit Tests

```
bazel test ...
bazel test //examples:hello_gtest
bazel test --test_output=errors //source/...
bazel test --test_output=errors //source:test_camera_and_projection
```

## Run Hello World and Examples

```
bazel run -- //examples:hello_world --name Moo
```

## VSCode Tips

Fix constant 100% CPU usage for C++ indexing:
    
Settings -> C_Cpp.workspaceParsingPriority -> Low

OR Disable microsoft's C++ extension and use the clangd extension instead

## clangd completion

run the following command to enable full intellisense:

```
bazel run //:refresh_compile_commands
```

## A note about coordinate systems

Unless otherwise noted, *everything* is in a coordinate system where:
+X = right
+Y = up
+Z = forward

This includes the coordinate systems of OpenGL rendering, local cameras, and odometry.
IMU data might not be like this as it comes off the sensor, but we flip it to this early.
