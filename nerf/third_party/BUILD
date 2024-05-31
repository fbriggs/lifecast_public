# MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
load("@rules_cuda//cuda:defs.bzl", "cuda_library")

cc_library(
  name = "dirent_windows",
  hdrs = ["dirent.h"],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "json",
  hdrs = ["json.h"],
  visibility = ["//visibility:public"],
)

cc_library(
  name = "fmt",
  srcs = ["tiny_cuda_nn/dependencies/fmt/src/format.cc"],
  hdrs = glob(["tiny_cuda_nn/dependencies/fmt/include/**/*.h"]),
  includes = ["tiny_cuda_nn/dependencies/fmt/include/"],
)

cuda_library(
  name = "tiny_cuda_nn",
  srcs = [
    "tiny_cuda_nn/src/common_host.cu",
    "tiny_cuda_nn/src/cpp_api.cu",
    "tiny_cuda_nn/src/encoding.cu",
  ],
  hdrs = glob(["tiny_cuda_nn/**/*.h"]),
  defines = [
    "TCNN_MIN_GPU_ARCH=75",  # at least 2000 series (Titan RTX)
    "TCNN_NO_NETWORKS",
  ],
  includes = [
    "tiny_cuda_nn/include/",
    "tiny_cuda_nn/dependencies/",
    "tiny_cuda_nn/dependencies/cutlass/include",
  ],
  deps = [
    ":fmt",
    ":json",
    "@local_cuda//:cuda"
  ],
  target_compatible_with = select({
    "@platforms//os:osx": ["@platforms//os:incompatible"],
    "//conditions:default": [],
  }),
  visibility = ["//visibility:public"],
)
