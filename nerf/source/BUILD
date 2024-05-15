# MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

load("//:p11.bzl", "p11_cc_library", "p11_cuda_library", "p11_cc_binary", "p11_cc_test", "p11_windows_resource", "DEFAULT_COPTS")

### cross-platform dependency selection ############################################################

p11_cc_library(
    name = "gflags_xplatform",
    deps = select({
      "@platforms//os:osx":     ["//external:gflags",],
      "@platforms//os:linux":   ["//external:gflags",],
      "@platforms//os:windows": ["@gflags_windows_vcpkg//:gflags"],
    })
)

p11_cc_library(
    name = "opencv_xplatform",
    deps = select({
      "@platforms//os:osx":     ["@opencv_osx_homebrew//:opencv"],
      "@platforms//os:linux":   ["@opencv_linux//:opencv"],
      "@platforms//os:windows": ["@opencv_windows_vcpkg//:opencv"],
    })
)

p11_cc_library(
    name = "glfw_xplatform",
    deps = select({
      "@platforms//os:osx":     ["@glfw_osx_homebrew//:glfw"],
      "@platforms//os:linux":   ["@glfw_linux//:glfw"],
      "@platforms//os:windows": ["@glfw_windows_vcpkg//:glfw", "@glew_windows_vcpkg//:glew"],
    })
)

p11_cc_library(
    name = "libtorch_xplatform",
    deps = select({
        "@platforms//os:osx":     ["@libtorch_osx_homebrew//:libtorch"],
        "@platforms//os:linux":   ["@libtorch_linux//:libtorch"],
        "@platforms//os:windows": ["@libtorch_windows//:libtorch"],
    }),
    linkopts = select({
        "@platforms//os:osx": ["-mmacosx-version-min=14.2.0"],
        "//conditions:default": [],
    })
)


p11_cc_library(
    name = "ceres_xplatform",
    deps = select({
      "@platforms//os:osx":   ["@ceres_osx_homebrew//:ceres"],
      "@platforms//os:linux": ["@ceres_linux//:ceres"],
      "@platforms//os:windows": ["@ceres_windows_vcpkg//:ceres"],
    })
)

p11_cc_library(
    name = "dirent_xplatform",
    deps = select({
      "@platforms//os:osx":     [],
      "@platforms//os:linux":   [],
      "@platforms//os:windows": ["//third_party:dirent_windows"],      
    })
)

### libraries ######################################################################################


p11_cc_library (
  name = "logger",
  hdrs = [
      "check.h",
      "logger.h"
  ],
  srcs = ["logger.cc"],
  deps = [],
)

p11_cc_library (
  name = "util_file",
  hdrs = ["util_file.h"],
  deps = [
    ":util_string",
    ":dirent_xplatform",
  ],
)

p11_cc_library (
  name = "util_math",
  hdrs = ["util_math.h"],
  deps = ["//external:eigen"],
)

p11_cc_library (
  name = "util_string",
  hdrs = ["util_string.h"],
  deps = [],
)

p11_cc_library (
  name = "util_time",
  hdrs = ["util_time.h"],
  deps = [],
)

p11_cc_library (
  name = "util_torch",
  hdrs = ["util_torch.h"],
  deps = [
    ":libtorch_xplatform",
    ":logger",
  ],
)

p11_cc_library (
  name = "util_opencv",
  hdrs = ["util_opencv.h"],
  deps = [
    "//external:eigen",
    ":opencv_xplatform",
    ":util_math",
  ],
)

p11_cc_library (
  name = "opengl_xplatform_includes",
  hdrs = ["opengl_xplatform_includes.h"],
  deps = [
    ":glfw_xplatform",
    "//third_party/dear_imgui:dear_imgui",
  ]
)

p11_cc_library (
  name = "util_opengl",
  hdrs = ["util_opengl.h"],
  srcs = ["util_opengl.cc"],
  deps = [
    ":logger",
    "//external:eigen",
    ":glfw_xplatform",
    ":opencv_xplatform",
    ":opengl_xplatform_includes",
    ":util_math",
  ]
)

p11_cc_library (
  name = "fisheye_camera",
  hdrs = ["fisheye_camera.h"],
  deps = [
    ":ceres_xplatform",
    "//external:eigen",
    ":logger",
    ":util_math",
  ],
)

p11_cc_library (
  name = "rectilinear_camera",
  hdrs = ["rectilinear_camera.h"],
  deps = [
    ":ceres_xplatform",
    "//external:eigen",
    ":logger",
    ":util_math",
  ],
)

p11_cc_library (
  name = "pose_param",
  hdrs = ["pose_param.h"],
  deps = [
    ":ceres_xplatform",
    "//external:eigen",
  ],
)

p11_cc_library (
  name = "point_cloud",
  hdrs = ["point_cloud.h"],
  deps = [
    "//external:eigen",
    ":logger",
    ":util_math",
    ":util_string",
  ],
)

p11_cc_library (
  name = "vignette",
  hdrs = ["vignette.h"],
  srcs = ["vignette.cc"],
  deps = [
    "//external:eigen",
    ":logger",
    ":opencv_xplatform",
    ":fisheye_camera",
    ":util_math",
    ":util_opencv",
  ],
)

p11_cc_library (
  name = "keypoint_tracking",
  hdrs = ["keypoint_tracking.h"],
  srcs = ["keypoint_tracking.cc"],
  deps = [
    ":logger",
    "//external:eigen",
    ":opencv_xplatform",
  ],
)

p11_cc_library(
  name = "nerf_heuristic_seg",
  hdrs = ["nerf_heuristic_seg.h"],
  deps = [
    ":logger",
    ":opencv_xplatform",
    ":util_time",
    ":util_math",
    ":ceres_xplatform",
    "//external:eigen",
  ],
)

p11_cc_library(
  name = "deepview_dataset",
  srcs = ["deepview_dataset.cc"],
  hdrs = ["deepview_dataset.h"],
  deps = [
    ":logger",
    ":util_opencv",
    ":rectilinear_camera",
    "//external:eigen",
  ],
)

p11_cc_binary(
  name = "point_cloud_viz",
  srcs = ["point_cloud_viz.cc"],
  deps = [
    "//source:gflags_xplatform",
    ":logger",
    "//external:eigen",
    ":opencv_xplatform",
    ":glfw_xplatform",
    ":opengl_xplatform_includes",
    ":point_cloud",
    ":util_file",
    ":util_math",
    ":util_opengl",
    ":util_string",
    "//third_party:json",
  ],
)

p11_cc_library(
  name = "ngp_radiance_model",
  srcs = ["ngp_radiance_model.cc"],
  hdrs = ["ngp_radiance_model.h"],
  deps = [
    ":logger",
    ":libtorch_xplatform",
  ],
)

p11_cc_library (
  name = "lifecast_nerf_lib",
  hdrs = ["lifecast_nerf_lib.h"],
  srcs = ["lifecast_nerf_lib.cc"],
  deps = [
    ":logger",
    "//source:gflags_xplatform",
    ":opencv_xplatform",
    ":libtorch_xplatform",
    ":util_math",
    ":util_file",
    ":util_string",
    ":util_time",
    ":util_opencv",
    "//third_party:json",
    ":ngp_radiance_model",
    ":util_torch",
    ":rectilinear_camera",
    ":fisheye_camera",
    ":ldi_common",
    ":vignette",
    ":deepview_dataset",
    ":nerf_heuristic_seg"
  ],
)

p11_cc_binary(
  name = "lifecast_nerf",
  srcs = ["lifecast_nerf.cc"],
  deps = [
    ":logger",
    "//source:gflags_xplatform",
    ":util_file",
    ":lifecast_nerf_lib",
  ],
)

p11_cc_library(
  name = "ldi_common",
  hdrs = ["ldi_common.h"],
  srcs = ["ldi_common.cc"],
  deps = [
    ":logger",
    "//external:eigen",
    ":opencv_xplatform",
    ":fisheye_camera",
    ":util_math",
    ":util_time",
    ":util_opencv",
    ":vignette",
  ],
)

p11_cc_library (
  name = "rectilinear_sfm_lib",
  hdrs = ["rectilinear_sfm_lib.h"],
  srcs = ["rectilinear_sfm_lib.cc"],
  deps = [
    ":logger",
    "//source:gflags_xplatform",
    "//external:eigen",
    ":opencv_xplatform",
    ":ceres_xplatform",
    ":keypoint_tracking",
    ":rectilinear_camera",
    ":pose_param",
    ":point_cloud",
    ":util_math",
    ":util_time",
    ":util_file",
    ":util_string",
    ":util_opencv",
    "//third_party:json",
  ],
)

p11_cc_binary(
  name = "rectilinear_sfm",
  srcs = ["rectilinear_sfm.cc"],
  deps = [
    ":logger",
    "//source:gflags_xplatform",
    ":rectilinear_sfm_lib",
  ], 
)