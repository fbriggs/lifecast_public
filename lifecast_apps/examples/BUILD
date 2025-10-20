# MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

load("//:p11.bzl", "p11_cc_library", "p11_cuda_library", "p11_cc_binary", "p11_windows_resource", "DEFAULT_COPTS")

### windows resource shenanagins ###

# Windows bazel doesn't copy data correctly so we have to brute force it to get it to copy 
# resources to the build directory. Unfortunately, this block is copy-pasted once in /source and once in /examples.

p11_windows_resource(name="winres_rof_cuda", src="//ml_models:rof_cuda.pt", out="rof_cuda.pt")
p11_windows_resource(name="winres_helvetica", src="//fonts:Helvetica.ttf", out="Helvetica.ttf")
p11_windows_resource(name="winres_depth_anything3_cuda", src="//ml_models:depth_anything3_cuda.pt", out="depth_anything3_cuda.pt")

### end windows resource shenangins ###

p11_cc_binary(
  name = "hello_world",
  srcs = ["hello_world.cc"],
  deps = [
    "//source:logger",
    "//source:gflags_xplatform",
  ],
)

p11_cc_binary(
  name = "hello_logger",
  srcs = ["hello_logger.cc"],
  deps = [
    "//source:logger",
  ],
)

p11_cc_binary(
  name = "hello_check",
  srcs = ["hello_check.cc"],
  deps = [
    "//source:logger",
  ],
)

p11_cc_binary(
  name = "hello_json",
  srcs = ["hello_json.cc"],
  deps = [
    "//source:logger",
    "//third_party:json",
  ],
)

p11_cc_binary(
  name = "hello_eigen",
  srcs = ["hello_eigen.cc"],
  deps = [
    "//external:eigen",
  ],
)

p11_cc_binary(
  name = "hello_glfw",
  srcs = ["hello_glfw.cc"],
  deps = [
    "//source:logger",
    "//source:glfw_xplatform",
  ]
)

p11_cc_binary(
  name = "hello_glfw_mvp",
  srcs = ["hello_glfw_mvp.cc"],
  deps = [
    "//source:logger",
    "//source:glfw_xplatform",
    "//source:util_opengl"
  ]
)

p11_cc_binary(
  name = "hello_imgui",
  srcs = ["hello_imgui.cc"],
  deps = [
    "//source:logger",
    "//source:gflags_xplatform",
    "//source:dear_imgui_app",
    "//third_party/dear_imgui:dear_imgui",
    "//source:opencv_xplatform",
  ]
)

p11_cc_binary(
  name = "hello_tinyplot",
  srcs = ["hello_tinyplot.cc"],
  deps = [
    "//source:logger",
    "//source:dear_imgui_app",
    "//source:imgui_tinyplot",
    "//third_party/dear_imgui:dear_imgui",
  ]
)

p11_cc_binary(
  name = "hello_opencv",
  srcs = ["hello_opencv.cc"],
  deps = [
    "//source:logger",
    "//source:gflags_xplatform",
    "//source:opencv_xplatform",
  ],
)

p11_cc_binary(
  name = "hello_jpegturbo",
  srcs = ["hello_jpegturbo.cc"],
  deps = [
    "//source:logger",
    "//source:gflags_xplatform",
    "//source:opencv_xplatform",
    "//source:jpegturbo_xplatform",
  ]
)

cc_test(
  name = "hello_gtest",
  srcs = ["hello_gtest.cc"],
  deps = [
    "//external:gtest_main",
  ],
)

p11_cc_binary(
  name = "hello_dialog",
  srcs = ["hello_dialog.cc"],
  deps = [
    "//source:logger",
    "//source:gflags_xplatform",
    "//third_party:tinyfiledialogs",
  ],
)

p11_cc_binary(
  name = "hello_torch",
  srcs = ["hello_torch.cc"],
  deps = ["//source:libtorch_xplatform"],
)

p11_cc_binary(
  name = "hello_rof",
  srcs = ["hello_rof.cc"],
  deps = [
    "//source:logger",
    "//source:gflags_xplatform",
    "//source:opencv_xplatform",
    "//source:rof",
  ],
  data = select({
      "@platforms//os:windows": [":winres_rof_cuda"],
      "//conditions:default":   [],
  })
)

p11_cc_binary(
  name = "hello_depth_anything2",
  srcs = ["hello_depth_anything2.cc"],
  deps = [
    "//source:logger",
    "//source:gflags_xplatform",
    "//source:opencv_xplatform",
    "//source:depth_anything2",
  ],
  data = select({
    "@platforms//os:osx":    ["//ml_models:depth_anything3_cuda.pt"],
    "@platforms//os:linux":  ["//ml_models:depth_anything3_cuda.pt"],
    "@platforms//os:windows": [":winres_depth_anything3_cuda"],
  })
)

p11_cuda_library(
  name = "hello_cuda_lib",
  srcs = ["hello_cuda.cu"],
  hdrs = ["hello_cuda.h"],
  deps = ["//source:util_cuda"],
  target_compatible_with = select({
    "@platforms//os:osx": ["@platforms//os:incompatible"],
    "//conditions:default": [],
  }),
)

p11_cc_binary(
  name = "hello_cuda",
  srcs = ["hello_cuda.cc"],
  deps = [
    "//source:logger",
    "//source:gflags_xplatform",
    "//source:util_cuda",
    ":hello_cuda_lib",
  ],
  target_compatible_with = select({
    "@platforms//os:osx": ["@platforms//os:incompatible"],
    "//conditions:default": [],
  }),
)

p11_cc_binary(
  name = "hello_write_mp4",
  srcs = ["hello_write_mp4.cc"],
  deps = [
    "//source:logger",
    "//source:gflags_xplatform",
    "//source:opencv_xplatform",
  ],
)

p11_cc_binary(
  name = "hello_preferences",
  srcs = ["hello_preferences.cc"],
  deps = [
    "//source:logger",
    "//source:preferences",
  ],
  linkopts = select({
      "@platforms//os:windows": ["Shell32.lib"],
      "//conditions:default":   []
  })
)

p11_cc_binary(
  name = "hello_httplib",
  srcs = ["hello_httplib.cc"],
  deps = [
    "//source:logger",
    "//source:gflags_xplatform",
    "//third_party:httplib",
  ],
)

p11_cc_binary(
  name = "hello_libav",
  srcs = ["hello_libav.cc"],
  deps = [
    "//source:logger",
    "//source:gflags_xplatform",
    "//source:opencv_xplatform",
    "//source:ffmpeg_xplatform",
  ],
)

p11_cc_binary(
  name = "hello_transcode",
  srcs = ["hello_transcode.cc"],
  deps = [
    "//source:logger",
    "//source:gflags_xplatform",
    "//source:opencv_xplatform",
    "//source:video_transcode_lib",
  ],
)

p11_cc_binary(
  name = "hello_transcode_all_streams",
  srcs = ["hello_transcode_all_streams.cc"],
  deps = [
    "//source:logger",
    "//source:gflags_xplatform",
    "//source:opencv_xplatform",
    "//source:video_transcode_lib",
  ],
)

p11_cc_binary(
  name = "hello_transcode_multithread",
  srcs = ["hello_transcode_multithread.cc"],
  deps = [
    "//source:logger",
    "//source:gflags_xplatform",
    "//source:opencv_xplatform",
    "//source:video_transcode_lib",
  ],
)
