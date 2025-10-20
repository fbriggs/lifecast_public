# MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

load("//:p11.bzl", "p11_cc_library", "p11_cuda_library", "p11_cc_binary", "p11_cc_test", "p11_windows_resource", "DEFAULT_COPTS")

### cross-platform dependency selection ############################################################

# A dummy main() file to be used in selects for platforms that don't support a target.
# This is exported so it can also be referenced in /examples.
exports_files(["unsupported_main.cc"])

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
    name = "ffmpeg_xplatform",
    deps = select({
      "@platforms//os:osx":     ["@ffmpeg_osx_homebrew//:ffmpeg"],
      "@platforms//os:linux":   ["@ffmpeg_linux//:ffmpeg"],
      "@platforms//os:windows": ["@ffmpeg_windows_vcpkg//:ffmpeg"],
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
    name = "jpegturbo_xplatform",
    deps = select({
      "@platforms//os:osx":     ["@jpegturbo_osx_homebrew//:jpegturbo"],
      "@platforms//os:linux":   ["@jpegturbo_linux//:jpegturbo"],
      "@platforms//os:windows": ["@libjpeg-turbo_windows_vcpkg//:jpegturbo"],
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
    }),
    defines = ["GLOG_USE_GLOG_EXPORT"], # Avoids C1189: #error:  <glog/logging.h> was not included correctly
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
    name = "cpprest_xplatform",
    deps = select({
      "@platforms//os:osx":   ["@cpprest_osx_homebrew//:cpprest"],
      "@platforms//os:linux": ["@cpprest_linux//:cpprest"],
      "@platforms//os:windows": ["@cpprestsdk_windows_vcpkg//:cpprest"],
      
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

### windows resource shenanagins ###################################################################

# Windows bazel doesn't copy data correctly so we have to brute force it to get it to copy 
# resources to the build directory. Unfortunately, this block is copy-pasted once in /source and once in /examples.

p11_windows_resource(name="winres_rof_cuda", src="//ml_models:rof_cuda.pt", out="rof_cuda.pt")
p11_windows_resource(name="winres_edsr2_noaug", src="//ml_models:edsr2_noaug.pt", out="edsr2_noaug.pt")
p11_windows_resource(name="winres_edsr2_mystery", src="//ml_models:edsr2_mystery.pt", out="edsr2_mystery.pt")
p11_windows_resource(name="winres_unet_noaug", src="//ml_models:unet_noaug.pt", out="unet_noaug.pt")
p11_windows_resource(name="winres_unet_mystery", src="//ml_models:unet_mystery.pt", out="unet_mystery.pt")
p11_windows_resource(name="winres_depth_anything3_cuda", src="//ml_models:depth_anything3_cuda.pt", out="depth_anything3_cuda.pt")
p11_windows_resource(name="winres_helvetica", src="//fonts:Helvetica.ttf", out="Helvetica.ttf")
p11_windows_resource(name="winres_eula", src="//windows:EULA.rtf", out="EULA.rtf")
p11_windows_resource(name="winres_icon", src="//windows:icon_256.ico", out="icon_256.ico")

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

p11_cc_library(
  name = "scope_exit",
  hdrs = [
    "scope_exit.h"
  ],
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
  deps = [
    "//external:eigen",
    ":logger",
  ],
)

p11_cc_library (
  name = "util_string",
  hdrs = ["util_string.h"],
  deps = [],
)

p11_cc_library (
  name = "util_command",
  hdrs = ["util_command.h"],
  srcs = ["util_command.cc"],
  deps = [":logger"],
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
  name = "util_browser",
  hdrs = ["util_browser.h"],
  deps = [],
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
  name = "torch_opencv",
  hdrs = ["torch_opencv.h"],
  srcs = ["torch_opencv.cc"],
  deps = [
    ":libtorch_xplatform",
    ":opencv_xplatform",
    ":util_torch",
    ":logger",
  ],
)

p11_cc_library (
  name = "torch_opengl",
  hdrs = ["torch_opengl.h"],
  srcs = ["torch_opengl.cc"],
  deps = [
    ":libtorch_xplatform",
    ":opencv_xplatform",
    ":opengl_xplatform_includes",
    ":util_torch",
    ":util_opengl",
    ":logger",
    "@rules_cuda//cuda:runtime",
  ],
)

p11_cc_library (
  name = "http_strings",
  hdrs = ["http_strings.h"],
  deps = [
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

config_setting(
  name = "util_runfile_config",
  values = {
    "define": "look_for_runfiles_in_mac_app=true"
  }
)
p11_cc_library (
  name = "util_runfile",
  hdrs = ["util_runfile.h"],
  srcs = ["util_runfile.cc"],
  deps = [
    ":logger",
    "@bazel_tools//tools/cpp/runfiles"
  ],
  defines = select({
    ":util_runfile_config": ["LOOK_FOR_RUNFILES_IN_MAC_APP"],
    "//conditions:default": [],
  })

)

p11_cc_library(
  name = "util_cuda",
  hdrs = ["util_cuda.h"],
  deps = [
    ":logger",
  ],
)

p11_cc_library(
  name = "concurrency_lib",
  hdrs = ["concurrency_lib.h"],
  deps = [
    ":logger",
  ],
)

p11_cc_library(
  name = "video_transcode_lib",
  hdrs = ["video_transcode_lib.h"],
  srcs = ["video_transcode_lib.cc"],
  deps = [
    ":logger",
    ":scope_exit",
    ":opencv_xplatform",
    ":ffmpeg_xplatform",
    ":util_file",
    ":util_string",
    ":util_opencv",
    ":concurrency_lib",
  ],
)

p11_cc_library(
  name = "view_interpolation",
  hdrs = ["view_interpolation.h"],
  srcs = ["view_interpolation.cc"],
  deps = [
    "//source:logger",
    "//source:opencv_xplatform",
    "//source:rof",
    "//source:projection",
  ],
  data = select({
      "@platforms//os:windows": [":winres_rof_cuda"],
      "//conditions:default":   [],
  })
)


p11_cc_library (
  name = "dear_imgui_app",
  hdrs = ["dear_imgui_app.h"],
  srcs = ["dear_imgui_app.cc"],
  deps = [
    ":logger",
    ":opengl_xplatform_includes",
    "//third_party/dear_imgui:dear_imgui",
  ],
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
  name = "equirectangular_camera",
  hdrs = ["equirectangular_camera.h"],
  deps = [
    "//external:eigen",
    ":logger",
    ":util_math",
  ],
)

p11_cc_library (
  name = "nerf_kludge_camera",
  hdrs = ["nerf_kludge_camera.h"],
  deps = [
    ":ceres_xplatform",
    ":opencv_xplatform",
    "//external:eigen",
    ":logger",
    ":util_math",
    ":fisheye_camera",
    ":rectilinear_camera",
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
  name = "projection",
  hdrs = ["projection.h"],
  srcs = ["projection.cc"],
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
  name = "depth_estimation",
  hdrs = ["depth_estimation.h"],
  srcs = ["depth_estimation.cc"],
  deps = [
    "//external:eigen",
    ":logger",
    "//third_party:turbo_colormap",
    ":opencv_xplatform",
    ":fisheye_camera",
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
    "//third_party:json",
  ],
)

p11_cc_library (
  name = "imgui_cvmat",
  hdrs = ["imgui_cvmat.h"],
  srcs = ["imgui_cvmat.cc"],
  deps = [
    ":logger",
    ":opencv_xplatform",
    ":glfw_xplatform",
    ":util_opengl",
    ":dear_imgui_app",
  ],
)

p11_cc_library (
  name = "imgui_tinyplot",
  hdrs = ["imgui_tinyplot.h"],
  srcs = ["imgui_tinyplot.cc"],
  deps = [
    ":logger",
    "//third_party/dear_imgui:dear_imgui",
  ],
)

p11_cc_library (
  name = "imgui_filedialog",
  hdrs = ["imgui_filedialog.h"],
  deps = [
    ":logger",
    ":opencv_xplatform",
    ":glfw_xplatform",
    "//third_party:tinyfiledialogs",
    ":dear_imgui_app",
  ],
)

p11_cc_library (
  name = "imgui_run_cancel_progress",
  hdrs = ["imgui_run_cancel_progress.h"],
  deps = [
    ":dear_imgui_app",
  ],
)

p11_cc_library(
  name = "turbojpeg_wrapper",
  hdrs = ["turbojpeg_wrapper.h"],
  srcs = ["turbojpeg_wrapper.cc"],
  deps = [
    ":jpegturbo_xplatform",
    ":opencv_xplatform",
    ":logger",
    ":util_file"
  ],
)

p11_cc_library(
  name = "rof",
  hdrs = ["rof.h"],
  srcs = ["rof.cc"],
  deps = [
    ":logger",
    "//source:gflags_xplatform",
    ":opencv_xplatform",
    ":libtorch_xplatform",
    ":util_runfile",
    ":util_time",
    ":util_math",
    ":util_torch",
  ],
  data = select({
      #"@platforms//os:osx":    ["//ml_models:rof_mps.pt"],
      "@platforms//os:osx":    ["//ml_models:rof_cpu.pt"],
      "@platforms//os:linux":  ["//ml_models:rof_cuda.pt"],
      "@platforms//os:windows": ["//ml_models:rof_cuda.pt"],
    })
)

p11_cc_library(
  name = "depth_anything2",
  hdrs = ["depth_anything2.h"],
  srcs = ["depth_anything2.cc"],
  deps = [
    ":logger",
    "//source:gflags_xplatform",
    ":opencv_xplatform",
    ":libtorch_xplatform",
    ":util_runfile",
    ":util_time",
    ":util_math",
    ":util_torch",
    "//external:eigen",
  ],
  data = select({
      "@platforms//os:osx":    ["//ml_models:depth_anything3_cuda.pt"],
      "@platforms//os:linux":  ["//ml_models:depth_anything3_cuda.pt"],
      "@platforms//os:windows": ["//source:winres_depth_anything3_cuda"],
    })
)

p11_cc_library(
  name = "inpaint_ceres",
  hdrs = ["inpaint_ceres.h"],
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

p11_cc_library (
  name = "preferences",
  hdrs = ["preferences.h"],
  srcs = ["preferences.cc"],
  deps = [
    ":logger",
    ":util_string",
  ],
)


### binaries ###

p11_cc_library(
  name = "gsplat_lib",
  srcs = ["gsplat_lib.cc"],
  hdrs = ["gsplat_lib.h"],
  deps = [
    "//third_party:gsplat_cuda",
    ":logger",
    ":util_torch",
    ":rectilinear_camera",
    ":lifecast_splat_io_lib",
  ],
)

p11_cc_library (
  name = "tinysr_lib",
  hdrs = ["tinysr_lib.h", "tinysr_zoo.h"],
  srcs = ["tinysr_lib.cc"],
  deps = [
    "//external:eigen",
    ":opencv_xplatform",
    ":libtorch_xplatform",
    ":logger",
    ":util_string",
    ":util_file",
    ":util_torch",
    ":util_opencv",
    ":util_time",
    ":torch_opencv",
  ],
  data = [
    "//ml_models:edsr2_noaug.pt",
    "//ml_models:edsr2_mystery.pt",
    "//ml_models:unet_noaug.pt",
    "//ml_models:unet_mystery.pt",
  ],
)

p11_cc_binary(
  name = "tinysr",
  srcs = ["tinysr.cc"],
  deps = [
    "//source:gflags_xplatform",
    ":logger",
    ":tinysr_lib",
  ],
)

p11_cc_binary(
  name = "lifecast_upscalevideoai",
  srcs = ["lifecast_upscalevideoai.cc"],
  deps = [
    "//external:eigen",
    "//third_party:json",
    ":gflags_xplatform",
    ":libtorch_xplatform",
    ":logger",
    ":fisheye_camera",
    ":rectilinear_camera",
    ":util_string",
    ":util_file",
    ":util_torch",
    ":video_transcode_lib",
    ":util_time",
    ":projection",
    ":tinysr_lib",
  ],
)

p11_cc_library(
  name = "convert_to_obj",
  hdrs = ["convert_to_obj.h"],
  srcs = ["convert_to_obj.cc"],
  deps = [
    "//source:gflags_xplatform",
    ":logger",
    ":depth_estimation",
    ":fisheye_camera",
    ":projection",
    ":rof",
    ":turbojpeg_wrapper",
    ":ldi_common",
    ":util_file",
    ":util_string",
    ":util_time",
    ":util_opencv",
  ],
)

p11_cc_library(
  name = "vve_lib",
  srcs = ["vve.cc"],
  deps = [
    ":logger",
    "//third_party/dear_imgui:dear_imgui",
    ":dear_imgui_app",
    ":imgui_filedialog",
    ":opencv_xplatform",
    ":torch_opencv",
    ":util_opengl",
    ":imgui_cvmat",
    ":util_runfile",
    ":util_math",
    ":util_file",
    ":util_browser",
    ":util_opencv",
    ":util_command",
    ":util_torch",
    ":projection",
    ":fisheye_camera",
    ":turbojpeg_wrapper",
    ":ldi_common",
    ":ldi_pipeline_lib",
    ":preferences",
    "//third_party:turbo_colormap",
    "//third_party:httplib",
    ":convert_to_obj",
    ":http_strings",
  ] + select({
    "@platforms//os:macos": ["@homebrew_gcc_libs//:gcc_runtime", "@homebrew_openblas_libs//:openblas"],
    "//conditions:default": [],
  }),

  data = [
    "//fonts:Helvetica.ttf",
  ],
)

p11_cc_binary(
  name = "vve",
  deps = [":vve_lib"],
  data = select({
      "@platforms//os:windows": [":winres_rof_cuda", ":winres_helvetica", ":winres_eula", ":winres_icon"],
      "//conditions:default":   [],
  }),
)

p11_cc_library(
  name = "volurama_helpers",
  hdrs = [
    "volurama_timeline.h",
    "volurama_camera_polymorphism.h",
  ],
  srcs = [
    "volurama_timeline.cc",
    "volurama_camera_polymorphism.cc",
  ],
  deps = [
    ":logger",
    "//third_party/dear_imgui:dear_imgui",
    ":rectilinear_camera",
    ":equirectangular_camera",
    ":opencv_xplatform",
    ":lifecast_nerf_lib",
  ],
)

p11_cc_library(
  name = "volurama_lib",
  srcs = ["volurama.cc"],
  deps = [
    ":logger",
    "//third_party/dear_imgui:dear_imgui",
    ":dear_imgui_app",
    ":imgui_filedialog",
    ":opencv_xplatform",
    ":util_opengl",
    ":imgui_cvmat",
    ":util_runfile",
    ":util_math",
    ":util_file",
    ":util_time",
    ":util_browser",
    ":util_opencv",
    ":util_command",
    ":util_torch",
    ":projection",
    ":fisheye_camera",
    ":turbojpeg_wrapper",
    ":preferences",
    ":point_cloud",
    ":rectilinear_sfm_lib",
    ":lifecast_nerf_lib",
    ":multicamera_dataset",
    ":imgui_tinyplot",
    ":volurama_helpers",
    ":vignette",
    ":convert_to_obj",
  ] + select({
    "@platforms//os:macos": ["@homebrew_gcc_libs//:gcc_runtime", "@homebrew_openblas_libs//:openblas"],
    "//conditions:default": [],
  }),
  data = ["//fonts:Helvetica.ttf"],
)

# This is some shenanagins for the mac app build
p11_cc_binary(
  name = "volurama",
  deps = [":volurama_lib"],
  data = select({
      "@platforms//os:windows": [":winres_helvetica", ":winres_eula", ":winres_icon"],
      "//conditions:default":   [],
  }),
)

p11_cc_library( # This is the mac / windows GUI app for upscaling
  name = "upscale_video_lib",
  srcs = ["upscale_video.cc"],
  hdrs = ["char_traits_uc.h"], # workaround for Xcode 16.3 breaking it
  deps = [
    ":logger",
    "//third_party/dear_imgui:dear_imgui",
    ":dear_imgui_app",
    ":imgui_filedialog",
    ":opencv_xplatform",
    ":cpprest_xplatform",
    ":util_opengl",
    ":imgui_cvmat",
    ":util_runfile",
    ":util_math",
    ":util_file",
    ":util_time",
    ":util_browser",
    ":util_command",
    ":util_torch",
    ":util_opencv",
    ":video_transcode_lib",
    ":preferences",
    ":tinysr_lib",
    ":torch_opencv",
    ":view_interpolation",
  ],
  data = ["//fonts:Helvetica.ttf"],
)

# This is some shenanagins for the mac app build
p11_cc_binary(
  name = "upscale_video",
  deps = [":upscale_video_lib"],
  data = select({
      "@platforms//os:windows": [
          ":winres_helvetica",
          ":winres_eula",
          ":winres_icon",
          ":winres_edsr2_noaug",
          ":winres_edsr2_mystery",
          ":winres_unet_noaug",
          ":winres_unet_mystery",
      ],
      "//conditions:default":   [],
  }),
)

p11_cc_library(
  name = "4dg_studio_helpers",
  hdrs = [
    "4dgstudio_timeline.h",
    "4dgstudio_web_template.h",
  ],
  srcs = [
    "4dgstudio_timeline.cc",
  ],
  deps = [
    ":logger",
    ":rectilinear_camera",
    "//third_party/dear_imgui:dear_imgui",
  ],
)

p11_cc_binary(
  name = "4dgstudio",
  srcs = ["4dgstudio.cc"],
  deps = [
    ":logger",
    "//third_party/dear_imgui:dear_imgui",
    ":dear_imgui_app",
    ":imgui_filedialog",
    ":opencv_xplatform",
    ":util_opengl",
    ":util_runfile",
    ":util_math",
    ":util_file",
    ":util_browser",
    ":util_command",
    ":util_torch",
    ":video_transcode_lib",
    ":preferences",
    ":torch_opencv",
    ":torch_opengl",
    ":4dg_studio_helpers",
    ":incremental_sfm_lib",
    ":lifecast_splat_lib",
    "//third_party:httplib",
  ],
  data = select({
      "@platforms//os:windows": [
          ":winres_helvetica",
          ":winres_eula",
          ":winres_icon",
          ":winres_edsr2_noaug",
          ":winres_edsr2_mystery",
          ":winres_unet_noaug",
          ":winres_unet_mystery",
          "//fonts:Helvetica.ttf",
      ],
      "//conditions:default":   ["//fonts:Helvetica.ttf"],
  }),
  #linkopts = ["/SUBSYSTEM:CONSOLE"]
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
  name = "noise",
  hdrs = ["noise.h"],
  srcs = ["noise.cc"],
  deps = [],
)

p11_cc_library(
  name = "neural_hashmap_torch",
  srcs = ["neural_hashmap_torch.cc"],
  hdrs = ["ngp_radiance_model.h"],
  deps = [
    ":logger",
    ":libtorch_xplatform",
    ":util_time",
  ],
)

p11_cuda_library(
  name = "tcnn_module",
  srcs = ["tcnn_module.cu"],
  hdrs = ["tcnn_module.h"],
  deps = [
    ":logger",
    ":util_torch",
    ":libtorch_xplatform",
    "//third_party:tiny_cuda_nn",
  ],
  target_compatible_with = select({
    "@platforms//os:osx": ["@platforms//os:incompatible"],
    "//conditions:default": [],
  }),
)

p11_cuda_library(
  name = "neural_hashmap_cuda",
  srcs = ["neural_hashmap_cuda.cu"],
  hdrs = ["ngp_radiance_model.h"],
  deps = [
    ":logger",
    ":tcnn_module",
  ],
  target_compatible_with = select({
    "@platforms//os:osx": ["@platforms//os:incompatible"],
    "//conditions:default": [],
  }),
)

p11_cc_library(
  name = "ngp_radiance_model",
  hdrs = ["ngp_radiance_model.h"],
  srcs = ["ngp_radiance_model.cc"],
  deps = select({
    "@platforms//os:osx": [":neural_hashmap_torch"],
    "//conditions:default": [":neural_hashmap_cuda"],  # TODO: make a command line flag to use torch on linux/win?
    #"//conditions:default": [":neural_hashmap_torch"],
  }),
)

p11_cc_library (
  name = "multicamera_dataset",
  hdrs = ["multicamera_dataset.h"],
  srcs = ["multicamera_dataset.cc"],
  deps = [
    ":logger",
    ":opencv_xplatform",
    ":libtorch_xplatform",
    ":util_math",
    ":util_file",
    ":util_string",
    ":util_time",
    ":util_torch",
    ":torch_opencv",
    "//third_party:json",
    ":rectilinear_camera",
    ":fisheye_camera",
    ":nerf_kludge_camera",
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
    ":multicamera_dataset",
    ":rectilinear_camera",
    ":fisheye_camera",
    ":nerf_kludge_camera",
    ":ldi_common",
    ":vignette",
    ":nerf_heuristic_seg",
    ":ldi_image_based_rendering",
  ],
)

p11_cc_library (
  name = "lifecast_splat_io_lib",
  hdrs = [
    "lifecast_splat_math.h",
    "lifecast_splat_config.h",
    "lifecast_splat_io.h",
  ],
  srcs = [
    "lifecast_splat_io.cc",
  ],
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
    ":util_torch",
    ":rectilinear_camera",
    ":fisheye_camera",
    ":nerf_kludge_camera",
    ":multicamera_dataset",
    ":point_cloud",
  ],
)

p11_cc_library (
  name = "lifecast_splat_lib",
  hdrs = [
    "lifecast_splat_lib.h",
    "lifecast_splat_population.h",
  ],
  srcs = [
    "lifecast_splat_lib.cc",
    "lifecast_splat_population.cc",
  ],
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
    ":util_torch",
    ":torch_opencv",
    ":rectilinear_camera",
    ":fisheye_camera",
    ":nerf_kludge_camera",
    ":multicamera_dataset",
    ":point_cloud",
    ":lifecast_splat_io_lib",
    ":gsplat_lib",
    ":depth_anything2",
    ":rof",
    "//third_party:turbo_colormap",
  ],
)

p11_cc_binary(
  name = "lifecast_splat",
  srcs = ["lifecast_splat.cc"],
  deps = [
    ":logger",
    "//source:gflags_xplatform",
    ":util_file",
    ":lifecast_splat_lib",
  ],
)

p11_cc_library(
  name = "ldi_segmentation",
  hdrs = ["ldi_segmentation.h", "ldi_segmentation2.h"],
  srcs = ["ldi_segmentation.cc", "ldi_segmentation2.cc"],
  deps = [
    ":logger",
    "//external:eigen",
    ":opencv_xplatform",
    ":libtorch_xplatform",
    ":fisheye_camera",
    ":util_math",
    ":util_time",
    ":util_opencv",
  ],
)

p11_cc_library(
  name = "ldi_image_based_rendering",
  hdrs = ["ldi_image_based_rendering.h"],
  srcs = ["ldi_image_based_rendering.cc"],
  deps = [
    ":logger",
    "//external:eigen",
    ":opencv_xplatform",
    ":libtorch_xplatform",
    ":util_torch",
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
    ":ldi_segmentation",
    ":inpaint_ceres",
    ":projection",
    ":vignette",
  ],
)

p11_cc_library (
  name = "ldi_pipeline_lib",
  hdrs = ["ldi_pipeline_lib.h"],
  srcs = ["ldi_pipeline_lib.cc"],
  deps = [
    ":logger",
    "//source:gflags_xplatform",
    ":opencv_xplatform",
    ":libtorch_xplatform",
    ":turbojpeg_wrapper",
    ":util_math",
    ":util_time",
    ":util_file",
    ":util_string",
    ":fisheye_camera",
    ":projection",
    ":vignette",
    ":rof",
    ":depth_estimation",
    ":util_opengl",
    ":util_opencv",
    ":ldi_common",
    ":ldi_segmentation",
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
    ":imgui_cvmat",
    "//third_party:json",
  ],
)

p11_cc_library (
  name = "incremental_sfm_lib",
  hdrs = ["incremental_sfm_lib.h"],
  srcs = ["incremental_sfm_lib.cc"],
  deps = [
    ":logger",
    "//external:eigen",
    ":opencv_xplatform",
    ":ceres_xplatform",
    ":fisheye_camera",
    ":rectilinear_camera",
    ":pose_param",
    ":point_cloud",
    ":util_math",
    ":util_time",
    ":util_file",
    ":util_opencv",
    "//third_party:vfc",
    "//third_party:json",
    ":rof",
    ":depth_anything2",
    ":libtorch_xplatform",
    ":keypoint_tracking",
    ":multicamera_dataset",
  ],
)

p11_cc_binary(
  name = "incremental_sfm",
  srcs = ["incremental_sfm.cc"],
  deps = [
    "//source:gflags_xplatform",
    ":logger",
    ":incremental_sfm_lib",
    ":point_cloud",
    ":util_string",
    ":util_file",
    ":video_transcode_lib",
  ],
)

p11_cc_test(
  name = "test_camera_and_projection",
  srcs = ["test_camera_and_projection.cc"],
  deps = [
    "//external:gtest_main",
    ":logger",
    "//external:eigen",
    ":fisheye_camera",
    ":projection",
  ],
)
