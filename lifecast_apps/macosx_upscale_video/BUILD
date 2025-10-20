# MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

load("//:p11.bzl", "p11_cc_library", "p11_cc_binary")

load("@build_bazel_apple_support//lib:repositories.bzl", "apple_support_dependencies")
load("@build_bazel_rules_apple//apple:macos.bzl", "macos_application")
load("@build_bazel_rules_apple//apple:versioning.bzl", "apple_bundle_version")

apple_bundle_version(
    name = "upscale_video_version",
    build_version = "1.17",
)

filegroup(
    name = "MacAppIcon.xcassets",
    srcs = glob(["MacAppIcon.xcassets/**"]),
)

macos_application(
    name = "upscale_video",
    app_icons = [":MacAppIcon.xcassets"],
    bundle_id = "com.lifecast.UpscaleVideo",
    infoplists = [":Info.plist"],
    minimum_os_version = "14.0",
    version = ":upscale_video_version",
    deps = ["//source:upscale_video_lib"],
    additional_contents = {
      "//fonts:Helvetica.ttf" : "fonts",
      "//ml_models:edsr2_noaug.pt" : "ml_models",
      "//ml_models:edsr2_mystery.pt" : "ml_models",
      "//ml_models:unet_noaug.pt" : "ml_models",
      "//ml_models:unet_mystery.pt" : "ml_models",
      "//ml_models:rof_cpu.pt" : "ml_models",
    }
)
