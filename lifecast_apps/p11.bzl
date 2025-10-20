# MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

load("@rules_cuda//cuda:defs.bzl", "cuda_library")

# TODO: use different optimization options in debug builds

MAC_LINUX_COPTS = ["-std=c++17", "-O3", "-funroll-loops", "-pipe", "-Wno-unused-function", "-Wno-deprecated-declarations", "-Wno-int-in-bool-context", "-Wno-int-to-void-pointer-cast", "-Wno-format-security", "-Wno-writable-strings", "-Wno-comment", "-Wno-macro-redefined", "-Wno-sign-compare"]

DEFAULT_DEFINES = ["GLOG_USE_GLOG_EXPORT"] # Avoids C1189: #error:  <glog/logging.h> was not included correctly

# On windows we compile with a different compiler that has different syntax than clang.
# nvcc defers to vc++ so we need to pass along some of those options to both
WINDOWS_COMMON_COPTS = ["/O2", "/wd4244", "/wd4624", "/wd4805", "/wd4067", "/wd4005", "/wd4267"]
#WINDOWS_COMMON_COPTS = WINDOWS_COMMON_COPTS + ["/fsanitize=address"]
WINDOWS_COPTS = ["/std:c++17", "/DGLOG_NO_ABBREVIATED_SEVERITIES"] + WINDOWS_COMMON_COPTS
WINDOWS_LINKOPTS = []
#WINDOWS_LINKOPTS = ["clang_rt.asan_dynamic-x86_64.lib", "clang_rt.asan_dynamic_runtime_thunk-x86_64.lib"]

CUDA_COPTS = ["-std=c++17", "-O2", "-diag-suppress=20012,186,1394,1388,1390"]
CUDA_PLATFORM_COPTS = select({
  "@platforms//os:windows": ["-Xcompiler=" + ",".join(WINDOWS_COMMON_COPTS)],
  "//conditions:default": [],
})

DEFAULT_COPTS = select({
  "@platforms//os:osx":     MAC_LINUX_COPTS,
  "@platforms//os:linux":   MAC_LINUX_COPTS,
  "@platforms//os:windows": WINDOWS_COPTS,
})

DEFAULT_LINKOPTS = select({
  "@platforms//os:osx":     [],
  "@platforms//os:linux":   [],
  "@platforms//os:windows": WINDOWS_LINKOPTS,
})

def p11_cc_library(name, copts = None, visibility = None, linkopts = None, defines = None, **kwargs):
  native.cc_library(
    name = name,
    copts = (copts or []) + DEFAULT_COPTS,
    linkopts = (linkopts or []) + DEFAULT_LINKOPTS,
    defines = (defines or []) + DEFAULT_DEFINES,
    visibility = visibility or ["//visibility:public"],
    **kwargs,
  )

def p11_cuda_library(name, copts = None, visibility = None, linkopts = None, defines = None, **kwargs):
  cuda_library(
    name = name,
    copts = (copts or []) + CUDA_COPTS + CUDA_PLATFORM_COPTS,
    linkopts = (linkopts or []) + DEFAULT_LINKOPTS,
    defines = (defines or []) + DEFAULT_DEFINES,
    visibility = visibility or ["//visibility:public"],
    **kwargs,
  )

def p11_cc_binary(name, copts = None, linkopts = None, defines = None, **kwargs):
  native.cc_binary(
    name = name,
    copts = (copts or []) + DEFAULT_COPTS,
    linkopts = (linkopts or []) + DEFAULT_LINKOPTS,
    defines = (defines or []) + DEFAULT_DEFINES,
    **kwargs,
  )

def p11_cc_test(name, copts = None, linkopts = None, defines = None, **kwargs):
  native.cc_test(
    name = name,
    copts = (copts or []) + DEFAULT_COPTS,
    linkopts = (linkopts or []) + DEFAULT_LINKOPTS,
    defines = (defines or []) + DEFAULT_DEFINES,
    **kwargs,
  )

def p11_windows_resource(name, src, out):
  native.genrule(
    name = name,
    srcs = [src],
    outs = [out],
    cmd = "cp $(SRCS) $(@D)",
    local = 1,
    output_to_bindir = 1,
    visibility = ["//visibility:public"],
  )
