cc_library(
  name = "dear_imgui",
  hdrs = [
    "imgui.h",
    "imgui_internal.h",
    "imgui_impl_opengl3.h",
    "imgui_impl_glfw.h",
    "imconfig.h",
    "imstb_rectpack.h",
    "imstb_textedit.h",
    "imstb_truetype.h",
    "GL/gl3w.h",
    "GL/glcorearb.h"
  ],
  srcs = [
    "imgui.cpp",
    "imgui_demo.cpp",
    "imgui_draw.cpp",
    "imgui_widgets.cpp",
    "imgui_impl_opengl3.cpp",
    "imgui_impl_glfw.cpp",
    "GL/gl3w.c",
  ],
  deps = ["//source:glfw_xplatform"],
  visibility = ["//visibility:public"],
)
