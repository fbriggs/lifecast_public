// MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// To avoid problems with OpenGL include order on multiple platforms, include util_opengl.h first!
// CAVEAT: this brings in all of dear_imgui, because we are piggy-backing on its copy of GL/glcorearb.h
#pragma once

//#ifndef _WIN32
//#define GLFW_INCLUDE_GLCOREARB  // Needed for GL_R32F texture mode
//#endif
//#define GL_GLEXT_PROTOTYPES

#include "third_party/dear_imgui/GL/gl3w.h"
#include "third_party/dear_imgui/imgui.h"
#include "third_party/dear_imgui/imgui_impl_glfw.h"
#include "third_party/dear_imgui/imgui_impl_opengl3.h"
// The Glfw include must go after imgui, and adding a comment prevents clang-format from sorting.
#include "GLFW/glfw3.h"

#ifdef __linux__
#include <GL/gl.h>
#endif

//#ifdef __APPLE__
//#include <OpenGL/gl3.h>  // Needed for glGenVertexArrays on OS X.
//#endif

#define GL_CHECK_ERROR \
  XCHECK_EQ(glGetError(), GL_FALSE) << "opengl error at: " << __FILE__ << " " << __LINE__;
