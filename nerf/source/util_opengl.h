// MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#pragma once

#include "opengl_xplatform_includes.h"
#include "logger.h"
#include "Eigen/Core"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

namespace p11 { namespace opengl {

Eigen::Matrix4f perspectiveProjectionMatrix(double fovy, double aspect, double zNear, double zFar);

Eigen::Matrix4f lookAtMatrix(
    const Eigen::Vector3f& cam_pos, const Eigen::Vector3f& look_at, const Eigen::Vector3f& up);

GLuint makeGlTextureFromCvMat(const cv::Mat& image);

// if we did stuff with shaders, this can get back to immediate mode
void restoreImmediateMode();

void checkShaderCompileError(GLuint shader);

struct GlShaderProgram {
  GLuint vertex_shader, fragment_shader, program;

  void compile(const char* vertex_shader_src, const char* fragment_shader_src);
  GLint getUniform(const GLchar* name) { return glGetUniformLocation(program, name); }
  GLint getAttrib(const GLchar* name) { return glGetAttribLocation(program, name); }
  void bind() { glUseProgram(program); }
};

struct GlVertexDataXYZRGB {
  static constexpr int kPositionAttribSize = 3;
  static constexpr int kPositionAttribOffset = 0;
  static constexpr int kColorAttribSize = 3;
  static constexpr int kColorAttribOffset = 3;
  static constexpr int kNumVertexAttribs = kPositionAttribSize + kColorAttribSize;

  float x, y, z, r, g, b;
  GlVertexDataXYZRGB(
      const float x, const float y, const float z, const float r, const float g, const float b)
      : x(x), y(y), z(z), r(r), g(g), b(b)
  {}

  static void setupVertexAttributes(
      GlShaderProgram& shader, const GLchar* name_xyz, const GLchar* name_rgb);
};

struct GlVertexDataXYZRGBA {
  static constexpr int kPositionAttribSize = 3;
  static constexpr int kPositionAttribOffset = 0;
  static constexpr int kColorAttribSize = 4;
  static constexpr int kColorAttribOffset = 3;
  static constexpr int kNumVertexAttribs = kPositionAttribSize + kColorAttribSize;

  float x, y, z, r, g, b, a;
  GlVertexDataXYZRGBA(
      const float x, const float y, const float z, const float r, const float g, const float b, const float a)
      : x(x), y(y), z(z), r(r), g(g), b(b), a(a)
  {}

  static void setupVertexAttributes(
      GlShaderProgram& shader, const GLchar* name_xyz, const GLchar* name_rgb);
};

struct GlVertexDataXYZUV {
  static constexpr int kPositionAttribSize = 3;
  static constexpr int kPositionAttribOffset = 0;
  static constexpr int kTexCoordAttribSize = 2;
  static constexpr int kTexCoordAttribOffset = 3;
  static constexpr int kNumVertexAttribs = kPositionAttribSize + kTexCoordAttribSize;

  float x, y, z, u, v;
  GlVertexDataXYZUV(const float x, const float y, const float z, const float u, const float v)
      : x(x), y(y), z(z), u(u), v(v)
  {}

  static void setupVertexAttributes(
      GlShaderProgram& shader, const GLchar* name_xyz, const GLchar* name_uv);
};

// A wrapper for a vertex array object and a vertex buffer object paired together.
template <typename TVertexData>
struct GlVertexBuffer {
  GLuint vbo, vao;
  std::vector<TVertexData> vertex_data;

  void init()
  {
    glGenBuffers(1, &vbo);
    glGenVertexArrays(1, &vao);
    GL_CHECK_ERROR;
  }

  void bind()
  {
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    GL_CHECK_ERROR;
  }

  void copyVertexDataToGPU(GLenum usage = GL_DYNAMIC_DRAW)
  {
    glBufferData(
        GL_ARRAY_BUFFER,
        sizeof(float) * TVertexData::kNumVertexAttribs * vertex_data.size(),
        vertex_data.data(),
        GL_STATIC_DRAW);
  }
};

}}  // namespace p11::opengl
