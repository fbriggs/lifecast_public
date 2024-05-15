// MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include "util_opengl.h"

#include "util_math.h"
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "logger.h"

namespace p11 { namespace opengl {

Eigen::Matrix4f perspectiveProjectionMatrix(double fovy, double aspect, double zNear, double zFar)
{
  const double radf = M_PI * fovy / 180.0;
  const double tanHalfFovy = tan(radf / 2.0);
  Eigen::Matrix4f res = Eigen::Matrix4f::Zero();
  res(0, 0) = 1.0 / (aspect * tanHalfFovy);
  res(1, 1) = 1.0 / (tanHalfFovy);
  res(2, 2) = -(zFar + zNear) / (zFar - zNear);
  res(3, 2) = -1.0;
  res(2, 3) = -(2.0 * zFar * zNear) / (zFar - zNear);
  return res;
}

Eigen::Matrix4f lookAtMatrix(
    const Eigen::Vector3f& cam_pos, const Eigen::Vector3f& look_at, const Eigen::Vector3f& up)
{
  const Eigen::Vector3f f = (look_at - cam_pos).normalized();
  Eigen::Vector3f u = up.normalized();
  const Eigen::Vector3f s = f.cross(u).normalized();
  u = s.cross(f);
  Eigen::Matrix4f res;
  // This sets it us as: +X, +Y, +Z = right, up, forward
  // clang-format off
  res <<  +s.z(), +s.y(), +s.x(), -s.dot(cam_pos),
          +u.z(), +u.y(), +u.x(), -u.dot(cam_pos),
          -f.z(), -f.y(), -f.x(), +f.dot(cam_pos),
          0,      0,      0,      1;
  // clang-format on
  return res;
}

GLuint makeGlTextureFromCvMat(const cv::Mat& image)
{
  XCHECK(image.type() == CV_8UC3 || image.type() == CV_8UC4 || image.type() == CV_32F);

  GLuint gl_texture_id;
  glGenTextures(1, &gl_texture_id);
  glBindTexture(GL_TEXTURE_2D, gl_texture_id);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  
  // ChatGPT says, "OpenGL expects the row alignment of textures to be 4 bytes by default.
  // This can be an issue if your image data does not meet this requirement, particularly
  // at lower resolutions." This fixes a bug where the textures decode incorrectly if they
  //  are certain sizes.
  if (image.rows % 4 != 0 || image.cols % 4 != 0)
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  
  if (image.channels() == 3) {
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows, 0, GL_BGR, GL_UNSIGNED_BYTE, image.ptr());
  } else if (image.channels() == 4) {
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        image.cols,
        image.rows,
        0,
        GL_BGRA,
        GL_UNSIGNED_BYTE,
        image.ptr());
  } else if (image.type() == CV_32F) {  // 1 channel float
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB32F, image.cols, image.rows, 0, GL_RED, GL_FLOAT, image.ptr());

    // Swizzle the R, G, and B channels into the one channel that actually has data (R).
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_R, GL_RED);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_G, GL_RED);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_B, GL_RED);
  } else {
    XCHECK(false) << "Unsupported image type for GL texture";
  }
  glBindTexture(GL_TEXTURE_2D, 0);
  return gl_texture_id;
}

void restoreImmediateMode()
{
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, 0);
  glDisable(GL_TEXTURE_3D);
  glDisable(GL_TEXTURE_2D);
  glUseProgram(0);
}

void checkShaderCompileError(GLuint shader)
{
  GLint compiled_ok = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled_ok);
  if (compiled_ok == GL_FALSE) {
    GLint log_length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
    std::vector<char> v(log_length);
    glGetShaderInfoLog(shader, log_length, NULL, v.data());
    std::string s(begin(v), end(v));
    XPLINFO << "shader compile error: " << s;
    glDeleteShader(shader);
    exit(1);
  }
}

void GlShaderProgram::compile(const char* vertex_shader_src, const char* fragment_shader_src)
{
  // make vertex shader
  vertex_shader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertex_shader, 1, &vertex_shader_src, NULL);
  glCompileShader(vertex_shader);
  checkShaderCompileError(vertex_shader);

  // make fragment shader
  fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragment_shader, 1, &fragment_shader_src, NULL);
  glCompileShader(fragment_shader);
  checkShaderCompileError(fragment_shader);

  // make shader program
  program = glCreateProgram();
  glAttachShader(program, vertex_shader);
  glAttachShader(program, fragment_shader);
  glBindFragDataLocation(program, 0, "frag_color");
  glLinkProgram(program);
  GL_CHECK_ERROR;
}

void GlVertexDataXYZRGB::setupVertexAttributes(
    GlShaderProgram& shader, const GLchar* name_xyz, const GLchar* name_rgb)
{
  glEnableVertexAttribArray(shader.getAttrib(name_xyz));
  glVertexAttribPointer(
      shader.getAttrib(name_xyz),
      kPositionAttribSize,
      GL_FLOAT,
      GL_FALSE,
      sizeof(float) * kNumVertexAttribs,
      (void*)(sizeof(float) * kPositionAttribOffset));

  glEnableVertexAttribArray(shader.getAttrib(name_rgb));
  glVertexAttribPointer(
      shader.getAttrib(name_rgb),
      kColorAttribSize,
      GL_FLOAT,
      GL_FALSE,
      sizeof(float) * kNumVertexAttribs,
      (void*)(sizeof(float) * kColorAttribOffset));
  GL_CHECK_ERROR;
}


void GlVertexDataXYZRGBA::setupVertexAttributes(
    GlShaderProgram& shader, const GLchar* name_xyz, const GLchar* name_rgba)
{
  glEnableVertexAttribArray(shader.getAttrib(name_xyz));
  glVertexAttribPointer(
      shader.getAttrib(name_xyz),
      kPositionAttribSize,
      GL_FLOAT,
      GL_FALSE,
      sizeof(float) * kNumVertexAttribs,
      (void*)(sizeof(float) * kPositionAttribOffset));

  glEnableVertexAttribArray(shader.getAttrib(name_rgba));
  glVertexAttribPointer(
      shader.getAttrib(name_rgba),
      kColorAttribSize,
      GL_FLOAT,
      GL_FALSE,
      sizeof(float) * kNumVertexAttribs,
      (void*)(sizeof(float) * kColorAttribOffset));
  GL_CHECK_ERROR;
}

void GlVertexDataXYZUV::setupVertexAttributes(
    GlShaderProgram& shader, const GLchar* name_xyz, const GLchar* name_uv)
{
  glEnableVertexAttribArray(shader.getAttrib(name_xyz));
  glVertexAttribPointer(
      shader.getAttrib(name_xyz),
      kPositionAttribSize,
      GL_FLOAT,
      GL_FALSE,
      sizeof(float) * kNumVertexAttribs,
      (void*)(sizeof(float) * kPositionAttribOffset));

  glEnableVertexAttribArray(shader.getAttrib(name_uv));
  glVertexAttribPointer(
      shader.getAttrib(name_uv),
      kTexCoordAttribSize,
      GL_FLOAT,
      GL_FALSE,
      sizeof(float) * kNumVertexAttribs,
      (void*)(sizeof(float) * kTexCoordAttribOffset));
  GL_CHECK_ERROR;
}

}}  // namespace p11::opengl
