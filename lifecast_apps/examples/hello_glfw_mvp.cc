// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
An example of setting up the Model-View-Projection matrix and using it in a vertex shader.

bazel run //examples:hello_glfw_mvp
*/
#include "source/util_opengl.h"

#include <cmath>
#include "source/logger.h"
#include "Eigen/Core"

namespace p11 {

opengl::GlShaderProgram basic_shader;
opengl::GlVertexBuffer<opengl::GlVertexDataXYZRGB> vertex_buffer;

static const char* kVertexShader_Basic = R"END(
#version 150 core
uniform mat4 uModelViewProjectionMatrix;
in vec3 aVertexPos;
in vec3 aVertexRGB;
out vec3 vRGB;
void main()
{
  gl_Position = uModelViewProjectionMatrix * vec4(aVertexPos, 1.0);
  vRGB = aVertexRGB;
}
)END";

static const char* kFragmentShader_Basic = R"END(
#version 150 core
in vec3 vRGB;
out vec4 frag_color;
void main()
{
  frag_color = vec4(vRGB, 1.0);
}
)END";

void drawGlStuff(GLFWwindow* window)
{
  int width, height;
  glfwGetFramebufferSize(window, &width, &height);
  glViewport(0, 0, width, height);

  glClearColor(0, 0, 0, 1);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glDisable(GL_CULL_FACE);

  // Setup camera model view projection
  static constexpr float kVerticalFovDeg = 60;
  static constexpr float kZNear = 0.01;
  static constexpr float kZFar = 1000.0;
  const float aspect_ratio = float(width) / float(height);
  const Eigen::Matrix4f projection_matrix =
      p11::opengl::perspectiveProjectionMatrix(kVerticalFovDeg, aspect_ratio, kZNear, kZFar);

  Eigen::Vector3f cam_pos = Eigen::Vector3f(0, 0, -5);
  Eigen::Vector3f look_at(0, 0, 0);
  const Eigen::Vector3f up(0, 1, 0);
  const Eigen::Matrix4f model_view_matrix = p11::opengl::lookAtMatrix(cam_pos, look_at, up);

  const Eigen::Matrix4f model_view_projection_matrix = projection_matrix * model_view_matrix;

  basic_shader.bind();
  glUniformMatrix4fv(
      basic_shader.getUniform("uModelViewProjectionMatrix"),
      1,
      false,
      model_view_projection_matrix.data());

  vertex_buffer.bind();
  opengl::GlVertexDataXYZRGB::setupVertexAttributes(basic_shader, "aVertexPos", "aVertexRGB");

  const float t = glfwGetTime();
  const float t2 = 0.5 + 0.5 * cos(t);
  vertex_buffer.vertex_data.clear();
  vertex_buffer.vertex_data.emplace_back(t2, 0.0, 0.0, 1.0, 0.0, 0.0);
  vertex_buffer.vertex_data.emplace_back(0.0, t2, 0.0, 0.0, 1.0, 0.0);
  vertex_buffer.vertex_data.emplace_back(0.0, 0.0, t2, 0.0, 0.0, 1.0);

  vertex_buffer.copyVertexDataToGPU(GL_DYNAMIC_DRAW);

  glDrawArrays(GL_TRIANGLES, 0, vertex_buffer.vertex_data.size());

  glfwSwapBuffers(window);

  GL_CHECK_ERROR;
}

}  // namespace p11

int main(int argc, char** argv)
{
  XCHECK(glfwInit());

  // specify the version of opengl to use
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

  glfwWindowHint(GLFW_VISIBLE, true);
  glfwWindowHint(GLFW_RESIZABLE, false);

  GLFWwindow* window =
      glfwCreateWindow(640, 480, "Model View Projection Example", nullptr, nullptr);
  XCHECK(window);

  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);  // Enable vsync

  // IMPORTANT- without this line, it will crash on Windows.
  XCHECK_EQ(gl3wInit(), 0) << "Failed to initialize gl3w.";

  GL_CHECK_ERROR;

  p11::basic_shader.compile(p11::kVertexShader_Basic, p11::kFragmentShader_Basic);

  p11::vertex_buffer.init();


  while (!glfwWindowShouldClose(window)) {
    p11::drawGlStuff(window);
    glfwPollEvents();
  }
  glfwTerminate();

  return EXIT_SUCCESS;
}
