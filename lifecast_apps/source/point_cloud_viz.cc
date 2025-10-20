// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
bazel run -- //source:point_cloud_viz \
--point_size 2 \
--point_cloud ~/Desktop/rectilinear_sfm/pointcloud_sfm.bin \
--cam_json ~/Desktop/rectilinear_sfm/dataset.json
*/
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "source/util_opengl.h"

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "gflags/gflags.h"
#include "logger.h"

#include "point_cloud.h"
#include "util_file.h"
#include "util_math.h"
#include "util_opengl.h"
#include "util_string.h"
#include "third_party/json.h"

DEFINE_string(point_cloud, "", "path to point cloud csv file");
DEFINE_string(anim_dir, "", "if non empty, well go in animation mode using files from this dir");
DEFINE_string(cam_json, "", "path to json file with camera poses to be visualized");
DEFINE_string(cam_json2, "", "path to a second camera json file, which can be toggled");
DEFINE_double(point_size, 2, "draw size for points");
DEFINE_bool(vsync, true, "vsync");
DEFINE_double(vfov_deg, 60.0, "vertical FOV (degrees)");
DEFINE_double(znear, 0.01, "near clipping plane");
DEFINE_double(zfar, 100000.0, "far clipping plane");
DEFINE_double(subsample, 1.0, "if less than 1, a fraction of the point cloud is discarded");

namespace p11 { namespace point_cloud_viz {
namespace {

static const char* kVertexShader_Basic = R"END(
#version 150 core
uniform mat4 uModelViewProjectionMatrix;
in vec3 aVertexPos;
in vec4 aVertexRGBA;
out vec4 vRGBA;
void main()
{
  gl_Position = uModelViewProjectionMatrix * vec4(aVertexPos, 1.0);
  vRGBA = aVertexRGBA;
}
)END";

static const char* kFragmentShader_Basic = R"END(
#version 150 core
in vec4 vRGBA;
out vec4 frag_color;
void main()
{
  frag_color = vRGBA;
}
)END";
}

struct PointCloudVizApp {
  GLFWwindow* window;
  opengl::GlShaderProgram basic_shader;
  opengl::GlVertexBuffer<opengl::GlVertexDataXYZRGBA> lines_vb;
  opengl::GlVertexBuffer<opengl::GlVertexDataXYZRGBA> points_vb;

  double camera_radius = 4.0;
  double camera_theta = M_PI;
  double camera_phi = M_PI / 2.0;
  Eigen::Vector3f camera_orbit_center = Eigen::Vector3f(0, 0, 0);
  double offset_cam_right = 0;
  double offset_cam_up = 0;
  Eigen::Matrix4f model_view_projection_matrix;

  bool left_mouse_down = false;
  bool right_mouse_down = false;
  double prev_mouse_x, prev_mouse_y, curr_mouse_x, curr_mouse_y;

  bool key_w_down = false;
  bool key_a_down = false;
  bool key_s_down = false;
  bool key_d_down = false;
  bool key_space_down = false;

  std::vector<std::string> anim_filenames;
  int anim_counter = 0;

  std::vector<Eigen::Vector3f> point_cloud;
  std::vector<Eigen::Vector3f> point_cloud_colors;
  std::vector<Eigen::Isometry3d> world_from_cam_poses; // camera poses to visualize

  void loadCameraJsonFile(const std::string& cam_json_path) {
    XPLINFO << "Loading camera json: " << cam_json_path;
    using json = nlohmann::json;
    std::ifstream f(cam_json_path);
    json json_data = json::parse(f);
    world_from_cam_poses.clear();
    for (auto v : json_data["frames_data"]) {
      std::vector<double> world_from_cam_data;
      for (const double x : v["world_from_cam"]) {
        world_from_cam_data.push_back(x);
      }
      const Eigen::Matrix4d world_from_cam = Eigen::Map<const Eigen::Matrix<double, 4, 4>>(world_from_cam_data.data());
      world_from_cam_poses.emplace_back(world_from_cam);
    }
  }

  void init() {
    XCHECK(glfwInit());

    // specify the version of opengl to use
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    glfwWindowHint(GLFW_VISIBLE, true);
    glfwWindowHint(GLFW_RESIZABLE, true);
    glfwWindowHint(GLFW_SAMPLES, 4); // anti-alias

    window = glfwCreateWindow(640, 480, "Point Cloud Viz", nullptr, nullptr);
    XCHECK(window);

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);  // Enable vsync

    // IMPORTANT- without this line, it will crash on Windows.
    XCHECK_EQ(gl3wInit(), 0) << "Failed to initialize gl3w.";

    GL_CHECK_ERROR;

    basic_shader.compile(kVertexShader_Basic, kFragmentShader_Basic);

    GL_CHECK_ERROR;

    lines_vb.init();
    lines_vb.bind();
    opengl::GlVertexDataXYZRGBA::setupVertexAttributes(basic_shader, "aVertexPos", "aVertexRGBA");

    points_vb.init();
    points_vb.bind();
    // We have to call this for each vertex buffer that uses it
    opengl::GlVertexDataXYZRGBA::setupVertexAttributes(basic_shader, "aVertexPos", "aVertexRGBA");

    GL_CHECK_ERROR;

    if (!FLAGS_anim_dir.empty()) {
      anim_filenames = file::getFilesInDir(FLAGS_anim_dir);
      p11::point_cloud::loadPointCloudBinary(
          FLAGS_anim_dir + "/" + anim_filenames[anim_counter],
          point_cloud,
          point_cloud_colors,
          FLAGS_subsample);
    } else {
      if (file::filenameExtension(FLAGS_point_cloud) == "csv") {
        point_cloud::loadPointCloudCsv(
            FLAGS_point_cloud, point_cloud, point_cloud_colors, FLAGS_subsample);
      } else if (!FLAGS_point_cloud.empty()) {
        point_cloud::loadPointCloudBinary(
            FLAGS_point_cloud, point_cloud, point_cloud_colors, FLAGS_subsample);
      }
    }

    if (!FLAGS_cam_json.empty()) {
      loadCameraJsonFile(FLAGS_cam_json);
    }
  }

  void updateApp()
  {
    prev_mouse_x = curr_mouse_x;
    prev_mouse_y = curr_mouse_y;
    glfwGetCursorPos(window, &curr_mouse_x, &curr_mouse_y);
    const float dx = prev_mouse_x - curr_mouse_x;
    const float dy = prev_mouse_y - curr_mouse_y;

    if (left_mouse_down) {
      camera_theta -= dx * 0.005;
      camera_phi += dy * 0.005;
      camera_phi = math::clamp<float>(camera_phi, 1e-5, M_PI - 1e-5);
    }
    if (right_mouse_down) {
      offset_cam_right += dx * 0.01;
      offset_cam_up += dy * 0.01;
    }

    if (!anim_filenames.empty()) {
      anim_filenames = file::getFilesInDir(FLAGS_anim_dir);

      static int slower = 0;
      slower = (slower + 1) % 10;
      if (slower == 9) anim_counter = (anim_counter + 1) % anim_filenames.size();
      point_cloud.clear();
      point_cloud_colors.clear();

      const std::string frame_filename =
          FLAGS_anim_dir + "/" + anim_filenames[anim_counter];
      XPLINFO << frame_filename;
      p11::point_cloud::loadPointCloudBinary(
          frame_filename, point_cloud, point_cloud_colors, FLAGS_subsample);
    }
  }

  void drawGlStuff()
  {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);

    glClearColor(0.2, 0.2, 0.2, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_CULL_FACE);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    const float aspect_ratio = float(width) / float(height);
    const Eigen::Matrix4f projection_matrix = p11::opengl::perspectiveProjectionMatrix(
        FLAGS_vfov_deg, aspect_ratio, FLAGS_znear, FLAGS_zfar);

    Eigen::Vector3f cam_pos = Eigen::Vector3f(
                                sin(camera_phi) * cos(camera_theta),
                                cos(camera_phi),
                                sin(camera_phi) * sin(camera_theta)) *
                            camera_radius;
  Eigen::Vector3f look_at(0, 0, 0);
    const Eigen::Vector3f up(0, 1, 0);
    const Eigen::Vector3f forward = (look_at - cam_pos).normalized();
    const Eigen::Vector3f right = forward.cross(up).normalized();
    const Eigen::Vector3f new_up = forward.cross(-right).normalized();

    const Eigen::Vector3f offset =
        offset_cam_right * right - offset_cam_up * new_up;
    cam_pos += offset + camera_orbit_center;
    look_at += offset + camera_orbit_center;

    const Eigen::Matrix4f model_view_matrix = p11::opengl::lookAtMatrix(cam_pos, look_at, up);

    model_view_projection_matrix = projection_matrix * model_view_matrix;

    basic_shader.bind();
    glUniformMatrix4fv(
        basic_shader.getUniform("uModelViewProjectionMatrix"),
        1,
        false,
        model_view_projection_matrix.data());

    lines_vb.bind();
    lines_vb.vertex_data.clear();
    // Draw coordinate axes
    lines_vb.vertex_data.emplace_back(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5);
    lines_vb.vertex_data.emplace_back(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5);
    lines_vb.vertex_data.emplace_back(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5);
    lines_vb.vertex_data.emplace_back(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5);
    lines_vb.vertex_data.emplace_back(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5);
    lines_vb.vertex_data.emplace_back(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5);

    const Eigen::Vector3d zero(0, 0, 0);
    const Eigen::Vector3d dx(0.35, 0, 0);
    const Eigen::Vector3d dy(0, 0.35, 0);
    const Eigen::Vector3d dz(0, 0, 0.35);

    for (const Eigen::Isometry3d& w_from_c : world_from_cam_poses) {
      const Eigen::Vector3d a = w_from_c * zero;
      const Eigen::Vector3d b = w_from_c * dx;
      const Eigen::Vector3d c = w_from_c * dy;
      const Eigen::Vector3d d = w_from_c * dz;

      lines_vb.vertex_data.emplace_back(a.x(), a.y(), a.z(), 1.0, 0.0, 0.0, 0.5);
      lines_vb.vertex_data.emplace_back(b.x(), b.y(), b.z(), 1.0, 0.0, 0.0, 0.5);
      lines_vb.vertex_data.emplace_back(a.x(), a.y(), a.z(), 0.0, 1.0, 0.0, 0.5);
      lines_vb.vertex_data.emplace_back(c.x(), c.y(), c.z(), 0.0, 1.0, 0.0, 0.5);
      lines_vb.vertex_data.emplace_back(a.x(), a.y(), a.z(), 0.0, 0.0, 1.0, 0.5);
      lines_vb.vertex_data.emplace_back(d.x(), d.y(), d.z(), 0.0, 0.0, 1.0, 0.5);
    }

    // Draw a line connecting all of the camera poses. Gradient in alpha indicates direction of time.
    for (int i = 0; i < std::min(0, int(world_from_cam_poses.size()) - 1); ++i) {
      const float alpha1 = 0.1 + 0.9 * float(i) / world_from_cam_poses.size();
      const float alpha2 = 0.1 + 0.9 * float(i + 1) / world_from_cam_poses.size();
      const Eigen::Vector3d p1 = world_from_cam_poses[i] * zero;
      const Eigen::Vector3d p2 = world_from_cam_poses[i + 1] * zero;
      lines_vb.vertex_data.emplace_back(p1.x(), p1.y(), p1.z(), 1.0, 1.0, 1.0, alpha1);
      lines_vb.vertex_data.emplace_back(p2.x(), p2.y(), p2.z(), 1.0, 1.0, 1.0, alpha2);
    }

    lines_vb.copyVertexDataToGPU(GL_DYNAMIC_DRAW);

    glDrawArrays(GL_LINES, 0, lines_vb.vertex_data.size());

    // Draw the point cloud
    points_vb.vertex_data.clear();
    points_vb.bind();
    for (int i = 0; i < point_cloud.size(); ++i) {
      points_vb.vertex_data.emplace_back(
        point_cloud[i].x(), point_cloud[i].y(), point_cloud[i].z(),
        point_cloud_colors[i].x(), point_cloud_colors[i].y(), point_cloud_colors[i].z(), 1.0);
    }

    points_vb.copyVertexDataToGPU(GL_DYNAMIC_DRAW);
    glPointSize(FLAGS_point_size);
    glDrawArrays(GL_POINTS, 0, points_vb.vertex_data.size());

    glfwSwapBuffers(window);

    GL_CHECK_ERROR;
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
      updateApp();
      drawGlStuff();
      glfwPollEvents();
    }
    glfwTerminate();
  }
};

}}  // namespace p11::point_cloud_viz

namespace {

p11::point_cloud_viz::PointCloudVizApp app;

static void error_callback(int error, const char* description) {}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  // clang-format off
  if (key == GLFW_KEY_W && action == GLFW_PRESS)   { app.key_w_down = true; }
  if (key == GLFW_KEY_W && action == GLFW_RELEASE) { app.key_w_down = false; }
  if (key == GLFW_KEY_A && action == GLFW_PRESS)   { app.key_a_down = true; }
  if (key == GLFW_KEY_A && action == GLFW_RELEASE) { app.key_a_down = false; }
  if (key == GLFW_KEY_S && action == GLFW_PRESS)   { app.key_s_down = true; }
  if (key == GLFW_KEY_S && action == GLFW_RELEASE) { app.key_s_down = false; }
  if (key == GLFW_KEY_D && action == GLFW_PRESS)   { app.key_d_down = true; }
  if (key == GLFW_KEY_D && action == GLFW_RELEASE) { app.key_d_down = false; }
  if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)   { app.key_space_down = true; }
  if (key == GLFW_KEY_SPACE && action == GLFW_RELEASE) { app.key_space_down = false; }

  if (key == GLFW_KEY_1 && action == GLFW_RELEASE) { app.loadCameraJsonFile(FLAGS_cam_json); }
  if (key == GLFW_KEY_2 && action == GLFW_RELEASE) { app.loadCameraJsonFile(FLAGS_cam_json2); }

  // clang-format on
}

static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
  if (yoffset > 0) {
    app.camera_radius *= 0.9;
  }
  if (yoffset < 0) {
    app.camera_radius *= 1.1;
  }
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
    app.left_mouse_down = true;
  }
  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
    app.left_mouse_down = false;
  }
  if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
    app.right_mouse_down = true;
  }
  if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) {
    app.right_mouse_down = false;
  }
}

} // namespace

int main(int argc, char** argv)
{
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  app.init();
  glfwSetKeyCallback(app.window, key_callback);
  glfwSetScrollCallback(app.window, scroll_callback);
  glfwSetMouseButtonCallback(app.window, mouse_button_callback);

  app.mainLoop();

  return EXIT_SUCCESS;
}
