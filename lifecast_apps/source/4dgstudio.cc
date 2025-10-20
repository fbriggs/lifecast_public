// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
on Mac with ARM, to properly use Metal:

PYTORCH_ENABLE_MPS_FALLBACK=1 bazel run -- //source:4dgstudio

bazel run --local_cpu_resources=10 --jobs=1 --cuda_archs=compute_86 -- //source:4dgstudio
*/

// Make the application run without a terminal in Windows.
#if defined(windows_hide_console) && defined(_WIN32)
#pragma comment(linker, "/SUBSYSTEM:WINDOWS /ENTRY:WinMainCRTStartup")
#endif

#ifdef _WIN32
  // slim down windows.h and stop it pulling in winsock.h (fixes conflict in httplib.h)
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #define _WIN32_WINNT 0x0A00 // Targeting Windows 10

  // pull in WinSock2 first
  #include <winsock2.h>
  #include <ws2tcpip.h>

  // now include the rest of the Win32 API
  #include <windows.h>
  #include <io.h>
  #include <fcntl.h>
#endif


#include "logger.h"
#include "dear_imgui_app.h"
#include "third_party/dear_imgui/imgui_internal.h" // For PushItemFlag on Windows
#include "imgui_filedialog.h"
#include "util_runfile.h"
#include "util_file.h"
#include "util_math.h"
#include "util_command.h"
#include "util_browser.h"
#include "util_torch.h"
#include "torch_opencv.h"
#include "torch_opengl.h"
#include "video_transcode_lib.h"
#include "preferences.h"
#include "4dgstudio_timeline.h"
#include "4dgstudio_web_template.h"
#include "incremental_sfm_lib.h"
#include "lifecast_splat_lib.h"
#include "lifecast_splat_io.h"
#include "lifecast_splat_math.h"
#include "lifecast_splat_population.h"
#include <regex>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <filesystem>
#include <locale>

#ifdef __linux__
#include <GL/gl.h>
#endif

#include "third_party/httplib.h"

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

static const char* kVertexShader_TexturedQuad = R"END(
#version 150 core
in vec3 aVertexPos;
in vec2 aTexCoord;
out vec2 vTexCoord;
void main()
{
  gl_Position = vec4(aVertexPos, 1.0);
  vTexCoord = aTexCoord;
}
)END";

static const char* kFragmentShader_TexturedQuad = R"END(
#version 150 core
uniform sampler2D uTexture;
in vec2 vTexCoord;
out vec4 frag_color;
void main()
{
  frag_color = texture(uTexture, vTexCoord);
}
)END";

}  // namespace

namespace p11 { namespace studio4dgs {

// This is compared to a number stored at https://lifecastvr.com/4dg_studio_version.txt to check for updates.
constexpr float kSoftwareVersion = 0.1;

enum OptionsPanelState {
  OPTIONS_PANEL_PROJECT,
  OPTIONS_PANEL_3D_CAMERA_TRACKING,
  OPTIONS_PANEL_GAUSSIAN,
  OPTIONS_PANEL_WORLD_TRANSFORM,
  OPTIONS_PANEL_CROP,
  OPTIONS_PANEL_WEB_PLAYER,
  OPTIONS_PANEL_VIRTUAL_CAMERA,
};

enum SfmCameraType {
  CAMERA_TYPE_RECTILINEAR,
  CAMERA_TYPE_FISHEYE,
};

std::string sanitizeForTFD(const std::string& s) {
  std::string result = s;
  for (char& c : result) if (c == '\'' || c == '\"') c = ' ';
  return result;
}

struct ImguiInputMultiFileSelect {
  static constexpr int kBufferSize = 1024;
 private:
  char path[kBufferSize] = "";
 public:

  std::vector<std::string> paths;
  std::string label;
  std::string hash1, hash2;  // used to prevent button collisions in Imgui
  bool editable = true;
  bool required = true;
  std::function<void()> on_change_callback;
  std::unordered_set<std::string> valid_extensions;
  std::vector<const char*> valid_extension_cstrs;

  ImguiInputMultiFileSelect(const std::string& label = "") : label(label)
  {
    hash1 = std::to_string(rand());
    hash2 = std::to_string(rand());
    // TODO: there is a small chance of hash collisions that would break the GUI

    for (const auto& ext : video::image_extensions) {
      valid_extensions.insert("*." + ext);
    }
    for (const auto& ext : video::video_extensions) {
      valid_extensions.insert("*." + ext);
    }
    for (const auto& ext : valid_extensions) {
      valid_extension_cstrs.push_back(ext.c_str());
    }
  }

  void setPath(const char* new_path) {
    string::copyBuffer(path, new_path, kBufferSize);
    if (path[0] != 0) paths = {std::string(new_path)};
  }

  void setPaths(std::vector<std::string> new_paths) {
    paths = std::move(new_paths);
    if (paths.size() == 0) {
      string::copyBuffer(path, "", kBufferSize);
    } else {
      string::copyBuffer(path, paths[0].c_str(), kBufferSize);
    }
  }

  void drawAndUpdate()
  {
    if (paths.size() <= 1) {
      if (!label.empty()) ImGui::Text("%s", label.c_str());

      if (ImGui::Button(("...##" + hash2).c_str())) {
        const char* result = tinyfd_openFileDialog(
          label.c_str(),
          nullptr,
          0,
          valid_extension_cstrs.data(),
          nullptr,
          1 /*multi select*/);

        if (result != nullptr) {
          paths = string::split(result, '|');
          if (paths.size() > 0) {
            string::copyBuffer(path, paths[0].c_str(), kBufferSize);
          }
          if (on_change_callback) on_change_callback();
        }
      }
      ImGui::SameLine();
      ImGui::Dummy(ImVec2(4.0f, 0.0f));
      ImGui::SameLine();
      if (editable) {
        if(ImGui::InputText(("##" + hash1).c_str(), path, IM_ARRAYSIZE(path))) {
          if (path == std::string()) {
            paths.clear();
          } else {
            paths = {std::string(path)};
          }
        }
      } else {
        ImGui::Text("%s", path);
      }

      if (required && (paths.empty() || paths[0].empty())) {
        ImGui::SameLine();
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));
        ImGui::Text("!");
        ImGui::PopStyleColor();
      }
    } else {
      if (ImGui::SmallButton("Clear")) {
        paths.clear();
        string::copyBuffer(path, "", kBufferSize);
      }
      ImGui::BeginChild("InputVideosScrolling", ImVec2(ImGui::CalcItemWidth() + 30, 100), true);
      for (auto& v : paths) {
        ImGui::TextUnformatted(v.c_str());
      }
      ImGui::EndChild();
    }
  }
};

struct GStudio4DApp : public DearImGuiApp {
  // Application preferences
  std::map<std::string, std::string> prefs;

  // Commands, workflow config
  CommandRunner command_runner;

  torch::DeviceType device;

  // Vertex buffer, shaders, 3D data
  opengl::GlShaderProgram basic_shader;
  opengl::GlVertexBuffer<opengl::GlVertexDataXYZRGBA> lines_vb, lines_vb2; // lines_vb2 gets world_transform, 1 does not
  opengl::GlVertexBuffer<opengl::GlVertexDataXYZRGBA> points_vb; // For GUI stuff mostly
  opengl::GlVertexBuffer<opengl::GlVertexDataXYZRGBA> pointcloud_vb;

  // For full-screen quad rendering
  opengl::GlShaderProgram textured_quad_shader;
  opengl::GlVertexBuffer<opengl::GlVertexDataXYZUV> quad_vb;

  // Visualization data
  calibration::incremental_sfm::IncrementalSfmGuiData sfm_gui_data;
  splat::GaussianSplatGuiData gaussian_splat_gui_data;
  torch_opengl::CudaGLTexture splat_texture;
  std::shared_ptr<splat::SplatModel> model_from_timeline = nullptr;
  std::shared_ptr<splat::SplatModel> which_model_to_draw = nullptr;

  // Camera
  double camera_radius, camera_theta, camera_phi;
  Eigen::Vector3f camera_orbit_center;
  double offset_cam_right, offset_cam_up;
  Eigen::Matrix4f model_view_projection_matrix;
  Eigen::Matrix4f view_matrix;
  Eigen::Matrix4f projection_matrix;
  Eigen::Matrix4d world_transform = Eigen::Matrix4d::Identity(); // TODO: GUI for world transform?
  calibration::RectilinearCamerad gl_camera_as_rectilinear; // a rectilinear camera that should match up with the opengl camera
  Eigen::Matrix4d splat_world_transform = Eigen::Matrix4d::Identity();

  float world_transform_scale;
  float world_transform_rx;
  float world_transform_ry;
  float world_transform_rz;
  float world_transform_tx;
  float world_transform_ty;
  float world_transform_tz;

  // 3D crop parameters
  bool apply_crop3d = false;
  float crop3d_radius = 1.0;

  // Mouse
  bool left_mouse_down = false;
  bool right_mouse_down = false;
  double prev_mouse_x = 0, prev_mouse_y = 0, curr_mouse_x = 0, curr_mouse_y = 0;
  bool mouse_in_3d_viewport = false;
  bool mouse_in_timeline = false;
  bool click_started_in_3d_viewport = false;
  bool main_menu_is_hovered = false;

  // 3D / GUI
  VideoTimelineWidget timeline;
  ImVec2 preview3d_viewport_min, preview3d_viewport_size;
  bool show_origin_in_3d_view = true;
  bool show_grid_in_3d_view = false;
  bool show_sfm_cameras_in_3d_view = true;
  bool show_sfm_pointcloud_in_3d_view = true;
  bool show_virtual_cameras_in_3d_view = true;
  bool show_gaussian_splats = true;
  bool show_crop3d_wireframe = false;
  bool animate_camera_rotation = false;
  
  OptionsPanelState options_panel_state = OPTIONS_PANEL_PROJECT;

  // Project settings
  ImguiInputMultiFileSelect input_files_select;
  gui::ImguiFolderSelect project_dir_select = gui::ImguiFolderSelect("Project Directory:");
  int camera_tracking_frame_stride = 10;
  std::map<int, std::string> frame_num_to_splat_img; // updated by scanProjectDirForSplatVideoFrames
  bool project_is_static = true; // static = not moving, dynamic = 4d video
  int project_camera_type = CAMERA_TYPE_RECTILINEAR;

  int sfm_cfg_resize_max_dim = 768;
  float sfm_cfg_guess_fov = 70.0;
  bool sfm_cfg_filter_with_flow = false;
  bool sfm_cfg_share_intrinsics = false;
  bool sfm_cfg_reorder_cameras = false;
  bool sfm_cfg_use_intrinsic_prior = false;
  float sfm_cfg_inlier_frac = 0.8;
  float sfm_cfg_depth_weight = 0.00001;
  int sfm_cfg_max_solver_itrs = 100;

  // Gaussain training settings (GUI state)
  int splat_cfg_max_num_splats = 262144;
  int splat_cfg_num_itrs = 2000;
  int splat_cfg_first_frame_warmup_itrs = 2000;
  int splat_cfg_popi = 100;
  int splat_cfg_resize_max_dim = 1024;
  float splat_cfg_learning_rate = 1e-2;
  int splat_cfg_images_per_batch = 4;
  bool splat_cfg_init_with_monodepth = false;
  bool splat_cfg_use_depth_loss = true;
  int splat_cfg_encode_w = 4096;
  int splat_cfg_encode_h = 2048;
  int splat_encode_select = 0; // 0 = h264, 1 = h265

  bool web_template_cfg_support_webxr = false;

  int render2d_width = 1920;
  int render2d_height = 1080;
  float render2d_vfov = 60;
  int render2d_num_frames = 120; // only used for static scenes
  int render2d_frame_rate = 30; // only used for static scenes

  void initGStudio4DApp()
  {
    device = util_torch::findBestTorchDevice();

    initGlBuffersAndShaders();
    resetCamera();
    timeline.curr_frame_change_callback = [&] {
      XPLINFO << "timeline.curr_frame=" << timeline.curr_frame;
      loadSplatFrameFromFileBasedOnTimeline();
    };

    prefs = preferences::getPrefs();

    setWorldTransformDefaults();
    setPresetGaussianOptions("medium");
  }

  void initGlBuffersAndShaders()
  {
    basic_shader.compile(kVertexShader_Basic, kFragmentShader_Basic);
    basic_shader.bind();

    textured_quad_shader.compile(kVertexShader_TexturedQuad, kFragmentShader_TexturedQuad);
    textured_quad_shader.bind();

    lines_vb.init();
    lines_vb.bind();
    opengl::GlVertexDataXYZRGBA::setupVertexAttributes(basic_shader, "aVertexPos", "aVertexRGBA");

    lines_vb2.init();
    lines_vb2.bind();
    opengl::GlVertexDataXYZRGBA::setupVertexAttributes(basic_shader, "aVertexPos", "aVertexRGBA");

    points_vb.init();
    points_vb.bind();
    opengl::GlVertexDataXYZRGBA::setupVertexAttributes(basic_shader, "aVertexPos", "aVertexRGBA");

    pointcloud_vb.init();
    pointcloud_vb.bind();
    opengl::GlVertexDataXYZRGBA::setupVertexAttributes(basic_shader, "aVertexPos", "aVertexRGBA");    

    quad_vb.init();
    quad_vb.bind();
    opengl::GlVertexDataXYZUV::setupVertexAttributes(textured_quad_shader, "aVertexPos", "aTexCoord");
    quad_vb.vertex_data.clear();
    quad_vb.vertex_data.emplace_back(-1.0f, -1.0f, 0.0f, 0.0f, 1.0f); // Bottom-left
    quad_vb.vertex_data.emplace_back( 1.0f, -1.0f, 0.0f, 1.0f, 1.0f); // Bottom-right
    quad_vb.vertex_data.emplace_back( 1.0f,  1.0f, 0.0f, 1.0f, 0.0f); // Top-right
    quad_vb.vertex_data.emplace_back(-1.0f, -1.0f, 0.0f, 0.0f, 1.0f); // Bottom-left
    quad_vb.vertex_data.emplace_back( 1.0f,  1.0f, 0.0f, 1.0f, 0.0f); // Top-right
    quad_vb.vertex_data.emplace_back(-1.0f,  1.0f, 0.0f, 0.0f, 0.0f); // Top-left
    quad_vb.copyVertexDataToGPU(GL_STATIC_DRAW);

    GL_CHECK_ERROR;
  }

  void updatePointCloudVertexBuffer() {
    sfm_gui_data.pointcloud_needs_update = false;
    sfm_gui_data.mutex.lock();

    pointcloud_vb.vertex_data.clear();
    for (int i = 0; i < sfm_gui_data.point_cloud.size(); ++i) {
      pointcloud_vb.vertex_data.emplace_back(
        sfm_gui_data.point_cloud[i].x(), 
        sfm_gui_data.point_cloud[i].y(), 
        sfm_gui_data.point_cloud[i].z(),
        sfm_gui_data.point_cloud_colors[i].x(),
        sfm_gui_data.point_cloud_colors[i].y(),
        sfm_gui_data.point_cloud_colors[i].z(),
        sfm_gui_data.point_cloud_colors[i].w());
    }
    pointcloud_vb.bind();  
    pointcloud_vb.copyVertexDataToGPU(GL_STATIC_DRAW);
    sfm_gui_data.mutex.unlock();
  }

  void handleKeyboard()
  {
    ImGuiIO& io = ImGui::GetIO();
    int delta = io.KeysDown[GLFW_KEY_LEFT_SHIFT] ? 10 : 1;
    if (io.KeysDown[GLFW_KEY_LEFT]) moveTimelineByDelta(-delta);
    if (io.KeysDown[GLFW_KEY_RIGHT]) moveTimelineByDelta(delta);
    // TODO: camea controls
  }

  void moveTimelineByDelta(int delta)
  {
    timeline.curr_frame += delta;
    if (timeline.curr_frame < 0) timeline.curr_frame = 0;
    if (timeline.curr_frame >= timeline.num_frames) {
      timeline.curr_frame = timeline.num_frames - 1;
    }
    timeline.curr_frame_change_callback();
  }

  void handleMouseDown(int button)
  {
    if (button == GLFW_MOUSE_BUTTON_LEFT) left_mouse_down = true;
    if (button == GLFW_MOUSE_BUTTON_RIGHT) right_mouse_down = true;

    click_started_in_3d_viewport = false;

    if (mouse_in_3d_viewport) click_started_in_3d_viewport = true;
  }

  void handleMouseUp(int button)
  {
    if (button == GLFW_MOUSE_BUTTON_LEFT) left_mouse_down = false;
    if (button == GLFW_MOUSE_BUTTON_RIGHT) right_mouse_down = false;
  }

  void updateMouse()
  {
    prev_mouse_x = curr_mouse_x;
    prev_mouse_y = curr_mouse_y;
    glfwGetCursorPos(window, &curr_mouse_x, &curr_mouse_y);

    mouse_in_3d_viewport = false;
    if (curr_mouse_x >= preview3d_viewport_min.x && 
        curr_mouse_y >= preview3d_viewport_min.y + ImGui::GetFrameHeight() && // hack: subtract menu bar height to make it work out right for some reason
        curr_mouse_x <= preview3d_viewport_min.x + preview3d_viewport_size.x &&
        curr_mouse_y <= preview3d_viewport_min.y + preview3d_viewport_size.y + ImGui::GetFrameHeight()) {
      mouse_in_3d_viewport = true;
    }

    mouse_in_timeline = false;
    if (curr_mouse_x >= timeline.viewport_min.x && curr_mouse_y >= timeline.viewport_min.y &&
        curr_mouse_x <= timeline.viewport_min.x + timeline.viewport_size.x &&
        curr_mouse_y <= timeline.viewport_min.y + timeline.viewport_size.y) {
      mouse_in_timeline = true;
    }

    if (main_menu_is_hovered) {
      mouse_in_3d_viewport = false;
      mouse_in_timeline = false;
    }
  }

  void restoreDefaultCursor()
  {
    glfwSetCursor(window, glfwCreateStandardCursor(GLFW_ARROW_CURSOR));
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
  }

  void updateCursor()
  {
    //restoreDefaultCursor();

    //ImGui::GetIO().ConfigFlags &= ~ImGuiConfigFlags_NoMouseCursorChange;

    if (main_menu_is_hovered) return;
  }

  void resetCamera()
  {
    camera_radius = 5.0;
    //camera_theta = 2.9298;
    //camera_phi = 1.13164;//M_PI / 2.0;
    camera_theta = M_PI;
    camera_phi = M_PI / 2.0;

    camera_orbit_center = Eigen::Vector3f(0, 0, 0);
    offset_cam_right = 0;
    offset_cam_up = 0;
  }

  void setWorldTransformDefaults() {
    world_transform_scale = 1;
    world_transform_rx = 0;
    world_transform_ry = 0;
    world_transform_rz = 0;
    world_transform_tx = 0;
    world_transform_ty = 0;
    world_transform_tz = 0;
  }

  Eigen::Matrix4d getWorldTransformFromGuiSettings() {
    std::vector<double> rvec = {
      world_transform_rx * M_PI / 180,
      world_transform_ry * M_PI / 180,
      world_transform_rz * M_PI / 180};
    Eigen::Matrix3d rotation;
    ceres::AngleAxisToRotationMatrix(rvec.data(), rotation.data());
    Eigen::Matrix4d world_transform = Eigen::Matrix4d::Zero();
    world_transform(0,0) = world_transform_scale;
    world_transform(1,1) = world_transform_scale;
    world_transform(2,2) = world_transform_scale;
    world_transform.block<3,3>(0,0) *= rotation;

    Eigen::Vector3d translation(-world_transform_tx, -world_transform_ty, -world_transform_tz);

    translation = -rotation * translation * world_transform_scale;

    world_transform(0,3) = translation.x();
    world_transform(1,3) = translation.y();
    world_transform(2,3) = translation.z();
    world_transform(3,3) = 1.0;

    return world_transform;
  }

  void updateCamera()
  {
    // Update orbit camera state from mouse
    if (click_started_in_3d_viewport) {
      const float dx = prev_mouse_x - curr_mouse_x;
      const float dy = prev_mouse_y - curr_mouse_y;
      if (left_mouse_down) {
        camera_theta -= dx * 0.005;
        camera_phi += dy * 0.005;
        camera_phi = math::clamp<float>(camera_phi, 1e-5, M_PI - 1e-5);
      }
      if (right_mouse_down) {
        offset_cam_right += dx * 0.001;
        offset_cam_up += dy * 0.001;
      }
    }

    if (animate_camera_rotation) {
      camera_theta += 0.003;
    }

    // Setup projection matrix
    static constexpr float kVerticalFovDeg = 60;
    static constexpr float kZNear = 0.01;
    static constexpr float kZFar = 1000.0;
    const float aspect_ratio = float(preview3d_viewport_size.x) / float(preview3d_viewport_size.y);
    projection_matrix =
        p11::opengl::perspectiveProjectionMatrix(kVerticalFovDeg, aspect_ratio, kZNear, kZFar);

    // Setup model-view matrix for orbit cam
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
    Eigen::Vector3f offset = offset_cam_right * right - offset_cam_up * new_up;

    cam_pos += offset + camera_orbit_center;
    look_at += offset + camera_orbit_center;

    view_matrix = p11::opengl::lookAtMatrix(cam_pos, look_at, up);

    world_transform = getWorldTransformFromGuiSettings();
    
    // Compute the model-view-projection matrix that will be used in shaders.
    // By default, the model matrix is identity and can be omitted.
    model_view_projection_matrix = projection_matrix * view_matrix;
 
    // Setup transform and camera model for splat rendering to match opengl
    Eigen::Matrix3d world_R_cam;
    world_R_cam.col(0) = right.cast<double>();
    world_R_cam.col(1) = new_up.cast<double>();
    world_R_cam.col(2) = forward.cast<double>();
    gl_camera_as_rectilinear.cam_from_world.linear() = world_R_cam.transpose();
    gl_camera_as_rectilinear.setPositionInWorld(cam_pos.cast<double>());
    gl_camera_as_rectilinear.width = preview3d_viewport_size.x;
    gl_camera_as_rectilinear.height = preview3d_viewport_size.y;
    gl_camera_as_rectilinear.optical_center = Eigen::Vector2d(gl_camera_as_rectilinear.width/2.0, gl_camera_as_rectilinear.height/2.0);
    double vertical_focal = gl_camera_as_rectilinear.height / (2.0 * tan(kVerticalFovDeg * M_PI / 180.0 / 2.0));
    gl_camera_as_rectilinear.focal_length = Eigen::Vector2d(
      vertical_focal, // Use the same focal length for both dimensions
      vertical_focal);
    
    constexpr int kMaxRenderSize = 2048; // TODO: config, option for no limit?
    if (gl_camera_as_rectilinear.width > kMaxRenderSize || gl_camera_as_rectilinear.height > kMaxRenderSize) {
      gl_camera_as_rectilinear.resizeToMaxDim(kMaxRenderSize);
    }
    
    // Flip coordinate systems to align splat and gl render
    splat_world_transform = Eigen::Matrix4d::Identity();
    Eigen::AngleAxisd rotation(-M_PI/2, Eigen::Vector3d::UnitY());
    Eigen::Matrix3d rotation_matrix = rotation.toRotationMatrix();
    Eigen::Matrix3d reflection = Eigen::Matrix3d::Identity();
    reflection(2, 2) = -1.0; // Flip the Z coordinate
    Eigen::Matrix3d combined = rotation_matrix * reflection; // Rotate first, then reflect
    splat_world_transform.block<3,3>(0,0) = combined;
    splat_world_transform *= world_transform;

    // Update camera state from keyboard. We do this last because we need forward/right/up as
    // calculated above
    ImGuiIO& io = ImGui::GetIO();
    if (mouse_in_3d_viewport) {
      static constexpr float kCamSpeed = 0.05;
      if (io.KeysDown[GLFW_KEY_W]) {
        camera_orbit_center += forward * kCamSpeed;
      }
      if (io.KeysDown[GLFW_KEY_S]) {
        camera_orbit_center -= forward * kCamSpeed;
      }
      if (io.KeysDown[GLFW_KEY_A]) {
        camera_orbit_center -= right * kCamSpeed;
      }
      if (io.KeysDown[GLFW_KEY_D]) {
        camera_orbit_center += right * kCamSpeed;
      }
    }
  }

  void beginModalRoundedBox(const char* child_id) {
    ImVec2 box_size(getScaledFontSize() * 44, getScaledFontSize() * 20);
    ImVec2 avail = ImGui::GetContentRegionAvail();
    // Center horizontally and vertically
    ImGui::SetNextWindowPos(ImVec2(
      ImGui::GetCursorPosX() + (avail.x - box_size.x) * 0.5,
      ImGui::GetCursorPosY() + (avail.y - box_size.y) * 0.5));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 4.0);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(getScaledFontSize() * 2, getScaledFontSize() * 2));
    ImGui::BeginChild(child_id, box_size, true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove);
  }

  void endModalRoundedBox() {
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::PopStyleVar();
  }

  void splashDragDropInstructionScreen() {
    ImVec2 s = ImGui::GetContentRegionAvail();
    ImVec2 p = ImGui::GetCursorScreenPos();
    if (s.y > p.y + 60) { // Only show this part if there is enough space in the window
      float pad = 20;
      const char* t = "Drag and drop video or image files here...";
      ImVec2 ts = ImGui::CalcTextSize(t);
      ImVec2 cp((s.x-ts.x)*0.5f,(s.y-ts.y)*0.5f);
      ImVec2 start(p.x+cp.x-pad, p.y+cp.y-pad);
      ImVec2 end(start.x+ts.x+pad*2, start.y+ts.y+pad*2);
      ImVec2 text_offset(start.x + pad, start.y + pad);
      ImGui::GetWindowDrawList()->AddRect(start, end, IM_COL32(128,128,128,255),10);
      ImGui::SetCursorPos(text_offset);
      ImGui::TextUnformatted(t);
    }
  }

  ~GStudio4DApp() {
    splat_texture.cleanup();
  }

  std::string formatVersionNumber(float v) {
    std::string ver = std::to_string(v);
    ver.erase(ver.find_last_not_of('0') + 1, std::string::npos);
    ver.erase(ver.find_last_not_of('.') + 1, std::string::npos);    
    return ver;
  }

  void showAboutDialog() {
    tinyfd_messageBox(
      "UpscaleVideo.ai by Lifecast",
      ("Version: " + formatVersionNumber(kSoftwareVersion) +
      "\nTorch Device: " + util_torch::deviceTypeToString(device)).c_str(), "ok", "info", 1);
  }

  void drawMainMenu()
  {
    if (ImGui::BeginMenuBar()) {
      // Add some more padding around the menu items... otherwise it looks wack.
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(10, 10));
      ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10, 10));
      if (ImGui::BeginMenu(" Settings ")) {
        ImGui::Dummy(ImVec2(0, 8));

        if (ImGui::MenuItem("About 4D Gaussian Studio by Lifecast.ai")) {
          showAboutDialog();
        }

        ImGui::Dummy(ImVec2(0, 8));
        ImGui::EndMenu();
      }

      if (ImGui::BeginMenu(" View ")) {
        ImGui::Dummy(ImVec2(0, 8));

        ImGui::MenuItem("Show Origin", nullptr, &show_origin_in_3d_view);
        ImGui::MenuItem("Show Grid", nullptr, &show_grid_in_3d_view);
        ImGui::MenuItem("Show Estimated Camera Path", nullptr, &show_sfm_cameras_in_3d_view);
        ImGui::MenuItem("Show SFM Point Cloud", nullptr, &show_sfm_pointcloud_in_3d_view);
        ImGui::MenuItem("Show Virtual Camera Path", nullptr, &show_virtual_cameras_in_3d_view);
        ImGui::MenuItem("Show Gaussians", nullptr, &show_gaussian_splats);
        ImGui::MenuItem("Show Crop 3D Bounds", nullptr, &show_crop3d_wireframe);
        ImGui::MenuItem("Auto-rotate camera", nullptr, &animate_camera_rotation);

        if (ImGui::MenuItem("Reset Camera")) {
          resetCamera();
        }
        
        
        ImGui::EndMenu();
      }

      ImGui::PopStyleVar(2);

      // NOTE: I don't know why this is negated.
      main_menu_is_hovered = !ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows);

      ImGui::EndMenuBar();
    }
  }


  void drawGlStuff()
  {
    GL_CHECK_ERROR;

    const float s = ImGui::GetIO().DisplayFramebufferScale.x;  // For Retina display

    constexpr int kMenuBarHeight = 20;
    glViewport(
        s * preview3d_viewport_min.x,
        s * (preview3d_viewport_min.y - kMenuBarHeight + VideoTimelineWidget::kTimelineHeight),
        s * preview3d_viewport_size.x,
        s * preview3d_viewport_size.y);

    // Draw a textured quad for the splat render below the rest of the OpenGL drawing
    if (show_gaussian_splats && which_model_to_draw != nullptr && splat_texture.isInitialized() && splat_texture.getWidth() > 0) {
      glDisable(GL_DEPTH_TEST);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, splat_texture.getTextureId());
      textured_quad_shader.bind();
      glUniform1i(textured_quad_shader.getUniform("uTexture"), 0);
      quad_vb.bind();
      glDrawArrays(GL_TRIANGLES, 0, quad_vb.vertex_data.size());
    }

    //glEnable(GL_DEPTH_TEST);
    //glDepthMask(GL_TRUE);
    //glDepthFunc(GL_LEQUAL);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT,  GL_NICEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    update3DGuiVertexData();

    // Construct a different MVP for the point cloud only (not the rest of the 3D UI),
    // which incorporates the World Transform Editor's scaling and rotation.
    Eigen::Matrix4f transformed_mvp = projection_matrix * view_matrix * world_transform.cast<float>();

    // Draw point cloud
    pointcloud_vb.bind();
    basic_shader.bind();
    glUniformMatrix4fv(
        basic_shader.getUniform("uModelViewProjectionMatrix"),
        1,
        false,
        transformed_mvp.data());
    glPointSize(3);
    if (show_sfm_pointcloud_in_3d_view) {
      glDrawArrays(GL_POINTS, 0, pointcloud_vb.vertex_data.size());
    }
    
    // Draw lines for axes
    lines_vb.bind();
    glUniformMatrix4fv(
        basic_shader.getUniform("uModelViewProjectionMatrix"),
        1,
        false,
        model_view_projection_matrix.data());
    glDrawArrays(GL_LINES, 0, lines_vb.vertex_data.size());

    lines_vb2.bind();
    glUniformMatrix4fv(
        basic_shader.getUniform("uModelViewProjectionMatrix"),
        1,
        false,
        transformed_mvp.data()); // use the version of the transform with user transform
    glDrawArrays(GL_LINES, 0, lines_vb2.vertex_data.size());

    // Draw points that aren't part of the pointcloud (e.g., for GUI purpose)
    points_vb.bind();
    glPointSize(4);
    glDrawArrays(GL_POINTS, 0, points_vb.vertex_data.size());

    GL_CHECK_ERROR;
  }

  void onSelectInputFiles() {
    if (input_files_select.paths.empty()) return;
    options_panel_state = OPTIONS_PANEL_PROJECT;

    // Make a default project directory path
    if (input_files_select.paths.size() == 1) {
      // Create a new folder at the same level for single input video.
      std::string prefix = file::filenamePrefixFromPath(input_files_select.paths[0]);
      project_dir_select.setPath(
        file::getDirectoryName(input_files_select.paths[0]) + "/" + prefix + "_4dgs");

      timeline.num_frames = render2d_num_frames;

      project_is_static = true;
      sfm_cfg_share_intrinsics = true;
    } else {
      // Put the project files one level up from the multiple inputs
      std::string prefix = file::filenamePrefixFromPath(input_files_select.paths[0]);
      project_dir_select.setPath(
        file::getDirectoryName(input_files_select.paths[0]) + "_4dgs");

      timeline.num_frames = 0;
      // If its a folder of images, static 3d, folder of videos, dynamic 4d
      if (video::hasImageExt(input_files_select.paths[0])) {
        project_is_static = true;
      } else {
        project_is_static = false;
      }
      sfm_cfg_share_intrinsics = false;
    }


    std::string project_dir = project_dir_select.getPath();
    xpl::stdoutLogger.attachTextFileLog(project_dir + "/log.txt");

    timeline.keyframes.clear();

    sfm_gui_data.mutex.lock();
    gaussian_splat_gui_data.current_model = nullptr;
    model_from_timeline = nullptr;
    which_model_to_draw = nullptr;
    sfm_gui_data.mutex.unlock();

    timeline.curr_frame = 0;
    scanProjectDirForSplatVideoFrames();
    loadPrecomputedStaticSplatModelIfExists();
    setWorldTransformDefaults();
    setPresetGaussianOptions("medium");
  }

  float getScaledFontSize() {
    ImGuiIO& io = ImGui::GetIO();
    return io.Fonts->Fonts[0]->FontSize * io.FontGlobalScale;
  }

  void drawFrame()
  {
    gl_context_mutex.lock();
    glfwMakeContextCurrent(window);

    if (sfm_gui_data.pointcloud_needs_update) {
      updatePointCloudVertexBuffer();
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_NoTitleBar;
    window_flags |= ImGuiWindowFlags_NoScrollbar;
    window_flags |= ImGuiWindowFlags_NoBackground;
    window_flags |= ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoResize;
    window_flags |= ImGuiWindowFlags_NoCollapse;
    window_flags |= ImGuiWindowFlags_AlwaysAutoResize;
    window_flags |= ImGuiWindowFlags_MenuBar;
    
    // Resize the ImGui window to the GLFW window
    int glfw_window_w, glfw_window_h;
    glfwGetWindowSize(window, &glfw_window_w, &glfw_window_h);
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(glfw_window_w, glfw_window_h), ImGuiCond_Always);

    std::string window_title = "4D Gaussian Studio by Lifecast.ai";
    
    //if (!input_files_select.paths.empty()) {
    //  std::string title = input_files_select.paths[0];
    //  if (!title.empty()) window_title += " - " + title;
    //}

    glfwSetWindowTitle(window, window_title.c_str());
    
    if (ImGui::Begin("Main Window", nullptr, window_flags)) {
      handleKeyboard();
      updateMouse();
      drawMainMenu();

      if (input_files_select.paths.empty()) {
        splashDragDropInstructionScreen();
      } else {
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
        drawMain3DView();
        ImGui::PopStyleVar();
      }

      ImGui::End();
    }

    // Here we would normally call finishDrawingImguiAndGl(), but don't because we are mixing in custom opengl
    beginGlDrawingForImgui();
  
    if (!input_files_select.paths.empty()) {
      drawGlStuff();
    }

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    
    updateCursor();

    glfwSwapBuffers(window);
    glfwMakeContextCurrent(nullptr);
    gl_context_mutex.unlock();
  }
  
  bool Splitter(bool split_vertically, float thickness, float* size1, float* size2, float min_size1, float min_size2, float splitter_long_axis_size = -1.0f) {
    using namespace ImGui;
    ImGuiContext& g = *GImGui;
    ImGuiWindow* window = g.CurrentWindow;
    ImGuiID id = window->GetID("##Splitter");
    ImRect bb;

    ImGui::GetStyle().Colors[ImGuiCol_Separator] = ImVec4(0.25, 0.25, 0.25, 0.0f);  // Normal state (less opaque)
    ImGui::GetStyle().Colors[ImGuiCol_SeparatorHovered] = ImVec4(0.6, 0.6, 0.6, 0.3);  // When hovered
    ImGui::GetStyle().Colors[ImGuiCol_SeparatorActive] = ImVec4(0.7, 0.7, 0.7, 0.5);  // When being dragged

    ImVec2 pos = window->DC.CursorPos;
    ImVec2 available_size = ImGui::GetContentRegionAvail();

    if (splitter_long_axis_size < 0.0f)
      splitter_long_axis_size = split_vertically ? available_size.y : available_size.x;

    bb.Min = pos;
    if (split_vertically) {
      bb.Max.x = pos.x + thickness;
      bb.Max.y = pos.y + splitter_long_axis_size - ImGui::GetStyle().FramePadding.y * 2;
    } else {
      bb.Max.x = pos.x + splitter_long_axis_size;
      bb.Max.y = pos.y + thickness;
    }

    ImGui::ItemSize(bb);
    if (!ImGui::ItemAdd(bb, id)) return false;

    return SplitterBehavior(bb, id, split_vertically ? ImGuiAxis_X : ImGuiAxis_Y,
        size1, size2, min_size1, min_size2, 0.0f);
  }

  // Splitter state
  bool show_options_panel = true;
  float split_width_right = 200;
  float split_width_left = 0;
  static constexpr int kMinSplitPanelSize = 360;
  static constexpr int kSplitterDragSize = 8;

  std::atomic<bool> needs_unhide_options_panel = false; // This can happen from another thread with drag/drop
  void unhideOptionsPanel() {
    show_options_panel = true;
    ImVec2 available = ImGui::GetContentRegionAvail();
    if (split_width_right <= kMinSplitPanelSize) { 
      split_width_right = kMinSplitPanelSize;
    }
    ImGuiStyle& style = ImGui::GetStyle();
    split_width_left = available.x - split_width_right - style.FramePadding.x;
  }

  void render2DVideoWithVirtualCamera() {
    using namespace p11::video;
  
    std::string project_dir = project_dir_select.getPath();
  
    std::vector<calibration::RectilinearCamerad> interp_cams = interpolateCameraPath(
      timeline.keyframes, timeline.num_frames);

    std::shared_ptr<InputVideoStream> original_video = nullptr;
    int read_frame_count = 0;
    MediaFrame read_frame;
    VideoStreamResult result = VideoStreamResult::OK;
    std::string first_video_path = input_files_select.paths[0];

    if (!project_is_static) {
      std::map<std::string, int> time_offsets = calibration::readTimeOffsetJsonAsMap(
        project_dir + "/time_offsets.json");
    
      // Open one of the source videos to get frame rate and audio stream
      original_video = std::make_shared<InputVideoStream>(first_video_path);
      XCHECK(original_video->valid()) << "Invalid input video stream: " << first_video_path;
    
      std::string first_cam_name = file::filenamePrefixFromPath(first_video_path);
      XPLINFO << "first_cam_name: " <<  first_cam_name;
      
      // Skip initial frames (probably needed to keep audio in sync).
      int frames_to_skip = time_offsets.count(first_cam_name) ? time_offsets.at(first_cam_name) : 0;
      XPLINFO << "Skipping " << frames_to_skip << " frames for camera " << first_cam_name;

      while(read_frame_count <= frames_to_skip && (result = original_video->readFrame(read_frame, CV_32FC3)) == VideoStreamResult::OK) {
        XCHECK_NE(int(result), int(VideoStreamResult::ERR)) << "There was an error decoding a frame from: " << first_video_path;
        if (!read_frame.is_video()) continue; // skip non-video frames (i.e. audio)
        ++read_frame_count;

        if (cancel_render2d_requested && *cancel_render2d_requested) return;
      }
    }

    std::string video_encoder = "libx264";
    if (splat_encode_select == 1) video_encoder = "libx265";
    
    video::EncoderConfig encode_cfg;
    encode_cfg.crf = 20; // TODO: config
    auto out_stream = project_is_static ? 
      std::make_shared<OutputVideoStream>(
        project_dir + "/render.mp4", // TODO: add a timestamp
        render2d_width,
        render2d_height,
        std::pair<int,int>(render2d_frame_rate, 1),
        video_encoder,
        encode_cfg) :
      std::make_shared<OutputVideoStream>(
        *original_video, // setup for audio stream copy
        project_dir + "/render.mp4", // TODO: add a timestamp
        render2d_width,
        render2d_height,
        original_video->guessFrameRate(),
        video_encoder,
        encode_cfg);

    std::shared_ptr<splat::SplatModel> render_model = nullptr;
    if (project_is_static) {
      gaussian_splat_gui_data.mutex.lock();
      if (gaussian_splat_gui_data.current_model) {
        if (apply_crop3d) {
          render_model = std::make_shared<splat::SplatModel>();
          render_model->copyFrom(gaussian_splat_gui_data.current_model);
        } else {
          render_model = gaussian_splat_gui_data.current_model;
        }
      }
      gaussian_splat_gui_data.mutex.unlock();
  
      render_model = applyCrop3DIfEnabled(render_model);
    }

    for (int frame = 0; frame < timeline.num_frames; ++frame) { // TODO: what if timeline.num_Frames changes while this is running?
      XPLINFO << "Rendering frame: " << frame << " / " << timeline.num_frames;
      if (cancel_render2d_requested && *cancel_render2d_requested) break;
    
      if (!project_is_static) {
        // Copy audio stream from the original video until we get one video frame from it
        read_frame_count = 0;
        result = VideoStreamResult::OK;
        while(read_frame_count < 1 && (result = original_video->readFrame(read_frame, CV_32FC3)) == VideoStreamResult::OK) {
          XCHECK_NE(int(result), int(VideoStreamResult::ERR)) << "There was an error decoding a frame from: " << first_video_path;
          if (!read_frame.is_video()) {
            out_stream->writeFrame(read_frame);
          } else {
            ++read_frame_count;
          }
        }
      }

      // Render a frame with the virtual camera
      calibration::RectilinearCamerad frame_cam = interp_cams[frame];
      int max_dim = std::max(render2d_width, render2d_height);
      frame_cam.resizeToMaxDim(max_dim);
      frame_cam.width = render2d_width;
      frame_cam.height = render2d_height;
      frame_cam.optical_center = Eigen::Vector2d(render2d_width / 2.0, render2d_height / 2.0);
  
      if (!project_is_static) {
        std::string frame_filename = frame_num_to_splat_img[frame];
        cv::Mat splat_image = cv::imread(frame_filename);
        std::vector<splat::SerializableSplat> splats = splat::decodeSplatImage(splat_image);
        render_model = serializableSplatsToModel(device, splats);

        render_model = applyCrop3DIfEnabled(render_model);
      }

      auto [image_tensor, alpha_map, depth_map, _0, _1, _2, metas] = splat::renderSplatImageGsplat(
        device,
        frame_cam,
        render_model,
        c10::nullopt,
        splat_world_transform);

      MediaFrame output_frame;
      torch_opencv::fastTensor_To_CvMat(image_tensor, output_frame.img);
      if (!out_stream->writeFrame(output_frame)) {
        XPLERROR << "Error writing frame";
        break;
      }
    } // end loop over render frames

    if (tinyfd_messageBox("Finished Rendering",
      sanitizeForTFD("Video is in the project directory: " + project_dir + "\nDo you want to open the folder?").c_str(),
        "okcancel", "question", 1)) {
      file::openFileExplorer(project_dir);
    }
  }

  std::shared_ptr<std::atomic<bool>> cancel_render2d_requested = std::make_shared<std::atomic<bool>>(false);
  void runRender2DVideoInCommandRunner() {

    if (timeline.keyframes.empty()) {
      tinyfd_messageBox(
        "Cannot Render 2D Video without Keyframes",
        sanitizeForTFD("You must create at least one keyframe to render.").c_str(),
        "ok", "error", 1);
      return;
    }

    command_runner.setCompleteCallback([] { XPLINFO << "Thread command finished!"; });
    command_runner.setKilledCallback([] { XPLINFO << "Thread command killed!"; });
    
    auto render_progress_parser = [](const std::string& line, CommandProgressDescription& p) {
      std::smatch matches;

      std::regex skip_regex("Skipping\\s+(\\d+)\\s+frames.*");
      if (std::regex_search(line, matches, skip_regex) && matches.size() == 2) {
        int num_skip = std::stoi(matches[1].str());
        p.progress_str = "Synching  " + std::to_string(num_skip) + " frames for audio";
        p.frac = 1;
      }

      std::regex render_regex("Rendering frame: (\\d+) / (\\d+).*");
      if (std::regex_search(line, matches, render_regex) && matches.size() == 3) {
        int curr_frame = std::stoi(matches[1].str());
        int total_frames = std::stoi(matches[2].str());
        p.progress_str = "Rendering frame: " + std::to_string(curr_frame + 1) + " / " + std::to_string(total_frames);
        p.frac = static_cast<float>(curr_frame) / total_frames;
      }
    };
    
    command_runner.queueThreadCommand(
      cancel_render2d_requested,
      [&] {
        render2DVideoWithVirtualCamera();
      },
      render_progress_parser);
    command_runner.runCommandQueue();
  }

  std::shared_ptr<std::atomic<bool>> cancel_sfm_requested = std::make_shared<std::atomic<bool>>(false);
  void runIncrementalSfmInCommandRunner() {
    if (project_dir_select.getPath().empty()) {
      tinyfd_messageBox(
        "You must set the Project Directory",
        sanitizeForTFD("In the Project settings, choose a Project Directory. Output files are saved here.").c_str(),
        "ok", "error", 1);
      return;
    }
    
    command_runner.setCompleteCallback([] { XPLINFO << "Thread command finished!"; });
    command_runner.setKilledCallback([] { XPLINFO << "Thread command killed!"; });
    
    // Clear old viz data before starting a new run
    sfm_gui_data.mutex.lock();
    sfm_gui_data.viz_cameras.clear();
    sfm_gui_data.point_cloud.clear();
    sfm_gui_data.point_cloud_colors.clear();
    sfm_gui_data.pointcloud_needs_update = true;
    gaussian_splat_gui_data.current_model = nullptr;
    model_from_timeline = nullptr;
    which_model_to_draw = nullptr;
    timeline.num_frames = render2d_num_frames;
    timeline.curr_frame = 0;
    sfm_gui_data.mutex.unlock();
    
    auto sfm_progress_parser = [](const std::string& line, CommandProgressDescription& p) {
      std::smatch matches;
      std::regex decode_regex("Decoded frame: (\\d+) / (\\d+).*");
      if (std::regex_search(line, matches, decode_regex) && matches.size() == 3) {
        int curr_frame = std::stoi(matches[1].str());
        int total_frames = std::stoi(matches[2].str());
        p.progress_str = "Decoded frame: " + std::to_string(curr_frame + 1) + " / " + std::to_string(total_frames);
        p.frac = static_cast<float>(curr_frame) / total_frames;
      }

      std::regex keypoint_regex("Extracting keypoints and descriptors for image (\\d+) / (\\d+).*");
      if (std::regex_search(line, matches, keypoint_regex) && matches.size() == 3) {
        int curr_frame = std::stoi(matches[1].str());
        int total_frames = std::stoi(matches[2].str());
        p.progress_str = "Finding keypoints: " + std::to_string(curr_frame + 1) + " / " + std::to_string(total_frames);
        p.frac = static_cast<float>(curr_frame + 1) / total_frames;
      }
  
      std::regex matching_regex("Matching keypoints between image pair: (\\d+), (\\d+).*");
      if (std::regex_search(line, matches, matching_regex) && matches.size() == 3) {
        int image1 = std::stoi(matches[1].str());
        int image2 = std::stoi(matches[2].str());
        p.progress_str = "Matching images: " + std::to_string(image1 + 1) + " - " + std::to_string(image2 + 1);
        p.frac = 1.0; // TODO: estimate?
      }

      std::regex monodepth_regex("Estimating mono depthmap: (\\d+) / (\\d+).*");
      if (std::regex_search(line, matches, monodepth_regex) && matches.size() == 3) {
        int curr_frame = std::stoi(matches[1].str());
        int total_frames = std::stoi(matches[2].str());
        p.progress_str = "Mono depth map: " + std::to_string(curr_frame + 1) + " / " + std::to_string(total_frames);
        p.frac = static_cast<float>(curr_frame + 1) / total_frames;
      }

      std::regex sfm_regex("==== # active cameras: (\\d+) / (\\d+).*");
      if (std::regex_search(line, matches, sfm_regex) && matches.size() == 3) {
        int curr_frame = std::stoi(matches[1].str());
        int total_frames = std::stoi(matches[2].str());
        p.progress_str = "Solving camera: " + std::to_string(curr_frame) + " / " + std::to_string(total_frames);
        p.frac = static_cast<float>(curr_frame) / total_frames;
      }
    };
    
    command_runner.queueThreadCommand(
      cancel_sfm_requested,
      [&] {
        runIncrementalSfm();
      },
      sfm_progress_parser);
    command_runner.runCommandQueue();
  }

  void resizeImageToMaxDim(cv::Mat& image, int max_image_dim) {
    if(image.cols > max_image_dim || image.rows > max_image_dim) {
      float scale = max_image_dim / static_cast<float>(std::max(image.cols, image.rows));
      cv::resize(image, image, cv::Size(image.cols * scale, image.rows * scale), 0, 0, cv::INTER_AREA);
    }
  }

  void runIncrementalSfm() {
    using namespace calibration;
    using namespace video;

    const int max_image_dim = sfm_cfg_resize_max_dim;

    std::vector<cv::Mat> images;
    std::vector<std::string> camera_names;

    std::string project_dir = project_dir_select.getPath(); // capture the value now in case it changes while running
    file::createDirectoryIfNotExists(project_dir);

    int w = 0, h = 0;
    
    if (input_files_select.paths.size() == 1) { // single input file

      const std::string input_video_path = input_files_select.paths[0];
      
      InputVideoStream in_stream(input_video_path);
      XCHECK(in_stream.valid()) << "Invalid input video stream: " << input_video_path;
    
      w = in_stream.getWidth();
      h = in_stream.getHeight();
      std::pair<int, int> frame_rate = in_stream.guessFrameRate();
      double est_duration = in_stream.guessDurationSec();
      int est_num_frames = in_stream.guessNumFrames();
      XPLINFO << "width, height: " << w << ", " << h;
      XPLINFO << "frame rate: " << frame_rate.first << "/" << frame_rate.second << " = " << (float(frame_rate.first) / frame_rate.second);
      XPLINFO << "estimated duration(sec): " << est_duration;
      XPLINFO << "estimated # frames: " << est_num_frames;

      int decode_type = CV_32FC3;
      MediaFrame frame;
      int frame_count = 0;

      VideoStreamResult result;
      while((result = in_stream.readFrame(frame, decode_type)) == VideoStreamResult::OK) {
        if (*cancel_sfm_requested) return;
        if (!frame.is_video()) continue;
        XPLINFO << "Decoded frame: " << frame_count << " / " << est_num_frames;

        if (frame_count % camera_tracking_frame_stride == 0) {
          std::string image_filename = "frame_" + string::intToZeroPad(frame_count, 6) + ".png";
          cv::imwrite(project_dir + "/" + image_filename, frame.img * 255.0);

          resizeImageToMaxDim(frame.img, max_image_dim);
          w = frame.img.cols;
          h = frame.img.rows;

          images.push_back(frame.img);
          camera_names.push_back(image_filename);
        }

        ++frame_count;
      }

      if (result == VideoStreamResult::FINISHED) {
        XPLINFO << "Finished successfully.";
      } else {
        XCHECK_EQ(int(result), int(VideoStreamResult::ERR)) << "There was an error during transcoding.";
      }
    } else { // multiple input files

      // Sort inputs so they come out in proper order for SFM
      std::sort(input_files_select.paths.begin(), input_files_select.paths.end());

      if (video::hasImageExt(input_files_select.paths[0])) { // read multiple images
        for (auto& p : input_files_select.paths) { // read one frame from multiple videos with time offset
          std::string cam_name = file::filenameFromPath(p);
          XPLINFO << "camera name: " <<  cam_name;
          cv::Mat image = cv::imread(p);
          XCHECK(!image.empty()) << "Failed to read image: " << p;

          // HACK: Save a copy of the image in the project directory (wasteful but simplifies code)
          // We'll need it later, and the depthmaps are there already.
          // TODO: maybe store data directory in dataset.json or project.json
          cv::imwrite(project_dir + "/" + cam_name, image);

          image.convertTo(image, CV_32FC3, 1.0 / 255.0);
          resizeImageToMaxDim(image, max_image_dim);
          w = image.cols;
          h = image.rows;

          images.push_back(image);
          camera_names.push_back(cam_name);
        }
      } else {

        // Attempt to read time offsets. If it doesn't exist, the map is empty, and well create it later
        std::string time_offset_json_path = project_dir + "/time_offsets.json";
        std::map<std::string, int> time_offsets = calibration::readTimeOffsetJsonAsMap(time_offset_json_path);
    
        for (auto& p : input_files_select.paths) { // read one frame from multiple videos with time offset
          std::string cam_name = file::filenameFromPath(p);
          XPLINFO << "camera name: " <<  cam_name;
    
          InputVideoStream in_stream(p);
          XCHECK(in_stream.valid()) << "Invalid input video stream: " << p;
        
          int frames_to_skip = time_offsets.count(cam_name) ? time_offsets.at(cam_name) : 0;
          XPLINFO << "Skipping " << frames_to_skip << " frames for camera " << cam_name;
          int frame_count = 0;
          MediaFrame frame;
          VideoStreamResult result = VideoStreamResult::OK;
          while(frame_count <= frames_to_skip && (result = in_stream.readFrame(frame, CV_32FC3)) == VideoStreamResult::OK) {
            XCHECK_NE(int(result), int(VideoStreamResult::ERR)) << "There was an error decoding a frame from: " << p;
            if (!frame.is_video()) continue; // skip non-video frames (i.e. audio)
            ++frame_count;
          }
    
          resizeImageToMaxDim(frame.img, max_image_dim);
          w = frame.img.cols;
          h = frame.img.rows;
    
          images.push_back(frame.img);
          camera_names.push_back(cam_name);
        }
    
        // Create a default time_offets.json file if it does not exist.
        if (!file::fileExists(time_offset_json_path)) {
          calibration::createEmptyTimeOffsetJson(time_offset_json_path, camera_names);
        }
      }
    } // end of read multiple inputs

    double dist_a_to_b = 0.0;
    bool show_keypoints = false;
    bool show_matches = false;
    float flow_err_threshold = 20.0;
    float match_ratio_threshold = 0.9;
    int time_window_size = project_is_static ? 5 : images.size();

    if (*cancel_sfm_requested) return;
    
    if (project_camera_type == CAMERA_TYPE_RECTILINEAR) {
      std::vector<RectilinearCamerad> guess_intrinsics_rectilinear(
        images.size(), calibration::guessRectilinearIntrinsics(w, h, sfm_cfg_guess_fov));
      std::vector<RectilinearCamerad> optimized_cameras_rectilinear =
        incremental_sfm::estimateCameraPosesAndKeypoint3DWithIncrementalSfm(
          cancel_sfm_requested,
          &sfm_gui_data,
          images,
          camera_names,
          guess_intrinsics_rectilinear,
          dist_a_to_b,
          project_dir,
          show_keypoints,
          show_matches,
          flow_err_threshold,
          match_ratio_threshold,
          sfm_cfg_inlier_frac,
          sfm_cfg_depth_weight,
          time_window_size,
          sfm_cfg_filter_with_flow,
          sfm_cfg_share_intrinsics,
          sfm_cfg_reorder_cameras,
          sfm_cfg_use_intrinsic_prior,
          sfm_cfg_max_solver_itrs,
          sfm_gui_data.point_cloud,
          sfm_gui_data.point_cloud_colors);
      calibration::incremental_sfm::makeJsonSingleMovingCamera(optimized_cameras_rectilinear, project_dir + "/dataset.json");
    
      sfm_gui_data.mutex.lock();
      sfm_gui_data.viz_cameras = optimized_cameras_rectilinear;
      sfm_gui_data.mutex.unlock();
    }

    if (project_camera_type == CAMERA_TYPE_FISHEYE) {
      std::vector<FisheyeCamerad> guess_intrinsics_fisheye(
        images.size(), calibration::guessGoProIntrinsics(w, h));
      std::vector<FisheyeCamerad> optimized_cameras_fisheye =
        incremental_sfm::estimateCameraPosesAndKeypoint3DWithIncrementalSfm(
          cancel_sfm_requested,
          &sfm_gui_data,
          images,
          camera_names,
          guess_intrinsics_fisheye,
          dist_a_to_b,
          project_dir,
          show_keypoints,
          show_matches,
          flow_err_threshold,
          match_ratio_threshold,
          sfm_cfg_inlier_frac,
          sfm_cfg_depth_weight,
          time_window_size,
          sfm_cfg_filter_with_flow,
          sfm_cfg_share_intrinsics,
          sfm_cfg_reorder_cameras,
          sfm_cfg_use_intrinsic_prior,
          sfm_cfg_max_solver_itrs,
          sfm_gui_data.point_cloud,
          sfm_gui_data.point_cloud_colors);

      calibration::incremental_sfm::makeJsonSingleMovingCamera(optimized_cameras_fisheye, project_dir + "/dataset.json");
          
      // TODO: what about sfm_gui_data.viz_cameras in this fisheye case? is this handled by estimateCameraPosesAndKeypoint3DWithIncrementalSfm?
    }

    // Convert the point-cloud from RGBA to RGB
    std::vector<Eigen::Vector3f> sfm_point_cloud_colors3f;
    for (auto& c : sfm_gui_data.point_cloud_colors) {
      sfm_point_cloud_colors3f.emplace_back(c.x(), c.y(), c.z());
    }
    point_cloud::savePointCloudBinary(
      project_dir + "/pointcloud_sfm.bin", sfm_gui_data.point_cloud, sfm_point_cloud_colors3f);
  }

  void runGaussianVideo3DReconstruction() {
    using namespace p11::splat;
    using namespace p11::video;
    std::string project_dir = project_dir_select.getPath(); // capture the value now in case it changes while running
    file::createDirectoryIfNotExists(project_dir);
    //std::filesystem::remove_all(project_dir + "/trainsplat"); // TODO: shouldnt need this with gui
    std::filesystem::remove_all(project_dir + "/splat_frames");
    //file::createDirectoryIfNotExists(project_dir + "/trainsplat");
    file::createDirectoryIfNotExists(project_dir + "/splat_frames");
    scanProjectDirForSplatVideoFrames();

    SplatConfig cfg;
    cfg.resize_max_dim = splat_cfg_resize_max_dim;
    cfg.output_dir = project_dir;
    cfg.vid_dir = project_dir;
    cfg.sfm_pointcloud = project_dir + "/pointcloud_sfm.bin";
    cfg.save_steps = false;
    cfg.calc_psnr = false;
    cfg.max_num_splats = splat_cfg_max_num_splats;
    cfg.num_itrs = splat_cfg_num_itrs;
    cfg.first_frame_warmup_itrs = splat_cfg_first_frame_warmup_itrs;
    cfg.images_per_batch = splat_cfg_images_per_batch;
    cfg.train_vis_interval = 0;
    cfg.population_update_interval = splat_cfg_popi;
    cfg.learning_rate = splat_cfg_learning_rate;
    cfg.init_with_monodepth = splat_cfg_init_with_monodepth;
    cfg.use_depth_loss = splat_cfg_use_depth_loss;
    cfg.is_video = true;
    
    torch::jit::script::Module mono_depth;
    if (cfg.use_depth_loss || cfg.init_with_monodepth) {
      XPLINFO << "Loading mono depth model";
      torch::jit::getProfilingMode() = false;
      depth_estimation::getTorchModelDepthAnything2(mono_depth);
      XPLINFO << "Finished loading mono depth model";
    }

    std::vector<calibration::NerfKludgeCamera> cameras = calibration::readDatasetCameraJson(project_dir + "/dataset.json");
    calibration::readTimeOffsetJson(project_dir + "/time_offsets.json", cameras);

    std::vector<torch::Tensor> ignore_mask_tensors;
    std::vector<calibration::NerfKludgeCamera> rectified_cameras;
    std::vector<std::vector<cv::Mat>> camera_to_rectify_warp(cameras.size(), std::vector<cv::Mat>());    
  
    int decode_type = CV_8UC3; // CV_8UC3 or CV_32FC3
    MediaFrame frame;
    VideoStreamResult result;
    std::vector<std::shared_ptr<InputVideoStream>> video_streams(cameras.size());
    
    XCHECK(cameras.size() <= input_files_select.paths.size());

    for (int i = 0; i < cameras.size(); ++i) {
      // TODO: correct solution is probably to store data directory in dataset.json, dont use input_files
      std::string data_dir = file::getDirectoryName(input_files_select.paths[0]); // HACK
      std::string video_path = data_dir + "/" + cameras[i].name();
      XPLINFO << "video path: " << video_path;

      video_streams[i] = std::make_shared<InputVideoStream>(video_path);

      // Pre-advance the video to the time_offset_frames for each camera, so later 
      // decoded frames are synched across all cameras
      XPLINFO << "Skipping " << cameras[i].time_offset_frames << " frames for camera: " << cameras[i].name();
      int vid_frame_count = 0;
      while(vid_frame_count < cameras[i].time_offset_frames && (result = video_streams[i]->readFrame(frame, decode_type)) == VideoStreamResult::OK) {
        if (!frame.is_video()) continue;
        ++vid_frame_count;

        if (cancel_gaussian_requested && *cancel_gaussian_requested) return;
      }
    }

    file::createDirectoryIfNotExists(project_dir + "/web_template");

    std::string video_encoder = "libx264";
    if (splat_encode_select == 1) video_encoder = "libx265";

    video::EncoderConfig encode_cfg;
    encode_cfg.crf = 0;
    OutputVideoStream out_stream(
      *video_streams[0],
      project_dir + "/web_template/splatvid.mp4",
      splat_cfg_encode_w,
      splat_cfg_encode_h,
      video_streams[0]->guessFrameRate(),
      video_encoder,
      encode_cfg);
    XCHECK(out_stream.valid()) << "Invalid output video stream: " << project_dir + "/splatvid.mp4";

    std::shared_ptr<SplatModel> model = nullptr;
    std::shared_ptr<SplatModel> prev_model = nullptr;
  
    int frame_num = 0;
    bool have_more_frames = true;
    while(have_more_frames) {
      if (cancel_gaussian_requested && *cancel_gaussian_requested) {
        have_more_frames = false;
        break;
      }

      XPLINFO << "=========== frame_num: " << frame_num;
      // Re-seed RNG every frame for more temporally stable results
      torch::manual_seed(123);  // For reproducible initialization of weights
      srand(123);               // For calls to rand()
      
      calibration::MultiCameraDataset frame_dataset;
      
      // Get one frame from each camera's video, maybe preprocess
      for (int cam_idx = 0; cam_idx < cameras.size(); ++cam_idx) {
        if (!have_more_frames) break;

        XPLINFO << "Decoding video frame from camera: " << cameras[cam_idx].name();
        while((result = video_streams[cam_idx]->readFrame(frame, decode_type)) == VideoStreamResult::OK) {
          if (!frame.is_video()) {
            // Copy the audio stream from camera 0 to the output video
            if (cam_idx == 0) {
              if (!out_stream.writeFrame(frame)) {
                XPLINFO << "Error copying audio from camera 0's video to output video stream";
              }
            }
            continue;
          }  
          break; // stop after one actual video frame
        }

        // TODO: we could grab audio from camera 0 here and put it in the output stream
  
        if (frame.img.empty() || result == VideoStreamResult::FINISHED || result == VideoStreamResult::ERR) {
          have_more_frames = false;
          break;
        }
        cv::Mat image = frame.img;

        // Precompute warps from fisheye to rectified and recitifed camera intrinsics
        // This is deferred because we need to know the image size.
        if (rectified_cameras.empty()) {
          for (int i = 0; i < cameras.size(); ++i) {
            if (cancel_gaussian_requested && *cancel_gaussian_requested) {
              have_more_frames = false;
              break;
            }

            XPLINFO << "Precomputing rectification for camera: " << cameras[i].name();
            cameras[i].resizeToWidth(image.cols); // Fix case where intrinsics dont match video file size
            if (cameras[i].is_fisheye) {
              constexpr float kGoProHero12MagicFovConstant = 0.4;
              calibration::NerfKludgeCamera rectified_cam(precomputeFisheyeToRectilinearWarp(
                cameras[i].fisheye,
                camera_to_rectify_warp[i],
                kGoProHero12MagicFovConstant,
                cfg.resize_max_dim));
              rectified_cameras.push_back(rectified_cam);
            } else if (cameras[i].is_rectilinear) {
              // NOTE: nothing to do here for now since the camera model only has focal length, its already rectified.
              // TODO: if we add more intrinsics parameters, we need to properly rectify.
              rectified_cameras.push_back(cameras[i]);
            } else {
              XCHECK(false) << "Camera should be fisheye or rectilinear";
            }
          }

          // Read ignore mask images. Make a binary mask from pixels that are pure red
          // in the image (0, 0, 255). The masks need to be rectified.
          if (file::directoryExists(cfg.vid_dir + "/masks")) {
            XPLINFO << "Found /masks folder. Reading masks...";
      
            for (int i = 0; i < cameras.size(); ++i) {
              if (cancel_gaussian_requested && *cancel_gaussian_requested) {
                have_more_frames = false;
                break;
              }

              auto& cam = cameras[i];
              const std::string mask_filename = cfg.vid_dir + "/masks/" + file::filenamePrefix(cam.name()) + ".png";
              if (file::fileExists(mask_filename)) {
                XPLINFO << "Reading mask: " << mask_filename;
                cv::Mat mask_rgb = cv::imread(mask_filename);

                if (cameras[i].is_fisheye) {
                  // Warp the mask from fisheye to rectilnear
                  cv::remap(mask_rgb, mask_rgb, camera_to_rectify_warp[i][0], camera_to_rectify_warp[i][1], cv::INTER_AREA, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 0));     
                } else {
                  // Just resize for now for rectilinear
                  cv::Size rectified_size(rectified_cameras[i].getWidth(), rectified_cameras[i].getHeight());
                  cv::resize(mask_rgb, mask_rgb, rectified_size, 0, 0, cv::INTER_AREA);
                }

                cv::Mat mask_binary(mask_rgb.size(), CV_8U, cv::Scalar(0));
                for (int y = 0; y < mask_rgb.rows; ++y) {
                  for (int x = 0; x < mask_rgb.cols; ++x) {
                    mask_binary.at<uint8_t>(y, x) = mask_rgb.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 0, 255) ? 0 : 255;
                  }
                }
                ignore_mask_tensors.push_back(torch_opencv::cvMat8UC1_to_Tensor(device, mask_binary).permute({1, 2, 0}));
                //cv::imshow("mask_binary", mask_binary); cv::waitKey(0);
              }
            }
          }      
        } // end if (rectified_cameras.empty())

        if (cancel_gaussian_requested && *cancel_gaussian_requested) { have_more_frames = false; break; }

        XPLINFO << "Resizing and rectifying camera: " << cameras[cam_idx].name();

        cv::Mat rectified_image;
        if (cameras[cam_idx].is_fisheye) {
          // cv::remap with INTER_CUBIC can alias badly when mapping from a high
          // res image to a low res image. Pre-blurring reduces this issue, and can dramatically improve PSNR.
          const float downscale_ratio = rectified_cameras[cam_idx].getWidth() / float(image.cols);
          if (downscale_ratio <= 0.25) {
            const double sigma = 0.25 / downscale_ratio; // TODO: this # is hand-tuned, could be further tuned to maximize PSNR
            const int kernel_size = 2 * static_cast<int>(std::ceil(3 * sigma)) + 1;
            cv::GaussianBlur(image, image, cv::Size(kernel_size, kernel_size), sigma, sigma);
          }
          cv::remap(image, rectified_image, camera_to_rectify_warp[cam_idx][0], camera_to_rectify_warp[cam_idx][1], cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 0));
        } else if (cameras[cam_idx].is_rectilinear) {
          cv::Size rectified_size(rectified_cameras[cam_idx].getWidth(), rectified_cameras[cam_idx].getHeight());
          cv::resize(image, rectified_image, rectified_size, 0, 0, cv::INTER_AREA);
        } else {
          XCHECK(false) << "Camera should be fisheye or rectilinear";
        }

        if (frame_num == 0) {
          XPLINFO << "Saving frame image for camera " << cameras[cam_idx].name();
          cv::imwrite(cfg.output_dir + "/" + cameras[cam_idx].name() + ".jpg", rectified_image);
        }

        frame_dataset.images.push_back(rectified_image); // TODO: do we need this given that we have the tensor?
        // TODO: calibration::cvMat8UC3_to_Tensor is deprecated, should use torch_opencv::cvMat_to_Tensor
        frame_dataset.image_tensors.push_back(calibration::cvMat8UC3_to_Tensor(device, rectified_image));
      
        if (cfg.use_depth_loss || (frame_num == 0 && cfg.init_with_monodepth)) {
          cv::Mat depthmap = depth_estimation::estimateMonoDepthWithDepthAnything2(mono_depth, frame_dataset.images[cam_idx], true, true);
          //for (int y = 0; y < depthmap.rows; ++y) {
          //  for (int x = 0; x < depthmap.cols; ++x) {
          //    // NOTE: magic number here is hand tuned for da2 trying to get meters 
          //    depthmap.at<float>(y, x) = math::clamp(2.0f / depthmap.at<float>(y, x), 0.1f, 50.0f);
          //  }
          //}
          torch::Tensor depthmap_tensor = torch_opencv::cvMat_to_Tensor(device, depthmap);
          frame_dataset.depthmaps.push_back(depthmap);
          frame_dataset.depthmap_tensors.push_back(depthmap_tensor);
        }
      }

      frame_dataset.cameras = rectified_cameras;
      frame_dataset.ignore_masks = ignore_mask_tensors;
    
      if (cancel_gaussian_requested && *cancel_gaussian_requested) { have_more_frames = false; break; }

      if (frame_num == 0) {
        model = initSplatPopulation(device, cfg, frame_dataset);
      }

      // Get the timeline ready so we can show the current frame in-progress
      gl_context_mutex.lock();
      timeline.num_frames = frame_num + 1;
      gl_context_mutex.unlock();
      
      XPLINFO << "Training splat model for frame";
      trainSplatModel(cfg, device, frame_dataset, model, prev_model, &gaussian_splat_gui_data, cancel_gaussian_requested);

      // Copy the model for use in regularizing the next frame
      prev_model = std::make_shared<SplatModel>();
      prev_model->copyFrom(model);
  
      cfg.save_steps = false; // HACK: turn off save steps after the first frame (it may or may not be on)
      
      cv::Mat encoded_splat_image = encodeSplatModelWithSizzleZ(model, splat_cfg_encode_w, splat_cfg_encode_h);
      cv::imwrite(cfg.output_dir + "/splat_frames/" + string::intToZeroPad(frame_num, 6) + ".bmp", encoded_splat_image);

      video::MediaFrame output_frame;
      output_frame.img = encoded_splat_image;
      if (!out_stream.writeFrame(output_frame)) {
        XPLERROR << "Error writing frame";
        break;
      }
      
      scanProjectDirForSplatVideoFrames();

      if (cancel_gaussian_requested && *cancel_gaussian_requested) { have_more_frames = false; break; }
      
      if (cfg.calc_psnr && frame_num == 0) {
        calculatePsnrForDataset(cfg, device, frame_dataset, model);
      }

      if (cancel_gaussian_requested && *cancel_gaussian_requested) {
        have_more_frames = false;
        break;
      }

      if( !have_more_frames) break;
      ++frame_num;
    }
    XPLINFO << "Finished rendering video!";
  }

  void runGaussainStatic3DReconstruction() {
    using namespace p11::splat;
    std::string project_dir = project_dir_select.getPath();
    file::createDirectoryIfNotExists(project_dir);

    SplatConfig cfg;
    cfg.vid_dir = "";
    cfg.train_images_dir = project_dir;
    cfg.train_json = project_dir + "/dataset.json";
    cfg.sfm_pointcloud = project_dir + "/pointcloud_sfm.bin";
    cfg.output_dir = project_dir;
    cfg.save_steps = false;
    cfg.calc_psnr = true;
    cfg.resize_max_dim = splat_cfg_resize_max_dim;
    cfg.max_num_splats = splat_cfg_max_num_splats;
    cfg.num_itrs = splat_cfg_num_itrs;
    cfg.first_frame_warmup_itrs = splat_cfg_first_frame_warmup_itrs;
    cfg.images_per_batch = splat_cfg_images_per_batch;
    cfg.train_vis_interval = 10; // 0 = disable
    cfg.population_update_interval = splat_cfg_popi;
    cfg.learning_rate = splat_cfg_learning_rate;
    cfg.init_with_monodepth = splat_cfg_init_with_monodepth;
    cfg.use_depth_loss = splat_cfg_use_depth_loss;
    cfg.is_video = false;

    calibration::MultiCameraDataset train_dataset = calibration::readDataset(cfg.train_images_dir, cfg.train_json, device, cfg.resize_max_dim);
    
    // Load ignore masks if available
    XPLINFO << "Checking for ignore masks at: " << project_dir + "/masks"; 
    if (file::directoryExists(project_dir + "/masks")) {
      XPLINFO << "Found /masks folder. Reading masks...";
      bool got_valid_masks = false;
      for (int i = 0; i < train_dataset.cameras.size(); ++i) {
        auto& cam = train_dataset.cameras[i];
        std::string mask_path = project_dir + "/masks/" + file::filenamePrefix(cam.name()) + ".png";
        XPLINFO << "mask path: " << mask_path;
        if (!file::fileExists(mask_path)) { break; }
        cv::Mat mask_rgb = cv::imread(mask_path);
        if (mask_rgb.empty()) { break; }
        got_valid_masks = true;
        cv::resize(mask_rgb, mask_rgb, train_dataset.images[i].size(), 0, 0, cv::INTER_LINEAR);
        cv::Mat mask_binary(mask_rgb.size(), CV_8U, cv::Scalar(0));
        for (int y = 0; y < mask_rgb.rows; ++y) {
          for (int x = 0; x < mask_rgb.cols; ++x) {
            mask_binary.at<uint8_t>(y, x) = mask_rgb.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 0, 255) ? 0 : 255;
          }
        }
        // TODO: do we need to rectify the mask for fisheye cameras?
        train_dataset.ignore_masks.push_back(torch_opencv::cvMat8UC1_to_Tensor(device, mask_binary).permute({1, 2, 0}));
      }
      // Avoid returning a half-filled vector of masks if there was any error
      if (!got_valid_masks) train_dataset.ignore_masks.clear();
    }
    
    // Copy cameras to viz (this will update the timeline).
    // TODO: can we load this earlier from json as soon as project dir updates?
    sfm_gui_data.mutex.lock();
    sfm_gui_data.viz_cameras.clear();
    for (auto & cam : train_dataset.cameras) {
      XCHECK(cam.is_rectilinear);
      sfm_gui_data.viz_cameras.push_back(cam.rectilinear);
    }
    sfm_gui_data.mutex.unlock();

    if (cancel_gaussian_requested && *cancel_gaussian_requested) { return;} 

    std::shared_ptr<SplatModel> model = initSplatPopulation(device, cfg, train_dataset);
    trainSplatModel(cfg, device, train_dataset, model, nullptr, &gaussian_splat_gui_data, cancel_gaussian_requested);
    
    file::createDirectoryIfNotExists(project_dir + "/web_template");
    saveEncodedSplatFileWithSizzleZ(project_dir + "/web_template/splats.png", model, splat_cfg_encode_w, splat_cfg_encode_h);

    if (cancel_gaussian_requested && *cancel_gaussian_requested) return;


    if (cfg.calc_psnr) {
      calculatePsnrForDataset(cfg, device, train_dataset, model);
    }
  }

  std::shared_ptr<std::atomic<bool>> cancel_gaussian_requested = std::make_shared<std::atomic<bool>>(false);
  void runGaussainReconstructionInCommandRunner() {
    if (project_dir_select.getPath().empty()) {
      tinyfd_messageBox(
        "You must set the Project Directory",
        sanitizeForTFD("In the Project settings, choose a Project Directory. Output files are saved here.").c_str(),
        "ok", "error", 1);
      return;
    }

    // TODO: verify we have an SFM solution first
    
    command_runner.setCompleteCallback([] { });
    command_runner.setKilledCallback([] { });

    auto gaussian_progress_parser = [](const std::string& line, CommandProgressDescription& p) {
      std::smatch matches;
      std::regex decode_regex("train: (\\d+) / (\\d+).*");
      if (std::regex_search(line, matches, decode_regex) && matches.size() == 3) {
        int curr_itr = std::stoi(matches[1].str());
        int total_itrs = std::stoi(matches[2].str());
        p.progress_str = "Training: " + std::to_string(curr_itr) + " / " + std::to_string(total_itrs);
        p.frac = static_cast<float>(curr_itr) / total_itrs;
      }
      std::regex skip_regex(R"(Skipping\s+(\d+)\s+frames\s+for\s+camera:\s*(\S+))");
      if (std::regex_search(line, matches, skip_regex) && matches.size() == 3) {
        std::string cam_name = matches[2].str();
        p.progress_str = "Synchronizing " + cam_name;
        p.frac = 1;
      }      
    };

    *cancel_gaussian_requested = false; // TODO: DO WE NEED TO RESET THIS?

    command_runner.queueThreadCommand(
      cancel_gaussian_requested,
      [&] {
        if (project_is_static) {
          runGaussainStatic3DReconstruction();
        } else {
          runGaussianVideo3DReconstruction();
        }
      },
      gaussian_progress_parser);
    command_runner.runCommandQueue();
  }

  httplib::Server http_svr;
  void startServer(const std::string& serve_dir) {
    http_svr.set_mount_point("/", serve_dir.c_str()); // Serve static files from project directory

    // Add CORS headers
    http_svr.set_pre_routing_handler([](const httplib::Request &req, httplib::Response &res) {
      XPLINFO << "Request: " << req.method << " " << req.path;
      res.set_header("Access-Control-Allow-Origin", "*");
      return httplib::Server::HandlerResponse::Unhandled;
    });

    // Run http_svr.listen() in a separate thread, then return to the cancel check loop
    std::thread server_thread([&] {
      XPLINFO << "Serving Files at " << serve_dir;
      XPLINFO << "Webserver starting at http://localhost:8000";
      http_svr.listen("0.0.0.0", 8000);
    });
    server_thread.detach();
  }

  std::shared_ptr<std::atomic<bool>> cancel_webserver_requested = std::make_shared<std::atomic<bool>>(false);
  void createWebTemplateAndRunWebServerCommand() {
    std::string project_dir = project_dir_select.getPath();

    scanProjectDirForSplatVideoFrames(); // Do this in the main thread

    auto webserver_progress_parser = [](const std::string& line, CommandProgressDescription& p) {
      p.progress_str = "Running local web server";
      p.frac = 1;

      std::smatch matches;
      std::regex decode_regex("Re-encoding frame: (\\d+) / (\\d+).*");
      if (std::regex_search(line, matches, decode_regex) && matches.size() == 3) {
        int curr_frame = std::stoi(matches[1].str());
        int total_frames = std::stoi(matches[2].str());
        p.progress_str = "Cropping Frame: " + std::to_string(curr_frame + 1) + " / " + std::to_string(total_frames);
        p.frac = static_cast<float>(curr_frame) / total_frames;
      }
    };

    command_runner.setCompleteOrKilledCallback(
      [&] { XPLINFO  << "Local web server stopped."; });
      command_runner.queueThreadCommand(cancel_webserver_requested,
        [&, project_dir] { createWebTemplateAndReEncode(); },
        webserver_progress_parser);

    command_runner.queueThreadCommand(cancel_webserver_requested,
      [&, project_dir] {
        startServer(project_dir +  "/web_template");
        
        browser::openURL("http://localhost:8000/index.html");
        
        // Spin in a loop waiting for cancel_webserver_requested to be set
        while(true) {
          if (*cancel_webserver_requested) {
            XPLINFO  << "Cancel button pressed; Stopping HTTP server";
            http_svr.stop();
            return;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
      },
      webserver_progress_parser);
    command_runner.runCommandQueue();
  }

  
  static std::string matrix4ToJsArray(const Eigen::Matrix4d& m) {
    std::ostringstream oss;
    oss << std::setprecision(17) << '[';
    for (int col = 0; col < 4; ++col) {
      for (int row = 0; row < 4; ++row) {
        oss << m(row, col);
        if (!(col == 3 && row == 3))
          oss << ", ";
      }
    }
    oss << ']';
    return oss.str();
  }
  
  void createWebTemplateAndReEncode() {
    if (apply_crop3d) {
      XPLINFO << "Re-encoding splats with 3D crop";
      if (frame_num_to_splat_img.empty()) {
        reEncodeStaticSplatsWithCrop3D();
      } else {
        reEncodeVideoSplatsWithCrop3D();
      }
    }

    world_transform = getWorldTransformFromGuiSettings();
    Eigen::Matrix4d z_reflection = Eigen::Matrix4d::Identity();
    z_reflection(2, 2) = -1.0;
    std::string world_transform_str = matrix4ToJsArray(z_reflection * world_transform * z_reflection);

    std::string project_dir = project_dir_select.getPath();
    file::createDirectoryIfNotExists(project_dir + "/web_template");

    // Switch player input file based on video or static scene
    std::string splat_filename = file::fileExists(project_dir + "/web_template/splatvid.mp4")
      ? "splatvid.mp4" : "splats.png";

    if (file::fileExists(project_dir + "/web_template/splats_cropped.png")) {
      splat_filename = "splats_cropped.png"; 
    }

    if (file::fileExists(project_dir + "/web_template/splatvid_cropped.mp4")) {
      splat_filename = "splatvid_cropped.mp4"; 
    }

    std::ofstream f_template(project_dir + "/web_template/index.html");
    f_template << 
      string::replaceAll(string::replaceAll(string::replaceAll(string::replaceAll(
        web_template_cfg_support_webxr ? kWebXRTemplateHTML : kSimplifiedWebTemplateHTML,
          "{INPUT_FILE}", splat_filename),
          "{NUM_SPLATS}", std::to_string(splat_cfg_max_num_splats)),
          "{SPLAT_SCALE}", std::to_string(world_transform_scale)),
          "{MESH_TRANSFORM}", world_transform_str);
    f_template.close();

    std::ofstream f_splatmesh(project_dir + "/web_template/LifecastSplatMesh.js");
    f_splatmesh << kLifecastSplatMeshJs;
    f_splatmesh.close();
  }

  void scanProjectDirForSplatVideoFrames() {
    frame_num_to_splat_img.clear();

    if (project_is_static) {
      timeline.num_frames = render2d_num_frames;
      return;
    }

    std::string project_dir = project_dir_select.getPath();
    if (!file::directoryExists(project_dir + "/splat_frames")) return;
    std::vector<std::string> filenames = file::getFilesInDir(project_dir + "/splat_frames");
    int max_n = 0;
    for (auto f : filenames) {
      std::string prefix = file::filenamePrefix(f);
      std::string ext = file::filenameExtension(f);
      if (ext != "bmp") continue;
      std::string combine = prefix + "." + ext;
      if (combine != f) continue;
      int n = std::atoi(prefix.c_str());
      std::string sn = string::intToZeroPad(n, 6);
      if (sn != prefix) continue;

      frame_num_to_splat_img[n] = project_dir + "/splat_frames/" + f;
      max_n = std::max(max_n, n);
      XPLINFO << n << " -> " << frame_num_to_splat_img[n];
    }
    if (max_n != frame_num_to_splat_img.size() - 1) {
      XPLINFO << "WARNING: contents of /splat_frames not as expected!";
    }
    timeline.num_frames = max_n + 1;
    timeline.curr_frame = std::min(timeline.curr_frame, timeline.num_frames - 1);
    timeline.curr_frame_change_callback();
    XPLINFO << "timeline.num_frames=" << timeline.num_frames;
  }

  void addVirtualCameraKeyframe() {
    calibration::RectilinearCamerad keyframe_cam = 
      calibration::guessRectilinearIntrinsics(render2d_width, render2d_height, render2d_vfov);
    keyframe_cam.cam_from_world = gl_camera_as_rectilinear.cam_from_world;

    timeline.keyframes[timeline.curr_frame] = keyframe_cam;
  }

  void deleteVirtualCameraKeyframe() {
    timeline.keyframes.erase(timeline.curr_frame);
  }

  void drawCommandCancelButton(float width) {
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(224/255.0, 115/255.0, 20/255.0, 0.8));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f); // Make the button rounded
    if (ImGui::Button("  Cancel  ", ImVec2(width, 32))) {
      command_runner.kill();
    }
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
    ImGui::Dummy(ImVec2(0, 20));
  }

  void drawCommandRunButton(float width, std::function<void()> on_click, std::string label = "  Run  ") {
    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2, 0.5, 0.2, 0.8));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f); // Make the button rounded
    if (ImGui::Button(label.c_str(), ImVec2(width, 32))) {
      on_click();
    }
    ImGui::PopStyleVar();
    ImGui::PopStyleColor();
  }

  void drawCommandProgressBar(float width) {
    CommandProgressDescription progress_description = command_runner.updateAndGetProgress();
    ImVec2 progress_bar_pos = ImGui::GetCursorPos();
    ImGui::PushItemWidth(width);
    ImGui::ProgressBar(progress_description.frac, ImVec2(width, 32), "");
    ImGui::PopItemWidth();
    ImGui::SetCursorPos(ImVec2(progress_bar_pos.x + 10, progress_bar_pos.y + 4)); // HACK tweaks to center text in progress bar
    ImGui::TextUnformatted(*cancel_sfm_requested ? "Cancelling..." : progress_description.progress_str.c_str());
  }

  void setPresetGaussianOptions(std::string config_name) {
    if (config_name == "vr") {
      splat_cfg_max_num_splats = 65536;
      splat_cfg_num_itrs = project_is_static ? 2000 : 200;
      splat_cfg_first_frame_warmup_itrs = 2000;
      splat_cfg_popi = 100;
      splat_cfg_images_per_batch = 8;
      splat_cfg_encode_w = 1920;
      splat_cfg_encode_h = 1080;
      splat_cfg_resize_max_dim = 1024;
    }
    if (config_name == "medium") {
      splat_cfg_max_num_splats = 262144;
      splat_cfg_num_itrs = project_is_static ? 2000 : 200;
      splat_cfg_first_frame_warmup_itrs = 2000;
      splat_cfg_popi = 100;
      splat_cfg_images_per_batch = 16;
      splat_cfg_encode_w = 4096;
      splat_cfg_encode_h = 2048;
      splat_cfg_resize_max_dim = 1024;
    }
    if (config_name == "ultra") {
      splat_cfg_max_num_splats = 1670000;
      splat_cfg_num_itrs = project_is_static ? 25000 : 5000;
      splat_cfg_first_frame_warmup_itrs = 25000;
      splat_cfg_popi = 200;
      splat_cfg_images_per_batch = 32;
      splat_cfg_encode_w = 8192;
      splat_cfg_encode_h = 4096;
      splat_cfg_resize_max_dim = 1280;
    }
  }

  void drawOptionsPanel() {
    ImGuiStyle& style = ImGui::GetStyle();
    float w = ImGui::GetContentRegionAvailWidth();
    float padding = style.FramePadding.x;
    int slider_width = w - 84 - 2 * padding;

    switch(options_panel_state) {
    case OPTIONS_PANEL_PROJECT:
      ImGui::Text("Project");
      ImGui::Dummy(ImVec2(0, 20));

      ImGui::Text("Input Videos");
      input_files_select.on_change_callback = [this] { this->onSelectInputFiles(); };
      ImGui::PushItemWidth(w - padding * 8);
      input_files_select.drawAndUpdate();
      ImGui::PopItemWidth();

      ImGui::Dummy(ImVec2(0, 20));
      ImGui::PushItemWidth(w - padding * 8);
      project_dir_select.drawAndUpdate();
      ImGui::PopItemWidth();

      ImGui::Dummy(ImVec2(0, 20));
      
      if (project_is_static) {
        ImGui::Text("Static (3D)");
      } else {
        ImGui::Text("Dynamic (4D)");
      }
      break;

    case OPTIONS_PANEL_3D_CAMERA_TRACKING:
      ImGui::Text("3D Camera Tracking");
      ImGui::Dummy(ImVec2(0, 20));

      if (command_runner.isRunning()) {
        drawCommandCancelButton(w);
        drawCommandProgressBar(w);
      } else {
        drawCommandRunButton(w, [this] { runIncrementalSfmInCommandRunner(); });
      }

      ImGui::Dummy(ImVec2(0, 40));
      if (ImGui::CollapsingHeader("Options", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Dummy(ImVec2(0, 20));
        ImGui::PushItemWidth(w);

        ImGui::Text("Lens Type");
        if (ImGui::RadioButton(" Rectilinear (e.g., iPhone)", project_camera_type == CAMERA_TYPE_RECTILINEAR)) { project_camera_type = CAMERA_TYPE_RECTILINEAR; }
        if (ImGui::RadioButton(" Fisheye (e.g., GoPro)", project_camera_type == CAMERA_TYPE_FISHEYE)) { project_camera_type = CAMERA_TYPE_FISHEYE; }

        ImGui::Dummy(ImVec2(0, 20));
        ImGui::Text("Resize Max Dim");
        ImGui::InputInt("##SfmCfgResizeMaxDim", &sfm_cfg_resize_max_dim, 0);
        ImGui::Dummy(ImVec2(0, 20));

        ImGui::Text("Guess FOV (degrees)");
        ImGui::InputFloat("##SfmCfgGuessFov", &sfm_cfg_guess_fov);

        if (project_is_static && input_files_select.paths.size() == 1) {
          ImGui::Dummy(ImVec2(0, 20));
          ImGui::Text("Frame Stride");
          ImGui::InputInt("##FradeStride", &camera_tracking_frame_stride, 0);
          camera_tracking_frame_stride = math::clamp(camera_tracking_frame_stride, 1, 1000);
        }

        ImGui::Dummy(ImVec2(0, 20));
        ImGui::Text("Inlier Fraction");
        ImGui::InputFloat("##SfmCfgInlierFrac", &sfm_cfg_inlier_frac);

        ImGui::Dummy(ImVec2(0, 20));
        ImGui::Text("Mono Depth Weight");
        ImGui::InputFloat("##SfmCfgDepthWeight", &sfm_cfg_depth_weight, 0.0f, 0.0f, "%.6g");

        ImGui::Dummy(ImVec2(0, 20));
        ImGui::Text("Max Solver Iterations");
        ImGui::InputInt("##SfmCfgMaxSolverItrs", &sfm_cfg_max_solver_itrs, 0);
        
        ImGui::Dummy(ImVec2(0, 20));
        ImGui::Checkbox("Filter with optical flow", &sfm_cfg_filter_with_flow);

        ImGui::Dummy(ImVec2(0, 20));
        ImGui::Checkbox("Share intrinsics all cameras", &sfm_cfg_share_intrinsics);

        ImGui::Dummy(ImVec2(0, 20));
        ImGui::Checkbox("Re-order cameras", &sfm_cfg_reorder_cameras);

        ImGui::Dummy(ImVec2(0, 20));
        ImGui::Checkbox("Use intrinsic prior", &sfm_cfg_use_intrinsic_prior);

        ImGui::PopItemWidth();
      }

      break;
    case OPTIONS_PANEL_GAUSSIAN:
      ImGui::Text("Gaussian");
      ImGui::Dummy(ImVec2(0, 20));

      if (command_runner.isRunning()) {
        drawCommandCancelButton(w);
        drawCommandProgressBar(w);
      } else {
        drawCommandRunButton(w, [this] { runGaussainReconstructionInCommandRunner(); });
      }

      ImGui::Dummy(ImVec2(0, 40));
    
      ImGui::Text("Presets");
      if (ImGui::Button("VR")) {
        setPresetGaussianOptions("vr");
      }
      ImGui::SameLine();
      ImGui::Dummy(ImVec2(20, 0));
      ImGui::SameLine();
      if (ImGui::Button("Medium")) {
        setPresetGaussianOptions("medium");
      }
      ImGui::SameLine();
      ImGui::Dummy(ImVec2(20, 0));
      ImGui::SameLine();
      if (ImGui::Button("Ultra")) {
        setPresetGaussianOptions("ultra");
      }

      ImGui::Dummy(ImVec2(0, 40));
      if (ImGui::CollapsingHeader("Gaussian Solver", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Dummy(ImVec2(0, 20));

        ImGui::PushItemWidth(w);

        ImGui::Text("Max Gaussians");
        ImGui::InputInt("##SplatCfgMaxNumSplats", &splat_cfg_max_num_splats, 0);
        ImGui::Dummy(ImVec2(0, 20));

        ImGui::Text("# Iterations");
        ImGui::InputInt("##SplatCfgNumItrs", &splat_cfg_num_itrs, 0);
        ImGui::Dummy(ImVec2(0, 20));

        if (!project_is_static) {
          ImGui::Text("# Iterations (1st Frame)");
          ImGui::InputInt("##SplatCfgWarmupNumItrs", &splat_cfg_first_frame_warmup_itrs, 0);
          ImGui::Dummy(ImVec2(0, 20));
        }

        ImGui::Text("Images Per Batch");
        ImGui::InputInt("##SplatCfgImagesPerBatch", &splat_cfg_images_per_batch, 0);
        ImGui::Dummy(ImVec2(0, 20));

        ImGui::Text("Population Update Interval");
        ImGui::InputInt("##SplatCfgPopi", &splat_cfg_popi, 0);
        ImGui::Dummy(ImVec2(0, 20));

        ImGui::Text("Resize Max Dim");
        ImGui::InputInt("##SplatCfgResizeMaxDim", &splat_cfg_resize_max_dim, 0);
        ImGui::Dummy(ImVec2(0, 20));

        ImGui::Text("Learning Rate");
        ImGui::InputFloat("##SplatCfgLearningRate", &splat_cfg_learning_rate);
        ImGui::Dummy(ImVec2(0, 20));

        ImGui::Checkbox("Initialize With Mono Depth", &splat_cfg_init_with_monodepth);
        ImGui::Dummy(ImVec2(0, 20));

        ImGui::Checkbox("Depth Loss", &splat_cfg_use_depth_loss);
  
        ImGui::PopItemWidth();
      }
      
      ImGui::Dummy(ImVec2(0, 40));

      if (ImGui::CollapsingHeader("Gaussian Compression", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Dummy(ImVec2(0, 20));
        ImGui::PushItemWidth(w);

        ImGui::Text("Compressed Encoding Width");
        ImGui::InputInt("##SplatCfgEncodeW", &splat_cfg_encode_w, 0);
        ImGui::Dummy(ImVec2(0, 20));

        ImGui::Text("Compressed Encoding Height");
        ImGui::InputInt("##SplatCfgEncodeH", &splat_cfg_encode_h, 0);
        ImGui::PopItemWidth();

        ImGui::Dummy(ImVec2(0, 20));
        ImGui::Text("Video Encoder");
        if (ImGui::RadioButton(" h264  ", splat_encode_select == 0)) { splat_encode_select = 0; }
        ImGui::SameLine();
        if (ImGui::RadioButton(" h265", splat_encode_select == 1)) { splat_encode_select = 1; }
      }

      break;
    
    case OPTIONS_PANEL_WEB_PLAYER:
      ImGui::Text("Web Template");
      ImGui::Dummy(ImVec2(0, 20));

      if (command_runner.isRunning()) {
        drawCommandCancelButton(w);
        drawCommandProgressBar(w);
      } else {
        drawCommandRunButton(w, [this] { createWebTemplateAndRunWebServerCommand(); }, "  Launch Web Template  ");
      }
      
      ImGui::Dummy(ImVec2(0, 40));
      if (ImGui::CollapsingHeader("Options", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::Dummy(ImVec2(0, 20));

        ImGui::Checkbox("Include WebXR Support", &web_template_cfg_support_webxr);
      }

      break;

    case OPTIONS_PANEL_CROP:
      ImGui::Text("Crop Gaussians in 3D");
      ImGui::Dummy(ImVec2(0, 20));

      ImGui::Checkbox("Apply 3D Crop", &apply_crop3d);
      ImGui::Dummy(ImVec2(0, 20));

      ImGui::Text("Radius");
      //ImGui::SameLine();
      //ImGui::SetCursorPosX(170);
      ImGui::PushItemWidth(w - 84 - 2 * padding);
      ImGui::SliderFloat("##CropRadius", &crop3d_radius, 0.1f, 10.0f, "");
      ImGui::PopItemWidth();
      ImGui::SameLine();
      ImGui::SetCursorPosX(w - 84 + 8);
      ImGui::PushItemWidth(84);
      ImGui::InputFloat("##CropRadius Text", &crop3d_radius);
      ImGui::PopItemWidth();

      break;

    case OPTIONS_PANEL_VIRTUAL_CAMERA:
      ImGui::Text("Virtual Camera");
      ImGui::Dummy(ImVec2(0, 20));

      if (command_runner.isRunning()) {
        drawCommandCancelButton(w);
        drawCommandProgressBar(w);
      } else {
        drawCommandRunButton(w, [this] { runRender2DVideoInCommandRunner(); }, "  Render 2D Video  ");
      }

      ImGui::Dummy(ImVec2(0, 40));

      if (ImGui::Button("Create Keyframe", ImVec2(w, 0))) {
        addVirtualCameraKeyframe();
      }

      ImGui::Dummy(ImVec2(0, 20));

      if (ImGui::Button("Delete Keyframe", ImVec2(w, 0))) {
        deleteVirtualCameraKeyframe();
      }

      ImGui::Dummy(ImVec2(0, 40));

      if (ImGui::CollapsingHeader("Render Options", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::PushItemWidth(w);
        
        ImGui::Dummy(ImVec2(0, 20));

        if (project_is_static) {
          ImGui::Text("Number of Frames");
          if (ImGui::InputInt("##RenderNumFrames", &render2d_num_frames, 0)) {
            // If the number of frames changed, resize the timeline and clear any keyframes past the end
            timeline.num_frames = render2d_num_frames;
            for (auto it = timeline.keyframes.begin(); it != timeline.keyframes.end(); ) {
              if (it->first >= timeline.num_frames) {
                it = timeline.keyframes.erase(it);
              } else {
                ++it;
              }
            }
          }
          ImGui::Dummy(ImVec2(0, 20));

          ImGui::Text("Frame Rate");
          ImGui::InputInt("##RenderFrameRate", &render2d_frame_rate, 0);
          ImGui::Dummy(ImVec2(0, 20));
        }

        ImGui::Text("Render Width");
        ImGui::InputInt("##RenderWidth", &render2d_width, 0);
        ImGui::Dummy(ImVec2(0, 20));

        ImGui::Text("Render Height");
        ImGui::InputInt("##RenderHeight", &render2d_height, 0);
        ImGui::Dummy(ImVec2(0, 20));

        ImGui::Text("Vertical FOV (degrees)");
        ImGui::InputFloat("##RenderVFOV", &render2d_vfov);

        ImGui::PopItemWidth();
      }

      break;

    case OPTIONS_PANEL_WORLD_TRANSFORM:
      ImGui::Text("World Transform");
  
      ImGui::Dummy(ImVec2(0, 20));
      ImGui::Text("Scale");
      ImGui::PushItemWidth(slider_width);
      ImGui::SliderFloat("##WTScale", &world_transform_scale, 0.1f, 10.0f, "");
      ImGui::PopItemWidth();
      ImGui::SameLine();
      ImGui::SetCursorPosX(slider_width + 16);
      ImGui::PushItemWidth(84);
      ImGui::InputFloat("##WTScale Text", &world_transform_scale);
      ImGui::PopItemWidth();
  
      ImGui::Dummy(ImVec2(0, 20));
      ImGui::Text("X Rotation");
      ImGui::PushItemWidth(slider_width);
      ImGui::SliderFloat("##WTX Rotation", &world_transform_rx, -180, 180, "");
      ImGui::PopItemWidth();
      ImGui::SameLine();
      ImGui::SetCursorPosX(slider_width + 16);
      ImGui::PushItemWidth(84);
      ImGui::InputFloat("##WTX Rotation Text", &world_transform_rx);
      ImGui::PopItemWidth();

      ImGui::Dummy(ImVec2(0, 20));
      ImGui::Text("Y Rotation");
      ImGui::PushItemWidth(slider_width);
      ImGui::SliderFloat("##WTY Rotation", &world_transform_ry, -180, 180, "");
      ImGui::PopItemWidth();
      ImGui::SameLine();
      ImGui::SetCursorPosX(slider_width + 16);
      ImGui::PushItemWidth(84);
      ImGui::InputFloat("##WTY Rotation Text", &world_transform_ry);
      ImGui::PopItemWidth();
  
      ImGui::Dummy(ImVec2(0, 20));
      ImGui::Text("Z Rotation");
      ImGui::PushItemWidth(slider_width);
      ImGui::SliderFloat("##WTZ Rotation", &world_transform_rz, -180, 180, "");
      ImGui::PopItemWidth();
      ImGui::SameLine();
      ImGui::SetCursorPosX(slider_width + 16);
      ImGui::PushItemWidth(84);
      ImGui::InputFloat("##WTZ Rotation Text", &world_transform_rz);
      ImGui::PopItemWidth();

      ImGui::Dummy(ImVec2(0, 20));
      ImGui::Text("X Translation");
      ImGui::PushItemWidth(slider_width);
      ImGui::SliderFloat("##WTX Translation", &world_transform_tx, -10, 10, "");
      ImGui::PopItemWidth();
      ImGui::SameLine();
      ImGui::SetCursorPosX(slider_width + 16);
      ImGui::PushItemWidth(84);
      ImGui::InputFloat("##WTX Translation Text", &world_transform_tx);
      ImGui::PopItemWidth();

      ImGui::Dummy(ImVec2(0, 20));
      ImGui::Text("Y Translation");
      ImGui::PushItemWidth(slider_width);
      ImGui::SliderFloat("##WTY Translation", &world_transform_ty, -10, 10, "");
      ImGui::PopItemWidth();
      ImGui::SameLine();
      ImGui::SetCursorPosX(slider_width + 16);
      ImGui::PushItemWidth(84);
      ImGui::InputFloat("##WTY Translation Text", &world_transform_ty);
      ImGui::PopItemWidth();

      ImGui::Dummy(ImVec2(0, 20));
      ImGui::Text("Z Translation");
      ImGui::PushItemWidth(slider_width);
      ImGui::SliderFloat("##WTZ Translation", &world_transform_tz, -10, 10, "");
      ImGui::PopItemWidth();
      ImGui::SameLine();
      ImGui::SetCursorPosX(slider_width + 16);
      ImGui::PushItemWidth(84);
      ImGui::InputFloat("##WTZ Translation Text", &world_transform_tz);
      ImGui::PopItemWidth();

      ImGui::Dummy(ImVec2(0, 20));
      if (ImGui::Button("Reset to Default")) {
        setWorldTransformDefaults();
      }
      break;
    }
  }

  void loadSplatFrameFromFileBasedOnTimeline() {
    XPLINFO << "Loading frame: " << timeline.curr_frame;
    if (frame_num_to_splat_img.count(timeline.curr_frame) == 0) {
      XPLINFO << "Frame not found";
      model_from_timeline = nullptr;
      return;
    }
    std::string frame_filename = frame_num_to_splat_img.at(timeline.curr_frame);
    std::vector<splat::SerializableSplat> splats = splat::loadSplatImageFile(frame_filename);
    XPLINFO << "# loaded: " << splats.size() << " from " << frame_filename;
    
    gaussian_splat_gui_data.mutex.lock();
    model_from_timeline = serializableSplatsToModel(device, splats);
    torch::cuda::synchronize();
    gaussian_splat_gui_data.mutex.unlock();
  }

  void loadPrecomputedStaticSplatModelIfExists() {
    std::string project_dir = project_dir_select.getPath();
    std::string splat_filename = project_dir + "/web_template/splats.png"; 
    if (!file::fileExists(splat_filename)) return;

    std::vector<splat::SerializableSplat> splats = splat::loadSplatImageFile(splat_filename);
    XPLINFO << "# loaded: " << splats.size() << " from " << splat_filename;
    
    gaussian_splat_gui_data.mutex.lock();
    gaussian_splat_gui_data.current_model = serializableSplatsToModel(device, splats);
    torch::cuda::synchronize();
    gaussian_splat_gui_data.mutex.unlock();
  }

  void reEncodeStaticSplatsWithCrop3D() {
    std::string project_dir = project_dir_select.getPath();
    std::string splat_filename = project_dir + "/web_template/splats.png"; 
    if (!file::fileExists(splat_filename)) return; // TODO: error message. also handle video
    
    cv::Mat splat_image = cv::imread(splat_filename);
    std::vector<splat::SerializableSplat> splats = splat::decodeSplatImage(splat_image);
    XPLINFO << "# loaded: " << splats.size() << " from " << splat_filename;

    world_transform = getWorldTransformFromGuiSettings();
    Eigen::Matrix4f world_transform_f = world_transform.cast<float>();

    for (auto& s : splats) {
      Eigen::Vector4f pos_homogeneous(s.pos.x(), s.pos.y(), s.pos.z(), 1.0);
      Eigen::Vector4f transformed_pos = world_transform_f * pos_homogeneous;
      Eigen::Vector3f transformed_pos3(transformed_pos.x(), transformed_pos.y(), transformed_pos.z());
      if (transformed_pos3.norm() > crop3d_radius) {
        s.color = Eigen::Vector4f(0, 0, 0, 0);
        s.pos = Eigen::Vector3f(0, 0, 0);
        s.scale = Eigen::Vector3f(0, 0, 0);
        s.quat = Eigen::Vector4f(0, 0, 0, 0);
      }
    }

    // TODO: resize / re-pack option?

    cv::Mat new_splat_image = encodeSerializableSplatsWithSizzleZ(splats, splat_image.cols, splat_image.rows);
    cv::imwrite(project_dir + "/web_template/splats_cropped.png", new_splat_image); 
  }


  void reEncodeVideoSplatsWithCrop3D() {
    using namespace p11::video;

    XCHECK(!input_files_select.paths.empty()); // TODO: get this from project config json

    std::string project_dir = project_dir_select.getPath();

    std::map<std::string, int> time_offsets = calibration::readTimeOffsetJsonAsMap(
      project_dir + "/time_offsets.json");

    // Open one of the source videos to get frame rate and audio stream
    std::string first_video_path = input_files_select.paths[0];
    InputVideoStream original_video(first_video_path);
    XCHECK(original_video.valid()) << "Invalid input video stream: " << first_video_path;
  
    std::string first_cam_name = file::filenamePrefixFromPath(first_video_path);
    XPLINFO << "first_cam_name: " <<  first_cam_name;
    
    // Skip initial frames (probably needed to keep audio in sync).
    int frames_to_skip = time_offsets.count(first_cam_name) ? time_offsets.at(first_cam_name) : 0;
    XPLINFO << "Skipping " << frames_to_skip << " frames for camera " << first_cam_name;
    int read_frame_count = 0;
    MediaFrame read_frame;
    VideoStreamResult result = VideoStreamResult::OK;
    while(read_frame_count <= frames_to_skip && (result = original_video.readFrame(read_frame, CV_32FC3)) == VideoStreamResult::OK) {
      XCHECK_NE(int(result), int(VideoStreamResult::ERR)) << "There was an error decoding a frame from: " << first_video_path;
      if (!read_frame.is_video()) continue; // skip non-video frames (i.e. audio)
      ++read_frame_count;
    }

    std::string video_encoder = "libx264";
    if (splat_encode_select == 1) video_encoder = "libx265";
   
    video::EncoderConfig encode_cfg;
    encode_cfg.crf = 0;
    std::shared_ptr<OutputVideoStream> out_stream = nullptr;

    world_transform = getWorldTransformFromGuiSettings();
    Eigen::Matrix4f world_transform_f = world_transform.cast<float>();

    int num_frames = frame_num_to_splat_img.size();
    for (int frame = 0; frame < num_frames; ++frame) {
      XPLINFO << "Re-encoding frame: " << frame << " / " << num_frames;
      std::string frame_filename = frame_num_to_splat_img[frame];
      cv::Mat splat_image = cv::imread(frame_filename);
      std::vector<splat::SerializableSplat> splats = splat::decodeSplatImage(splat_image);
      
      // Wait until we know the size to create the output video stream
      if (frame == 0) {
        out_stream = std::make_shared<OutputVideoStream>(
          original_video,
          project_dir + "/web_template/splatvid_cropped.mp4",
          splat_image.cols,
          splat_image.rows,
          original_video.guessFrameRate(),
          video_encoder,
          encode_cfg);
        XCHECK(out_stream->valid()) << "Invalid output video stream: " << project_dir + "/splatvid_cropped.mp4";
      }
  
      // Copy audio stream from the original video until we get one video frame from it
      read_frame_count = 0;
      result = VideoStreamResult::OK;
      while(read_frame_count < 1 && (result = original_video.readFrame(read_frame, CV_32FC3)) == VideoStreamResult::OK) {
        XCHECK_NE(int(result), int(VideoStreamResult::ERR)) << "There was an error decoding a frame from: " << first_video_path;
        if (!read_frame.is_video()) {
          out_stream->writeFrame(read_frame);
        } else {
          ++read_frame_count;
        }
      }

      // Crop the splats, applying world transform
      for (auto& s : splats) {
        Eigen::Vector4f pos_homogeneous(s.pos.x(), s.pos.y(), s.pos.z(), 1.0);
        Eigen::Vector4f transformed_pos = world_transform_f * pos_homogeneous;
        Eigen::Vector3f transformed_pos3(transformed_pos.x(), transformed_pos.y(), transformed_pos.z());
        if (transformed_pos3.norm() > crop3d_radius) {
          s.color = Eigen::Vector4f(0, 0, 0, 0);
          s.pos = Eigen::Vector3f(0, 0, 0);
          s.scale = Eigen::Vector3f(0, 0, 0);
          s.quat = Eigen::Vector4f(0, 0, 0, 0);
        }
      }
      
      cv::Mat new_splat_image = encodeSerializableSplatsWithSizzleZ(splats, splat_image.cols, splat_image.rows);
      video::MediaFrame output_frame;
      output_frame.img = new_splat_image;
      if (!out_stream->writeFrame(output_frame)) {
        XPLERROR << "Error writing frame";
        return;
      }

      // TODO: handle audio frames
    }
  }

  std::shared_ptr<splat::SplatModel> applyCrop3DIfEnabled(std::shared_ptr<splat::SplatModel> input_model) {
    if (!apply_crop3d || input_model == nullptr) { return input_model; }
    world_transform = getWorldTransformFromGuiSettings();
    Eigen::Matrix4f world_transform_f = world_transform.cast<float>();
    torch::Tensor world_transform_tensor = torch::from_blob(
      world_transform_f.data(), {4, 4}, {torch::kFloat32}).transpose(0,1).to(device);

    auto splat_pos_linear = splat::splatPosActivation(input_model->splat_pos);
    auto ones = torch::ones({splat_pos_linear.size(0), 1}, splat_pos_linear.options());
    auto hom_pos = torch::cat({splat_pos_linear, ones}, 1);
    auto transformed = (world_transform_tensor.mm(hom_pos.t())).t();
    auto transformed_pos = transformed.slice(1, 0, 3);

    input_model->splat_alive = torch::where(
      torch::norm(transformed_pos, 2, 1, true) < crop3d_radius,
      input_model->splat_alive,
      torch::full_like(input_model->splat_alive, false));

    return input_model;
  }

  void updateGaussianSplatTexture() {
    std::shared_ptr<splat::SplatModel> cropped_model = nullptr;

    gaussian_splat_gui_data.mutex.lock();

    which_model_to_draw = gaussian_splat_gui_data.current_model;
    if (model_from_timeline && (gaussian_splat_gui_data.current_model == nullptr || timeline.curr_frame != timeline.num_frames - 1)) {
      which_model_to_draw = model_from_timeline;
    }

    // Do the copy while still holding the lock
    if (apply_crop3d && which_model_to_draw != nullptr) {
      cropped_model = std::make_shared<splat::SplatModel>();
      cropped_model->copyFrom(which_model_to_draw);
    }

    gaussian_splat_gui_data.mutex.unlock();
 
    if (which_model_to_draw == nullptr) { return; }

    // Dissable drawing splats outside the crop volume. This is a 
    // non-destructive preview.
    if (apply_crop3d) {
      which_model_to_draw = applyCrop3DIfEnabled(cropped_model);
    }

    // Update Gaussian splat rendering if we have a model
    if (show_gaussian_splats ) {
      // Render the splats
      gaussian_splat_gui_data.mutex.lock();
      auto [rendered_image, alpha_map, depth_map, _0, _1, _2, metas] = splat::renderSplatImageGsplat(
          device,
          gl_camera_as_rectilinear,
          which_model_to_draw,
          c10::nullopt,
          splat_world_transform);
      gaussian_splat_gui_data.mutex.unlock();

      // Make sure the tensor is contiguous for proper memory layout
      if (!rendered_image.is_contiguous()) {
        rendered_image = rendered_image.contiguous();
      }
      
      rendered_image = rendered_image.index({torch::indexing::Ellipsis, 
        torch::Tensor(torch::tensor({2, 1, 0}))});
        
      if (!splat_texture.isInitialized()) {
        XCHECK(splat_texture.init(rendered_image.size(1), rendered_image.size(0))) << "Failed to initialize Gaussian rendering texture";
      }

      // Direct update from CUDA tensor to OpenGL texture
      splat_texture.updateFromTensor(rendered_image);
    }
  }

  void drawMain3DView() {
    ImGuiStyle& style = ImGui::GetStyle();
    const float button_size = 32.0f;
    const float toolbar_height = button_size + style.FramePadding.y * 2;

    ImGui::BeginChild("Toolbar", ImVec2(0, toolbar_height), false, ImGuiWindowFlags_NoScrollbar);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f);
    if (ImGui::Button("  Project  ", ImVec2(0, button_size))) {
      options_panel_state = OPTIONS_PANEL_PROJECT;
      unhideOptionsPanel();
    }
    ImGui::SameLine();
    ImGui::Dummy(ImVec2(10, 0));
    ImGui::SameLine();
    if (ImGui::Button("  3D Camera Tracking  ", ImVec2(0, button_size))) {
      options_panel_state = OPTIONS_PANEL_3D_CAMERA_TRACKING;
      unhideOptionsPanel();
    }
    ImGui::SameLine();
    ImGui::Dummy(ImVec2(10, 0));
    ImGui::SameLine();
    if (ImGui::Button("  Gaussian  ", ImVec2(0, button_size))) {
      options_panel_state = OPTIONS_PANEL_GAUSSIAN;
      unhideOptionsPanel();
    }
    ImGui::SameLine();
    ImGui::Dummy(ImVec2(10, 0));
    ImGui::SameLine();
    if (ImGui::Button("  World Transform  ", ImVec2(0, button_size))) {
      options_panel_state = OPTIONS_PANEL_WORLD_TRANSFORM;
      unhideOptionsPanel();
    }
    ImGui::SameLine();
    ImGui::Dummy(ImVec2(10, 0));
    ImGui::SameLine();
    if (ImGui::Button("  Crop  ", ImVec2(0, button_size))) {
      options_panel_state = OPTIONS_PANEL_CROP;
      unhideOptionsPanel();
    }
    ImGui::SameLine();
    ImGui::Dummy(ImVec2(10, 0));
    ImGui::SameLine();
    if (ImGui::Button("  Web Template  ", ImVec2(0, button_size))) {
      options_panel_state = OPTIONS_PANEL_WEB_PLAYER;
      unhideOptionsPanel();
    }
    ImGui::SameLine();
    ImGui::Dummy(ImVec2(10, 0));
    ImGui::SameLine();
    if (ImGui::Button("  Virtual Camera  ", ImVec2(0, button_size))) {
      options_panel_state = OPTIONS_PANEL_VIRTUAL_CAMERA;
      unhideOptionsPanel();
    }
    ImGui::PopStyleVar();
    ImGui::EndChild();

    if (needs_unhide_options_panel) {
      needs_unhide_options_panel = false;
      unhideOptionsPanel();
    }

    ImVec2 available = ImGui::GetContentRegionAvail();

    static float last_window_width = 0;
    float current_width = ImGui::GetWindowWidth();
    if (current_width != last_window_width || !show_options_panel) {
      if (show_options_panel) {
        // First ensure right panel maintains minimum size
        split_width_right = std::max(split_width_right, (float)kMinSplitPanelSize);
        
        // Then give remaining space to left panel
        float max_left_width = available.x - split_width_right - kSplitterDragSize;
        split_width_left = std::max((float)kMinSplitPanelSize, max_left_width);
      } else {
        // If right panel is hidden, left panel takes full width
        split_width_left = available.x;
      }
      last_window_width = current_width;
    }

    int view3d_height = ImGui::GetWindowHeight() - VideoTimelineWidget::kTimelineHeight -
      toolbar_height - style.WindowPadding.y * 4 - ImGui::GetFrameHeight();

    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 0));
    ImGui::BeginChild("3D View", ImVec2(split_width_left, view3d_height), false);
    preview3d_viewport_min = ImVec2(ImGui::GetCursorPosX(), toolbar_height + 2 * style.FramePadding.y);
    preview3d_viewport_size = ImGui::GetWindowSize();

    updateCamera();
    updateGaussianSplatTexture();

    ImGui::EndChild();
    ImGui::PopStyleColor();

    if (show_options_panel) {
      ImGui::SameLine();
      Splitter(true, kSplitterDragSize, &split_width_left, &split_width_right, kMinSplitPanelSize, kMinSplitPanelSize, view3d_height);
      ImGui::SameLine();

      ImGui::BeginChild("Right Panel", ImVec2(split_width_right, view3d_height), false);
      
      ImGui::Indent(style.FramePadding.x);
      
      ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
      ImGui::BeginChild("Right Outline", ImVec2(split_width_right - style.FramePadding.x * 2, view3d_height - style.FramePadding.y * 2), true);
      
      drawOptionsPanel();

      // Add close button in the top-right corner
      float close_btn_size = 24.0;
      ImGui::SetCursorPos(ImVec2(ImGui::GetWindowWidth() - close_btn_size - style.FramePadding.x, 4));
      ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 12.0f);
      ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, 
        ImVec2(ImGui::GetStyle().FramePadding.x, ImGui::GetStyle().FramePadding.y - 8));
      if (ImGui::Button("x", ImVec2(close_btn_size, close_btn_size))) {
        show_options_panel = false;
      }
  
      ImGui::PopStyleVar(2);        
      ImGui::EndChild();
      ImGui::PopStyleVar(); 
      ImGui::Unindent();
      ImGui::EndChild();
    }

    timeline.render();
  }

  // Calling this instead of DearImGuiApp::finishDrawingImguiAndGl gives a chance to draw on top of
  // ImGui.
  void beginGlDrawingForImgui()
  {
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(window, &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(0.1, 0.1, 0.1, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // glfwSwapBuffers(window);// This is done in finishDrawingImguiAndGl() but not here, you must
    // call it later!
  }

  void makeCameraPoseVertexData(
    const std::vector<calibration::RectilinearCamerad>& cameras,
    const Eigen::Vector4f line_color,
    opengl::GlVertexBuffer<opengl::GlVertexDataXYZRGBA>& which_lines_vb
  ) {
    // Draw coordinate axes representing pose of each camera
    const Eigen::Vector3d zero(0, 0, 0);
    const Eigen::Vector3d dx(0.05, 0, 0);
    const Eigen::Vector3d dy(0, 0.05, 0);
    const Eigen::Vector3d dz(0, 0, 0.05);
    for(int i = 0; i < cameras.size(); ++i) {
      auto& cam = cameras[i];
      makeCameraFrustumLinesVertexData(cam, which_lines_vb);

      //const Eigen::Isometry3d& w_from_c = cam.cam_from_world.inverse();
      //const Eigen::Vector3d a = w_from_c * zero;
      //const Eigen::Vector3d b = w_from_c * dx;
      //const Eigen::Vector3d c = w_from_c * dy;
      //const Eigen::Vector3d d = w_from_c * dz;
      //float alpha = 0.33;
      //lines_vb2.vertex_data.emplace_back(a.x(), a.y(), a.z(), 1.0, 0.0, 0.0, alpha);
      //lines_vb2.vertex_data.emplace_back(b.x(), b.y(), b.z(), 1.0, 0.0, 0.0, alpha);
      //lines_vb2.vertex_data.emplace_back(a.x(), a.y(), a.z(), 0.0, 1.0, 0.0, alpha);
      //lines_vb2.vertex_data.emplace_back(c.x(), c.y(), c.z(), 0.0, 1.0, 0.0, alpha);
      //lines_vb2.vertex_data.emplace_back(a.x(), a.y(), a.z(), 0.0, 0.0, 1.0, alpha);
      //lines_vb2.vertex_data.emplace_back(d.x(), d.y(), d.z(), 0.0, 0.0, 1.0, alpha);
    }

    // Draw a line connecting all of the camera poses. Gradient in alpha indicates direction of time.
    //for (int i = 0; i < int(cameras.size()) - 1; ++i) {
    //  const Eigen::Vector3d p1 = cameras[i].getPositionInWorld();
    //  const Eigen::Vector3d p2 = cameras[i+1].getPositionInWorld();
    //  lines_vb2.vertex_data.emplace_back(p1.x(), p1.y(), p1.z(), line_color.x(), line_color.y(), line_color.z(), line_color.w());
    //  lines_vb2.vertex_data.emplace_back(p2.x(), p2.y(), p2.z(), line_color.x(), line_color.y(), line_color.z(), line_color.w());
    //}
  }

  std::vector<calibration::RectilinearCamerad> interpolateCameraPath(
      const std::map<int, calibration::RectilinearCamerad>& keyframes,
      int num_frames)
  {
    std::vector<calibration::RectilinearCamerad> path(num_frames);
    if (keyframes.empty()) return {};

    // pull keyframes into a vector
    std::vector<std::pair<int, calibration::RectilinearCamerad>> kf;
    kf.reserve(keyframes.size());
    for (auto& kv : keyframes) kf.push_back(kv);
    int K = int(kf.size());

    // single-keyframe → constant
    if (K == 1) {
      auto cam = kf[0].second;
      int md = std::max(render2d_width, render2d_height);
      cam.resizeToMaxDim(md);
      for (int f = 0; f < num_frames; ++f) path[f] = cam;
      return path;
    }

    // before first & after last fills
    int f_first = kf.front().first,
        f_last  = kf.back ().first;
    for (int f = 0; f < f_first;   ++f) path[f] = kf.front().second;
    for (int f = f_last+1; f < num_frames; ++f) path[f] = kf.back().second;

    // positions & position-tangents (Hermite)
    std::vector<Eigen::Vector3d> P(K);
    std::vector<double>          F(K);
    for (int i = 0; i < K; ++i) {
      P[i] = kf[i].second.getPositionInWorld();
      F[i] = double(kf[i].first);
    }
    std::vector<Eigen::Vector3d> Tn(K);
    for (int i = 0; i < K; ++i) {
      if (i == 0)
        Tn[i] = (P[1] - P[0]) / (F[1] - F[0]);
      else if (i == K-1)
        Tn[i] = (P[K-1] - P[K-2]) / (F[K-1] - F[K-2]);
      else
        Tn[i] = (P[i+1] - P[i-1]) / (F[i+1] - F[i-1]);
    }
    auto hermitePos = [&](int i, double u){
      double D   = F[i+1] - F[i];
      double h00 =  2*u*u*u - 3*u*u + 1;
      double h10 =      u*u*u - 2*u*u + u;
      double h01 = -2*u*u*u + 3*u*u;
      double h11 =      u*u*u -   u*u;
      return h00*P[i]
          + h10*Tn[i]*D
          + h01*P[i+1]
          + h11*Tn[i+1]*D;
    };

    // precompute rotation “chords” a[i]=log( R0ᵀ R1 ) and tangents t[i]
    std::vector<Eigen::Vector3d> a(K-1), t(K);
    for (int i = 0; i+1 < K; ++i) {
      Eigen::Matrix3d R0 = kf[i  ].second.cam_from_world.linear();
      Eigen::Matrix3d R1 = kf[i+1].second.cam_from_world.linear();
      bool improper = (R0.determinant() < 0);
      if ((R1.determinant() < 0) != improper) R1 = -R1;
      Eigen::AngleAxisd aa(R0.transpose() * R1);
      a[i] = aa.axis() * aa.angle();
    }
    for (int i = 0; i < K; ++i) {
      if (i == 0)        t[i] = a[0];
      else if (i == K-1) t[i] = a[K-2];
      else {
        double dt_lo = F[i]   - F[i-1];
        double dt_hi = F[i+1] - F[i];
        t[i] = (a[i-1]*dt_hi + a[i]*dt_lo) / (dt_lo + dt_hi);
      }
    }

    // interpolate each segment
    for (int i = 0; i + 1 < K; ++i) {
      int f0 = kf[i  ].first;
      int f1 = kf[i+1].first;
      auto cam0 = kf[i  ].second;
      auto cam1 = kf[i+1].second;
      int md = std::max(render2d_width, render2d_height);
      cam0.resizeToMaxDim(md);
      cam1.resizeToMaxDim(md);

      Eigen::Matrix3d R0 = cam0.cam_from_world.linear();
      bool improper = (R0.determinant() < 0);

      for (int f = f0; f <= f1 && f < num_frames; ++f) {
        double u = (f1 == f0) ? 0.0 : double(f - f0)/(f1 - f0);
        calibration::RectilinearCamerad out = cam0;

        // so(3)-Hermite rotation
        double h10 =      u*u*u - 2*u*u + u;
        double h01 = -2*u*u*u + 3*u*u;
        double h11 =      u*u*u -   u*u;
        Eigen::Vector3d alpha = h10*t[i]
                            + h01*a[i]
                            + h11*t[i+1];
        double theta = alpha.norm();
        Eigen::Matrix3d dR = (theta > 1e-8
          ? Eigen::AngleAxisd(theta, alpha/theta).toRotationMatrix()
          : Eigen::Matrix3d::Identity());
        Eigen::Matrix3d Rm = R0 * dR;

        // SVD-orthonormalize + improper fix
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(Rm,
            Eigen::ComputeFullU|Eigen::ComputeFullV);
        Eigen::Matrix3d ortho = svd.matrixU()*svd.matrixV().transpose();
        if (improper && ortho.determinant()>0) ortho = -ortho;
        Rm = ortho;

        out.cam_from_world.linear() = Rm;
        out.setPositionInWorld(hermitePos(i,u));
        out.focal_length = (1-u)*cam0.focal_length + u*cam1.focal_length;

        path[f] = out;
      }
    }

    return path;
  }

  void makeVertexDataForVirtualCameraPath() {
    // we store gl_camera_as_rectilinear in keyframes, which is in splat coordinate
    // system. to draw it in GL coordinates, we need to invert that transform,
    // but not apply user world transform
    Eigen::Matrix4d coord_transform = Eigen::Matrix4d::Identity();
    Eigen::AngleAxisd rotation(-M_PI/2, Eigen::Vector3d::UnitY());
    Eigen::Matrix3d rotation_matrix = rotation.toRotationMatrix();
    Eigen::Matrix3d reflection = Eigen::Matrix3d::Identity();
    reflection(2, 2) = -1.0;
    Eigen::Matrix3d combined = rotation_matrix * reflection;
    coord_transform.block<3,3>(0,0) = combined;
    const Eigen::Matrix4d inv = coord_transform.inverse();

    std::vector<calibration::RectilinearCamerad> interp_cams = interpolateCameraPath(
      timeline.keyframes, timeline.num_frames);

    std::vector<calibration::RectilinearCamerad> viz_cams;
    for (auto& cam : interp_cams) {
      calibration::RectilinearCamerad viz_cam(cam);
      int max_dim = std::max(render2d_width, render2d_height);
      viz_cam.resizeToMaxDim(max_dim);
      viz_cam.cam_from_world.matrix() = cam.cam_from_world.matrix() * inv;
      viz_cams.emplace_back(std::move(viz_cam));
    }

    std::lock_guard<std::mutex> lock(sfm_gui_data.mutex);
    makeCameraPoseVertexData(viz_cams, {1.0f, 0.5f, 0.3f, 1.0f}, lines_vb);
  }

  void makeFloorGrid() {
    for (int x = -50; x <= 50; ++x) {
      lines_vb.vertex_data.emplace_back(x, 0, -50, 1.0, 1.0, 1.0, 0.05);
      lines_vb.vertex_data.emplace_back(x, 0, +50, 1.0, 1.0, 1.0, 0.05);
    }
    for (int z = -50; z <= 50; ++z) {
      lines_vb.vertex_data.emplace_back(-50, 0, z, 1.0, 1.0, 1.0, 0.05);
      lines_vb.vertex_data.emplace_back(+50, 0, z, 1.0, 1.0, 1.0, 0.05);
    }
  }

  void makeCrop3DWireframe() {
    const float& r = crop3d_radius;
    int n = 64;
    for (int i = 0; i < n; ++i) {
      float theta0 = 2.0 * M_PI * float(i) / float(n);
      float theta1 = 2.0 * M_PI * float(i + 1) / float(n);
      float rcos0 = r * std::cos(theta0);
      float rsin0 = r * std::sin(theta0);
      float rcos1 = r * std::cos(theta1);
      float rsin1 = r * std::sin(theta1);
      lines_vb.vertex_data.emplace_back(rcos0, rsin0, 0, 1.0, 1.0, 1.0, 0.95);
      lines_vb.vertex_data.emplace_back(rcos1, rsin1, 0, 1.0, 1.0, 1.0, 0.95);
      lines_vb.vertex_data.emplace_back(rcos0, 0, rsin0, 1.0, 1.0, 1.0, 0.95);
      lines_vb.vertex_data.emplace_back(rcos1, 0, rsin1,1.0, 1.0, 1.0, 0.95);
      lines_vb.vertex_data.emplace_back(0, rcos0, rsin0, 1.0, 1.0, 1.0, 0.95);
      lines_vb.vertex_data.emplace_back(0, rcos1, rsin1, 1.0, 1.0, 1.0, 0.95);
    } 
  }

  void makeCameraFrustumLinesVertexData(const calibration::RectilinearCamerad& cam, opengl::GlVertexBuffer<opengl::GlVertexDataXYZRGBA>& which_lines_vb) {
    // Draw a dot on top of the camera
    const Eigen::Vector3d cam_pos = cam.getPositionInWorld();

    // Draw the frustum as a ray going from the origin through each corner pixel
    const double kFrustLineLen = 0.1;
    const Eigen::Matrix3d w_R_c = cam.cam_from_world.linear().transpose();

    const Eigen::Vector3d frust00 = cam_pos + kFrustLineLen * w_R_c * cam.rayDirFromPixel(Eigen::Vector2d(0, 0));
    const Eigen::Vector3d frust10 = cam_pos + kFrustLineLen * w_R_c * cam.rayDirFromPixel(Eigen::Vector2d(cam.width - 1, 0));
    const Eigen::Vector3d frust11 = cam_pos + kFrustLineLen * w_R_c * cam.rayDirFromPixel(Eigen::Vector2d(cam.width - 1, cam.height - 1));
    const Eigen::Vector3d frust01 = cam_pos + kFrustLineLen * w_R_c * cam.rayDirFromPixel(Eigen::Vector2d(0, cam.height - 1));

    which_lines_vb.vertex_data.emplace_back(cam_pos.x(), cam_pos.y(), cam_pos.z(), 1.0, 1.0, 1.0, 1.0);
    which_lines_vb.vertex_data.emplace_back(frust00.x(), frust00.y(), frust00.z(), 1.0, 1.0, 1.0, 0.2);
    which_lines_vb.vertex_data.emplace_back(cam_pos.x(), cam_pos.y(), cam_pos.z(), 1.0, 1.0, 1.0, 1.0);
    which_lines_vb.vertex_data.emplace_back(frust10.x(), frust10.y(), frust10.z(), 1.0, 1.0, 1.0, 0.2);
    which_lines_vb.vertex_data.emplace_back(cam_pos.x(), cam_pos.y(), cam_pos.z(), 1.0, 1.0, 1.0, 1.0);
    which_lines_vb.vertex_data.emplace_back(frust11.x(), frust11.y(), frust11.z(), 1.0, 1.0, 1.0, 0.2);
    which_lines_vb.vertex_data.emplace_back(cam_pos.x(), cam_pos.y(), cam_pos.z(), 1.0, 1.0, 1.0, 1.0);
    which_lines_vb.vertex_data.emplace_back(frust01.x(), frust01.y(), frust01.z(), 1.0, 1.0, 1.0, 0.2);
  
    which_lines_vb.vertex_data.emplace_back(frust00.x(), frust00.y(), frust00.z(), 1.0, 1.0, 1.0, 0.2);
    which_lines_vb.vertex_data.emplace_back(frust10.x(), frust10.y(), frust10.z(), 1.0, 1.0, 1.0, 0.2);
    which_lines_vb.vertex_data.emplace_back(frust10.x(), frust10.y(), frust10.z(), 1.0, 1.0, 1.0, 0.2);
    which_lines_vb.vertex_data.emplace_back(frust11.x(), frust11.y(), frust11.z(), 1.0, 1.0, 1.0, 0.2);
    which_lines_vb.vertex_data.emplace_back(frust11.x(), frust11.y(), frust11.z(), 1.0, 1.0, 1.0, 0.2);
    which_lines_vb.vertex_data.emplace_back(frust01.x(), frust01.y(), frust01.z(), 1.0, 1.0, 1.0, 0.2);    
    which_lines_vb.vertex_data.emplace_back(frust01.x(), frust01.y(), frust01.z(), 1.0, 1.0, 1.0, 0.2);
    which_lines_vb.vertex_data.emplace_back(frust00.x(), frust00.y(), frust00.z(), 1.0, 1.0, 1.0, 0.2);
  }

  void update3DGuiVertexData() {
    // Update the vertex buffer for lines
    lines_vb.vertex_data.clear();
    lines_vb2.vertex_data.clear();
    points_vb.vertex_data.clear();

    if (show_origin_in_3d_view) {
      lines_vb.vertex_data.emplace_back(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5);
      lines_vb.vertex_data.emplace_back(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5);
      lines_vb.vertex_data.emplace_back(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5);
      lines_vb.vertex_data.emplace_back(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5);
      lines_vb.vertex_data.emplace_back(0.0, 0.0, 0.0, 0.3, 0.3, 1.0, 0.5);
      lines_vb.vertex_data.emplace_back(0.0, 0.0, 1.0, 0.3, 0.3, 1.0, 0.5);
    }

    if (show_grid_in_3d_view) { makeFloorGrid(); }

    if (show_crop3d_wireframe) { makeCrop3DWireframe(); }

    if (show_sfm_cameras_in_3d_view) {
      sfm_gui_data.mutex.lock();
      makeCameraPoseVertexData(sfm_gui_data.viz_cameras, Eigen::Vector4f(0.65,0.65,1,0.25), lines_vb2);
      sfm_gui_data.mutex.unlock();
    }

    if (show_virtual_cameras_in_3d_view) {
      makeVertexDataForVirtualCameraPath();
    }

    lines_vb.bind();
    lines_vb.copyVertexDataToGPU(GL_DYNAMIC_DRAW);

    lines_vb2.bind();
    lines_vb2.copyVertexDataToGPU(GL_DYNAMIC_DRAW);

    points_vb.bind();
    points_vb.copyVertexDataToGPU(GL_DYNAMIC_DRAW);
  }

  void handleDragDrop(std::vector<std::string> drop_paths) {
    // Only all drag-drop on the main view of the app
    input_files_select.setPaths(drop_paths);
    needs_unhide_options_panel = true; // This may happen outside the main thread, so we defer it

    onSelectInputFiles();
  }
};

std::shared_ptr<GStudio4DApp> app;

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) app->handleMouseDown(button);
  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) app->handleMouseUp(button);
  if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) app->handleMouseDown(button);
  if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) app->handleMouseUp(button);
}

static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
  if (app->mouse_in_3d_viewport) {
    if (yoffset > 0) app->camera_radius *= 0.9;
    if (yoffset < 0) app->camera_radius *= 1.1;
  }
  if (app->mouse_in_timeline) {
    if (yoffset > 0) app->timeline.pixels_per_frame *= 1.1;
    if (yoffset < 0) app->timeline.pixels_per_frame *= 0.9;
    app->timeline.pixels_per_frame = math::clamp<float>(app->timeline.pixels_per_frame, 0.1, 100.0);
  }

  // TODO: call this?
  // ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  //if (key == GLFW_KEY_H && action == GLFW_PRESS) app->selected_tool = kScroll;
  //if (key == GLFW_KEY_SPACE && action == GLFW_RELEASE)
  //  app->toggle_play_video = !app->toggle_play_video;
  //if (app->mouse_in_3d_viewport) {
  //  if (key == GLFW_KEY_1 && action == GLFW_RELEASE) ...
  //}

  ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
}

void drop_callback(GLFWwindow* window, int count, const char** paths) {
  std::vector<std::string> unfiltered_paths;
  std::vector<std::string> filtered_paths;

  // Drop a single folder
  if (count == 1 && std::filesystem::is_directory(paths[0])) {
    std::vector<std::string> filenames = file::getFilesInDir(paths[0]);
    for (auto& f : filenames) {
      unfiltered_paths.push_back(std::string(paths[0]) + "/" + f);
    }
  } else {
    for (int i = 0; i < count; ++i) {
      unfiltered_paths.push_back(paths[i]);
    }
  }
  for (auto& f : unfiltered_paths) {
    if (!std::filesystem::is_regular_file(f)) continue;
    if (!(video::hasVideoExt(f) || video::hasImageExt(f))) continue;
    filtered_paths.push_back(f);
  }
  if (filtered_paths.empty()) {
    tinyfd_messageBox(
      "Invalid drag drop",
      "Only videos, images, or a single folder can be drag-dropped.", "ok", "info", 1);
  }
  app->handleDragDrop(filtered_paths);
}

}} // namespace p11::studio4dgs

#if defined(windows_hide_console) && defined(_WIN32)
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
#else
int main(int argc, char** argv) {
#endif

#ifdef _WIN32
  LoadLibraryA("torch_cuda.dll"); // Fix CUDA not found in libtorch, https://github.com/pytorch/pytorch/issues/72396
#endif
  // Fix a bug where some other libraries mess up the locale and cause random commas to be inserted in numbers
  std::locale::global(std::locale("C"));

  p11::studio4dgs::app = std::make_shared<p11::studio4dgs::GStudio4DApp>();

  p11::studio4dgs::app->init("4D Gaussian Studio", 1280, 720);

  glfwSetWindowUserPointer(p11::studio4dgs::app->window, (void*)&p11::studio4dgs::app);

  glfwSetMouseButtonCallback(p11::studio4dgs::app->window, p11::studio4dgs::mouse_button_callback);
  glfwSetDropCallback(p11::studio4dgs::app->window, p11::studio4dgs::drop_callback);
  glfwSetScrollCallback(p11::studio4dgs::app->window, p11::studio4dgs::scroll_callback);
  glfwSetKeyCallback(p11::studio4dgs::app->window, p11::studio4dgs::key_callback);

  p11::studio4dgs::app->initGStudio4DApp();
  p11::studio4dgs::app->setProStyle();

#ifdef _WIN32
  const std::string font_path = "Helvetica.ttf";  // On Windows, the directory structure is flat.
#else
  const std::string font_path = "fonts/Helvetica.ttf";
#endif
  ImGuiIO& io = ImGui::GetIO();
  io.Fonts->AddFontFromFileTTF(p11::runfile::getRunfileResourcePath(font_path).c_str(), 24.0);
  io.FontGlobalScale = 1.0f;
  
  ImGui::GetIO().IniFilename = nullptr;  // Disable layout saving.

  p11::studio4dgs::app->guiDrawLoop();
  p11::studio4dgs::app->cleanup();

  return EXIT_SUCCESS;
}
