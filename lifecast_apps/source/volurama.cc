// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
On Windows / Linux:

NOTE: we compile tinycudann for specific GPU types, see https://developer.nvidia.com/cuda-gpus

bazel run --cuda_archs=compute_75,sm_75,sm_86,sm_89 -- //source:volurama

on Mac with ARM, to properly use Metal:

PYTORCH_ENABLE_MPS_FALLBACK=1 bazel run -- //source:volurama

*/

// Make the application run without a terminal in Windows.
#if defined(windows_hide_console) && defined(_WIN32)
#pragma comment(linker, "/SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
#endif

#include "util_opengl.h"
#include "logger.h"
#include "dear_imgui_app.h"
#include "third_party/dear_imgui/imgui_internal.h" // For PushItemFlag on Windows
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "imgui_filedialog.h"
#include "imgui_cvmat.h"
#include "imgui_tinyplot.h"
#include "util_runfile.h"
#include "util_file.h"
#include "util_math.h"
#include "util_time.h"
#include "util_command.h"
#include "util_opencv.h"
#include "util_browser.h"
#include "util_torch.h"
#include "point_cloud.h"
#include "turbojpeg_wrapper.h"
#include "preferences.h"
#include "vignette.h"
#include "rectilinear_sfm_lib.h"
#include "lifecast_nerf_lib.h"
#include "multicamera_dataset.h"
#include "ngp_radiance_model.h"
#include "volurama_timeline.h"
#include "volurama_camera_polymorphism.h"
#include "convert_to_obj.h"
#include <regex>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <filesystem>
#include <locale>

#ifdef _WIN32
#include <Windows.h>
#endif
#ifdef __linux__
#include <GL/gl.h>
#endif


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

}  // namespace

namespace p11 {

constexpr float kSoftwareVersion = 1.7;

constexpr bool kEnableAutoLogging = true;

enum VoluramaAppState {
  STATE_SPLASH,
  STATE_EXPLAIN,
  STATE_SELECT_PROJECT_DIR,
  STATE_CONFIG_FFMPEG,
  STATE_SFM_NERF_CONFIG,
  STATE_3D_VIEW,
  STATE_RENDER_VID_CONFIG,
  STATE_EXPORT_LDI_CONFIG,
  STATE_EXPORT_MESH_CONFIG,
};

struct FloatingWindow {
  std::string title;
  bool visible;
  bool first_time_showing;
  std::function<void()> first_time_window_setup_func;
  std::function<void()> draw_body_func;

  void init(
    std::string title,
    bool visible,
    std::function<void()> first_time_window_setup_func,
    std::function<void()> draw_body_func)
   {
    this->title = title;
    this->visible = visible;
    this->first_time_showing = true;
    this->first_time_window_setup_func = first_time_window_setup_func;
    this->draw_body_func = draw_body_func;
  }

  void draw() {
    if (visible) {
      if (first_time_showing) {
        first_time_window_setup_func();
        first_time_showing = false;
      }

      ImGui::Begin(
        title.c_str(), &visible,
        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoScrollbar);
      //floatingWindowFocusShenanagins(); // TODO: it would be nice if this was here once instead of in every body function
      draw_body_func();
      ImGui::End();
    }
  }
};

struct VoluramaApp : public DearImGuiApp {
  // Application preferences
  std::map<std::string, std::string> prefs;
  
  VideoTimelineWidget timeline;

  ImVec2 preview3d_viewport_min, preview3d_viewport_size;

  // Vertex buffer, shaders, 3D data
  opengl::GlShaderProgram basic_shader;
  opengl::GlVertexBuffer<opengl::GlVertexDataXYZRGBA> lines_vb;
  opengl::GlVertexBuffer<opengl::GlVertexDataXYZRGBA> points_vb; // For GUI stuff mostly
  opengl::GlVertexBuffer<opengl::GlVertexDataXYZRGBA> pointcloud_vb;
  std::atomic<bool> pointcloud_vb_needs_update = false;

  // Mouse
  bool left_mouse_down = false;
  bool right_mouse_down = false;
  double prev_mouse_x = 0, prev_mouse_y = 0, curr_mouse_x = 0, curr_mouse_y = 0;
  bool mouse_in_3d_viewport = false;
  bool mouse_in_timeline = false;
  bool click_started_in_3d_viewport = false;

  // Camera
  double camera_radius, camera_theta, camera_phi;
  Eigen::Vector3f camera_orbit_center;
  double offset_cam_right, offset_cam_up;
  Eigen::Matrix4f model_view_projection_matrix;
  Eigen::Matrix4f view_matrix;
  Eigen::Matrix4f projection_matrix;
  Eigen::Matrix4d world_transform;

  // External tool config
  gui::ImguiInputFileSelect ffmpeg_tool_select;
  gui::ImguiInputFileSelect python_tool_select;
  std::string ffmpeg;

  // Commands, workflow config
  CommandRunner command_runner;

  std::string project_dir;
  std::string source_video_path;

  bool need_to_reload_project = false;
  
  rectilinear_sfm::RectilinearSfmConfig sfm_cfg;
  rectilinear_sfm::RectliinearSfmGuiData sfm_gui_data;
  nerf::NeoNerfConfig nerf_cfg;
  nerf::NeoNerfGuiData nerf_gui_data;
  int hdr_tonemap_choice = 0;
  int virtual_camera_type_choice = CAM_TYPE_RECTILINEAR;
  int virtual_camera_width = 0; // These are filled in at the end of loadProject() based on the sfm camera properties.
  int virtual_camera_height = 0;
  float virtual_camera_hfov = 0;
  bool virtual_camera_draw_horizon_line = true;

  int virtual_camera_vr180_size = 512;
  int virtual_camera_eqr_width = 512;
  int virtual_camera_eqr_height = 256;
  float virtual_stereo_baseline = 0.1;

  bool checkbox_run_extract_frames = true;
  bool checkbox_run_sfm = true;
  bool checkbox_run_nerf = true;

  // Application state and workflow
  VoluramaAppState app_state;
  bool is_creating_new_project = false;

  bool show_origin_in_3d_view = true;
  bool show_sfm_cameras_in_3d_view = true;
  bool show_virtual_cameras_in_3d_view = true;

  // The current dataset- nerf model, camera poses, point clouds, etc
  torch::DeviceType device;
  std::shared_ptr<nerf::NeoNerfModel> radiance_model;
  std::shared_ptr<nerf::ProposalDensityModel> proposal_model;

  std::vector<calibration::RectilinearCamerad> sfm_cameras;

  int point_cloud_num_samples;
  std::vector<Eigen::Vector3f> point_cloud;
  std::vector<Eigen::Vector4f> point_cloud_colors;

  gui::ImguiCvMat virtual_cam_preview_image;

  bool isAnyModalPopupOpen() {
    return commandWindowIsVisible() || (app_state != STATE_3D_VIEW);
  }

  void setAppState(VoluramaAppState new_state) {
    app_state = new_state;
    std::string state_str;
    switch (app_state) {
      case STATE_SPLASH: state_str = "STATE_SPLASH"; break;
      case STATE_EXPLAIN: state_str = "STATE_EXPLAIN"; break;
      case STATE_SELECT_PROJECT_DIR: state_str = "STATE_SELECT_PROJECT_DIR"; break;
      case STATE_CONFIG_FFMPEG: state_str = "STATE_CONFIG_FFMPEG"; break;
      case STATE_SFM_NERF_CONFIG: state_str = "STATE_SFM_NERF_CONFIG"; break;
      case STATE_3D_VIEW: state_str = "STATE_3D_VIEW"; break;
      case STATE_RENDER_VID_CONFIG: state_str = "STATE_RENDER_VID_CONFIG"; break;
      case STATE_EXPORT_LDI_CONFIG: state_str = "STATE_EXPORT_LDI_CONFIG"; break;
      case STATE_EXPORT_MESH_CONFIG: state_str = "STATE_EXPORT_MESH_CONFIG"; break;
    }
  }

  void setExternalToolDefaultPaths() {
  #ifdef __APPLE__
    #ifdef __arm64__ // Best guess ffmpeg install location for Apple Silicon (M1, M2, M3, etc.)
      ffmpeg = "/opt/homebrew/bin/ffmpeg";
    #else // For Intel Macs, brew installs ffmpeg in a different place
      ffmpeg = "/usr/local/bin/ffmpeg";
    #endif
  #else
    ffmpeg = "ffmpeg";
  #endif
  }

  void getExternalToolFileSelects() {
    ffmpeg = ffmpeg_tool_select.path;
  }

  void setExternalToolFileSelects() {
    ffmpeg_tool_select.setPath(ffmpeg.c_str());
  }

  void initGlBuffersAndShaders()
  {
    basic_shader.compile(kVertexShader_Basic, kFragmentShader_Basic);
    basic_shader.bind();

    lines_vb.init();
    lines_vb.bind();
    opengl::GlVertexDataXYZRGBA::setupVertexAttributes(basic_shader, "aVertexPos", "aVertexRGBA");

    points_vb.init();
    points_vb.bind();
    opengl::GlVertexDataXYZRGBA::setupVertexAttributes(basic_shader, "aVertexPos", "aVertexRGBA");

    pointcloud_vb.init();
    pointcloud_vb.bind();
    opengl::GlVertexDataXYZRGBA::setupVertexAttributes(basic_shader, "aVertexPos", "aVertexRGBA");    

    GL_CHECK_ERROR;
  }

  bool floatInputWithSlider(const char* label,  const char* unique_id, float* val, float min, float max) {
    bool value_changed = false;
    ImGui::AlignTextToFramePadding();
    ImGui::Text("%s", label);
    ImGui::SameLine();
    ImGui::SetCursorPosX(190);
    ImGui::PushItemWidth(120);
    value_changed |= 
      ImGui::InputFloat((std::string(unique_id) + "9").c_str(), val); // construct an id that won't clash with the slider
    ImGui::PopItemWidth();
    sameLineHorizontalSpace();
    ImGui::PushItemWidth(ImGui::GetWindowWidth() - 370);
    value_changed |=
      ImGui::SliderFloat(unique_id, val, min, max, "");
    ImGui::PopItemWidth();
    ImGui::Dummy(ImVec2(0.0f, 10.0f));
    return value_changed;
  }

  float world_transform_scale;
  float world_transform_rx;
  float world_transform_ry;
  float world_transform_rz;
  void setWorldTransformDefaults() {
    world_transform_scale = 1;
    world_transform_rx = 0;
    world_transform_ry = 0;
    world_transform_rz = 0;
  }

  void updateWorldTransform() {
    std::vector<double> rvec = {
      world_transform_rx * M_PI / 180,
      world_transform_ry * M_PI / 180,
      world_transform_rz * M_PI / 180};
    Eigen::Matrix3d rotation;
    ceres::AngleAxisToRotationMatrix(rvec.data(), rotation.data());
    world_transform = Eigen::Matrix4d::Zero();
    world_transform(0,0) = world_transform_scale;
    world_transform(1,1) = world_transform_scale;
    world_transform(2,2) = world_transform_scale;
    world_transform.block<3,3>(0,0) *= rotation;
    world_transform(3,3) = 1.0;
  }

  // Orbit editor variables
  float orbit_editor_radius;
  float orbit_editor_center_x;
  float orbit_editor_center_y;
  float orbit_editor_center_z;
  float orbit_editor_start_theta;
  float orbit_editor_end_theta;
  void setOrbitEditorDefaults() {
    orbit_editor_radius = 3;
    orbit_editor_center_x = 0;
    orbit_editor_center_y = 0;
    orbit_editor_center_z = 3;
    orbit_editor_start_theta = 300;
    orbit_editor_end_theta = 240;
  }

  FloatingWindow floating_window_tracking_viz;
  FloatingWindow floating_window_sfm_cost_plot;
  FloatingWindow floating_window_nerf_cost_plot;
  FloatingWindow floating_window_preview_render;
  FloatingWindow floating_window_virtual_cam_settings;
  FloatingWindow floating_window_keyframe_editor;
  FloatingWindow floating_window_world_transform;
  FloatingWindow floating_window_orbit_editor;

  void setupFloatingWindows() {
    floating_window_tracking_viz.init("Keypoint Tracking", false,
      [&]{
        float window_w = ImGui::GetWindowWidth(), window_h = ImGui::GetWindowHeight();
        float w = window_w * 0.66, h = window_h * 0.66;
        ImGui::SetNextWindowSize(ImVec2(w, h));
        ImGui::SetNextWindowPos(ImVec2(window_w - w - 60, 100));
      },
      [&]{
        floatingWindowFocusShenanagins();
    
        if (sfm_gui_data.viz.size.width != 0) {
          sfm_gui_data.viz.scale_to_fit = false;
          sfm_gui_data.viz.center_and_expand = true;
          sfm_gui_data.mutex.lock();
          sfm_gui_data.viz.makeGlTexture(); // NOTE: its ok to call this every frame. it wont do anything unless an update is needed
          sfm_gui_data.mutex.unlock();
          sfm_gui_data.viz.drawInImGui();
        }
      });

    floating_window_sfm_cost_plot.init("Structure from Motion Optimization", false,
      [&]{
        float window_w = ImGui::GetWindowWidth(), window_h = ImGui::GetWindowHeight();
        float w = window_w * 0.66, h = window_h * 0.66;
        ImGui::SetNextWindowSize(ImVec2(w, h));
        ImGui::SetNextWindowPos(ImVec2(window_w - w - 100, 140));
      },
      [&]{
        floatingWindowFocusShenanagins();

        std::lock_guard<std::mutex> guard(sfm_gui_data.mutex);
        if (!sfm_gui_data.plot_data_x.empty()) {
          float y_max = *std::max_element(sfm_gui_data.plot_data_y.begin(), sfm_gui_data.plot_data_y.end());
          gui::plotLineGraph(sfm_gui_data.plot_data_x, sfm_gui_data.plot_data_y, 0.0, sfm_gui_data.ceres_iterations, 0.0, y_max);
        }
      });

    floating_window_nerf_cost_plot.init("Nerf Optimization", false,
      [&]{
        float window_w = ImGui::GetWindowWidth(), window_h = ImGui::GetWindowHeight();
        float w = window_w * 0.66, h = window_h * 0.66;
        ImGui::SetNextWindowSize(ImVec2(w, h));
        ImGui::SetNextWindowPos(ImVec2(window_w - w - 140, 180));
      },
      [&]{
        floatingWindowFocusShenanagins();

        std::lock_guard<std::mutex> guard(nerf_gui_data.mutex);
        if (!nerf_gui_data.plot_data_x.empty()) {
          float y_max = *std::max_element(nerf_gui_data.plot_data_y.begin(), nerf_gui_data.plot_data_y.end());
          gui::plotLineGraph(nerf_gui_data.plot_data_x, nerf_gui_data.plot_data_y, 0.0, nerf_cfg.num_training_itrs, 0.0, y_max, false);
        }
      });

    floating_window_preview_render.init("Preview Render", true,
      [&]{
        ImGui::SetNextWindowSize(ImVec2(480, 360));
        ImGui::SetNextWindowPos(ImVec2(40, 80));
      },
      [&]{
        floatingWindowFocusShenanagins();

        if (virtual_cam_preview_image.size.width != 0) {
          virtual_cam_preview_image.scale_to_fit = false;
          virtual_cam_preview_image.center_and_expand = true;
          virtual_cam_preview_image.makeGlTexture(); // NOTE: its ok to call this every frame. it wont do anything unless an update is needed.
          virtual_cam_preview_image.drawInImGui();
        }
      });

    floating_window_virtual_cam_settings.init("Virtual Camera Settings", true,
      [&]{
        ImGui::SetNextWindowSize(ImVec2(520, 360));
        ImGui::SetNextWindowPos(ImVec2(530, 80));
      },
      [&]{
        floatingWindowFocusShenanagins();

        ImGui::Dummy(ImVec2(0.0f, 20.0f));
        ImGui::Indent();

        bool value_changed = drawVirtualCameraSettingsWithoutWindow();

        ImGui::Unindent();
        if (value_changed) refreshPreviewRender();
      });

    floating_window_keyframe_editor.init("Keyframe Editor", true,
      [&]{
        ImGui::SetNextWindowSize(ImVec2(480, 400));
        ImGui::SetNextWindowPos(ImVec2(40, 450));
      },
      [&]{
        floatingWindowFocusShenanagins();

        ImGui::Dummy(ImVec2(0.0f, 20.0f));
        ImGui::Indent();

        if (timeline.keyframes.count(timeline.curr_frame) > 0) {
          bool disable_remove = (timeline.curr_frame == 0);
          bool remove_clicked = false;
          if (disable_remove) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2, 0.2, 0.2, 0.2));
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0, 1.0, 1.0, 0.2));
            ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
          }
          if (ImGui::Button("Remove Keyframe")) {
            timeline.keyframes.erase(timeline.curr_frame);
            remove_clicked = true;
            refreshPreviewRender();
          } 
          if (disable_remove) {
            ImGui::PopItemFlag();
            ImGui::PopStyleColor(2);
          }
          if (!remove_clicked) {
            ImGui::Dummy(ImVec2(0.0f, 20.0f));
    
            bool val_changed = false;
            auto& curr_kf = timeline.keyframes[timeline.curr_frame];
            val_changed |= floatInputWithSlider("X",        "##897421", &curr_kf.tx, -5, 5);
            val_changed |= floatInputWithSlider("Y",        "##093282", &curr_kf.ty, -5, 5);
            val_changed |= floatInputWithSlider("Z",        "##023333", &curr_kf.tz, -5, 5);
            val_changed |= floatInputWithSlider("Rotate X", "##523464", &curr_kf.rx, -180, 180);
            val_changed |= floatInputWithSlider("Rotate Y", "##462335", &curr_kf.ry, -180, 180);
            val_changed |= floatInputWithSlider("Rotate Z", "##885666", &curr_kf.rz, -180, 180);
            if (val_changed) { refreshPreviewRender(); }
          }
        } else {
          if (ImGui::Button("Add Keyframe")) {
            addKeyframe(timeline.curr_frame); // TODO: copy data from previous keyframe or interpolated bewteen prev and next
            refreshPreviewRender();
          }
        }

        ImGui::Unindent();
      });

    floating_window_world_transform.init("World Transform", true,
      [&]{
        ImGui::SetNextWindowSize(ImVec2(480, 310));
        ImGui::SetNextWindowPos(ImVec2(40, 870));
      },
      [&]{
        floatingWindowFocusShenanagins();

        ImGui::Dummy(ImVec2(0.0f, 20.0f));
        ImGui::Indent();

        if (ImGui::Button("Reset to Default")) {
          setWorldTransformDefaults();
          refreshPreviewRender();
        }
        ImGui::Dummy(ImVec2(0.0f, 20.0f));

        bool val_changed = false;
        val_changed |= floatInputWithSlider("Scale",    "##455263", &world_transform_scale, 0.1, 10);
        val_changed |= floatInputWithSlider("Rotate X", "##734566", &world_transform_rx, -180, 180);
        val_changed |= floatInputWithSlider("Rotate Y", "##478888", &world_transform_ry, -180, 180);
        val_changed |= floatInputWithSlider("Rotate Z", "##456666", &world_transform_rz, -180, 180);
        if (val_changed) { refreshPreviewRender(); }

        ImGui::Unindent();
      });

    floating_window_orbit_editor.init("Orbit Editor", true,
      [&]{
        ImGui::SetNextWindowSize(ImVec2(520, 400));
        ImGui::SetNextWindowPos(ImVec2(530, 450));
      },
      [&]{
        floatingWindowFocusShenanagins();

        ImGui::Dummy(ImVec2(0.0f, 20.0f));
        ImGui::Indent();

        if (ImGui::Button("Create Keyframes")) {
          makeVirtualCameraPreset("orbit");
        }
    
        ImGui::SameLine();
        ImGui::Dummy(ImVec2(20.0, 0.0));
        ImGui::SameLine();

        if (ImGui::Button("Reset to Default")) {
          setOrbitEditorDefaults();
        }
        ImGui::Dummy(ImVec2(0.0f, 20.0f));
        floatInputWithSlider("Radius",   "##873226", &orbit_editor_radius,    0,  10);
        floatInputWithSlider("Center X", "##245372", &orbit_editor_center_x, -10, 10);
        floatInputWithSlider("Center Y", "##234452", &orbit_editor_center_y, -10, 10);
        floatInputWithSlider("Center Z", "##512346", &orbit_editor_center_z, -10, 10);
        floatInputWithSlider("Start Angle", "##63452", &orbit_editor_start_theta, -360, 360);
        floatInputWithSlider("End Angle",   "##54345", &orbit_editor_end_theta, -360, 360);
        
        ImGui::Unindent();
      });
  }

  void initVoluramaApp()
  {
    device = util_torch::findBestTorchDevice();

    initGlBuffersAndShaders();
    resetCamera();
    timeline.curr_frame_change_callback = [&] {
      XPLINFO << "timeline.curr_frame=" << timeline.curr_frame;
      refreshPreviewRender();
    };

    prefs = preferences::getPrefs();
    if (prefs.count("project_dir")) project_dir_folder_select.setPath(prefs.at("project_dir").c_str());
    
    setExternalToolDefaultPaths();
    if (prefs.count("ffmpeg")) ffmpeg = prefs.at("ffmpeg");
    setExternalToolFileSelects();

    point_cloud_num_samples = 1000000;
    if (prefs.count("point_cloud_num_samples")) {
      point_cloud_num_samples = std::atoi(prefs.at("point_cloud_num_samples").c_str());
    }

    setSfmAndNerfDefaults();

    setupFloatingWindows();

    std::thread render_nerf_thread(&VoluramaApp::renderNerfThread, this);
    render_nerf_thread.detach();

    if (kEnableAutoLogging) xpl::stdoutLogger.attachTextFileLog(project_dir + "/log.txt");

    setAppState(STATE_SPLASH);
  }

  void updateNumProgressiveRenderSteps() {
    constexpr int kStereo = 2;
    // Pick a number of progressive rendering steps such that the coursest level is fast
    int max_dim;
    switch(virtual_camera_type_choice) {
    case CAM_TYPE_VR180:
      max_dim = virtual_camera_vr180_size * kStereo;
      break;
    case CAM_TYPE_EQUIRECTANGULAR:
      max_dim = std::max(virtual_camera_eqr_width, virtual_camera_eqr_height);
      break;
    case CAM_TYPE_RECTILINEAR_STEREO:
      max_dim = std::max(virtual_camera_width * 2, virtual_camera_height);
      break;
    default: 
      max_dim = std::max(virtual_camera_width, virtual_camera_height);
      break;
    }

    render_progression_steps = 2;
    constexpr int kTargetSize = 512;
    while (max_dim > kTargetSize && render_progression_steps < 8) {
      render_progression_steps++;
      max_dim /= 2;
    }

    if (virtual_camera_type_choice == CAM_TYPE_LOOKING_GLASS_PORTRAIT) {
      render_progression_steps = 6;
    }
  
    //XPLINFO << "render_progression_steps=" << render_progression_steps;
  }


  void refreshPreviewRender() {
    updateNumProgressiveRenderSteps();
    preview_render_progression = render_progression_steps;
    *cancel_preview_render = true;
  }

  // We will progressively render low-res up to higher res.
  int render_progression_steps = 2;
  // When this is 0, there is no downgrade from the full thing.
  int preview_render_progression = render_progression_steps;

  std::shared_ptr<std::atomic<bool>> cancel_preview_render = std::make_shared<std::atomic<bool>>(false);

  void renderNerfThread() {
    torch::Tensor image_code = torch::zeros({nerf::kImageCodeDim}, {torch::kFloat32}).to(device);

    while (true) {
      std::this_thread::sleep_for(std::chrono::milliseconds(25));  // avoid spamming
      
      const bool should_render = !isAnyModalPopupOpen() && floating_window_preview_render.visible && !timeline.keyframes.empty();
      if (should_render) {
        // Check if we have already reached the highest level of quality for the current
        // camera, and if so, don't render it again.
        if (preview_render_progression < 0) continue; 

        auto curr_cam = getInterpolatedCamera(timeline.curr_frame);

        const int downscale = (1 << preview_render_progression);
        curr_cam.width /= downscale;
        curr_cam.height /= downscale;
        curr_cam.focal_length /= downscale;
        curr_cam.optical_center /= downscale;

        auto render_timer = time::now();

        *cancel_preview_render = false;
        PolymorphicCameraOptions opts;
        opts.cam_type = (VirtualCameraType)virtual_camera_type_choice;
        opts.eqr_width = virtual_camera_eqr_width / downscale;
        opts.eqr_height = virtual_camera_eqr_height / downscale;
        opts.vr180_size = virtual_camera_vr180_size / downscale;
        opts.virtual_stereo_baseline = virtual_stereo_baseline;
        opts.looking_glass_hfov = virtual_camera_hfov;
        opts.looking_glass_downscale = downscale;

        cv::Mat image_mat = renderImageWithNerfPolymorphicCamera(
          opts,
          device,
          radiance_model,
          proposal_model,
          curr_cam,
          image_code,
          nerf_cfg.num_basic_samples,
          std::max(1, nerf_cfg.num_importance_samples / downscale),
          world_transform,
          cancel_preview_render);
        if (image_mat.empty()) continue; // This happens if the render is cancelled.

        // Draw horizon line
        if (virtual_camera_draw_horizon_line && virtual_camera_type_choice == CAM_TYPE_EQUIRECTANGULAR) {
          for(int x = 0; x < image_mat.cols; ++x) {
            image_mat.at<cv::Vec3b>(image_mat.rows/2, x) = cv::Vec3b(0, 0, 255);
          }
        }

        XPLINFO << "render time (sec): " << time::timeSinceSec(render_timer);

        virtual_cam_preview_image.setImage(image_mat);

        if (preview_render_progression >= 0) --preview_render_progression;
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

  void beginBigModalRoundedBox(const char* child_id) {
    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImVec2 box_size(avail.x - getScaledFontSize() * 4, avail.y - getScaledFontSize() * 4);
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

  void createNewProject() {
    const bool ffmpeg_ok = checkFFmpegInstallAndWarnIfNot();
    if (!ffmpeg_ok) {
      project_dir = ""; // Clearing this makes it so that if we exit the ffmpeg screen it goes back to splash
      if (kEnableAutoLogging) xpl::stdoutLogger.stopTextFileLog(); // Any time we no longer have a project directory, we must stop logging.
      setAppState(STATE_CONFIG_FFMPEG);
      return;
    }

    const char* cpath = tinyfd_openFileDialog(
      "Select input video (.mp4, .mov, .mkv)", nullptr, 0, nullptr, nullptr, 0);
    if (cpath != nullptr) {
      source_video_path = std::string(cpath);
      XPLINFO << "source_video_path: " << source_video_path;

      setAppState(STATE_SELECT_PROJECT_DIR);
      is_creating_new_project = true; // This changes where we go after selecting the project dir.
    } else {
      setAppState(STATE_SPLASH);
    }
  }

  void splashScreen() {
    ImVec2 button_size(getScaledFontSize() * 15, getScaledFontSize() * 10);
    const float buttons_width = 2 * button_size.x + 20;
    ImGui::SetCursorPosX((ImGui::GetWindowSize().x - buttons_width) * 0.5f);
    ImGui::SetCursorPosY((ImGui::GetWindowSize().y - button_size.y) * 0.5f);

    if (ImGui::Button("New Project", button_size)) {
      setAppState(STATE_EXPLAIN);
    }
    ImGui::SameLine();
    ImGui::Dummy(ImVec2(0.0f, 20.0f));
    ImGui::SameLine();
    if (ImGui::Button("Open Project", button_size)) {
      setAppState(STATE_SELECT_PROJECT_DIR);
      is_creating_new_project = false;
    }
  }

  void explainScreen() {
    beginModalRoundedBox("##Explain");
    ImGui::BulletText("Choose an input video file to begin (.mp4, .mov, .mkv).");
    ImGui::Dummy(ImVec2(20.0f, 0.0f));
    ImGui::SameLine();
    ImGui::Text("Ideal video is 10 seconds.");
    ImGui::Dummy(ImVec2(20.0f, 0.0f));
    ImGui::SameLine();
    ImGui::Text("Move camera in a square while pointing forward.");

    ImGui::BulletText("Choose a project folder where data will be stored.");
    ImGui::BulletText("Choose processing options, wait for NeRF to train.");
    ImGui::BulletText("Render your 3D scene from the point of view of a new camera.");

    ImGui::Dummy(ImVec2(0.0f, 50.0f));

    ImVec2 button_size(getScaledFontSize() * 38, getScaledFontSize() * 3);
    if (ImGui::Button("Continue", button_size)) {
      createNewProject();
    }

    endModalRoundedBox();
  }

  void hideSfmNerfVizWindows() {
    floating_window_tracking_viz.visible = false;
    floating_window_sfm_cost_plot.visible = false;
    floating_window_nerf_cost_plot.visible = false;
  }

  void runSfmAndNerfPipeline() {
    const std::string sfm_dir = project_dir + "/structure_from_motion";

    sfm_cfg.dest_dir = sfm_dir; 
    sfm_cfg.video_frames_dir = project_dir + "/video_frames";
    sfm_cfg.ffmpeg = ffmpeg;
    rectilinear_sfm::printConfig(sfm_cfg);

    hideSfmNerfVizWindows();

    sfm_gui_data.viz.reset();
    nerf_gui_data.plot_data_x.clear();
    nerf_gui_data.plot_data_y.clear();

    nerf_cfg.train_images_dir = sfm_cfg.video_frames_dir;
    nerf_cfg.train_json = sfm_dir + "/dataset_train.json";
    nerf_cfg.output_dir = project_dir + "/nerf";

    p11::file::createDirectoryIfNotExists(sfm_cfg.video_frames_dir);
    p11::file::createDirectoryIfNotExists(sfm_dir);
    p11::file::createDirectoryIfNotExists(nerf_cfg.output_dir);

    command_runner.setRedirectStdErr(true);
    command_runner.setCompleteCallback([&] {
      need_to_reload_project = true;
      hideSfmNerfVizWindows();
      setAppState(STATE_3D_VIEW);
    });
    command_runner.setKilledCallback([&] {
      hideSfmNerfVizWindows();
      setAppState(STATE_SPLASH);
    });

    const std::string scale_max_dim_maintain_aspect = 
      "scale=" + std::to_string(sfm_cfg.max_image_dim) + ":" + std::to_string(sfm_cfg.max_image_dim)
      + ":force_original_aspect_ratio=decrease";
    std::string ffmpeg_extract_frames = 
      ffmpeg + " -y -progress pipe:1 -i \"" + source_video_path + "\"  -vf \"";
#ifdef __APPLE__
    if (hdr_tonemap_choice == 0) { // iPhone 14
      ffmpeg_extract_frames += "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p,";
    }
#endif
    ffmpeg_extract_frames += scale_max_dim_maintain_aspect + "\" -start_number 0 \"" + sfm_cfg.video_frames_dir + "/frame_%06d.png\"";
    
    auto ffmpeg_progress_parser = [](const std::string& line, CommandProgressDescription& p) {
      std::regex frame_regex("frame=(\\d+)");
      std::smatch matches;
      if (std::regex_search(line, matches, frame_regex) && matches.size() > 1) {
        p.progress_str = "Extracting video frame: " + matches[1].str();
      }
    };

    auto sfm_progress_parser = [&](const std::string& line, CommandProgressDescription& p) {
      if (line == "Phase: Tracking keypoints") {
        floating_window_tracking_viz.visible = true;
      }
      if (string::beginsWith(line, "Phase: Solve structure from motion")) {
        floating_window_sfm_cost_plot.visible = true;
      }
      
      if (line == "Phase: Tracking keypoints" ||
          line == "Phase: Generating tracking and outlier visualization" ||
          line == "Phase: Cleaning up unused images" ||
          string::beginsWith(line, "Phase: Solve structure from motion")    
      ) {
        p.phase = line;
        p.progress_str = p.phase;
        return;
      }

      std::smatch matches;
      if (std::regex_search(line, matches, std::regex("Tracking keypoints, frame: (\\d+) / (\\d+)")) && matches.size() > 2) {
        int curr_frame = std::stoi(matches[1].str());
        int total_frames = std::stoi(matches[2].str());
        p.progress_str = p.phase + ", frame: " + std::to_string(curr_frame) + " / " + std::to_string(total_frames);
        p.frac = static_cast<float>(curr_frame) / total_frames;
      }

      if (string::beginsWith(p.phase, "Phase: Solve structure from motion")) {
        if (std::regex_search(line, matches, std::regex("(\\d+)\\s+\\d*\\.\\d+")) && matches.size() > 1) {
          int curr_iteration = std::stoi(matches[1].str());
          p.progress_str = p.phase + ", iteration: " + std::to_string(curr_iteration) + " / " + std::to_string(sfm_gui_data.ceres_iterations);
          p.frac = static_cast<float>(curr_iteration) / sfm_gui_data.ceres_iterations;
        }
      }
    };

    auto nerf_progress_parser = [&](const std::string& line, CommandProgressDescription& p) {
      if (line == "Phase: Train neural radiance field") {
        p.phase = line;
        p.progress_str = p.phase;
        floating_window_nerf_cost_plot.visible = true;
        return;
      } 
      if (p.phase == "Phase: Train neural radiance field") {
        std::smatch matches;
        if (std::regex_search(line, matches, std::regex("(\\d+)\\s+")) && matches.size() > 1) {
          int curr_iteration = std::stoi(matches[1].str());
          p.progress_str = p.phase + ", iteration: " + std::to_string(curr_iteration) + " / " + std::to_string(nerf_cfg.num_training_itrs);
          p.frac = static_cast<float>(curr_iteration) / nerf_cfg.num_training_itrs;
        }
      }
    };

    if (checkbox_run_extract_frames) {
      command_runner.queueShellCommand(ffmpeg_extract_frames, ffmpeg_progress_parser);
    }

    if (checkbox_run_sfm) {
      command_runner.queueThreadCommand(
        sfm_cfg.cancel_requested,
        [&] { rectilinear_sfm::runRectilinearSfmPipeline(sfm_cfg, &sfm_gui_data); },
        sfm_progress_parser);
    }

    if (checkbox_run_nerf) {
      command_runner.queueThreadCommand(
        nerf_cfg.cancel_requested,
        [&] { nerf::runNerfPipeline(nerf_cfg, &nerf_gui_data); },
        nerf_progress_parser);
    }

    command_runner.runCommandQueue();
  }

  void setSfmAndNerfDefaults() {
    hdr_tonemap_choice = 0;
    sfm_cfg.cancel_requested          = std::make_shared<std::atomic<bool>>(false);
    sfm_cfg.src_vid                   = ""; // We're running ffmpeg separately, unused
    sfm_cfg.video_frames_dir          = ""; // Filled in right before calling SFM
    sfm_cfg.dest_dir                  = ""; // Filled in right before calling SFM
    sfm_cfg.max_image_dim             = 640;
    sfm_cfg.num_train_frames          = 30;
    sfm_cfg.num_test_frames           = 10;
    sfm_cfg.outer_iterations          = 2;
    sfm_cfg.ceres_iterations          = 25;
    sfm_cfg.ceres_iterations2         = 5;
    sfm_cfg.outlier_percentile        = 0.8;
    sfm_cfg.outlier_weight_steepness  = 2.0;
    sfm_cfg.rm_unused_images          = true;
    sfm_cfg.no_ffmpeg                 = true;

    nerf_cfg.cancel_requested         = nullptr; // TODO this should be initialized so cancelling can work
    nerf_cfg.train_images_dir         = ""; // Filled in just before running pipeline
    nerf_cfg.train_json               = ""; // Filled in just before running pipeline
    nerf_cfg.output_dir               = ""; // Filled in just before running pipeline
    nerf_cfg.load_model_dir           = ""; // Empty means don't load a model
    nerf_cfg.num_training_itrs        = 5000;
    nerf_cfg.rays_per_batch           = 4096;
    nerf_cfg.num_basic_samples        = 128;
    nerf_cfg.num_importance_samples   = 64;
    nerf_cfg.warmup_itrs              = 100;
    nerf_cfg.num_novel_views          = 0;
    nerf_cfg.compute_train_psnr       = false;
    nerf_cfg.radiance_lr              = 1e-2;
    nerf_cfg.radiance_decay           = 1e-8;
    nerf_cfg.image_code_lr            = 1e-4;
    nerf_cfg.image_code_decay         = 1e-4;
    nerf_cfg.prop_lr                  = 1e-2;
    nerf_cfg.prop_decay               = 1e-6;
    nerf_cfg.adam_eps                 = 1e-17;
    nerf_cfg.floater_min_dist         = 1.0;
    nerf_cfg.floater_weight           = 1e-3;
    nerf_cfg.gini_weight              = 1e-5;
    nerf_cfg.distortion_weight        = 1e-2;
    nerf_cfg.density_weight           = 1e-6;
    nerf_cfg.far_away_weight          = 1e-4;
    nerf_cfg.visibility_weight        = 1e-4;
    nerf_cfg.num_visibility_points    = 1024;
    nerf_cfg.prev_density_weight      = 1e-4;
    nerf_cfg.prev_reg_num_samples     = 32;
    nerf_cfg.resize_max_dim           = 0; // 0 means dont resize
    nerf_cfg.show_epipolar_viz        = false; // This will crash if true due to using cv::imshow!
  }

  void setSfmAndNerfHDSettings() {
    setSfmAndNerfDefaults();
    sfm_cfg.max_image_dim             = 2048;
    nerf_cfg.num_training_itrs        = 10000;
    nerf_cfg.rays_per_batch           = 8192;
    nerf_cfg.num_basic_samples        = 256;
    nerf_cfg.num_importance_samples   = 128;
  }

  void renderVideoToFiles() {
    const std::string render_dir = project_dir + "/render_frames";
    XPLINFO << "Creating directory " << render_dir;
    p11::file::createDirectoryIfNotExists(render_dir);
    p11::file::clearDirectoryContents(render_dir);

    torch::Tensor image_code = torch::zeros({nerf::kImageCodeDim}, {torch::kFloat32}).to(device);
    
    for (int frame_idx = 0; frame_idx < timeline.num_frames; ++frame_idx) {
      if (*render_video_cancel_requested) return;

      XPLINFO << "Rendering frame " << frame_idx << " / " << timeline.num_frames;
      auto curr_cam = getInterpolatedCamera(frame_idx);

      PolymorphicCameraOptions opts;
      opts.cam_type = (VirtualCameraType)virtual_camera_type_choice;
      opts.eqr_width = virtual_camera_eqr_width;
      opts.eqr_height = virtual_camera_eqr_height;
      opts.vr180_size = virtual_camera_vr180_size;
      opts.virtual_stereo_baseline = virtual_stereo_baseline;
      opts.looking_glass_hfov = virtual_camera_hfov;
      opts.looking_glass_downscale = 1;
  
      cv::Mat image_mat = renderImageWithNerfPolymorphicCamera(
        opts,
        device,
        radiance_model,
        proposal_model,
        curr_cam,
        image_code,
        nerf_cfg.num_basic_samples,
        nerf_cfg.num_importance_samples,
        world_transform,
        render_video_cancel_requested);

      if (image_mat.empty()) return; // This happens if the render is cancelled.
      if (*render_video_cancel_requested) return;

      cv::imwrite(render_dir + "/" + string::intToZeroPad(frame_idx, 6) + ".png", image_mat);
    }
  }

  std::shared_ptr<std::atomic<bool>> render_video_cancel_requested = std::make_shared<std::atomic<bool>>(false);
  void runRenderVideoCommand() {
    if (!checkFFmpegInstallAndWarnIfNot()) {
      return;
    }

    const std::string render_dir = project_dir + "/render_frames";
    p11::file::createDirectoryIfNotExists(render_dir);
    p11::file::clearDirectoryContents(render_dir);

    command_runner.setRedirectStdErr(true);
    command_runner.setKilledCallback([&] { setAppState(STATE_3D_VIEW); });
    command_runner.setCompleteCallback([&] {
      setAppState(STATE_3D_VIEW);

      if (tinyfd_messageBox(
          "Render Video Complete",
          "Output files are in the project directory.\nDo you want to open the folder?",
          "okcancel", "question", 1)) {
        file::openFileExplorer(project_dir);
      }
    });

    auto render_progress_parser = [](const std::string& line, CommandProgressDescription& p) {
      std::regex frame_regex("Rendering frame (\\d+) / (\\d+)");
      std::smatch matches;
      if (std::regex_search(line, matches, frame_regex) && matches.size() == 3) {
        int curr_frame = std::stoi(matches[1].str());
        int total_frames = std::stoi(matches[2].str());
        p.progress_str = "Rendering frame: " + std::to_string(curr_frame) + " / " + std::to_string(total_frames);
        p.frac = static_cast<float>(curr_frame) / total_frames;
      }
    };

    command_runner.queueThreadCommand(
      render_video_cancel_requested,
      [&] { renderVideoToFiles(); },
      render_progress_parser);

    if (checkbox_encode_h264) {
      auto progress_parser = [](const std::string& line, CommandProgressDescription& p) {
        p.progress_str = "Encoding h264";
      };
      std::string output_name = p11::file::createTimestampFilename(project_dir, "render_h264", "mp4");
      command_runner.queueShellCommand(
          ffmpeg + " -y -framerate 30 -i \"" + render_dir + "/%06d.png\" "
          "-progress pipe:1 -c:v libx264 -preset fast -crf 24 -pix_fmt yuv420p -movflags faststart \"" +
          output_name + "\"",
          progress_parser);
    }

    if (checkbox_encode_h265) {
      auto progress_parser = [](const std::string& line, CommandProgressDescription& p) {
        p.progress_str = "Encoding h265";
      };
      std::string output_name = p11::file::createTimestampFilename(project_dir, "render_h265", "mp4");
      command_runner.queueShellCommand(
          ffmpeg + " -y -framerate 30 -i \"" + render_dir + "/%06d.png\" "
          "-progress pipe:1 -c:v libx265 -preset fast -crf 27 -pix_fmt yuv420p -movflags faststart \"" +
          output_name + "\"",
          progress_parser);
    }

    if (checkbox_encode_prores) {
      auto progress_parser = [](const std::string& line, CommandProgressDescription& p) {
        p.progress_str = "Encoding ProRes";
      };
      std::string output_name = p11::file::createTimestampFilename(project_dir, "render_prores", "mov");
      command_runner.queueShellCommand(
          ffmpeg + " -y -framerate 30 -i \"" + render_dir + "/%06d.png\" "
          "-progress pipe:1 -c:v prores_ks -profile:v 1 -vendor apl0 \"" +
          output_name + "\"",
          progress_parser);
    }

    command_runner.runCommandQueue();
  }

  void sameLineHorizontalSpace() {
    ImGui::SameLine();
    ImGui::Dummy(ImVec2(20.0f, 0.0f));
    ImGui::SameLine();
  }

  void sfmAndNerfProcessingSettingsScreen() {
    beginBigModalRoundedBox("##SFM_NERF_CFG");

    ImVec2 button_size(getScaledFontSize() * 10, getScaledFontSize() * 3);

    if (ImGui::Button("Cancel", button_size)) {
      setAppState(STATE_SPLASH);
    }

    sameLineHorizontalSpace();

    if (ImGui::Button("Restore Defaults", button_size)) {
      setSfmAndNerfDefaults();
    }

    sameLineHorizontalSpace();

    if (ImGui::Button("HD Settings", button_size)) {
      setSfmAndNerfHDSettings();
    }

    sameLineHorizontalSpace();
  
    if (ImGui::Button("Start Processing", button_size)) {
      runSfmAndNerfPipeline();
    }

    ImGui::Dummy(ImVec2(0.0f, 60.0f));
    ImGui::SetWindowFontScale(1.3); 
    ImGui::Text("Extract Video Frames as Images");
    ImGui::SetWindowFontScale(1.0); 
    ImGui::Dummy(ImVec2(0.0f, 20.0f));

    const float input_offset = getScaledFontSize() * 20; // Space for the label
    const float input_width = getScaledFontSize() * 10;

    ImGui::Text("Resize Max Width or Height");
    ImGui::SameLine();
    ImGui::SetCursorPosX(input_offset);
    ImGui::PushItemWidth(input_width);
    ImGui::InputInt("##654", &sfm_cfg.max_image_dim, 0);
    ImGui::PopItemWidth();

    ImGui::Text("HDR Tonemap");
    ImGui::SameLine();
    ImGui::SetCursorPosX(input_offset);
    ImGui::PushItemWidth(input_width);
    const char* hdr_tonemap_options[] = { "iPhone 14", "None" };
    ImGui::GetStyle().WindowPadding = ImVec2(0, 0); // Without this, some earlier stuff makes the popup look weird
    ImGui::Combo("##HDRTonemap", &hdr_tonemap_choice, hdr_tonemap_options, IM_ARRAYSIZE(hdr_tonemap_options));
    ImGui::PopItemWidth();

    ImGui::Text("Delete Unused Images");
    ImGui::SameLine();
    ImGui::SetCursorPosX(input_offset);
    ImGui::PushItemWidth(input_width);
    ImGui::Checkbox("##6345", &sfm_cfg.rm_unused_images);
    ImGui::PopItemWidth();

    ImGui::Dummy(ImVec2(0.0f, 60.0f));
    ImGui::SetWindowFontScale(1.3);
    ImGui::Text("Structure from Motion");
    ImGui::SetWindowFontScale(1);
    ImGui::Dummy(ImVec2(0.0f, 20.0f));

    ImGui::Text("# Iterations");
    ImGui::SameLine();
    ImGui::SetCursorPosX(input_offset);
    ImGui::PushItemWidth(input_width);
    ImGui::InputInt("##6545", &sfm_cfg.ceres_iterations, 0);
    ImGui::SameLine();
    ImGui::Text(" 2nd pass: ");
    ImGui::SameLine();
    ImGui::InputInt("##253234", &sfm_cfg.ceres_iterations2, 0);
    ImGui::PopItemWidth();
    
    ImGui::Text("Outlier Percentile");
    ImGui::SameLine();
    ImGui::SetCursorPosX(input_offset);
    ImGui::PushItemWidth(input_width);
    ImGui::InputDouble("##5341", &sfm_cfg.outlier_percentile);
    ImGui::PopItemWidth();
  

    ImGui::Dummy(ImVec2(0.0f, 60.0f));
    ImGui::SetWindowFontScale(1.3);
    ImGui::Text("Neural Radiance Field");
    ImGui::SetWindowFontScale(1.0);
    ImGui::Dummy(ImVec2(0.0f, 20.0f));

    ImGui::Text("Training Iterations");
    ImGui::SameLine();
    ImGui::SetCursorPosX(input_offset);
    ImGui::PushItemWidth(input_width);
    ImGui::InputInt("##6333345", &nerf_cfg.num_training_itrs, 0);
    ImGui::PopItemWidth();

    ImGui::Text("Rays Per Batch");
    ImGui::SameLine();
    ImGui::SetCursorPosX(input_offset);
    ImGui::PushItemWidth(input_width);
    ImGui::InputInt("##37345", &nerf_cfg.rays_per_batch, 0);
    ImGui::PopItemWidth();

    ImGui::Text("Basic Samples Per Ray");
    ImGui::SameLine();
    ImGui::SetCursorPosX(input_offset);
    ImGui::PushItemWidth(input_width);
    ImGui::InputInt("##1958", &nerf_cfg.num_basic_samples, 0);
    ImGui::PopItemWidth();

    ImGui::Text("Importance Samples Per Ray");
    ImGui::SameLine();
    ImGui::SetCursorPosX(input_offset);
    ImGui::PushItemWidth(input_width);
    ImGui::InputInt("##28394", &nerf_cfg.num_importance_samples, 0);
    ImGui::PopItemWidth();

    ImGui::Text("Min Dist");
    ImGui::SameLine();
    ImGui::SetCursorPosX(input_offset);
    ImGui::PushItemWidth(input_width);
    ImGui::InputDouble("##36452", &nerf_cfg.floater_min_dist);
    ImGui::PopItemWidth();

    endModalRoundedBox();
  }

  bool checkbox_encode_h264 = true;
  bool checkbox_encode_h265 = false;
  bool checkbox_encode_prores = false;
  void renderVideoSettingsScreen() {
    beginBigModalRoundedBox("##RENDER_NERF_CFG");

    ImVec2 button_size(getScaledFontSize() * 10, getScaledFontSize() * 3);

    if (ImGui::Button("Cancel", button_size)) {
      setAppState(STATE_3D_VIEW);
    }

    sameLineHorizontalSpace();

    //if (ImGui::Button("Restore Defaults", button_size)) {} // TODO?

    sameLineHorizontalSpace();
  
    if (ImGui::Button("Start Rendering", button_size)) {
      runRenderVideoCommand();
    }

/*
    ImGui::Dummy(ImVec2(0.0f, 60.0f));
    ImGui::SetWindowFontScale(1.3); 
    ImGui::Text("Radiance Field Sampling");
    ImGui::SetWindowFontScale(1.0); 
    ImGui::Dummy(ImVec2(0.0f, 20.0f));
    ImGui::Text("TODO: # of samples options here?");
*/

    ImGui::Dummy(ImVec2(0.0f, 60.0f));
    ImGui::SetWindowFontScale(1.3); 
    ImGui::Text("Virtual Camera Settings");
    ImGui::SetWindowFontScale(1.0); 
    ImGui::Dummy(ImVec2(0.0f, 20.0f));

    drawVirtualCameraSettingsWithoutWindow(/*max_width=*/600);

    ImGui::Dummy(ImVec2(0.0f, 60.0f));
    ImGui::SetWindowFontScale(1.3); 
    ImGui::Text("Encode Video");
    ImGui::SetWindowFontScale(1.0); 
    ImGui::Dummy(ImVec2(0.0f, 20.0f));

    ImGui::Checkbox("  Encode h264", &checkbox_encode_h264);
    ImGui::Checkbox("  Encode h265", &checkbox_encode_h265);
    ImGui::Checkbox("  Encode ProRes", &checkbox_encode_prores);

    endModalRoundedBox();
  }

  bool checkbox_ldi_transparent_bg = false;
  int ldi_export_resolution = 1920;
  void exportLdiSettingsScreen() {
    beginBigModalRoundedBox("##EXPORT_LDI_CFG");

    ImVec2 button_size(getScaledFontSize() * 10, getScaledFontSize() * 3);

    if (ImGui::Button("Cancel", button_size)) {
      setAppState(STATE_3D_VIEW);
    }

    sameLineHorizontalSpace();
  
    if (ImGui::Button("Start Rendering", button_size)) {
      runExportCommand("ldi3");
    }

    ImGui::Dummy(ImVec2(0.0f, 60.0f));
    ImGui::SetWindowFontScale(1.3); 
    ImGui::Text("Layered Depth Image (LDI3) Settings");
    ImGui::SetWindowFontScale(1.0); 
    ImGui::Dummy(ImVec2(0.0f, 20.0f));

    const float input_offset = getScaledFontSize() * 20; // Space for the label
    const float input_width = getScaledFontSize() * 10;
    ImGui::Text("Resolution Per Layer");
    ImGui::SameLine();
    ImGui::SetCursorPosX(input_offset);
    ImGui::PushItemWidth(input_width);
    ImGui::InputInt("##23473235", &ldi_export_resolution, 0);
    ImGui::PopItemWidth();

    ImGui::Checkbox("  Transparent Background", &checkbox_ldi_transparent_bg);

    endModalRoundedBox();
  }

  void exportMeshSettingsScreen() {
    beginBigModalRoundedBox("##EXPORT_MESH_CFG");

    ImVec2 button_size(getScaledFontSize() * 10, getScaledFontSize() * 3);

    if (ImGui::Button("Cancel", button_size)) {
      setAppState(STATE_3D_VIEW);
    }

    sameLineHorizontalSpace();
  
    if (ImGui::Button("Export (obj)", button_size)) {
      runExportCommand("obj");
    }

    ImGui::Dummy(ImVec2(0.0f, 60.0f));
    ImGui::SetWindowFontScale(1.3); 
    ImGui::Text("Mesh Export Settings");
    ImGui::SetWindowFontScale(1.0); 
    ImGui::Dummy(ImVec2(0.0f, 20.0f));

    const float input_offset = getScaledFontSize() * 20; // Space for the label
    const float input_width = getScaledFontSize() * 10;
    ImGui::Text("Resolution Per Layer");
    ImGui::SameLine();
    ImGui::SetCursorPosX(input_offset);
    ImGui::PushItemWidth(input_width);
    ImGui::InputInt("##23473235", &ldi_export_resolution, 0);
    ImGui::PopItemWidth();

    ImGui::Checkbox("  Transparent Background", &checkbox_ldi_transparent_bg);

    endModalRoundedBox();
  }

  ~VoluramaApp() {
    virtual_cam_preview_image.freeGlTexture();
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
    if (curr_mouse_x >= preview3d_viewport_min.x && curr_mouse_y >= preview3d_viewport_min.y &&
        curr_mouse_x <= preview3d_viewport_min.x + preview3d_viewport_size.x &&
        curr_mouse_y <= preview3d_viewport_min.y + preview3d_viewport_size.y) {
      mouse_in_3d_viewport = true;
    }

    mouse_in_timeline = false;
    if (curr_mouse_x >= timeline.viewport_min.x && curr_mouse_y >= timeline.viewport_min.y &&
        curr_mouse_x <= timeline.viewport_min.x + timeline.viewport_size.x &&
        curr_mouse_y <= timeline.viewport_min.y + timeline.viewport_size.y) {
      mouse_in_timeline = true;
    }

    if (isAnyModalPopupOpen() || main_menu_is_hovered) {
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
    restoreDefaultCursor();

    ImGui::GetIO().ConfigFlags &= ~ImGuiConfigFlags_NoMouseCursorChange;

    if (isAnyModalPopupOpen()) return;
    if (main_menu_is_hovered) return;
  }

  void resetCamera()
  {
    camera_radius = 3.0;
    camera_theta = M_PI;
    camera_phi = M_PI / 2.0;

    camera_orbit_center = Eigen::Vector3f(0, 0, 0);

    offset_cam_right = 0;
    offset_cam_up = 0;
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

    // Setup projection matrix
    static constexpr float kVerticalFovDeg = 80;
    static constexpr float kZNear = 0.1;
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

    // Compute the model-view-projection matrix that will be used in shaders.
    // By default, the model matrix is identity and can be omitted.
    model_view_projection_matrix = projection_matrix * view_matrix;

    // Update camera state from keyboard. We do this last because we need forward/right/up as
    // calculated above
    ImGuiIO& io = ImGui::GetIO();
    if (mouse_in_3d_viewport) {
      static constexpr float kCamSpeed = 0.01;
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

  void drawGlStuff()
  {
    GL_CHECK_ERROR;

    updateCamera();

    const float s = ImGui::GetIO().DisplayFramebufferScale.x;  // For Retina display

    constexpr int kMenuBarHeight = 20;
    glViewport(
        s * preview3d_viewport_min.x,
        s * (preview3d_viewport_min.y - kMenuBarHeight + VideoTimelineWidget::kTimelineHeight),
        s * preview3d_viewport_size.x,
        s * preview3d_viewport_size.y);

    //glEnable(GL_DEPTH_TEST);
    //glDepthMask(GL_TRUE);
    //glDepthFunc(GL_LEQUAL);

    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT,  GL_NICEST);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    update3DGuiVertexData();

    updateWorldTransform();
    // Construct a different MVP for the point cloud only (not the rest of the 3D UI),
    // which incorporates the World Transform Editor's scaling and rotation.
    Eigen::Matrix4f pointcloud_mvp = projection_matrix * view_matrix * world_transform.cast<float>();

    // Draw point cloud
    pointcloud_vb.bind();
    basic_shader.bind();
    glUniformMatrix4fv(
        basic_shader.getUniform("uModelViewProjectionMatrix"),
        1,
        false,
        pointcloud_mvp.data());
    glPointSize(4);
    glDrawArrays(GL_POINTS, 0, pointcloud_vb.vertex_data.size());

    // Draw lines for axes
    lines_vb.bind();
    glUniformMatrix4fv(
        basic_shader.getUniform("uModelViewProjectionMatrix"),
        1,
        false,
        model_view_projection_matrix.data());
    glDrawArrays(GL_LINES, 0, lines_vb.vertex_data.size());

    // Draw points that aren't part of the pointcloud (e.g., for GUI purpose)
    points_vb.bind();
    glPointSize(10);
    glDrawArrays(GL_POINTS, 0, points_vb.vertex_data.size());

    GL_CHECK_ERROR;
  }

  // TODO: it is redundant to have this keyboard handler and a lower-level GLfw handler.
  void handleKeyboard()
  {
    ImGuiIO& io = ImGui::GetIO();
    int delta = io.KeysDown[GLFW_KEY_LEFT_SHIFT] ? 10 : 1;
    if (io.KeysDown[GLFW_KEY_LEFT]) moveTimelineByDelta(-delta);
    if (io.KeysDown[GLFW_KEY_RIGHT]) moveTimelineByDelta(delta);
  }

  void moveTimelineByDelta(int delta)
  {
    if (project_dir.empty()) return;

    timeline.curr_frame += delta;
    if (timeline.curr_frame < 0) timeline.curr_frame = 0;
    if (timeline.curr_frame >= timeline.num_frames) {
      timeline.curr_frame = timeline.num_frames - 1;
    }
    timeline.curr_frame_change_callback();
  }

  void makeVirtualCameraPreset(const std::string& preset) {
    timeline.keyframes.clear();
    addKeyframe(0);
    if (preset != "origin") { 
      addKeyframe(timeline.num_frames - 1);
    }
    if (preset == "pan_right") {
      timeline.keyframes[0].tx = -0.5;
      timeline.keyframes[timeline.num_frames - 1].tx = 0.5;
    }
    if (preset == "pan_left") {
      timeline.keyframes[0].tx = 0.5;
      timeline.keyframes[timeline.num_frames - 1].tx = -0.5;
    }
    if (preset == "dolly_forward") {
      timeline.keyframes[0].tz = -0.5;
      timeline.keyframes[timeline.num_frames - 1].tz = 0.5;
    }
    if (preset == "dolly_backward") {
      timeline.keyframes[0].tz = 0.5;
      timeline.keyframes[timeline.num_frames - 1].tz = -0.5;
    }
    if (preset == "boom_up") {
      timeline.keyframes[0].ty = -0.5;
      timeline.keyframes[timeline.num_frames - 1].ty = 0.5;
    }
    if (preset == "boom_down") {
      timeline.keyframes[0].ty = 0.5;
      timeline.keyframes[timeline.num_frames - 1].ty = -0.5;
    }
    if (preset == "orbit") {
      timeline.keyframes.clear();
      for (int i = 0; i < timeline.num_frames; ++i) {
        const float alpha = float(i) / timeline.num_frames;
        const float theta_deg = alpha * orbit_editor_start_theta + (1.0 - alpha) * orbit_editor_end_theta;
        const float theta = (M_PI / 180.0) * theta_deg;

        addKeyframe(i);
        auto& kf = timeline.keyframes.at(i);
        kf.tx = orbit_editor_radius * cos(theta);
        kf.ty = 0;
        kf.tz = orbit_editor_radius * sin(theta);
        kf.rx = 0;
        kf.ry = 90 + (180 / M_PI) * atan2(kf.tz, kf.tx); // Calculate yaw to face the origin, in degrees
        kf.rz = 0;
        kf.tx += orbit_editor_center_x;
        kf.ty += orbit_editor_center_y;
        kf.tz += orbit_editor_center_z;
      }
    }
    refreshPreviewRender();
  }

  void resizeTimeline() {
    const char* new_num_frames_cstr = tinyfd_inputBox(
      "Change Timeline Duration",
      "Enter new number of frames in timeline",
      "300");
    if (new_num_frames_cstr == nullptr) return; // cancelled
    int new_num_frames = std::atoi(new_num_frames_cstr);
    if (new_num_frames < 1) return;

    timeline.num_frames = new_num_frames;
    timeline.curr_frame = 0;
    timeline.curr_frame_change_callback();
    
    // Erase any keyframes past the end of the new timeline
    auto it = timeline.keyframes.begin();
    while (it != timeline.keyframes.end()) {
      if (it->first > new_num_frames) {
        it = timeline.keyframes.erase(it);  // erase returns the next valid iterator
      } else {
        ++it;
      }
    }
  }

  void setNumberOfPointsInPointCloudAndResample() {
    constexpr const char* kPrefKey = "point_cloud_num_samples";
    std::string default_pc_size = "1000000";
    if (prefs.count(kPrefKey)) {
      default_pc_size = prefs.at(kPrefKey);
    }
    const char* pc_size_str = tinyfd_inputBox(
      "Set Number of Points for Point Cloud",
      "Enter a number between 1000 and 10,000,000",
      default_pc_size.c_str());
    if (pc_size_str == nullptr) return; // cancelled
    int pc_size = std::atoi(pc_size_str);
    if (pc_size < 1000 || pc_size > 10000000) return;

    point_cloud_num_samples = pc_size;

    prefs[kPrefKey] = std::to_string(point_cloud_num_samples);
    preferences::setPrefs(prefs);

    samplePointCloudFromNerf();
  }

  void exportPointCloud(const std::string ext) {
    const std::string default_path = project_dir + "/pointcloud." + ext;
    const char* dest_path = tinyfd_saveFileDialog("Select output path", default_path.c_str(), 0, nullptr, nullptr);
    if (dest_path == nullptr) return;

    if (ext == "csv") {
      point_cloud::savePointCloudCsv(dest_path, point_cloud, point_cloud_colors);
    }

    if (ext == "pcd") {
      point_cloud::savePointCloudPCL(dest_path, point_cloud, point_cloud_colors);
    }
  }

  std::shared_ptr<std::atomic<bool>> export_cancel_requested;
  calibration::FisheyeCamerad ldi_cam; // so it doesn't go out of scope
  cv::Mat ldi_vignette;
  void runExportCommand(const std::string& export_type) {
    std::string default_path = project_dir + "/";
    if (export_type == "ldi3") default_path += "ldi3.png";
    if (export_type == "obj") default_path += "mesh.obj";
    const char* dest_path = tinyfd_saveFileDialog("Select output path", default_path.c_str(), 0, nullptr, nullptr);
    if (dest_path == nullptr) return;

    command_runner.setRedirectStdErr(true);
    command_runner.setCompleteCallback([&] { setAppState(STATE_3D_VIEW); });
    command_runner.setKilledCallback([&] { setAppState(STATE_3D_VIEW); });
    auto progress_parser = [&](const std::string& line, CommandProgressDescription& p) {
      std::smatch matches;
      if (std::regex_search(line, matches, std::regex("y=(\\d+)")) && matches.size() == 2) {
        int current_y = std::stoi(matches[1].str());
        float frac = static_cast<float>(current_y) / (ldi_cam.height - 1);

        std::ostringstream percentage;
        percentage << std::fixed << std::setprecision(2) << (frac * 100.0);
        p.frac = frac;
        p.progress_str = "Rendering layered depth image: " + percentage.str() + "%%"; // Two %% for ImGui::Text
      }
    };

    export_cancel_requested = std::make_shared<std::atomic<bool>>(false);
    command_runner.queueThreadCommand(
      export_cancel_requested,
      [&, dest_path, export_type] {
        // TODO: several of these parameters should be configurable before rendering
        // Setup for baking an LDI
        constexpr float kFthetaScale = 1.15;
        ldi_cam.width = ldi_export_resolution;
        ldi_cam.height = ldi_export_resolution;
        ldi_cam.radius_at_90 = kFthetaScale * ldi_cam.width/2;
        ldi_cam.optical_center = Eigen::Vector2d(ldi_cam.width / 2.0, ldi_cam.height / 2.0);

        // Compute vignette
        ldi_vignette = projection::makeVignetteMap(
          ldi_cam, cv::Size(ldi_cam.width, ldi_cam.height),
          0.85, 0.86, 0.01, 0.02);

        torch::Tensor image_code = torch::zeros({nerf::kImageCodeDim}, {torch::kFloat32}).to(device);

        constexpr int kNumBasicSamples = 256;
        constexpr int kNumImportanceSamples = 128;
        constexpr float kInverseDepthCoef = 0.3;
        if (export_type == "ldi3") {
          cv::Mat fused_bgra;
          cv::Mat ldi3_image = distillNerfToLdi3(
            project_dir,
            device,
            radiance_model,
            proposal_model,
            ldi_cam,
            ldi_vignette,
            image_code,
            kNumBasicSamples,
            kNumImportanceSamples,
            checkbox_ldi_transparent_bg,
            fused_bgra,
            cv::Mat(),
            world_transform,
            export_cancel_requested);
          // If the user cancelled, the output is empty
          if (!ldi3_image.empty()) cv::imwrite(dest_path, ldi3_image);
        }
        if (export_type == "obj") {
          cv::Mat fused_bgra;
          std::vector<cv::Mat> layer_bgra, layer_invd;
          distillNerfToLdi3(
            project_dir,
            device,
            radiance_model,
            proposal_model,
            ldi_cam,
            ldi_vignette,
            image_code,
            kNumBasicSamples,
            kNumImportanceSamples,
            checkbox_ldi_transparent_bg,
            fused_bgra,
            world_transform,
            export_cancel_requested,
            layer_bgra,
            layer_invd);
          
          XPLINFO << "Generating obj...";
          vr180::ConvertToOBJConfig obj_cfg;
          obj_cfg.output_obj = dest_path;
          obj_cfg.ftheta_size = ldi_cam.width;
          obj_cfg.ftheta_scale = 1.15;
          obj_cfg.ftheta_inflation = 3.0;
          obj_cfg.inv_depth_encoding_coef = kInverseDepthCoef;
          vr180::writeTexturedMeshObj(obj_cfg, layer_bgra, layer_invd);
        }
      },
      progress_parser);
    command_runner.runCommandQueue();
  }

  std::string formatVersionNumber(float v) {
    std::string ver = std::to_string(v);
    ver.erase(ver.find_last_not_of('0') + 1, std::string::npos);
    ver.erase(ver.find_last_not_of('.') + 1, std::string::npos);    
    return ver;
  }

  void showAboutVoluramaDialog() {
    tinyfd_messageBox(
      "Volurama by Lifecast",
      ("Version: " + formatVersionNumber(kSoftwareVersion) +
      "\nTorch Device: " + util_torch::deviceTypeToString(device)).c_str(), "ok", "info", 1);
  }

  bool main_menu_is_hovered = false;
  void drawMainMenu()
  {
    if (ImGui::BeginMenuBar()) {
      // Add some more padding around the menu items... otherwise it looks wack.
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(30, 30));
      ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10, 10));

      if (ImGui::BeginMenu(" File ")) {
        ImGui::Dummy(ImVec2(0, 8));

        if (ImGui::MenuItem("New Project")) {
          setAppState(STATE_EXPLAIN);
        }

        if (ImGui::MenuItem("Open Project")) {
          setAppState(STATE_SELECT_PROJECT_DIR);
          is_creating_new_project = false;
        }

        ImGui::Dummy(ImVec2(0, 8));
        ImGui::Separator();
        ImGui::Dummy(ImVec2(0, 8));

        if (ImGui::MenuItem("Export Triangle Mesh (obj)")) {
          setAppState(STATE_EXPORT_MESH_CONFIG);
        }

        if (ImGui::MenuItem("Export Layered Depth Image (ldi3)")) {
          setAppState(STATE_EXPORT_LDI_CONFIG);
        }
        
        if (ImGui::MenuItem("Render Video with Moving Camera (mp4)")) {
          setAppState(STATE_RENDER_VID_CONFIG);
        }

        ImGui::EndMenu();
      }
      if (ImGui::BeginMenu(" View ")) {
        ImGui::Dummy(ImVec2(0, 8));

        ImGui::MenuItem("Show Origin", nullptr, &show_origin_in_3d_view);
        ImGui::MenuItem("Show Estimated Camera Path", nullptr, &show_sfm_cameras_in_3d_view);
        ImGui::MenuItem("Show Virtual Camera Path", nullptr, &show_virtual_cameras_in_3d_view);

        if (ImGui::MenuItem("Reset Camera")) {
          resetCamera();
        }

        if (ImGui::MenuItem("Reset Window Layout")) {
          resetWindowLayout();
        }

        ImGui::EndMenu();
      }


      if (ImGui::BeginMenu(" Virtual Camera ")) {
        ImGui::Dummy(ImVec2(0, 8));

        if (ImGui::MenuItem("Show Preview Render", nullptr, &floating_window_preview_render.visible)) {}
        if (ImGui::MenuItem("Show Virtual Camera Settings", nullptr, &floating_window_virtual_cam_settings.visible)) {}
        if (ImGui::MenuItem("Show Keyframe Editor", nullptr, &floating_window_keyframe_editor.visible)) {}
        if (ImGui::MenuItem("Show Orbit Editor", nullptr, &floating_window_orbit_editor.visible)) {}
        if (ImGui::MenuItem("Show World Transform", nullptr, &floating_window_world_transform.visible)) {}

        if (ImGui::MenuItem("Change Timeline Duration")) {
          resizeTimeline();
        }

        ImGui::Dummy(ImVec2(0, 8));
        ImGui::Separator();
        ImGui::Dummy(ImVec2(0, 8));

        ImGui::MenuItem("Motion Presets", NULL, false, false);
        ImGui::Dummy(ImVec2(0, 8));
        if (ImGui::MenuItem("  No Motion (Clear All Keyframes)")) {
          makeVirtualCameraPreset("origin");
        }
        if (ImGui::MenuItem("  Pan Right")) {
          makeVirtualCameraPreset("pan_right");
        }
        if (ImGui::MenuItem("  Pan Left")) {
          makeVirtualCameraPreset("pan_left");
        }
        if (ImGui::MenuItem("  Dolly Forward")) {
          makeVirtualCameraPreset("dolly_forward");
        }
        if (ImGui::MenuItem("  Dolly Backward")) {
          makeVirtualCameraPreset("dolly_backward");
        }
        if (ImGui::MenuItem("  Boom Up")) {
          makeVirtualCameraPreset("boom_up");
        }
        if (ImGui::MenuItem("  Boom Down")) {
          makeVirtualCameraPreset("boom_down");
        }
        if (ImGui::MenuItem("  Orbit")) {
          floating_window_orbit_editor.visible = true;
          floating_window_orbit_editor.first_time_showing = true;
        }
        ImGui::EndMenu();
      }

      if (ImGui::BeginMenu(" Point Cloud ")) {
        ImGui::Dummy(ImVec2(0, 8));

        if (ImGui::MenuItem("Set Number of Points")) {
          setNumberOfPointsInPointCloudAndResample();
        }

        if (ImGui::MenuItem("Export .pcd")) {
          exportPointCloud("pcd");
        }
        
        if (ImGui::MenuItem("Export .csv")) {
          exportPointCloud("csv");
        }

        ImGui::EndMenu();
      }

      if (ImGui::BeginMenu(" Settings ")) {
        ImGui::Dummy(ImVec2(0, 8));

        if (ImGui::MenuItem("About Volurama")) {
          showAboutVoluramaDialog();
        }

        ImGui::Dummy(ImVec2(0, 8));
        ImGui::Separator();
        ImGui::Dummy(ImVec2(0, 8));

        if (ImGui::MenuItem("Configure ffmpeg")) {
          setAppState(STATE_CONFIG_FFMPEG);
        }
        if (ImGui::MenuItem("Check ffmpeg installation")) {
          if (checkFFmpegInstallAndWarnIfNot()) {
            tinyfd_messageBox(
                "ffmpeg is installed OK",
                "ffmpeg appears to be installed correctly.",
                "ok",
                "info",
                1);
          }
        }

        ImGui::Dummy(ImVec2(0, 8));
        ImGui::Separator();
        ImGui::Dummy(ImVec2(0, 8));

        if (ImGui::MenuItem("Show Console")) {
          consoleSleepCommand();
        }
        if (ImGui::MenuItem("Debug Test 1")) {
          testCommand();
        }
        if (ImGui::MenuItem("Debug Test 2")) {
          testThreadCommand();
        }

        ImGui::Dummy(ImVec2(0, 8));
        ImGui::EndMenu();
      }

      // NOTE: I don't know why this is negated.
      main_menu_is_hovered = !ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows);

      ImGui::PopStyleVar(2);

      ImGui::EndMenuBar();
    }
  }

  bool checkFFmpegInstall(std::string& output) {
    try {
      output = execBlockingWithOutput(ffmpeg + " -version");
    } catch (...) {
      XPLERROR << "Failed to run " << ffmpeg << "\n";
      return false;
    }

    XPLINFO << "ffmpeg -version\n\n" << output;
    bool found_ffmpeg_version = output.find("ffmpeg version") != std::string::npos;
    return found_ffmpeg_version;
  }

  bool checkFFmpegInstallAndWarnIfNot() {
    std::string output;
    bool installed = checkFFmpegInstall(output);
    if (!installed) {
      tinyfd_messageBox(
          "ffmpeg not found or not configured OK",
          std::string("ffmpeg is required to encode and decode video.\nYou must install ffmpeg separately.\nffmpeg test output:" + output).c_str(),
          "ok",
          "error",
          1);
    }
    return installed;
  }

  // This command does nothing but wait. It is useful to keep the console open to see errors.
  std::shared_ptr<std::atomic<bool>> cancel_console_sleep_requested = std::make_shared<std::atomic<bool>>(false);
  void consoleSleepCommand()
  {
    command_runner.setCompleteOrKilledCallback([] {});
    command_runner.queueThreadCommand(
      cancel_console_sleep_requested,
      [&] {
        while(true) {
          if (*cancel_console_sleep_requested) return;
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      });
    command_runner.runCommandQueue();
  }

  void testCommand()
  {
    command_runner.setCompleteCallback([] { XPLINFO << "Command finished!"; });
    command_runner.setKilledCallback([] { XPLINFO << "Command killed!"; });
    command_runner.queueShellCommand("echo \"foo\"");
#ifdef _WIN32
    command_runner.queueShellCommand("ping -n 5 google.com");
#else
    command_runner.queueShellCommand("ping -c 20 google.com");
#endif
    command_runner.runCommandQueue();
  }

  std::shared_ptr<std::atomic<bool>> cancel_test_command_requested = std::make_shared<std::atomic<bool>>(false);
  void testThreadCommand()
  {
    command_runner.setCompleteCallback([] { XPLINFO << "Thread command finished!"; });
    command_runner.setKilledCallback([] { XPLINFO << "Thread command killed!"; });
    command_runner.queueThreadCommand(
      cancel_test_command_requested,
      [&] {
        for (int i = 0; i < 150; ++i) {
          XPLINFO << "hello " << i;
          if (*cancel_test_command_requested) return;
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      });
    command_runner.runCommandQueue();
  }

  // No actual drawing happens here. We just get the rectangle to be drawn in and save it for
  // later.
  void drawPreview3D()
  {
    preview3d_viewport_min = ImGui::GetWindowPos();
    preview3d_viewport_size = ImGui::GetWindowSize();
  }

  void drawConfigFFmpegScreen()
  {
    beginModalRoundedBox("##LicenseScreenRoundedFrame");

    ImGui::Text("ffmpeg command or location: ");
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - getScaledFontSize());
    ffmpeg_tool_select.drawAndUpdate();
    ImGui::PopItemWidth();

    ImGui::Dummy(ImVec2(0.0f, 40.0f));

    if (ImGui::Button("OK", ImVec2(0, 0))) {
      getExternalToolFileSelects();
      prefs["ffmpeg"] = ffmpeg;
      preferences::setPrefs(prefs);
  
      if (project_dir.empty()) {
        setAppState(STATE_SPLASH);
      } else {
        setAppState(STATE_3D_VIEW);
      }
    }
    
    ImGui::SameLine();
    
    if (ImGui::Button("Cancel", ImVec2(0, 0))) {
      ffmpeg_tool_select.setPath(ffmpeg.c_str());

      if (project_dir.empty()) {
        setAppState(STATE_SPLASH);
      } else {
        setAppState(STATE_3D_VIEW);
      }
    }

    ImGui::SameLine();

    if (ImGui::Button("Restore Defaults", ImVec2(0, 0))) {
      setExternalToolDefaultPaths();
      setExternalToolFileSelects();
    }
    
    ImGui::SameLine();

    if (ImGui::Button("Test", ImVec2(0, 0))) {
      getExternalToolFileSelects();
      if (checkFFmpegInstallAndWarnIfNot()) {
        tinyfd_messageBox(
            "ffmpeg is installed OK",
            "ffmpeg appears to be installed correctly.",
            "ok",
            "info",
            1);
      }
    }

    endModalRoundedBox();
  }

  // We have to do some shenanagins here to keep giving the window focus
  // so it will respond to dragging. It will lose it and never get it back
  // otherwise if we click anywhere else.
  void floatingWindowFocusShenanagins() { 
    if (main_menu_is_hovered || command_window_extras_popup_open) return;
    if (
      ImGui::GetMousePos().x >= ImGui::GetWindowPos().x && 
      ImGui::GetMousePos().x <= ImGui::GetWindowPos().x + ImGui::GetWindowSize().x &&
      ImGui::GetMousePos().y >= ImGui::GetWindowPos().y && 
      ImGui::GetMousePos().y <= ImGui::GetWindowPos().y + ImGui::GetWindowSize().y) {
      ImGui::SetWindowFocus(); // Set focus to the window
    }
  }

  void resetWindowLayout() {
    for (FloatingWindow* floating_window : {
      &floating_window_preview_render,
      &floating_window_virtual_cam_settings,
      &floating_window_keyframe_editor,
      &floating_window_world_transform,
      &floating_window_orbit_editor
    }) {
      floating_window->visible = true;
      floating_window->first_time_showing = true;
    }
  }

  // We factor it out like this so the same settings can appear in the Render Video screen.
  bool drawVirtualCameraSettingsWithoutWindow(const int max_width = 0) {
    bool value_changed = false;

    ImGui::Text("Camera Type:");
    if (max_width == 0) {
      ImGui::PushItemWidth(ImGui::GetWindowWidth() - 60);
    } else {
      ImGui::PushItemWidth(std::min(max_width, int(ImGui::GetWindowWidth() - 60)));
    }

    ImGui::GetStyle().WindowPadding = ImVec2(0, 0);

    static const char* kVirtualCameraTypeNames[] = {
      "Rectilinear",
      "Rectilinear (Stereo)",
      "Equirectangular (Mono 360)",
      "VR180 (Stereo)",
      "Looking Glass Portrait"
    };

    value_changed |= ImGui::Combo("##CameraType", &virtual_camera_type_choice, kVirtualCameraTypeNames, IM_ARRAYSIZE(kVirtualCameraTypeNames));
    ImGui::PopItemWidth();

    if (virtual_camera_type_choice == CAM_TYPE_RECTILINEAR || virtual_camera_type_choice == CAM_TYPE_RECTILINEAR_STEREO) {      
      ImGui::Dummy(ImVec2(0.0f, 20.0f));
      ImGui::AlignTextToFramePadding();
      ImGui::Text("Width");
      ImGui::SameLine();
      ImGui::SetCursorPosX(370);
      ImGui::PushItemWidth(120);
      value_changed |= ImGui::InputInt("##983275", &virtual_camera_width, 0);
      ImGui::PopItemWidth();
      
      ImGui::Dummy(ImVec2(0.0f, 10.0f));
      ImGui::AlignTextToFramePadding();
      ImGui::Text("Height");
      ImGui::SameLine();
      ImGui::SetCursorPosX(370);
      ImGui::PushItemWidth(120);
      value_changed |= ImGui::InputInt("##345623", &virtual_camera_height, 0);
      ImGui::PopItemWidth();
    }

    if (virtual_camera_type_choice == CAM_TYPE_EQUIRECTANGULAR) {
      ImGui::Dummy(ImVec2(0.0f, 20.0f));
      ImGui::AlignTextToFramePadding();
      ImGui::Text("Width");
      ImGui::SameLine();
      ImGui::SetCursorPosX(370);
      ImGui::PushItemWidth(120);
      value_changed |= ImGui::InputInt("##356634", &virtual_camera_eqr_width, 0);
      ImGui::PopItemWidth();
      
      ImGui::Dummy(ImVec2(0.0f, 10.0f));
      ImGui::AlignTextToFramePadding();
      ImGui::Text("Height");
      ImGui::SameLine();
      ImGui::SetCursorPosX(370);
      ImGui::PushItemWidth(120);
      value_changed |= ImGui::InputInt("##23443243", &virtual_camera_eqr_height, 0);
      ImGui::PopItemWidth();

      ImGui::Dummy(ImVec2(0.0f, 10.0f));
      value_changed |= ImGui::Checkbox("Draw Horizon Line", &virtual_camera_draw_horizon_line);
    }

    if (virtual_camera_type_choice == CAM_TYPE_RECTILINEAR || virtual_camera_type_choice == CAM_TYPE_RECTILINEAR_STEREO || virtual_camera_type_choice == CAM_TYPE_LOOKING_GLASS_PORTRAIT) {
      ImGui::Dummy(ImVec2(0.0f, 10.0f));
      ImGui::AlignTextToFramePadding();
      ImGui::Text("Horizontal FOV (deg)");
      ImGui::SameLine();
      ImGui::SetCursorPosX(370);
      ImGui::PushItemWidth(120);
      value_changed |= ImGui::InputFloat("##2345234544", &virtual_camera_hfov);
      ImGui::PopItemWidth();
    }

    if (virtual_camera_type_choice == CAM_TYPE_VR180) {
      ImGui::Dummy(ImVec2(0.0f, 10.0f));
      ImGui::AlignTextToFramePadding();
      ImGui::Text("Image Size (Per Eye)");
      ImGui::SameLine();
      ImGui::SetCursorPosX(370);
      ImGui::PushItemWidth(120);
      value_changed |= ImGui::InputInt("##2544355", &virtual_camera_vr180_size, 0);
      ImGui::PopItemWidth();
    }

    if (virtual_camera_type_choice == CAM_TYPE_VR180 || virtual_camera_type_choice == CAM_TYPE_LOOKING_GLASS_PORTRAIT || virtual_camera_type_choice == CAM_TYPE_RECTILINEAR_STEREO) {
      ImGui::Dummy(ImVec2(0.0f, 10.0f));
      ImGui::AlignTextToFramePadding();
      ImGui::Text("Stereo Baseline (unitless)");
      ImGui::SameLine();
      ImGui::SetCursorPosX(370);
      ImGui::PushItemWidth(120);
      value_changed |= ImGui::InputFloat("##235774", &virtual_stereo_baseline);
      ImGui::PopItemWidth();
    }

    virtual_camera_width = math::clamp(virtual_camera_width, 64, 4096 * 4);
    virtual_camera_height = math::clamp(virtual_camera_height, 64, 4096 * 4);
    virtual_camera_hfov = math::clamp(virtual_camera_hfov, 1.0f, 179.0f);
    virtual_camera_vr180_size = math::clamp(virtual_camera_vr180_size, 64, 4096 * 4);
    virtual_camera_eqr_width = math::clamp(virtual_camera_eqr_width, 64, 4096 * 4);
    virtual_camera_eqr_height = math::clamp(virtual_camera_eqr_height, 64, 4096 * 4);

    return value_changed;
  }

  calibration::RectilinearCamerad getVirtualCameraWithIntrinsicsFromSettings() {
    calibration::RectilinearCamerad cam(sfm_cameras[0]);
    cam.width = virtual_camera_width;
    cam.height = virtual_camera_height;
    cam.optical_center = Eigen::Vector2d(cam.width / 2.0, cam.height / 2.0);
    double f = cam.width / (2.0 * tan(virtual_camera_hfov * M_PI / 360.0));
    cam.focal_length = Eigen::Vector2d(f, f);
    return cam;
  }

  void addKeyframe(int frame_idx) {
    VirtualCamKeyframe k;
    k.cam = getVirtualCameraWithIntrinsicsFromSettings();
    k.cam.cam_from_world = Eigen::Isometry3d::Identity();
    timeline.keyframes[frame_idx] = k;
  }

  calibration::RectilinearCamerad applyKeyframeTransform(const VirtualCamKeyframe& keyframe) {
    calibration::RectilinearCamerad cam = getVirtualCameraWithIntrinsicsFromSettings();
    cam.cam_from_world = Eigen::Isometry3d::Identity();
    std::vector<double> rvec = {
      keyframe.rx * M_PI / 180,
      keyframe.ry * M_PI / 180,
      keyframe.rz * M_PI / 180};
    Eigen::Matrix3d rotation;
    ceres::AngleAxisToRotationMatrix(rvec.data(), rotation.data());
    cam.cam_from_world.linear() = rotation;
    cam.setPositionInWorld(Eigen::Vector3d(keyframe.tx, keyframe.ty, keyframe.tz));
    return cam;
  }

  calibration::RectilinearCamerad getInterpolatedCamera(int frame_idx) {
    // Find the previous and next keyframes
    int prev_idx = 0;
    int next_idx = -1;
    for (auto& [k, v] : timeline.keyframes) {
      if (k <= frame_idx) prev_idx = k;
      if (k > frame_idx) {
        next_idx = k;
        break;
      }
    }  
    const auto& prev_kf = timeline.keyframes.at(prev_idx);
    
    // If there is no next keyframe, there is nothing to interpolate. Use
    // the data from the previous keyframe directly.
    if (next_idx < 0) return applyKeyframeTransform(prev_kf);

    const auto& next_kf = timeline.keyframes.at(next_idx);

    const float alpha = float(frame_idx - prev_idx) / float(next_idx - prev_idx);

    VirtualCamKeyframe interpolated(prev_kf);
    interpolated.tx = next_kf.tx * alpha + prev_kf.tx * (1 - alpha);
    interpolated.ty = next_kf.ty * alpha + prev_kf.ty * (1 - alpha);
    interpolated.tz = next_kf.tz * alpha + prev_kf.tz * (1 - alpha);
    interpolated.rx = next_kf.rx * alpha + prev_kf.rx * (1 - alpha);
    interpolated.ry = next_kf.ry * alpha + prev_kf.ry * (1 - alpha);
    interpolated.rz = next_kf.rz * alpha + prev_kf.rz * (1 - alpha);
    return applyKeyframeTransform(interpolated);
  }

  void drawMainWindow()
  {
    // Make the background of the 3D view transparent
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 0));
    
    ImGuiStyle& style = ImGui::GetStyle();
    int view3d_height = ImGui::GetWindowHeight() - VideoTimelineWidget::kTimelineHeight -
                        style.WindowPadding.y * 2 - ImGui::GetFrameHeight();

    ImGui::BeginChild("3D View", ImVec2(0, view3d_height), false);
    drawPreview3D();
    ImGui::EndChild();
    ImGui::PopStyleColor();

    timeline.render();

    floating_window_preview_render.draw();
    floating_window_keyframe_editor.draw();
    floating_window_orbit_editor.draw();
    floating_window_virtual_cam_settings.draw();
    floating_window_world_transform.draw();
  }

  bool commandWindowIsVisible() { return command_runner.isRunning(); }

  bool command_window_extras_popup_open = false;
  void drawCommandWindow()
  {
    constexpr float kCancelButtonWidth = 200;
    constexpr float kWindowMenuButtonWidth = 80;
    constexpr float kPadding = 20;

    ImGui::Dummy(ImVec2(0.0, kPadding)); 

    ImGui::Dummy(ImVec2(kPadding, 0));
    ImGui::SameLine();

    if (ImGui::Button("Cancel", ImVec2(kCancelButtonWidth, 0))) {
      command_runner.kill();
    }

    ImGui::SameLine();
    ImGui::Dummy(ImVec2(kPadding, 0.0)); 

    if (command_runner.waitingForCancel()) {
      ImGui::SameLine();
      ImGui::Text("Cancelling...");
    } else { // Progress bar

      float available_width = 
        ImGui::GetContentRegionAvail().x - kWindowMenuButtonWidth - kCancelButtonWidth - kPadding * 3;
      
      CommandProgressDescription progress_description = command_runner.updateAndGetProgress();
      ImGui::SameLine();
      ImVec2 progressBarPos = ImGui::GetCursorPos();

      // Draw the progress bar
      ImGui::PushItemWidth(available_width);
      ImGui::ProgressBar(progress_description.frac, ImVec2(available_width, 0), "");
      ImGui::PopItemWidth();

      // Move the cursor back to overlay the text on the progress bar
      ImGui::SetCursorPos(ImVec2(progressBarPos.x + 10, progressBarPos.y));
      ImGui::TextUnformatted(progress_description.progress_str.c_str());

      // Reset the cursor position for the triangle button
      ImGui::SetCursorPos(ImVec2(progressBarPos.x + available_width + kPadding, progressBarPos.y));

      // Triangle button
      if (ImGui::ArrowButton("windows_button_id", ImGuiDir_Down)) {
        ImGui::OpenPopup("VizWindowsTogglePopup");
      }

      // Popup menu
      command_window_extras_popup_open = false;
      if (ImGui::BeginPopup("VizWindowsTogglePopup")) {
        command_window_extras_popup_open = true;
        ImGui::MenuItem(floating_window_tracking_viz.title.c_str(), nullptr, &floating_window_tracking_viz.visible);
        ImGui::MenuItem(floating_window_sfm_cost_plot.title.c_str(), nullptr, &floating_window_sfm_cost_plot.visible);
        ImGui::MenuItem(floating_window_nerf_cost_plot.title.c_str(), nullptr, &floating_window_nerf_cost_plot.visible);
        ImGui::EndPopup();
      }
    }

    ImGui::Dummy(ImVec2(0.0, kPadding)); 

    ImGui::BeginChild("CommandScroll", ImVec2(0, 0), false, ImGuiWindowFlags_None);

    std::string command_output = command_runner.getOutput();
#ifdef _WIN32
    command_output.erase(std::remove(command_output.begin(), command_output.end(), '\r'), command_output.end());
#else
    std::replace(command_output.begin(), command_output.end(), '\r', '\n');
#endif

    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 0.7f));
    ImGui::TextUnformatted(command_output.c_str());
    ImGui::PopStyleColor();

    ImGui::SetScrollY(ImGui::GetScrollMaxY());
  
    ImGui::EndChild();

    floating_window_tracking_viz.draw();
    floating_window_sfm_cost_plot.draw();
    floating_window_nerf_cost_plot.draw();
  }

  // Check if we have the minimum required files a in a project directory to consider it "valid"
  bool checkProjectDirForValidData() {
    if (!file::directoryExists(project_dir)) return false;
    if (!file::directoryExists(project_dir + "/video_frames")) return false;
    if (!file::directoryExists(project_dir + "/structure_from_motion")) return false;
    if (!file::directoryExists(project_dir + "/nerf")) return false;
    if (!file::fileExists(project_dir + "/structure_from_motion/dataset_all.json")) return false;
    //if (!file::fileExists(project_dir + "/structure_from_motion/pointcloud_sfm.bin")) return false;
    if (!file::fileExists(project_dir + "/nerf/radiance_model")) return false;
    if (!file::fileExists(project_dir + "/nerf/proposal_model")) return false;
    return true;
  }

  void loadRadianceAndProposalModels() {
    nerf::loadNerfAndProposalModels(project_dir + "/nerf", device, radiance_model, proposal_model);
  }

  void updatePointCloudVertexBuffer() {
    pointcloud_vb_needs_update = false;

    pointcloud_vb.vertex_data.clear();
    for (int i = 0; i < point_cloud.size(); ++i) {
      pointcloud_vb.vertex_data.emplace_back(
        point_cloud[i].x(), point_cloud[i].y(), point_cloud[i].z(),
        point_cloud_colors[i].x(), point_cloud_colors[i].y(), point_cloud_colors[i].z(), point_cloud_colors[i].w());
    }
    pointcloud_vb.bind();  
    pointcloud_vb.copyVertexDataToGPU(GL_STATIC_DRAW);
  }

  std::shared_ptr<std::atomic<bool>> sample_pointcloud_cancel_requested;
  std::atomic<bool> sampled_pointcloud_at_least_once = false;
  void samplePointCloudFromNerf() {
    point_cloud.clear();
    point_cloud_colors.clear();

    command_runner.setRedirectStdErr(true);
    command_runner.setCompleteCallback([&] {
      sampled_pointcloud_at_least_once = true;
      setAppState(STATE_3D_VIEW);
    });
    command_runner.setKilledCallback([&] {
      if (sampled_pointcloud_at_least_once) {
        setAppState(STATE_3D_VIEW);
      } else {
        setAppState(STATE_SPLASH);
      }
    });

    auto progress_parser = [&](const std::string& line, CommandProgressDescription& p) {
      std::smatch matches;
      if (std::regex_search(line, matches, std::regex("Batch: (\\d+) / (\\d+)\\s+# Points: (\\d+) / (\\d+)")) && matches.size() == 5) {
        int curr_batch = std::stoi(matches[1].str());
        int max_batches = std::stoi(matches[2].str());
        int num_points = std::stoi(matches[3].str());
        int max_points = std::stoi(matches[4].str());

        float frac_batch = static_cast<float>(curr_batch) / max_batches;
        float frac_points = static_cast<float>(num_points) / max_points;

        p.frac = std::max(frac_batch, frac_points);

        std::ostringstream percentage;
        percentage << std::fixed << std::setprecision(2) << (p.frac * 100.0);
        p.progress_str = "Sampling point cloud from NeRF " + percentage.str() + "%%"; // Two $'s for ImGui::Text
      }
    };

    sample_pointcloud_cancel_requested = std::make_shared<std::atomic<bool>>(false);
    command_runner.queueThreadCommand(
      sample_pointcloud_cancel_requested,
      [&] {
        XPLINFO << "Phase: Sampling point cloud from NeRF";
        torch::Tensor image_code = torch::zeros({nerf::kImageCodeDim}, {torch::kFloat32}).to(device);
        samplePointCloudFromRadianceField(device, radiance_model, image_code, point_cloud_num_samples, point_cloud, point_cloud_colors, sample_pointcloud_cancel_requested);
        
        // After the point cloud generation happens in a separate thread, we need to update
        // the vertex buffer from the main thread. Setting this will cause that to happen
        // at the start of the next frame.
        pointcloud_vb_needs_update = true;
      },
      progress_parser);
  
    command_runner.runCommandQueue();
  }

  void makeCameraPoseVertexData(
      const std::vector<calibration::RectilinearCamerad>& cameras,
      const Eigen::Vector3f line_color) {
    // Draw coordinate axes representing pose of each camera
    const Eigen::Vector3d zero(0, 0, 0);
    const Eigen::Vector3d dx(0.05, 0, 0);
    const Eigen::Vector3d dy(0, 0.05, 0);
    const Eigen::Vector3d dz(0, 0, 0.05);
    for(int i = 0; i < cameras.size(); ++i) {
      auto& cam = cameras[i];
      const Eigen::Isometry3d& w_from_c = cam.cam_from_world.inverse();
      const Eigen::Vector3d a = w_from_c * zero;
      const Eigen::Vector3d b = w_from_c * dx;
      const Eigen::Vector3d c = w_from_c * dy;
      const Eigen::Vector3d d = w_from_c * dz;

      float alpha = 0.25;
      lines_vb.vertex_data.emplace_back(a.x(), a.y(), a.z(), 1.0, 0.0, 0.0, alpha);
      lines_vb.vertex_data.emplace_back(b.x(), b.y(), b.z(), 1.0, 0.0, 0.0, alpha);
      lines_vb.vertex_data.emplace_back(a.x(), a.y(), a.z(), 0.0, 1.0, 0.0, alpha);
      lines_vb.vertex_data.emplace_back(c.x(), c.y(), c.z(), 0.0, 1.0, 0.0, alpha);
      lines_vb.vertex_data.emplace_back(a.x(), a.y(), a.z(), 0.0, 0.0, 1.0, alpha);
      lines_vb.vertex_data.emplace_back(d.x(), d.y(), d.z(), 0.0, 0.0, 1.0, alpha);
    }

    // Draw a line connecting all of the camera poses. Gradient in alpha indicates direction of time.
    for (int i = 0; i < int(cameras.size()) - 1; ++i) {
      const float alpha1 = 0.1 + 0.9 * float(i) / cameras.size();
      const float alpha2 = 0.1 + 0.9 * float(i + 1) / cameras.size();
      const Eigen::Vector3d p1 = cameras[i].getPositionInWorld();
      const Eigen::Vector3d p2 = cameras[i+1].getPositionInWorld();
      lines_vb.vertex_data.emplace_back(p1.x(), p1.y(), p1.z(), line_color.x(), line_color.y(), line_color.z(), alpha1);
      lines_vb.vertex_data.emplace_back(p2.x(), p2.y(), p2.z(), line_color.x(), line_color.y(), line_color.z(), alpha2);
    }
  }

  void makeCameraFrustumLinesVertexData(const calibration::RectilinearCamerad& cam) {
    // Draw a dot on top of the camera
    const Eigen::Vector3d cam_pos = cam.getPositionInWorld();
    points_vb.vertex_data.emplace_back(cam_pos.x(), cam_pos.y(), cam_pos.z(), 1.0, 1.0, 1.0, 1.0);

    // Draw the frustum as a ray going from the origin through each corner pixel
    const double kFrustLineLen = 0.25;
    const Eigen::Matrix3d w_R_c = cam.cam_from_world.linear().transpose();

    const Eigen::Vector3d frust00 = cam_pos + kFrustLineLen * w_R_c * cam.rayDirFromPixel(Eigen::Vector2d(0, 0));
    const Eigen::Vector3d frust10 = cam_pos + kFrustLineLen * w_R_c * cam.rayDirFromPixel(Eigen::Vector2d(cam.width - 1, 0));
    const Eigen::Vector3d frust11 = cam_pos + kFrustLineLen * w_R_c * cam.rayDirFromPixel(Eigen::Vector2d(cam.width - 1, cam.height - 1));
    const Eigen::Vector3d frust01 = cam_pos + kFrustLineLen * w_R_c * cam.rayDirFromPixel(Eigen::Vector2d(0, cam.height - 1));

    lines_vb.vertex_data.emplace_back(cam_pos.x(), cam_pos.y(), cam_pos.z(), 1.0, 1.0, 1.0, 1.0);
    lines_vb.vertex_data.emplace_back(frust00.x(), frust00.y(), frust00.z(), 1.0, 1.0, 1.0, 0.0);
    lines_vb.vertex_data.emplace_back(cam_pos.x(), cam_pos.y(), cam_pos.z(), 1.0, 1.0, 1.0, 1.0);
    lines_vb.vertex_data.emplace_back(frust10.x(), frust10.y(), frust10.z(), 1.0, 1.0, 1.0, 0.0);
    lines_vb.vertex_data.emplace_back(cam_pos.x(), cam_pos.y(), cam_pos.z(), 1.0, 1.0, 1.0, 1.0);
    lines_vb.vertex_data.emplace_back(frust11.x(), frust11.y(), frust11.z(), 1.0, 1.0, 1.0, 0.0);
    lines_vb.vertex_data.emplace_back(cam_pos.x(), cam_pos.y(), cam_pos.z(), 1.0, 1.0, 1.0, 1.0);
    lines_vb.vertex_data.emplace_back(frust01.x(), frust01.y(), frust01.z(), 1.0, 1.0, 1.0, 0.0);
  }

  void update3DGuiVertexData() {
    // Update the vertex buffer for lines
    lines_vb.vertex_data.clear();
    points_vb.vertex_data.clear();

    if (show_origin_in_3d_view) {
      lines_vb.vertex_data.emplace_back(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0);
      lines_vb.vertex_data.emplace_back(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0);
      lines_vb.vertex_data.emplace_back(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0);
      lines_vb.vertex_data.emplace_back(0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0);
      lines_vb.vertex_data.emplace_back(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0);
      lines_vb.vertex_data.emplace_back(0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0);
    }

    if (show_sfm_cameras_in_3d_view) {
      makeCameraPoseVertexData(sfm_cameras, Eigen::Vector3f(1,1,1));
    }

    if (show_virtual_cameras_in_3d_view) {
      std::vector<calibration::RectilinearCamerad> interpolated_cameras;
      for (int i = 0; i < timeline.num_frames; ++i) {
        interpolated_cameras.push_back(getInterpolatedCamera(i));
      }
      makeCameraPoseVertexData(interpolated_cameras, Eigen::Vector3f(1, 0, 1));

      // Make frustum lines for all keyframes
      for (auto& [k, v] : timeline.keyframes) {
        makeCameraFrustumLinesVertexData(getInterpolatedCamera(k));
      }
    }

    // Make a frustum for the current frame, even if it is interpolated
    makeCameraFrustumLinesVertexData(getInterpolatedCamera(timeline.curr_frame));

    if (floating_window_orbit_editor.visible) {
      for (int i = 0; i < timeline.num_frames; ++i) {
        const float alpha = float(i) / timeline.num_frames;
        const float theta_deg = alpha * orbit_editor_start_theta + (1.0 - alpha) * orbit_editor_end_theta;
        const float theta = (M_PI / 180.0) * theta_deg;
  
        float x = orbit_editor_radius * cos(theta);
        float y = 0;
        float z = orbit_editor_radius * sin(theta);
        x += orbit_editor_center_x;
        y += orbit_editor_center_y;
        z += orbit_editor_center_z;
        lines_vb.vertex_data.emplace_back(x, y, z, 1.0, 0.65, 0.2, 0.5 + 0.5 * alpha);
      }
    }

    lines_vb.bind();
    lines_vb.copyVertexDataToGPU(GL_DYNAMIC_DRAW);

    points_vb.bind();
    points_vb.copyVertexDataToGPU(GL_DYNAMIC_DRAW);
  }

  void loadProject() {
    need_to_reload_project = false;
    
    // Loading now returns a mix of rectilinear or fisheye cameras but Volurama only supports rectilinear for now.
    std::vector<calibration::NerfKludgeCamera> kluge_cameras = calibration::readDatasetCameraJson(project_dir + "/structure_from_motion/dataset_all.json");
    sfm_cameras.clear();
    for (const auto& cam : kluge_cameras) {
      XCHECK(cam.is_rectilinear);
      sfm_cameras.push_back(cam.rectilinear);
    }
     
    loadRadianceAndProposalModels();
    //renderTestImage();

    sampled_pointcloud_at_least_once = false;
    samplePointCloudFromNerf(); // If the user cancels during this, then we might not finish loading.

    timeline.num_frames = 300;
    timeline.keyframes.clear();
    addKeyframe(0); // Force a keyframe at frame 0

    setOrbitEditorDefaults();
    setWorldTransformDefaults();
  
    virtual_camera_width = sfm_cameras[0].width;
    virtual_camera_height = sfm_cameras[0].height;

    const float f = sfm_cameras[0].focal_length.x();
    virtual_camera_hfov = 2.0 * atan2(sfm_cameras[0].width / 2.0, f) * 180.0 / M_PI;

    refreshPreviewRender();
  }

  void checkForValidProjectDirAndLoadOrBounce() {
    if(!checkProjectDirForValidData()) {
      tinyfd_messageBox(
        "Invalid Project Directory",
        "The project directory does not contain all data for a fully processed NeRF. Unable to continue.",
        "ok", "error", 0);
      setAppState(STATE_SPLASH);
    } else {
      loadProject();
      setAppState(STATE_3D_VIEW);
    }
  }

  gui::ImguiFolderSelect project_dir_folder_select;
  void projectDirScreen() {
    beginModalRoundedBox("##LicenseScreenRoundedFrame");

    ImGui::Text("Select Project Directory");
    ImGui::Dummy(ImVec2(0.0f, 20.0f));
    ImGui::PushItemWidth(ImGui::GetContentRegionAvail().x - getScaledFontSize() * 6);
    project_dir_folder_select.drawAndUpdate();
    ImGui::PopItemWidth();

    if (!std::string(project_dir_folder_select.path).empty()) {
      ImGui::SameLine();
      ImGui::Dummy(ImVec2(getScaledFontSize() * 0.5, 0.0f));
      ImGui::SameLine();

      if (ImGui::Button("OK")) {
        if (file::directoryExists(project_dir_folder_select.path)) {
          project_dir = project_dir_folder_select.path;
          if (kEnableAutoLogging) xpl::stdoutLogger.attachTextFileLog(project_dir + "/log.txt");
          prefs["project_dir"] = project_dir;
          preferences::setPrefs(prefs);

          if (is_creating_new_project) {
            setAppState(STATE_SFM_NERF_CONFIG);
          } else {
            checkForValidProjectDirAndLoadOrBounce();
          }
        } else {
          tinyfd_messageBox("Invalid Project Directory Path", "The path does not appear to be a directory.", "ok", "error", 0);
        }
      }
    }

    ImGui::Dummy(ImVec2(0.0f, 20.0f));
    ImGui::BulletText("Data for the project will be stored in this folder.");
    ImGui::BulletText("Files in this directory may be deleted or irreversibly modified.");
    ImGui::BulletText("Large amounts of data may be generated.");

    endModalRoundedBox();
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

  float getScaledFontSize() {
    ImGuiIO& io = ImGui::GetIO();
    return io.Fonts->Fonts[0]->FontSize * io.FontGlobalScale;
  }

  void drawFrame()
  {
    gl_context_mutex.lock();
    glfwMakeContextCurrent(window);

    if (pointcloud_vb_needs_update) {
      updatePointCloudVertexBuffer();
    }

    if(need_to_reload_project) {
      loadProject();
    }

    // If a modal popup is open, we don't want to waste GPU resources rendering the nerf preview,
    // (e.g., because we might be trying to sample a point cloud). This will cancel the preview render.
    if (isAnyModalPopupOpen()) {
      refreshPreviewRender();
    }

    // Scale text for high DPI displays
    float xscale, yscale;
    glfwGetWindowContentScale(window, &xscale, &yscale);
    ImGui::GetIO().FontGlobalScale = xscale;

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    const bool hide_main_menu = isAnyModalPopupOpen();
    ImGuiWindowFlags window_flags = 0;
    window_flags |= ImGuiWindowFlags_NoTitleBar;
    window_flags |= ImGuiWindowFlags_NoScrollbar;
    window_flags |= ImGuiWindowFlags_NoBackground;
    window_flags |= ImGuiWindowFlags_NoMove;
    window_flags |= ImGuiWindowFlags_NoResize;
    window_flags |= ImGuiWindowFlags_NoCollapse;
    window_flags |= ImGuiWindowFlags_AlwaysAutoResize;
    if (!hide_main_menu) window_flags |= ImGuiWindowFlags_MenuBar;
    
    // Resize the ImGui window to the GLFW window
    int glfw_window_w, glfw_window_h;
    glfwGetWindowSize(window, &glfw_window_w, &glfw_window_h);
    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(glfw_window_w, glfw_window_h), ImGuiCond_Always);

    std::string window_title = std::string() + "Volurama";
    if (!project_dir.empty()) window_title += " - " + project_dir;
    glfwSetWindowTitle(window, window_title.c_str());
    
    if (ImGui::Begin("Main Window", nullptr, window_flags)) {
      if (commandWindowIsVisible()) {
        main_menu_is_hovered = false; // Kludge: this is required for floatingWindowFocusShenanagins()
        drawCommandWindow();
      } else {
        switch(app_state) {
        case STATE_SPLASH: splashScreen(); break;
        case STATE_EXPLAIN: explainScreen(); break;
        case STATE_SELECT_PROJECT_DIR: projectDirScreen(); break;
        case STATE_CONFIG_FFMPEG: drawConfigFFmpegScreen(); break;
        case STATE_SFM_NERF_CONFIG: sfmAndNerfProcessingSettingsScreen(); break;
        case STATE_RENDER_VID_CONFIG: renderVideoSettingsScreen(); break;
        case STATE_EXPORT_LDI_CONFIG: exportLdiSettingsScreen(); break;
        case STATE_EXPORT_MESH_CONFIG: exportMeshSettingsScreen(); break;
        case STATE_3D_VIEW: 
          handleKeyboard();
          updateMouse();

          drawMainMenu();

          ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
          drawMainWindow();
          ImGui::PopStyleVar();
          break;
        }
      }

      ImGui::End();
    }

    // Here we would normally call finishDrawingImguiAndGl(), but don't because we are mixing in custom opengl
    beginGlDrawingForImgui();
    //ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    if (!isAnyModalPopupOpen()) {
      drawGlStuff();
    }

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    updateCursor();

    glfwSwapBuffers(window);
    glfwMakeContextCurrent(nullptr);
    gl_context_mutex.unlock();
  }
} app;  // An instance of VoluramaApp named app exists in this namespace. This is used in glfw
        // callbacks.

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) app.handleMouseDown(button);
  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) app.handleMouseUp(button);
  if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) app.handleMouseDown(button);
  if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) app.handleMouseUp(button);
}

static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
  if (app.mouse_in_3d_viewport) {
    if (yoffset > 0) app.camera_radius *= 0.9;
    if (yoffset < 0) app.camera_radius *= 1.1;
  }
  if (app.mouse_in_timeline) {
    if (yoffset > 0) app.timeline.pixels_per_frame *= 1.1;
    if (yoffset < 0) app.timeline.pixels_per_frame *= 0.9;
    app.timeline.pixels_per_frame = math::clamp<float>(app.timeline.pixels_per_frame, 0.1, 10.0);
  }

  // TODO: call this?
  // ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  //if (key == GLFW_KEY_H && action == GLFW_PRESS) app.selected_tool = kScroll;
  //if (key == GLFW_KEY_SPACE && action == GLFW_RELEASE)
  //  app.toggle_play_video = !app.toggle_play_video;
  //if (app.mouse_in_3d_viewport) {
  //  if (key == GLFW_KEY_1 && action == GLFW_RELEASE) ...
  //}

  ImGui_ImplGlfw_KeyCallback(window, key, scancode, action, mods);
}

}  // namespace p11

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

  p11::app.init("Volurama", 1280, 720);
  glfwSetMouseButtonCallback(p11::app.window, p11::mouse_button_callback);
  glfwSetScrollCallback(p11::app.window, p11::scroll_callback);
  glfwSetKeyCallback(p11::app.window, p11::key_callback);

  p11::app.initVoluramaApp();
  p11::app.setProStyle();

#ifdef _WIN32
  const std::string font_path = "Helvetica.ttf";  // On Windows, the directory structure is flat.
#else
  const std::string font_path = "fonts/Helvetica.ttf";
#endif
  ImGuiIO& io = ImGui::GetIO();
  io.Fonts->AddFontFromFileTTF(p11::runfile::getRunfileResourcePath(font_path).c_str(), 14.0);

  ImGui::GetIO().IniFilename = nullptr;  // Disable layout saving.

  p11::app.guiDrawLoop();
  p11::app.cleanup();

  return EXIT_SUCCESS;
}
