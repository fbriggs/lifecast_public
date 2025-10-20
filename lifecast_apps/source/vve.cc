// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// Make the application run without a terminal in Windows.
#if defined(windows_hide_console) && defined(_WIN32)
#pragma comment(linker, "/SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
#endif
#ifdef _WIN32
#define _WIN32_WINNT 0x0A00 // Targeting Windows 10
#endif

#include "util_opengl.h"
#include "logger.h"
#include "dear_imgui_app.h"
#include "third_party/dear_imgui/imgui_internal.h" // For PushItemFlag on Windows
#include "third_party/turbo_colormap.h"
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "imgui_filedialog.h"
#include "imgui_cvmat.h"
#include "util_runfile.h"
#include "util_file.h"
#include "util_math.h"
#include "util_command.h"
#include "util_opencv.h"
#include "util_browser.h"
#include "util_torch.h"
#include "convert_to_obj.h"
#include "ldi_common.h"
#include "ldi_pipeline_lib.h"
#include "turbojpeg_wrapper.h"
#include "preferences.h"
#include <regex>
#include <algorithm>
#include <chrono>
#include <atomic>

// HTTP server
#include "third_party/httplib.h"
#include <thread>
#include <iostream>
#include "http_strings.h"


#ifdef _WIN32
#include <Windows.h>
#endif
#ifdef __linux__
#include <GL/gl.h>
#endif

namespace {

static float kTextScale = 1.0;
static float kGuiScaleHack = 1.0;

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

/////////////////////////////////////////////////

static const char* kVertexShader_LDI = R"END(
#version 150 core

uniform mat4 uModelViewProjectionMatrix;
uniform sampler2D uDepthTexture;

in vec3 aVertexPos;
in vec2 aVertexUV;
out vec2 vUV;

void main()
{
  float depth_sample = texture(uDepthTexture, aVertexUV).r;
  float s = clamp(0.3 / depth_sample, 0.01, 50.0);

  gl_Position = uModelViewProjectionMatrix * vec4(aVertexPos * s, 1.0);
  vUV = aVertexUV;
}
)END";

static const char* kFragmentShader_LDI = R"END(
#version 150 core

uniform sampler2D uImageTexture;

in vec2 vUV;
out vec4 frag_color;

void main()
{
  vec4 rgba = texture(uImageTexture, vUV);
  if (rgba.a < 0.05) discard; // Don't write invisible stuff in the depth buffer.

  frag_color = rgba;
}
)END";

}  // namespace

namespace p11 {

constexpr float kSoftwareVersion = 2.7;

static constexpr bool kEnableAutoLogging = false;

enum LdiStudioTool { kScroll = 0, kSelectBrush = 1, kGetDepthTool = 2, kDepthBrushTool = 3 };

enum LdiStudio_LayerChannelSelect {
  kLayer0Depth = 0,
  kLayer0RGBA = 1,
  kLayer0RGB = 2,
  kLayer0A = 3,
  kLayer1Depth = 4,
  kLayer1RGBA = 5,
  kLayer1RGB = 6,
  kLayer1A = 7,
  kLayer2Depth = 8,
  kLayer2RGBA = 9,
  kLayer2RGB = 10,
  kLayer2A = 11,
  kLayerSelection = 12
};

class VideoTimelineWidget {
 public:
  static constexpr int kTimelineHeight = 64;
  static constexpr float kHandleHeight = 10.0;
  static constexpr float kHandleHalfWidth = 5.0;
  static constexpr float kFirstAndLastHandleYOffset = 10.0;
  static constexpr int kHorizontalPadding = 20;

  float pixels_per_frame = 5.0;  // the width of the selection area for 1 frame
  int num_frames = 0;
  int first_frame = 0;
  int curr_frame = 0;
  int last_frame = 0;
  bool is_dragging_curr_handle = false;
  bool is_dragging_first_handle = false;
  bool is_dragging_last_handle = false;

  ImVec2 viewport_min, viewport_size;

  std::function<void()> curr_frame_change_callback;

  void render()
  {
    ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(32, 32, 32, 255));
    ImGui::BeginChild(
        "Timeline", ImVec2(0, kTimelineHeight), false, ImGuiWindowFlags_HorizontalScrollbar);

    if (num_frames == 0) {
      // Don't render a timeline
      ImGui::EndChild();
      ImGui::PopStyleColor();
      return;
    }

    ImVec2 child_topleft = ImGui::GetCursorScreenPos();
    float timeline_width = num_frames * pixels_per_frame;

    ImGui::Dummy(ImVec2(  // Force a horizontal scrollbar.
        timeline_width + kHorizontalPadding * 2,
        kTimelineHeight - ImGui::GetStyle().ScrollbarSize -
            ImGui::GetStyle().WindowPadding.y * 2.0f));

    int top_space = ImGui::GetTextLineHeightWithSpacing();

    // Draw ticks
    int label_interval = 30;
    if (pixels_per_frame < 2.0) label_interval = 90;
    if (pixels_per_frame < 0.5) label_interval = 300;
    if (pixels_per_frame < 0.2) label_interval = 900;
    int tick_interval = label_interval / 3;
    int mark_interval = tick_interval / 10;

    for (int i = 0; i < num_frames; ++i) {
      float x = child_topleft.x + i * pixels_per_frame + kHorizontalPadding;
      if (i % mark_interval == 0) {
        ImGui::GetWindowDrawList()->AddLine(
            ImVec2(
                x, child_topleft.y + top_space + (i % tick_interval != 0 ? kHandleHeight / 2 : 0)),
            ImVec2(x, child_topleft.y + top_space + kHandleHeight),
            IM_COL32(255, 255, 255, 32),
            1.0);
      }

      if (i % label_interval == 0) {
        static constexpr int kTweak = 3;
        ImGui::GetWindowDrawList()->AddText(
            ImGui::GetFont(),
            ImGui::GetFontSize() * 0.8,
            ImVec2(x - kTweak, child_topleft.y + kTweak),
            IM_COL32(255, 255, 255, 128),
            std::to_string(i).c_str());
      }
    }

    ImVec2 first_handle_pos(
        child_topleft.x + kHorizontalPadding + first_frame * pixels_per_frame,
        child_topleft.y + top_space + kFirstAndLastHandleYOffset);
    ImVec2 last_handle_pos(
        child_topleft.x + kHorizontalPadding + last_frame * pixels_per_frame,
        child_topleft.y + top_space + kFirstAndLastHandleYOffset);
    ImVec2 curr_handle_pos(
        child_topleft.x + kHorizontalPadding + curr_frame * pixels_per_frame,
        child_topleft.y + top_space);

    int curr_frame_before_update = curr_frame;
    if (!is_dragging_curr_handle && !is_dragging_last_handle)
      updateHandle(first_handle_pos, first_frame, is_dragging_first_handle, "first");
    if (!is_dragging_curr_handle && !is_dragging_first_handle)
      updateHandle(last_handle_pos, last_frame, is_dragging_last_handle, "last");
    if (!is_dragging_first_handle && !is_dragging_last_handle)
      updateHandle(curr_handle_pos, curr_frame, is_dragging_curr_handle, "curr");

    if (curr_frame_before_update != curr_frame) {
      if (curr_frame_change_callback) curr_frame_change_callback();
    }

    if (is_dragging_last_handle) first_frame = std::min(first_frame, last_frame);
    if (is_dragging_first_handle) last_frame = std::max(last_frame, first_frame);

    // Horizontal line for selected interval
    static constexpr float kThickness = 7;
    ImVec2 line_start(
        first_handle_pos.x,
        child_topleft.y + top_space + kFirstAndLastHandleYOffset + kHandleHeight / 2 - 0.5);
    ImVec2 line_end(last_handle_pos.x, line_start.y);
    ImGui::GetWindowDrawList()->AddLine(
        line_start, line_end, IM_COL32(128, 255, 128, 100), kThickness);

    renderHandle(
        first_handle_pos, IM_COL32(128, 255, 128, is_dragging_first_handle ? 255 : 128), "first");
    renderHandle(
        last_handle_pos, IM_COL32(128, 255, 128, is_dragging_last_handle ? 255 : 128), "last");
    renderHandle(
        curr_handle_pos, IM_COL32(100, 100, 255, is_dragging_curr_handle ? 255 : 200), "curr");

    viewport_min = ImGui::GetWindowPos();
    viewport_size = ImGui::GetContentRegionAvail();  // excludes scrollbars
    viewport_size.y = kTimelineHeight - ImGui::GetStyle().ScrollbarSize;

    ImGui::EndChild();
    ImGui::PopStyleColor();
  }

 private:
  void renderHandle(const ImVec2& pos, const ImU32& color, const std::string& handle_type)
  {
    if (handle_type == "curr") {
      ImGui::GetWindowDrawList()->AddTriangleFilled(
          ImVec2(pos.x - kHandleHalfWidth, pos.y),
          ImVec2(pos.x + kHandleHalfWidth, pos.y),
          ImVec2(pos.x, pos.y + kHandleHeight),
          color);
      ImGui::GetWindowDrawList()->AddLine(
          ImVec2(pos.x - 0.5, pos.y + kHandleHeight - 1),
          ImVec2(pos.x - 0.5, pos.y + kTimelineHeight),
          color,
          1.0);
    } else if (handle_type == "first" || handle_type == "last") {
      float flip = (handle_type == "last") ? -1 : 1;
      ImGui::GetWindowDrawList()->AddTriangleFilled(
          ImVec2(pos.x - flip * 2 * kHandleHalfWidth, pos.y),
          ImVec2(pos.x - flip * 2 * kHandleHalfWidth, pos.y + kHandleHeight),
          ImVec2(pos.x - flip * 0.5, pos.y + kHandleHeight / 2),
          color);
    }
  }

  void updateHandle(
      const ImVec2& pos, int& frame, bool& is_dragging, const std::string& handle_type)
  {
    if (!ImGui::IsMouseDown(0)) {
      is_dragging = false;
    }

    ImVec2 hover_min, hover_max;
    if (handle_type == "curr") {
      hover_min = ImVec2(std::numeric_limits<float>::lowest(), pos.y);
      hover_max = ImVec2(std::numeric_limits<float>::max(), pos.y + kTimelineHeight);
    } else if (handle_type == "first") {
      hover_min = ImVec2(pos.x - 2 * kHandleHalfWidth, pos.y);
      hover_max = ImVec2(pos.x, pos.y + kHandleHeight);
    } else if (handle_type == "last") {
      hover_min = ImVec2(pos.x, pos.y);
      hover_max = ImVec2(pos.x + 2 * kHandleHalfWidth, pos.y + kHandleHeight);
    }

    if (!is_dragging && ImGui::IsMouseDown(0)) {
      if (ImGui::IsMouseHoveringRect(hover_min, hover_max)) {
        is_dragging = true;
      }
    }

    if (is_dragging) {
      ImVec2 child_topleft = ImGui::GetCursorScreenPos();
      frame = (ImGui::GetIO().MousePos.x - child_topleft.x - kHorizontalPadding) / pixels_per_frame;
      if (handle_type == "first") frame += 1;  // fix a quirk
      frame = std::clamp(frame, 0, num_frames - 1);
    }
  }
};

struct LdiStudio : public DearImGuiApp {
  torch::DeviceType device; // TODO: use this

  // Application preferences
  std::map<std::string, std::string> prefs;
  // LDI data
  static constexpr int kNumLayers = 3;
  cv::Mat photo_ldi3;  // 8 bit format (typically with 12 bit depth encoding).
  std::vector<cv::Mat> layer_bgra, layer_invd;  // CV_32F format.

  std::vector<std::string> ldivid_filenames;
  bool have_video_data = false;

  // Texture and channel data
  std::atomic<bool> have_any_ldi_data = false;
  std::vector<uint32_t> layer_image_texture_id;
  std::vector<uint32_t> layer_depth_texture_id;

  // Editor LDI-related data
  cv::Mat selection;
  CommandRunner command_runner;

  // ImGui Elements
  float kToolbarWidth;
  int menu_bar_height;
  void calculateGuiScale() {
    kToolbarWidth = toggle_large_text ? 422 : 312;
    kTextScale = toggle_large_text ? 1.5 : 1.0;
    kGuiScaleHack = toggle_large_text ? 1.3 : 1.0; // Some things dont scale as much as text
    menu_bar_height = toggle_large_text ? 30 : 20;
    ImGui::GetIO().FontGlobalScale = kTextScale;
  }

  std::string photo_ldi3_path;
  std::string project_dir;
  gui::ImguiCvMat editor_image;
  VideoTimelineWidget timeline;

  ImVec2 preview3d_viewport_min, preview3d_viewport_size;
  ImVec2 editor_viewport_min, editor_viewport_size;

  int editor_scroll_x = 0, editor_scroll_y = 0;
  bool editor_needs_scroll_update = false;

  int selected_channel = kLayer2RGB;
  int prev_selected_channel = selected_channel;
  bool toggle_layer_visible[kNumLayers] = {true, true, true};
  int selected_tool = kScroll;

  static constexpr float kMinBrushSize = 1.0;
  static constexpr float kMaxBrushSize = 1000.0;
  float brush_size = 25.0;
  float brush_hardness = 0.1;
  float brush_depth = 0.0;

  bool toggle_colorize_depthmaps = true;
  bool prev_toggle_colorize_depthmaps = true;

  bool toggle_sway_3d_camera =
      false;  // when enabled, the 3D camera view automatically sways back and forth

  bool toggle_play_video = false;  // when true, the video plays back as fast as possible

  bool toggle_large_text = false;

  // Vertex buffer, shaders
  opengl::GlShaderProgram basic_shader;
  opengl::GlShaderProgram ldi_shader;
  opengl::GlVertexBuffer<opengl::GlVertexDataXYZRGB> lines_vb;
  opengl::GlVertexBuffer<opengl::GlVertexDataXYZUV> ftheta_mesh_vb;

  // Mouse
  bool left_mouse_down = false;
  bool right_mouse_down = false;
  double prev_mouse_x = 0, prev_mouse_y = 0, curr_mouse_x = 0, curr_mouse_y = 0;
  bool mouse_in_3d_viewport = false;
  bool mouse_in_editor_viewport = false;
  bool mouse_in_timeline = false;
  bool click_started_in_3d_viewport = false;
  bool click_started_in_editor_viewport = false;

  // Camera
  double camera_radius, camera_theta, camera_phi;
  Eigen::Vector3f camera_orbit_center;
  double offset_cam_right, offset_cam_up;
  Eigen::Matrix4f model_view_projection_matrix;

  // External tool config
  gui::ImguiInputFileSelect ffmpeg_tool_select;
  gui::ImguiInputFileSelect python_tool_select;
  std::string ffmpeg, python3;

  void setExternalToolDefaultPaths() {
  #ifdef __APPLE__
    ffmpeg = "/usr/local/bin/ffmpeg";
    python3 = "python3";
  #else
    ffmpeg = "ffmpeg";
    python3 = "python3";
  #endif
  }

  void getExternalToolFileSelects() {
    ffmpeg = ffmpeg_tool_select.path;
    python3 = python_tool_select.path;
  }

  void setExternalToolFileSelects() {
    python_tool_select.setPath(python3.c_str());
    ffmpeg_tool_select.setPath(ffmpeg.c_str());
  }

  void makeLdiMeshAndShaders()
  {
    ldi_shader.compile(kVertexShader_LDI, kFragmentShader_LDI);
    ldi_shader.bind();  // TODO: maybe not necessary

    ftheta_mesh_vb.init();
    ftheta_mesh_vb.bind();
    opengl::GlVertexDataXYZUV::setupVertexAttributes(ldi_shader, "aVertexPos", "aVertexUV");

    static constexpr float kFthetaScale = 1.15;
    static constexpr float kFthetaInflation = 3.0;
    static constexpr int kMeshResolution = 512;
    static constexpr int kMargin = 3;
    static constexpr int kClipRadius = kMeshResolution / 2;

    std::vector<Eigen::Vector3f> verts;
    std::vector<Eigen::Vector2f> uvs;
    for (int j = 0; j < kMeshResolution; ++j) {
      for (int i = 0; i < kMeshResolution; ++i) {
        const float u = i / float(kMeshResolution - 1);
        const float v = j / float(kMeshResolution - 1);
        const float a = 2.0 * (u - 0.5);
        const float b = 2.0 * (v - 0.5);

        const float theta = atan2(b, a);
        float r = sqrt(a * a + b * b) / kFthetaScale;
        r = 0.5 * r + 0.5 * std::pow(r, kFthetaInflation);
        const float phi = r * M_PI / 2.0;

        const float x = cos(theta) * sin(phi);
        const float y = cos(phi);
        const float z = -sin(theta) * sin(phi);

        verts.emplace_back(x, y, z);
        uvs.emplace_back(u, v);
      }
    }

    for (int j = 0; j < kMeshResolution; ++j) {
      for (int i = 0; i < kMeshResolution; ++i) {
        const int di = i - kClipRadius;
        const int dj = j - kClipRadius;
        if (di * di + dj * dj > (kClipRadius - kMargin) * (kClipRadius - kMargin)) continue;
        const int a = i + (kMeshResolution - 1 + 1) * j;
        const int b = a + 1;
        const int c = a + (kMeshResolution - 1 + 1);
        const int d = c + 1;

        // Triangle b, c, a
        ftheta_mesh_vb.vertex_data.emplace_back(
            verts[b].x(), verts[b].z(), verts[b].y(), uvs[b].x(), uvs[b].y());
        ftheta_mesh_vb.vertex_data.emplace_back(
            verts[c].x(), verts[c].z(), verts[c].y(), uvs[c].x(), uvs[c].y());
        ftheta_mesh_vb.vertex_data.emplace_back(
            verts[a].x(), verts[a].z(), verts[a].y(), uvs[a].x(), uvs[a].y());
        // Triangle b, d, c
        ftheta_mesh_vb.vertex_data.emplace_back(
            verts[b].x(), verts[b].z(), verts[b].y(), uvs[b].x(), uvs[b].y());
        ftheta_mesh_vb.vertex_data.emplace_back(
            verts[d].x(), verts[d].z(), verts[d].y(), uvs[d].x(), uvs[d].y());
        ftheta_mesh_vb.vertex_data.emplace_back(
            verts[c].x(), verts[c].z(), verts[c].y(), uvs[c].x(), uvs[c].y());
      }
    }

    ftheta_mesh_vb.copyVertexDataToGPU(GL_STATIC_DRAW);
  }

  void initGlBuffersAndShaders()
  {
    basic_shader.compile(kVertexShader_Basic, kFragmentShader_Basic);
    basic_shader.bind();

    // Lines for axes
    lines_vb.init();
    lines_vb.bind();
    opengl::GlVertexDataXYZRGB::setupVertexAttributes(basic_shader, "aVertexPos", "aVertexRGB");

    lines_vb.vertex_data.emplace_back(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
    lines_vb.vertex_data.emplace_back(0.02, 0.0, 0.0, 1.0, 0.0, 0.0);
    lines_vb.vertex_data.emplace_back(0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    lines_vb.vertex_data.emplace_back(0.0, 0.02, 0.0, 0.0, 1.0, 0.0);
    lines_vb.vertex_data.emplace_back(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    lines_vb.vertex_data.emplace_back(0.0, 0.0, 0.02, 0.0, 0.0, 1.0);

    lines_vb.copyVertexDataToGPU(GL_STATIC_DRAW);

    makeLdiMeshAndShaders();

    GL_CHECK_ERROR;
  }

  void initLdiStudio()
  {
    device = util_torch::findBestTorchDevice();
    
    initGlBuffersAndShaders();
    resetCamera();
    timeline.curr_frame_change_callback = [&] {
      if (!have_video_data) return;
      if (project_dir.empty()) return;
      XCHECK_LT(timeline.curr_frame, ldivid_filenames.size());

      photo_ldi3_path = project_dir + "/" + ldivid_filenames[timeline.curr_frame];
      XPLINFO << "photo_ldi3_path=" << photo_ldi3_path;
      loadLDI3PhotoInBackgroundThread();
    };

    prefs = preferences::getPrefs();
    if (prefs.count("project_dir")) project_dir_folder_select.setPath(prefs.at("project_dir").c_str());
    
    setExternalToolDefaultPaths();
    if (prefs.count("ffmpeg")) ffmpeg = prefs.at("ffmpeg");
    if (prefs.count("python3")) python3 = prefs.at("python3");
    setExternalToolFileSelects();

    if (prefs.count("large_text_mode")) {
      toggle_large_text = (prefs.at("large_text_mode") == "1");
    }
    calculateGuiScale();
  }

  ~LdiStudio() { editor_image.freeGlTexture(); }

  void handleMouseDown(int button)
  {
    if (button == GLFW_MOUSE_BUTTON_LEFT) left_mouse_down = true;
    if (button == GLFW_MOUSE_BUTTON_RIGHT) right_mouse_down = true;

    click_started_in_3d_viewport = false;
    click_started_in_editor_viewport = false;

    if (mouse_in_3d_viewport) click_started_in_3d_viewport = true;
    if (mouse_in_editor_viewport) click_started_in_editor_viewport = true;
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

    mouse_in_editor_viewport = false;
    if (curr_mouse_x >= editor_viewport_min.x && curr_mouse_y >= editor_viewport_min.y &&
        curr_mouse_x <= editor_viewport_min.x + editor_viewport_size.x &&
        curr_mouse_y <= editor_viewport_min.y + editor_viewport_size.y) {
      mouse_in_editor_viewport = true;
    }

    mouse_in_timeline = false;
    if (curr_mouse_x >= timeline.viewport_min.x && curr_mouse_y >= timeline.viewport_min.y &&
        curr_mouse_x <= timeline.viewport_min.x + timeline.viewport_size.x &&
        curr_mouse_y <= timeline.viewport_min.y + timeline.viewport_size.y) {
      mouse_in_timeline = true;
    }

    if (isAnyModalPopupOpen() || main_menu_is_hovered) {
      mouse_in_3d_viewport = false;
      mouse_in_editor_viewport = false;
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

    if (commandWindowIsVisible()) return;
    if (isAnyModalPopupOpen()) return;
    if (main_menu_is_hovered) return;

    if (mouse_in_editor_viewport) {
      ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;

      if (selected_tool == kScroll) {
        glfwSetCursor(window, glfwCreateStandardCursor(GLFW_HAND_CURSOR));
      }
      if (selected_tool == kSelectBrush || selected_tool == kGetDepthTool ||
          selected_tool == kDepthBrushTool) {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
      }
    }
  }

  void resetCamera()
  {
    camera_radius = 0.2;
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
    static constexpr float kZNear = 0.01;
    static constexpr float kZFar = 1000.0;
    const float aspect_ratio = float(preview3d_viewport_size.x) / float(preview3d_viewport_size.y);
    const Eigen::Matrix4f projection_matrix =
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
    if (toggle_sway_3d_camera) {
      static constexpr float kSwayRadius = 0.1;
      offset += right * sin(glfwGetTime()) * kSwayRadius;
    }
    cam_pos += offset + camera_orbit_center;
    look_at += offset + camera_orbit_center;

    const Eigen::Matrix4f model_view_matrix = p11::opengl::lookAtMatrix(cam_pos, look_at, up);

    // Compute the model-view-projection matrix that will be used in shaders.
    model_view_projection_matrix = projection_matrix * model_view_matrix;

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

    updateEditor();
    updateCamera();

    const float s = ImGui::GetIO().DisplayFramebufferScale.x;  // For Retina display

    glViewport(
        s * preview3d_viewport_min.x,
        s * (preview3d_viewport_min.y - menu_bar_height + VideoTimelineWidget::kTimelineHeight),
        s * preview3d_viewport_size.x,
        s * preview3d_viewport_size.y);

    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    glDepthFunc(GL_LEQUAL);

    // Draw lines for axes
    basic_shader.bind();
    lines_vb.bind();
    glUniformMatrix4fv(
        basic_shader.getUniform("uModelViewProjectionMatrix"),
        1,
        false,
        model_view_projection_matrix.data());
    glDrawArrays(GL_LINES, 0, lines_vb.vertex_data.size());

    // Draw LDI mesh
    if (!have_any_ldi_data || opengl_textures_need_update) return;

    ldi_shader.bind();
    ftheta_mesh_vb.bind();
    glUniformMatrix4fv(
        ldi_shader.getUniform("uModelViewProjectionMatrix"),
        1,
        false,
        model_view_projection_matrix.data());

    for (int l = 0; l < kNumLayers; ++l) {
      if (!toggle_layer_visible[l]) continue;

      if (l == 0) {
        glDisable(GL_BLEND);
      } else {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBlendEquation(GL_FUNC_ADD);
      }

      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_2D, layer_image_texture_id[l]);
      glUniform1i(
          ldi_shader.getUniform("uImageTexture"),
          /*texture #=*/0);  // texture # should match the call to glActiveTexture above

      glActiveTexture(GL_TEXTURE1);
      glBindTexture(GL_TEXTURE_2D, layer_depth_texture_id[l]);
      glUniform1i(
          ldi_shader.getUniform("uDepthTexture"),
          /*texture #=*/1);  // texture # should match the call to glActiveTexture above

      glDrawArrays(GL_TRIANGLES, 0, ftheta_mesh_vb.vertex_data.size());
    }

    GL_CHECK_ERROR;
  }

  // TODO: it is redundant to have this keyboard handler and a lower-level GLfw handler.
  void handleKeyboard()
  {
    ImGuiIO& io = ImGui::GetIO();
    int delta = io.KeysDown[GLFW_KEY_LEFT_SHIFT] ? 10 : 1;
    if (io.KeysDown[GLFW_KEY_LEFT] && !load_photo_in_progress) moveTimelineByDelta(-delta);
    if (io.KeysDown[GLFW_KEY_RIGHT] && !load_photo_in_progress) moveTimelineByDelta(delta);
  }

  void updateAllLdiTextures()
  {
    for (auto& t : layer_image_texture_id) glDeleteTextures(1, &t);
    for (auto& t : layer_depth_texture_id) glDeleteTextures(1, &t);
    layer_image_texture_id.clear();
    layer_depth_texture_id.clear();

    for (int l = 0; l < kNumLayers; ++l) {
      layer_image_texture_id.push_back(opengl::makeGlTextureFromCvMat(layer_bgra[l]));
      layer_depth_texture_id.push_back(opengl::makeGlTextureFromCvMat(layer_invd[l]));
    }
  }

  void resetSelection(const float val = 0.0)
  {
    if (layer_bgra.size() > 0) selection = cv::Mat(layer_bgra[0].size(), CV_32F, cv::Scalar(val));
  }

  bool warnIfSelectionIsEmpty()
  {
    float sum = cv::sum(selection)[0];
    if (sum == 0) {
      tinyfd_messageBox(
          "No Pixels Selected",
          "The filter won't do anything unless pixels are selected.",
          "ok",
          "error",
          0);

      return true;
    }

    return false;
  }

  void setInverseDepthInSelection(const std::vector<bool>& apply_to_layer, float val)
  {
    if (!have_any_ldi_data) return;
    if (warnIfSelectionIsEmpty()) return;
    XCHECK_EQ(apply_to_layer.size(), layer_invd.size());

    cv::Mat resized_selection;
    cv::resize(selection, resized_selection, layer_invd[0].size(), 0, 0, cv::INTER_AREA);

    for (int l = 0; l < layer_invd.size(); ++l) {
      if (!apply_to_layer[l]) continue;
      cv::Mat& invd = layer_invd[l];
      for (int y = 0; y < invd.rows; ++y) {
        for (int x = 0; x < invd.cols; ++x) {
          const float s = resized_selection.at<float>(y, x);
          invd.at<float>(y, x) = s * val + (1.0 - s) * invd.at<float>(y, x);
        }
      }
    }

    updateEditorImageWithSelectedChannel();
    updateAllLdiTextures();
  }

  void setAlphaInSelection(const std::vector<bool>& apply_to_layer, float val)
  {
    if (!have_any_ldi_data) return;
    if (warnIfSelectionIsEmpty()) return;
    XCHECK_EQ(apply_to_layer.size(), layer_invd.size());

    for (int l = 0; l < layer_invd.size(); ++l) {
      if (!apply_to_layer[l]) continue;
      cv::Mat& bgra = layer_bgra[l];
      for (int y = 0; y < bgra.rows; ++y) {
        for (int x = 0; x < bgra.cols; ++x) {
          const float s = selection.at<float>(y, x);
          bgra.at<cv::Vec4b>(y, x)[3] = val * 255.0f * s + bgra.at<cv::Vec4b>(y, x)[3] * (1.0 - s);
        }
      }
    }

    updateEditorImageWithSelectedChannel();
    updateAllLdiTextures();
  }

  std::atomic<bool> load_photo_in_progress = false;
  std::atomic<bool> interupt_load = false;
  std::atomic<bool> opengl_textures_need_update = false;

  // always running, waiting for a new photo to come in. well start it the
  // first time we need it and then re-use it.
  std::shared_ptr<std::thread> load_photo_thread;  
                          
  void loadLDI3PhotoInBackgroundThread()
  {
    // The first time we run this function, start a singleton thread that will run forever to handle
    // loading photos and updating textures.
    if (load_photo_thread == nullptr) {
      load_photo_thread = std::make_shared<std::thread>([&] {
        while (true) {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));  // avoid spamming
          if (!load_photo_in_progress) continue;
          if (interupt_load) {
            interupt_load = false;
            continue;
          }

          cv::Mat new_photo_ldi3 = turbojpeg::readJpeg(photo_ldi3_path);

          if (new_photo_ldi3.empty()) {
            tinyfd_messageBox(
                "LDI3 frame data is corrupt",
                ("Empty image at: " + photo_ldi3_path + "\nThis can happen due to cancelling ffmpeg.").c_str(),
                "ok",
                "error",
                0);
            load_photo_in_progress = false;
            interupt_load = false;
            have_any_ldi_data = false;
            continue;
          }

          if (interupt_load) {
            interupt_load = false;
            continue;
          }

          std::vector<cv::Mat> new_layer_bgra, new_layer_invd;
          ldi::unpackLDI3(new_photo_ldi3, new_layer_bgra, new_layer_invd);

          if (interupt_load) {
            interupt_load = false;
            continue;
          }

          // No turning back now, we have all of the data.
          photo_ldi3 = new_photo_ldi3;
          layer_bgra = new_layer_bgra;
          layer_invd = new_layer_invd;
          if (selection.size() != layer_bgra[0].size()) resetSelection();

          have_any_ldi_data = true;
          opengl_textures_need_update = true;
          load_photo_in_progress = false;
          interupt_load = false;
        }
      });
      load_photo_thread->detach();
    }

    // Request a load from the worker thread.
    // TODO: don't interupt if we are already loading the right frame. I dont know how often this
    // happens.
    if (load_photo_in_progress) interupt_load = true;
    load_photo_in_progress = true;
  }

  void moveTimelineByDelta(int delta)
  {
    if (!have_video_data) return;
    if (project_dir.empty()) return;

    timeline.curr_frame += delta;
    if (timeline.curr_frame < 0) timeline.curr_frame = 0;
    if (timeline.curr_frame >= timeline.num_frames) {
      timeline.curr_frame = timeline.num_frames - 1;
      toggle_play_video = false;
    }

    photo_ldi3_path = project_dir + "/" + ldivid_filenames[timeline.curr_frame];
    loadLDI3PhotoInBackgroundThread();
  }

  void handleMenuOpenLDI3Photo()
  {
    restoreDefaultCursor();

    const char* cpath = tinyfd_openFileDialog(
        "Select an LDI3 photo (.jpg or .png usually)", nullptr, 0, nullptr, nullptr, 0);
    if (cpath != nullptr) {
      photo_ldi3_path = std::string(cpath);
      XPLINFO << "path: " << photo_ldi3_path;
      loadLDI3PhotoInBackgroundThread();
    }
  }

  void reloadCurrentLDI3Photo()
  {
    if (photo_ldi3_path.empty()) return;
    XPLINFO << "Reloading " << photo_ldi3_path;
    loadLDI3PhotoInBackgroundThread();
  }

  void saveCurrentFrameAsLDI3(bool interactive) {
    if (!have_video_data) return;
    if (project_dir.empty()) return;
    const std::string default_path = project_dir + "/ldi3.png";
    std::string path = default_path;
    if (interactive) {
      path = tinyfd_saveFileDialog("Select LDI3 photo output path (.jpg or .png)", default_path.c_str(), 0, nullptr, nullptr);
    }

    cv::Mat ldi3_encoded = ldi::make6DofGrid(
      layer_bgra,
      layer_invd,
      "split12",
      {},
      /*dilate_invd=*/false);
    static constexpr int kJpegQuality = 100;
    turbojpeg::writeJpeg(path.c_str(), ldi3_encoded, kJpegQuality);

    if (interactive) {
      tinyfd_messageBox(
        "LDI3 frame saved",
        ("The LDI3 frame is at: " + path).c_str(),
        "ok",
        "info",
        1);
    }
  }

  std::shared_ptr<std::atomic<bool>> cancel_export_obj = std::make_shared<std::atomic<bool>>(false);
  void exportCurrentFrameAsOBJ() {
    if (!have_any_ldi_data) return;
    if (project_dir.empty()) return;

    const std::string default_path = project_dir + "/my_scene.obj";
    const char* dest_path = tinyfd_saveFileDialog("Select output path (.obj)", default_path.c_str(), 0, nullptr, nullptr);
    if (dest_path == nullptr) return;

    vr180::ConvertToOBJConfig cfg;
    cfg.input_vid = ""; // unused
    cfg.output_obj = dest_path;
    cfg.ftheta_size = 1920;
    cfg.ftheta_scale = 1.15;
    cfg.ftheta_inflation = 3;
    cfg.inv_depth_encoding_coef = 0.3;
    static constexpr int kTopLayer = 2;

    command_runner.setCompleteOrKilledCallback([] { XPLINFO << "exportCurrentFrameAsOBJ finished or cancelled"; });
    command_runner.queueThreadCommand(
      cancel_export_obj, // TODO: writeTexturedMeshObj ignores cancel requests
      [&, cfg] {
        XPLINFO << "Generating OBJ...";
        vr180::writeTexturedMeshObj(cfg, layer_bgra, layer_invd);
      });
    command_runner.runCommandQueue();
  }

  // We often want to call scanProjectDirForVideoFrames after a command completes,
  // but doing so is problematic because this executes in a different thread from the
  // main thread, so it can't easily update OpenGL textures. Instead, complete callbacks
  // should request a scan by setting this to true, and it will be done later in the main thread.
  std::atomic<bool> scan_project_dir_requested = false;

  // WARNING: do not try to call this outside of the main thread, OpenGL segfaults may occur.
  void scanProjectDirForVideoFrames()
  {
    have_any_ldi_data = false;
    have_video_data = false;
    timeline.num_frames = 0;
    timeline.curr_frame = 0;

    std::vector<std::string> filenames = file::getFilesInDir(project_dir);
    std::regex pattern("^ldi3_([0-9]{6})\\.(png|jpg)");
    ldivid_filenames.clear();
    std::vector<int> frame_numbers;
    for (const std::string& f : filenames) {
      std::smatch match;
      if (std::regex_search(f, match, pattern)) {
        ldivid_filenames.push_back(f);
        frame_numbers.push_back(std::atoi(match[1].str().c_str()));
      }
    }

    if (!ldivid_filenames.empty()) {
      int min_frame_num = *std::min_element(frame_numbers.begin(), frame_numbers.end());
      int max_frame_num = *std::max_element(frame_numbers.begin(), frame_numbers.end());

      if (min_frame_num != 0) {
        tinyfd_messageBox(
            "Video Frame Data Error",
            "Cannot find frame 0, it should be ldi3_000000.png or jpg.",
            "ok",
            "error",
            0);;
      } else if (!ldivid_filenames.empty() && ldivid_filenames.size() - 1 == max_frame_num - min_frame_num) {
        have_video_data = true;
        timeline.num_frames = ldivid_filenames.size();
      } else {
        tinyfd_messageBox(
            "Video Frame Data Error",
            "Total number of frames does not match first and last frame, maybe there is a missing "
            "frame.",
            "ok",
            "error",
            0);
      }
    }

    timeline.curr_frame_change_callback();
    updateEditorImageWithSelectedChannel();
  }

  void handleMenuSelectProjectDir()
  {
    const char* cpath =
        tinyfd_selectFolderDialog("Select a project folder (for video data)", nullptr);
    if (cpath != nullptr) {
      project_dir = std::string(cpath);
      prefs["project_dir"] = project_dir;
      preferences::setPrefs(prefs);
      if (kEnableAutoLogging) xpl::stdoutLogger.attachTextFileLog(project_dir + "/log.txt");

      scanProjectDirForVideoFrames();
    }
  }

  void handleMenuImportLDI3Video()
  {
    if (project_dir.empty()) {
      tinyfd_messageBox(
          "No Project Dir Set",
          "Select a project directory before importing video (File -> Select Project Dir).",
          "ok",
          "error",
          0);
      return;
    }

    if (!checkFFmpegInstallAndWarnIfNot()) return;

    const char* cpath =
        tinyfd_openFileDialog("Select an LDI3 video (.mp4, .mov)", nullptr, 0, nullptr, nullptr, 0);
    if (cpath != nullptr) {
      std::string ldi3_video_path = std::string(cpath);
      XPLINFO << "video path: " << ldi3_video_path;

      command_runner.setRedirectStdErr(true);
      command_runner.setCompleteOrKilledCallback(
        [&] { scan_project_dir_requested = true; });
      command_runner.queueShellCommand(
        ffmpeg + " -y -i " + ldi3_video_path + " -vn -c:a aac -b:a 192k " + project_dir + "/audio.aac");
      command_runner.queueShellCommand(
        ffmpeg + " -y -i " +  ldi3_video_path + " -progress pipe:1 -start_number 0 " +  project_dir + "/ldi3_%06d.png");
      command_runner.runCommandQueue();
    }
  }

  void timelineSelectCurrFrame()
  {
    timeline.first_frame = timeline.curr_frame;
    timeline.last_frame = timeline.curr_frame;
  }

  void timelineSelectAllFrames()
  {
    timeline.first_frame = 0;
    timeline.last_frame = timeline.num_frames - 1;
  }

  std::string formatVersionNumber(float v) {
    std::string ver = std::to_string(v);
    ver.erase(ver.find_last_not_of('0') + 1, std::string::npos);
    ver.erase(ver.find_last_not_of('.') + 1, std::string::npos);    
    return ver;
  }

  void showAboutVVEDialog() {
    tinyfd_messageBox(
      "Volurama by Lifecast",
      ("Version: " + formatVersionNumber(kSoftwareVersion) +
      "\nTorch Device: " + util_torch::deviceTypeToString(device)).c_str(), "ok", "info", 1);
  }

  bool main_menu_is_hovered = false;
  void drawMainMenu()
  {
    if (ImGui::BeginMenuBar()) {
      if (ImGui::BeginMenu("File")) {
        if (ImGui::MenuItem("Select Project Directory")) {
          handleMenuSelectProjectDir();
        }

        ImGui::Dummy(ImVec2(0, 8));
        ImGui::Separator();
        ImGui::Dummy(ImVec2(0, 8));

        if (ImGui::MenuItem("Open LDI3 Photo")) {
          handleMenuOpenLDI3Photo();
        }

        //if (ImGui::MenuItem("Reload from Disk", "R")) { // TODO: choose a different key, maybe F5
        //  reloadCurrentLDI3Photo();
        //}
        
        if (ImGui::MenuItem("Save current frame as LDI3")) {
          saveCurrentFrameAsLDI3(true);
        }

        if (ImGui::MenuItem("Export current frame as OBJ")) {
          exportCurrentFrameAsOBJ();
        }

        ImGui::Dummy(ImVec2(0, 8));
        ImGui::Separator();
        ImGui::Dummy(ImVec2(0, 8));

        if (ImGui::MenuItem("Import LDI3 Video (already rendered)")) {
          handleMenuImportLDI3Video();
        }

        if (ImGui::MenuItem("Import VR180, Render LDI3")) {
          open_popup_render_vr180_ldi = true;
        }

        if (ImGui::MenuItem("Encode Compressed LDI3 Video")) {
          open_popup_encode_video = true;
        }

        ImGui::Dummy(ImVec2(0, 8));
        ImGui::EndMenu();
      }

      if (ImGui::BeginMenu("Channel View")) {
        if (ImGui::MenuItem("Select All")) {
          resetSelection(1.0);
          updateEditorImageWithSelectedChannel();
        }

        if (ImGui::MenuItem("Select None")) {
          resetSelection(0.0);
          updateEditorImageWithSelectedChannel();
        }

        ImGui::Dummy(ImVec2(0, 8));
        ImGui::Separator();
        ImGui::Dummy(ImVec2(0, 8));

        prev_toggle_colorize_depthmaps = toggle_colorize_depthmaps;
        if (ImGui::MenuItem("Colorize Depth Maps", nullptr, &toggle_colorize_depthmaps)) {
        }
        if (prev_toggle_colorize_depthmaps != toggle_colorize_depthmaps) {
          updateEditorImageWithSelectedChannel();
        }

        ImGui::Dummy(ImVec2(0, 8));
        ImGui::EndMenu();
      }

      if (ImGui::BeginMenu("3D View")) {
        if (ImGui::MenuItem("Reset Camera")) {
          resetCamera();
        }

        if (ImGui::MenuItem("Sway 3D Camera", nullptr, &toggle_sway_3d_camera)) {
        }

        ImGui::Dummy(ImVec2(0, 8));
        ImGui::Separator();
        ImGui::Dummy(ImVec2(0, 8));

        if (ImGui::MenuItem("Show Layer 2", "3", &toggle_layer_visible[2])) {
        }
        if (ImGui::MenuItem("Show Layer 1", "2", &toggle_layer_visible[1])) {
        }
        if (ImGui::MenuItem("Show Layer 0", "1", &toggle_layer_visible[0])) {
        }
        
        ImGui::Dummy(ImVec2(0, 8));
        ImGui::EndMenu();
      }

      if (ImGui::BeginMenu("Timeline")) {
        if (ImGui::MenuItem("Play", "Space", &toggle_play_video)) {
        }

        ImGui::Dummy(ImVec2(0, 8));
        ImGui::Separator();
        ImGui::Dummy(ImVec2(0, 8));

        if (ImGui::MenuItem("Select Current Frame")) {
          timelineSelectCurrFrame();
        }
        if (ImGui::MenuItem("Select All Frames")) {
          timelineSelectAllFrames();
        }

        ImGui::Dummy(ImVec2(0, 8));
        ImGui::EndMenu();
      }

      if (ImGui::BeginMenu("Filter")) {
        ImGui::TextColored(ImVec4(0.4, 0.4, 1.0, 1.0), "Current Frame");
        ImGui::Dummy(ImVec2(0, 8));

        if (ImGui::MenuItem("Set Depth In Selection")) {
          set_invd_value = brush_depth;  // Pre-set the value to the brush depth value
          open_popup_set_invd = true;
        }
        if (ImGui::MenuItem("Set Alpha In Selection")) open_popup_set_alpha = true;

        ImGui::Dummy(ImVec2(0, 8));
        ImGui::Separator();
        ImGui::Dummy(ImVec2(0, 8));

        ImGui::TextColored(ImVec4(0.5, 1.0, 0.5, 1.0), "Selected Frames");
        ImGui::Dummy(ImVec2(0, 8));

        if (ImGui::MenuItem("Copy Channels To Selected Frames")) {
          open_popup_copy_channels_to_frames = true;
        }

        ImGui::Dummy(ImVec2(0, 8));
        ImGui::EndMenu();
      }

      if (ImGui::BeginMenu("Settings")) {
        if (ImGui::MenuItem("About Volumetric Video Editor")) {
          showAboutVVEDialog();
        }

        if (ImGui::MenuItem("Larger Text", "", &toggle_large_text)) {
          prefs["large_text_mode"] = toggle_large_text ? "1" : "0";
          preferences::setPrefs(prefs);
          calculateGuiScale();
        }

        ImGui::Dummy(ImVec2(0, 8));
        ImGui::Separator();
        ImGui::Dummy(ImVec2(0, 8));

        if (ImGui::MenuItem("Configure ffmpeg, python")) {
          open_popup_config_tools = true;
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
        if (ImGui::MenuItem("Check python3 installation")) {
          if (checkPython3InstallAndWarnIfNot()) {
            tinyfd_messageBox(
                "python3 is installed OK",
                "python3 appears to be installed correctly.",
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

      if (ImGui::BeginMenu("WebXR")) {
        if (ImGui::MenuItem("View Selected Frame in WebXR")) {
          runWebXRCurrentFrame();
        }

        if (ImGui::MenuItem("View Full Video in WebXR")) {
          runWebXRVideo();
        }

        ImGui::Dummy(ImVec2(0, 8));
        ImGui::EndMenu();
      }

      // NOTE: I don't know why this is negated.
      main_menu_is_hovered = !ImGui::IsWindowHovered(ImGuiHoveredFlags_RootAndChildWindows);

      ImGui::EndMenuBar();
    }
  }

  bool checkFFmpegInstall(std::string& output) {
    output = execBlockingWithOutput(ffmpeg + " -version");

    XPLINFO << "ffmpeg -version\n\n" << output;
    bool found_ffmpeg_version = output.find("ffmpeg version") != std::string::npos;
    return found_ffmpeg_version;
  }

  bool checkPython3Install(std::string& output) {
    output = execBlockingWithOutput(python3 + " --version");
    XPLINFO << "python3 --version\n\n" << output;
    bool found_python_version = output.find("Python 3") != std::string::npos;
    return found_python_version;
  }

  float getVideoDuration(const std::string& file_path) {
    cv::VideoCapture cap(file_path);
    if (!cap.isOpened()) {
        XPLINFO << "Error opening video file: " << file_path << std::endl;
        return -1.0;
    }
    
    const double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) {
        XPLINFO << "Error: FPS could not be determined" << std::endl;
        return -1.0;
    }

    double frame_count = cap.get(cv::CAP_PROP_FRAME_COUNT);
    if (frame_count <= 0) {
        XPLINFO << "Error: Frame count could not be determined" << std::endl;
        return -1.0;
    }
    cap.release();
    return frame_count / fps;
  }

  bool checkFFmpegInstallAndWarnIfNot() {
    std::string output;
    bool installed = checkFFmpegInstall(output);
    if (!installed) {
      tinyfd_messageBox(
          "ffmpeg not found",
          std::string("ffmpeg is required to encode and decode video.\nYou must install ffmpeg separately.\nffmpeg test output:" + output).c_str(),
          "ok",
          "error",
          1);
    }
    return installed;
  }

  bool checkPython3InstallAndWarnIfNot() {
    std::string output;
    bool installed = checkPython3Install(output);
    if (!installed) {
      tinyfd_messageBox(
          "python3 not found",
          ("python3 is required to run a local web server.\npython3 test command output: " + output).c_str(),
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

  void runWebXRCurrentFrame() {
    saveCurrentFrameAsLDI3(false);

    // Write kLifecastPlayerHTML to photo.html in the project directory
    std::ofstream client_html(project_dir + "/photo.html");
    client_html << kLifecastPlayerHTML;
    client_html.close();

    runLocalWebserver();
    browser::openURL("http://localhost:8000/photo.html");
  }

  void runWebXRVideo() {
    // TODO: Automatically run FFmpeg? Or, check if .mp4 files exist, then prompt the user to run FFmpeg?
    //encodeVideoFromFramesWithFfmpeg();

    // Write kLifecastPlayerHTML to video.html in the project directory
    std::ofstream client_html(project_dir + "/video.html");
    client_html << kLifecastPlayerVideoHTML;
    client_html.close();

    runLocalWebserver();
    browser::openURL("http://localhost:8000/video.html");
  }


  httplib::Server http_svr;
  void startServer(const std::string& project_dir) {
      http_svr.set_mount_point("/", project_dir.c_str()); // Serve static files from project directory

      // Add CORS headers
      http_svr.set_pre_routing_handler([](const httplib::Request &req, httplib::Response &res) {
        XPLINFO << "Request: " << req.method << " " << req.path;
        res.set_header("Access-Control-Allow-Origin", "*");
        return httplib::Server::HandlerResponse::Unhandled;
      });

      // Run http_svr.listen() in a separate thread, then return to the cancel check loop
      std::thread server_thread([&] {
        XPLINFO << "Serving Files at " << project_dir;
        XPLINFO << "Webserver starting at http://localhost:8000";
        http_svr.listen("0.0.0.0", 8000);
      });
      server_thread.detach();
  }

  std::shared_ptr<std::atomic<bool>> cancel_webserver_requested = std::make_shared<std::atomic<bool>>(false);
  void runLocalWebserver() {

    command_runner.setCompleteOrKilledCallback(
      [&] { XPLINFO  << "Local web server stopped."; });
    command_runner.queueThreadCommand(cancel_webserver_requested,
      [&] {
        startServer(project_dir);
        // Spin in a loop waiting for cancel_webserver_requested to be set
        while(true) {
          if (*cancel_webserver_requested) {
            XPLINFO  << "Cancel button pressed; Stopping HTTP server";
            http_svr.stop();
            return;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
      });
    command_runner.runCommandQueue();
  }

  // Takes the color, alpha, and depthmap from the current frame, and within the selection mask,
  // blends that into the depthmaps of all other frames selected in the timeline, updates their
  // LDI's and saves them.
  std::shared_ptr<std::atomic<bool>> cancel_copy_channel_data_requested = std::make_shared<std::atomic<bool>>(false);
  void copyChannelDataToSelectedFrames()
  {
    if (!have_any_ldi_data) return;
    if (!have_video_data) return;
    if (project_dir.empty()) return;
    if (warnIfSelectionIsEmpty()) return;

    command_runner.setCompleteOrKilledCallback(
      [&] { XPLINFO  << "copyChannelDataToSelectedFrames finished or cancelled"; });
    command_runner.queueThreadCommand(
        cancel_copy_channel_data_requested,
        [&] {
          cv::Mat selection_small;
          cv::resize(selection, selection_small, layer_invd[0].size(), 0.0, 0.0, cv::INTER_AREA);

          for (int f = timeline.first_frame; f <= timeline.last_frame; ++f) {
            if (*cancel_copy_channel_data_requested) return;

            XPLINFO << "Processing frame: " << f;
            const std::string frame_filename = project_dir + "/" + ldivid_filenames[f];
            cv::Mat frame_ldi3 = turbojpeg::readJpeg(frame_filename);

            std::vector<cv::Mat> frame_layer_bgra, frame_layer_invd;
            ldi::unpackLDI3(frame_ldi3, frame_layer_bgra, frame_layer_invd);

            for (int l = 0; l < kNumLayers; ++l) {
              XCHECK_EQ(layer_invd[l].size(), frame_layer_invd[l].size());
              if (layer_copy_invd[l]) {
                for (int y = 0; y < layer_invd[l].rows; ++y) {
                  for (int x = 0; x < layer_invd[l].cols; ++x) {
                    const float s = selection_small.at<float>(y, x);
                    frame_layer_invd[l].at<float>(y, x) =
                        s * layer_invd[l].at<float>(y, x) +
                        (1.0f - s) * frame_layer_invd[l].at<float>(y, x);
                  }
                }
              }
              if (layer_copy_rgb[l] || layer_copy_a[l]) {
                int first_channel = layer_copy_rgb[l] ? 0 : 3;
                int last_channel = layer_copy_a[l] ? 3 : 2;
                for (int y = 0; y < layer_bgra[l].rows; ++y) {
                  for (int x = 0; x < layer_bgra[l].cols; ++x) {
                    const float s = selection.at<float>(y, x);
                    for (int c = first_channel; c <= last_channel; ++c) {
                      frame_layer_bgra[l].at<cv::Vec4b>(y, x)[c] =
                          s * layer_bgra[l].at<cv::Vec4b>(y, x)[c] +
                          (1.0f - s) * frame_layer_bgra[l].at<cv::Vec4b>(y, x)[c];
                    }
                  }
                }
              }
            }

            cv::Mat new_frame_ldi3 = ldi::make6DofGrid(
                frame_layer_bgra,
                frame_layer_invd,
                "split12",
                {},
                /*dilate_invd=*/false);

            if (cancel_copy_channel_data_requested && *cancel_copy_channel_data_requested) return;
            cv::imwrite(frame_filename, new_frame_ldi3);
          }
        });
    command_runner.runCommandQueue();
  }

  void encodeVideoFromFramesWithFfmpeg()
  {
    if (project_dir.empty()) return;
    if (!have_video_data) return;

    if (!checkFFmpegInstallAndWarnIfNot()) return;

    bool encode_audio = file::fileExists(project_dir + "/audio.aac");

    command_runner.setRedirectStdErr(true);
    command_runner.setCompleteCallback([] { 
        tinyfd_messageBox(
            "Video Encoding Complete",
            "Output video files are in the project directory.",
            "ok",
            "info",
            1);
    });
    command_runner.setKilledCallback([] { 
        tinyfd_messageBox(
            "Video Encoding Cancelled",
            "Partial results may be in the project directory.",
            "ok",
            "warn",
            1);
    });

    std::string framerate = std::to_string(encode_video_fps);
    std::string audio_input = encode_audio ? " -i " + project_dir + "/audio.aac " : " ";
    std::string audio_codec = encode_audio ? " -c:a aac -map 1:a:0 " : " ";

    if (encode_h264_hq) {
        command_runner.queueShellCommand(
            ffmpeg + " -y -framerate " + framerate + " -i " + project_dir + "/ldi3_%06d.png" + audio_input +
            "-progress pipe:1 -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p -movflags faststart -map 0:v:0" + audio_codec +
            project_dir + "/ldi3_h264_high_quality_5760x5760.mp4");
    }

    if (encode_h264_streaming) {
        command_runner.queueShellCommand(
            ffmpeg + " -y -framerate " + framerate + " -i " + project_dir + "/ldi3_%06d.png" + audio_input +
            "-progress pipe:1 -c:v libx264 -preset fast -crf 29 -x264-params mvrange=511 -maxrate 50M -bufsize 25M " +
            "-pix_fmt yuv420p -movflags faststart -map 0:v:0" + audio_codec +
            project_dir + "/ldi3_h264_5760x5760.mp4");
    }

    if (encode_h264_mobile) {
        command_runner.queueShellCommand(
            ffmpeg + " -y -framerate " + framerate + " -i " + project_dir + "/ldi3_%06d.png" + audio_input +
            "-progress pipe:1 -vf scale=1920:1920 -c:v libx264 -preset fast -crf 29 -pix_fmt yuv420p -movflags faststart -map 0:v:0" + audio_codec +
            project_dir + "/ldi3_h264_1920x1920.mp4");
    }

    if (encode_h264_looking_glass) {
        command_runner.queueShellCommand(
            ffmpeg + " -y -framerate " + framerate + " -i " + project_dir + "/ldi3_%06d.png" + audio_input +
            "-progress pipe:1 -vf scale=3840:3840 -c:v libx264 -preset fast -crf 24 -pix_fmt yuv420p -movflags faststart -map 0:v:0" + audio_codec +
            project_dir + "/ldi3_h264_3840x3840.mp4");
    }

    if (encode_h265_oculus) {
        command_runner.queueShellCommand(
            ffmpeg + " -y -framerate " + framerate + " -i " + project_dir + "/ldi3_%06d.png" + audio_input +
            "-progress pipe:1 -c:v libx265 -preset medium -crf 29 -pix_fmt yuv420p -movflags faststart -profile:v main -tag:v hvc1 -map 0:v:0" + audio_codec +
            project_dir + "/ldi3_h265_hvc1_5760x5760.mp4");
    }

    if (encode_prores) {
        command_runner.queueShellCommand(
            ffmpeg + " -y -framerate " + framerate + " -i " + project_dir + "/ldi3_%06d.png" + audio_input +
            "-progress pipe:1 -c:v prores_ks -profile:v 1 -vendor apl0 -map 0:v:0" + audio_codec +
            project_dir + "/ldi3_prores.mov");
    }

    command_runner.runCommandQueue();
  }

  void renderVR180toLDI3() {
    vr180toLdi3_cfg.cancel_requested = std::make_shared<std::atomic<bool>>(false);
    vr180toLdi3_cfg.cwd = "";
    vr180toLdi3_cfg.src_vr180 = vr180toLdi3_input_file_select.path;
    vr180toLdi3_cfg.src_ftheta_image = "";
    vr180toLdi3_cfg.src_ftheta_depth = "";
    vr180toLdi3_cfg.dest_dir = project_dir;
    vr180toLdi3_cfg.output_filename = "ldi3_000000.png";
    vr180toLdi3_cfg.rm_dest_dir = false;
    vr180toLdi3_cfg.ftheta_size = 3840;
    vr180toLdi3_cfg.inflated_ftheta_size = 1920;
    vr180toLdi3_cfg.rectified_size_for_depth = 1280;
    vr180toLdi3_cfg.disparity_bias = 0.0;
    vr180toLdi3_cfg.baseline_m = vr180toLdi3_baseline_cm * 0.01;
    vr180toLdi3_cfg.inv_depth_coef = 0.3;
    vr180toLdi3_cfg.ftheta_scale = 1.15;
    vr180toLdi3_cfg.inpaint_method = "ceres";
    vr180toLdi3_cfg.seg_method = "heuristic";
    vr180toLdi3_cfg.sd_ver = "v2";
    vr180toLdi3_cfg.first_frame = 0;
    vr180toLdi3_cfg.last_frame = -1;
    vr180toLdi3_cfg.phase = "all";
    vr180toLdi3_cfg.stabilize_inpainting = vr180toLdi3_stabilize_inpainting;
    vr180toLdi3_cfg.run_seg_only = false;
    vr180toLdi3_cfg.write_seg = false;
    vr180toLdi3_cfg.use_cached_seg = false;
    vr180toLdi3_cfg.output_encoding = "split12";
    vr180toLdi3_cfg.make_fused_image = false;
    vr180toLdi3_cfg.inpaint_dilate_radius = 25;
    vr180toLdi3_cfg.skip_every_other_frame = false;

    if (!file::fileExists(vr180toLdi3_cfg.src_vr180)) {
      tinyfd_messageBox(
        "File Not Found",
        vr180toLdi3_cfg.src_vr180.c_str(),
        "ok",
        "error",
        1);
      return;
    }

    const std::string ext = file::filenameExtension(vr180toLdi3_cfg.src_vr180);
    std::string mode;
    if (ext == "png" || ext == "jpg" || ext == "bmp" || ext == "tiff") {
      mode = "photo";
    } else if (ext == "mov" || ext == "mp4" || ext == "mkv") {
      mode = "video";
    } else {
      tinyfd_messageBox(
        "Invalid input",
        std::string("Unrecognized file type: " + ext + " (Supported types: .mov, .mp4, .mkv, .png, .jpg, .bmp, .tiff)").c_str(),
        "ok",
        "error",
        1);
      return;
    }
    XPLINFO << "mode=" << mode;
    printConfig(vr180toLdi3_cfg);

    float video_duration = getVideoDuration(vr180toLdi3_cfg.src_vr180);
    XPLINFO << "video_duration=" << video_duration;

    command_runner.setCompleteCallback([&, mode] { 
        tinyfd_messageBox(
            "Render VR180 to LDI3 Complete",
            "Output frames are in the project directory.",
            "ok",
            "info",
            1);
        if (mode == "photo") {
          have_video_data = false;
          photo_ldi3_path = project_dir + "/" + vr180toLdi3_cfg.output_filename;
          loadLDI3PhotoInBackgroundThread();
        }
        if (mode == "video") {
          scan_project_dir_requested = true;
        }
      });
    command_runner.setKilledCallback([&] { 
        tinyfd_messageBox(
            "Render VR180 to LDI3 Cancelled",
            "Partial results may be in the project directory.",
            "ok",
            "warn",
            1); 
        scan_project_dir_requested = true;
      });

    // Attempt to extract audio from the input video.
    if (mode == "video") {
      command_runner.queueShellCommand(
        ffmpeg + " -y -i " + vr180toLdi3_cfg.src_vr180 + " -vn -c:a aac -b:a 192k " + project_dir + "/audio.aac");
    }

    command_runner.queueThreadCommand(
      vr180toLdi3_cfg.cancel_requested,
      [&, mode] {
        if (mode == "photo") {
          ldi::runVR180PhototoLdiPipeline(vr180toLdi3_cfg);
        } 
        if (mode == "video") {
          ldi::runVR180toLdi3VideoPipelineAllPhases(vr180toLdi3_cfg);
        }
      });
    command_runner.runCommandQueue();
  }

  std::tuple<int, int, int, int, int> computeBrushRect(
      const int image_width, const int image_height, float brush_size, float mouse_x, float mouse_y)
  {
    int brush_radius = brush_size + 1;
    int min_x = std::max(0, int(mouse_x) - brush_radius);
    int min_y = std::max(0, int(mouse_y) - brush_radius);
    int max_x = std::min(image_width - 1, int(mouse_x) + brush_radius);
    int max_y = std::min(image_height - 1, int(mouse_y) + brush_radius);
    return {brush_radius, min_x, min_y, max_x, max_y};
  }

  void useSelectionBrushTool()
  {
    const auto& [brush_radius, min_x, min_y, max_x, max_y] = computeBrushRect(
        selection.cols,
        selection.rows,
        brush_size,
        mouse_in_editor_coords_x,
        mouse_in_editor_coords_y);

    bool negative = ImGui::GetIO().KeysDown[GLFW_KEY_LEFT_SHIFT];

    float pseudo_hardness = brush_hardness >= 1.0 ? brush_hardness : 1.0;
    float brush_flow = brush_hardness < 1.0 ? brush_hardness : 1.0;

    for (int y = min_y; y <= max_y; ++y) {
      for (int x = min_x; x <= max_x; ++x) {
        const float dx = x - mouse_in_editor_coords_x;
        const float dy = y - mouse_in_editor_coords_y;
        const float dist = sqrt(dx * dx + dy * dy);
        const float fade =
            std::pow(math::clamp(1.0f - dist / brush_size, 0.0f, 1.0f), 1.0 / pseudo_hardness) *
            brush_flow;

        selection.at<float>(y, x) += fade * (negative ? -1.0 : 1.0);
        selection.at<float>(y, x) = math::clamp(selection.at<float>(y, x), 0.0f, 1.0f);
      }
    }

    updateEditorImageWithSelectedChannel();
  }

  cv::Mat getDepthmapForSelectedChannel()
  {
    switch (selected_channel) {
      case kLayer0Depth:
        return layer_invd[0];
      case kLayer0RGBA:
        return layer_invd[0];
      case kLayer1Depth:
        return layer_invd[1];
      case kLayer1RGBA:
        return layer_invd[1];
      case kLayer2Depth:
        return layer_invd[2];
      case kLayer2RGBA:
        return layer_invd[2];
      default:
        return cv::Mat();
    }
  }

  void useDepthBrushTool()
  {
    cv::Mat depthmap = getDepthmapForSelectedChannel();
    if (depthmap.empty())
      return;  // TODO: This might happen if the user uses this tool while in selection view. No
               // reasonable action to take. Warn the user somehow, or make this impossible.

    const auto& [brush_radius, min_x, min_y, max_x, max_y] = computeBrushRect(
        depthmap.cols,
        depthmap.rows,
        brush_size / 2.0,
        mouse_in_editor_coords_x / 2.0,
        mouse_in_editor_coords_y / 2.0);

    float pseudo_hardness = brush_hardness >= 1.0 ? brush_hardness : 1.0;
    float brush_flow = brush_hardness < 1.0 ? brush_hardness : 1.0;
    const float half_size = 0.5;
    for (int y = min_y; y <= max_y; ++y) {
      for (int x = min_x; x <= max_x; ++x) {
        const float dx = x - mouse_in_editor_coords_x * half_size;
        const float dy = y - mouse_in_editor_coords_y * half_size;
        const float dist = sqrt(dx * dx + dy * dy);
        const float fade = std::pow(
                               math::clamp(1.0f - dist / (brush_size * half_size), 0.0f, 1.0f),
                               1.0 / pseudo_hardness) *
                           brush_flow;

        depthmap.at<float>(y, x) = fade * brush_depth + (1.0 - fade) * depthmap.at<float>(y, x);
        depthmap.at<float>(y, x) = math::clamp(depthmap.at<float>(y, x), 0.0f, 1.0f);

        // enforce layer-ordering
        for (int l = 1; l >= 0; --l) {
          static constexpr float kEps = 1.0 / 255.0f;
          // Only enforce if alpha is non-zero in the upper layer.
          if (layer_bgra[l + 1].at<cv::Vec4b>(y, x)[3] > 0) {
            layer_invd[l].at<float>(y, x) = std::max(
                0.0f,
                std::min(layer_invd[l].at<float>(y, x), layer_invd[l + 1].at<float>(y, x) - kEps));
          }
        }
      }
    }

    updateEditorImageWithSelectedChannel();
    updateAllLdiTextures();
  }

  void useGetDepthTool()
  {
    cv::Mat src_depthmap = getDepthmapForSelectedChannel();
    if (src_depthmap.empty())
      return;  // TODO: This might happen if the user uses this tool while in selection view. No
               // reasonable action to take. Warn the user somehow, or make this impossible.

    static constexpr float kScale = 2;
    brush_depth = opencv::getPixelBilinear<float>(
        src_depthmap, mouse_in_editor_coords_x / kScale, mouse_in_editor_coords_y / kScale);
  }

  void updateEditor()
  {
    if (!have_any_ldi_data) return;

    brush_size = math::clamp(brush_size, 0.5f, 1000.0f);
    brush_hardness = math::clamp(brush_hardness, 0.01f, 10.0f);

    if (click_started_in_editor_viewport) {
      if (selected_tool == kScroll && left_mouse_down) {
        const float dx = prev_mouse_x - curr_mouse_x;
        const float dy = prev_mouse_y - curr_mouse_y;
        editor_scroll_x += dx;
        editor_scroll_y += dy;
        editor_needs_scroll_update = true;
      }

      if (selected_tool == kSelectBrush && left_mouse_down) {
        useSelectionBrushTool();
      }

      if (selected_tool == kGetDepthTool && left_mouse_down) {
        useGetDepthTool();
      }

      if (selected_tool == kDepthBrushTool && left_mouse_down) {
        useDepthBrushTool();
      }
    }
  }

  void setCvMatVec4bAlpha255(cv::Mat& img) {
    for (int y = 0; y < img.rows; ++y) {
      for (int x = 0; x < img.cols; ++x) {
        img.at<cv::Vec4b>(y, x)[3] = 255;
      }
    }
  }

  void updateEditorImageWithSelectedChannel()
  {
    if (!have_any_ldi_data) {
      editor_image.reset();
      return;
    }

    cv::Mat editor_augmented;
    bool scale_2x = false;
    bool is_depthmap = false;
    switch (selected_channel) {
      case kLayer0Depth:
        editor_augmented = layer_invd[0].clone();
        is_depthmap = true;
        scale_2x = true;
        break;
      case kLayer0RGBA:
        editor_augmented = layer_bgra[0].clone();
        break;
      case kLayer0RGB:
        editor_augmented = layer_bgra[0].clone();
        setCvMatVec4bAlpha255(editor_augmented);
        break;
      case kLayer0A:
        cv::extractChannel(layer_bgra[0], editor_augmented, 3);
        break;
      case kLayer1Depth:
        editor_augmented = layer_invd[1].clone();
        is_depthmap = true;
        scale_2x = true;
        break;
      case kLayer1RGBA:
        editor_augmented = layer_bgra[1].clone();
        break;
      case kLayer1RGB:
        editor_augmented = layer_bgra[1].clone();
        setCvMatVec4bAlpha255(editor_augmented);
        break;
      case kLayer1A:
        cv::extractChannel(layer_bgra[1], editor_augmented, 3);
        break;
      case kLayer2Depth:
        editor_augmented = layer_invd[2].clone();
        is_depthmap = true;
        scale_2x = true;
        break;
      case kLayer2RGBA:
        editor_augmented = layer_bgra[2].clone();
        break;
      case kLayer2RGB:
        editor_augmented = layer_bgra[2].clone();
        setCvMatVec4bAlpha255(editor_augmented);
        break;
      case kLayer2A:
        cv::extractChannel(layer_bgra[2], editor_augmented, 3);
        break;
      case kLayerSelection:
        editor_augmented = selection.clone();
      default:
        break;
    }

    if (is_depthmap && toggle_colorize_depthmaps) {
      XCHECK_EQ(editor_augmented.type(), CV_32F);
      cv::Mat colorized(editor_augmented.size(), CV_8UC4);
      // TODO: this could be multi-threaded or done in a shader to speed it up
      for (int y = 0; y < editor_augmented.rows; ++y) {
        for (int x = 0; x < editor_augmented.cols; ++x) {
          const float id = editor_augmented.at<float>(y, x);
          const Eigen::Vector3f color = turbo_colormap::float01ToColor(id);
          colorized.at<cv::Vec4b>(y, x) =
              cv::Vec4b(color.x() * 255, color.y() * 255, color.z() * 255, 255);
        }
      }
      editor_augmented = colorized;
    }

    if (scale_2x) {
      cv::resize(
          editor_augmented,
          editor_augmented,
          cv::Size(editor_augmented.cols * 2, editor_augmented.rows * 2),
          0,
          0,
          cv::INTER_LINEAR);
    }

    if (selected_channel != kLayerSelection) {
      if (editor_augmented.type() == CV_32F) {
        editor_augmented.convertTo(editor_augmented, CV_8U, 255.0);
      }
      if (editor_augmented.channels() == 1) {
        cv::cvtColor(editor_augmented, editor_augmented, cv::COLOR_GRAY2BGRA);
      }

      XCHECK_EQ(editor_augmented.size(), selection.size());
      for (int y = 0; y < editor_augmented.rows; ++y) {
        for (int x = 0; x < editor_augmented.cols; ++x) {

          const float checker = (x / 8 + y / 8) % 2 == 0 ? 1.0 : 0.0;
          const float s = selection.at<float>(y, x) * (0.3 + checker * 0.1);

          const cv::Vec4f base_color(editor_augmented.at<cv::Vec4b>(y, x));
          const cv::Vec4f selection_color = cv::Vec4f(0.0, 0.0, 255.0, 255.0) * 0.5 + 
                                            cv::Vec4f(255 - base_color[0], 255 - base_color[1], 255 - base_color[2], 255) * 0.5;

          editor_augmented.at<cv::Vec4b>(y, x) = s * selection_color + base_color * (1.0 - s);
        }
      }
    }

    editor_image.setImage(editor_augmented);
  }

  void drawToolbar()
  {
    if (prev_selected_channel != selected_channel) updateEditorImageWithSelectedChannel();
    prev_selected_channel = selected_channel;

    ImGuiStyle& style = ImGui::GetStyle();
    ImGui::BeginChild("##2DViewRoundedFrame", ImVec2(kToolbarWidth - style.WindowPadding.y, 140 * kGuiScaleHack), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse);

    ImGui::Text("Channel View");
    ImGui::Dummy(ImVec2(0, 8));

    ImGui::Text("All");
    ImGui::SameLine();
    ImGui::Dummy(ImVec2(toggle_large_text ? 51 : 36, 0));
    ImGui::SameLine();
    ImGui::SetCursorPosY(
        ImGui::GetCursorPosY() -
        ImGui::GetTextLineHeight() *
            0.1);  // Fix an inconsistency in ImGui::Text vs ImGui::RadioButton
    ImGui::RadioButton("Selection Mask", &selected_channel, kLayerSelection);

    ImGui::Dummy(ImVec2(0, 8));

    ImGui::Text("Layer 2");
    ImGui::SameLine();
    ImGui::Dummy(ImVec2(5, 0));
    ImGui::SameLine();
    ImGui::SetCursorPosY(
        ImGui::GetCursorPosY() -
        ImGui::GetTextLineHeight() *
            0.1);  // Fix an inconsistency in ImGui::Text vs ImGui::RadioButton
    ImGui::RadioButton("RGBA ##L3", &selected_channel, kLayer2RGBA);
    ImGui::SameLine();  // Note the trailing space is for formatting
    ImGui::RadioButton("RGB ##L3", &selected_channel, kLayer2RGB);
    ImGui::SameLine();
    ImGui::RadioButton("A ##L3", &selected_channel, kLayer2A);
    ImGui::SameLine();
    ImGui::RadioButton("Depth ##L3", &selected_channel, kLayer2Depth);

    ImGui::Dummy(ImVec2(0, 8));

    ImGui::Text("Layer 1");
    ImGui::SameLine();
    ImGui::Dummy(ImVec2(5, 0));
    ImGui::SameLine();
    ImGui::SetCursorPosY(
        ImGui::GetCursorPosY() -
        ImGui::GetTextLineHeight() *
            0.1);  // Fix an inconsistency in ImGui::Text vs ImGui::RadioButton
    ImGui::RadioButton("RGBA ##L2", &selected_channel, kLayer1RGBA);
    ImGui::SameLine();  // Note the trailing space is for formatting
    ImGui::RadioButton("RGB ##L2", &selected_channel, kLayer1RGB);
    ImGui::SameLine();
    ImGui::RadioButton("A ##L2", &selected_channel, kLayer1A);
    ImGui::SameLine();
    ImGui::RadioButton("Depth ##L2", &selected_channel, kLayer1Depth);

    ImGui::Dummy(ImVec2(0, 8));

    ImGui::Text("Layer 0");
    ImGui::SameLine();
    ImGui::Dummy(ImVec2(5, 0));
    ImGui::SameLine();
    ImGui::SetCursorPosY(
        ImGui::GetCursorPosY() -
        ImGui::GetTextLineHeight() *
            0.1);  // Fix an inconsistency in ImGui::Text vs ImGui::RadioButton
    ImGui::RadioButton("RGBA ##L1", &selected_channel, kLayer0RGBA);
    ImGui::SameLine();  // Note the trailing space is for formatting
    ImGui::RadioButton("RGB ##L1", &selected_channel, kLayer0RGB);
    ImGui::SameLine();
    ImGui::RadioButton("A ##L1", &selected_channel, kLayer0A);
    ImGui::SameLine();
    ImGui::RadioButton("Depth ##L1", &selected_channel, kLayer0Depth);

    ImGui::EndChild();

    ImGui::Dummy(ImVec2(0, 10));
    
    ImGui::BeginChild("##ToolsRoundedFrame", ImVec2(kToolbarWidth - style.WindowPadding.y, 120 * kGuiScaleHack), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse);

    ImGui::Text("Tools");
    ImGui::Dummy(ImVec2(0, 8));
    ImGui::RadioButton("Scroll (H)", &selected_tool, kScroll);
    ImGui::RadioButton("Select Brush (B)", &selected_tool, kSelectBrush);
    ImGui::RadioButton("Get Depth (G)", &selected_tool, kGetDepthTool);
    ImGui::RadioButton("Depth Brush (V)", &selected_tool, kDepthBrushTool);

    ImGui::EndChild();

    ImGui::Dummy(ImVec2(0, 10));

    ImGui::BeginChild("##BrushRoundedFrame", ImVec2(kToolbarWidth - style.WindowPadding.y, 90 * kGuiScaleHack), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse);

    ImGui::Text("Brush");
    ImGui::Dummy(ImVec2(0, 8));
    
    ImGui::PushItemWidth(50 * kTextScale);
    ImGui::InputFloat("Size", &brush_size);
    ImGui::SameLine();
    ImGui::Dummy(ImVec2(16, 0));
    ImGui::SameLine();
    ImGui::InputFloat("Hardness", &brush_hardness);
    ImGui::PopItemWidth();

    ImGui::Dummy(ImVec2(0, 10));

    // Draw depth value widget
    float depth_in_meters = math::clamp(0.3 / brush_depth, 0.01, 50.0);
    ImGui::Text("Depth [m]: %.2f", depth_in_meters);
    ImGui::PushStyleVar(ImGuiStyleVar_FrameBorderSize, 2);
    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(1, 1, 1, 0.25));
    const Eigen::Vector3f colormap = turbo_colormap::float01ToColor(brush_depth);
    ImVec4 widget_color = toggle_colorize_depthmaps
                              ? ImVec4(colormap.z(), colormap.y(), colormap.x(), 1.0f)
                              : ImVec4(brush_depth, brush_depth, brush_depth, 1.0);
    ImGui::PushStyleColor(ImGuiCol_FrameBg, widget_color);
    ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, widget_color);
    ImGui::PushStyleColor(ImGuiCol_FrameBgActive, widget_color);
    ImGui::SameLine(120 * kTextScale);
    ImGui::PushItemWidth(166);
    ImGui::SetCursorPosY(ImGui::GetCursorPosY() - ImGui::GetTextLineHeight() * 0.2);
    ImGui::SliderFloat("", &brush_depth, 0.0f, 1.0f);
    ImGui::PopStyleColor(4);
    ImGui::PopItemWidth();
    ImGui::PopStyleVar();

    ImGui::EndChild();
  }

  float mouse_in_editor_coords_x = 0;
  float mouse_in_editor_coords_y = 0;

  void drawEditor()
  {
    editor_viewport_min = ImGui::GetWindowPos();
    editor_viewport_size = ImGui::GetContentRegionAvail();  // excludes scrollbars

    mouse_in_editor_coords_x = curr_mouse_x - editor_viewport_min.x + ImGui::GetScrollX();
    mouse_in_editor_coords_y = curr_mouse_y - editor_viewport_min.y + ImGui::GetScrollY();

    editor_image.scale_to_fit = false;
    editor_image.makeGlTexture();

    // Center the image horizontally and vertically
    int w = editor_image.size.width;
    int h = editor_image.size.height;

    if (editor_viewport_size.x > w || editor_viewport_size.y > h) {
      ImVec2 offset{(editor_viewport_size.x - w) / 2, (editor_viewport_size.y - h) / 2};
      ImGui::SetCursorPos(offset);
    }
    editor_image.drawInImGui();

    if (mouse_in_editor_viewport) {
      // Draw brush circle for selection brush or depth brush
      if (selected_tool == kSelectBrush || selected_tool == kDepthBrushTool) {
        ImGui::GetWindowDrawList()->AddCircle(
            ImVec2(curr_mouse_x, curr_mouse_y), brush_size, IM_COL32(255, 255, 255, 196), 32, 1.0);
        ImGui::GetWindowDrawList()->AddCircle(
            ImVec2(curr_mouse_x, curr_mouse_y), brush_size + 1, IM_COL32(0, 0, 0, 64), 32, 1.0);
      }

      // Draw "Get Depth" tool. TODO: maybe draw a crosshair instead, to make it more distinct
      if (selected_tool == kGetDepthTool) {
        static constexpr float kEyedropperRadius = 4;
        ImGui::GetWindowDrawList()->AddCircle(
            ImVec2(curr_mouse_x, curr_mouse_y),
            kEyedropperRadius,
            IM_COL32(255, 255, 255, 196),
            32,
            1.0);
        ImGui::GetWindowDrawList()->AddCircle(
            ImVec2(curr_mouse_x, curr_mouse_y),
            kEyedropperRadius + 1,
            IM_COL32(0, 0, 0, 64),
            32,
            1.0);
      }
    }
  }

  void drawPreview3D()
  {
    // No actual drawing happens here. We just get the rectangle to be drawn in and save it for
    // later.
    preview3d_viewport_min = ImGui::GetWindowPos();
    preview3d_viewport_size = ImGui::GetWindowSize();
  }

  // Modal popup stuff
  bool open_popup_set_alpha = false;
  bool open_popup_set_invd = false;
  bool open_popup_copy_channels_to_frames = false;
  bool open_popup_encode_video = false;
  bool open_popup_render_vr180_ldi = false;
  bool open_popup_config_tools = false;

  bool isAnyModalPopupOpen()
  {
    return open_popup_set_alpha || open_popup_set_invd || open_popup_copy_channels_to_frames ||
           open_popup_encode_video || open_popup_render_vr180_ldi || open_popup_config_tools;
  }

  bool set_in_layer0 = false;
  bool set_in_layer1 = false;
  bool set_in_layer2 = false;
  float set_alpha_value = 1.0;
  float set_invd_value = 1.0;
  bool layer_copy_rgb[kNumLayers] = {false, false, false};
  bool layer_copy_a[kNumLayers] = {false, false, false};
  bool layer_copy_invd[kNumLayers] = {false, false, false};
  bool encode_h264_hq = false;
  bool encode_h264_streaming = true;
  bool encode_h264_mobile = true;
  bool encode_h264_looking_glass = true;
  bool encode_h265_oculus = true;
  bool encode_prores = false;
  float encode_video_fps = 29.97;

  gui::ImguiInputFileSelect vr180toLdi3_input_file_select;
  p11::ldi::LdiPipelineConfig vr180toLdi3_cfg;
  float vr180toLdi3_baseline_cm = 6.0; // The UI will show [cm], while the pipeline expects [m]
  bool vr180toLdi3_stabilize_inpainting = false;

  gui::ImguiFolderSelect project_dir_folder_select;

  void makeApplyToLayerCheckboxes()
  {
    ImGui::Text("Apply To:");
    ImGui::Checkbox("Layer 2", &set_in_layer2);
    ImGui::Checkbox("Layer 1", &set_in_layer1);
    ImGui::Checkbox("Layer 0", &set_in_layer0);
  }

  void makeApplyAndCancelButtons(bool& open_popup, bool enabled, std::function<void()> apply_func)
  {
    ImGui::Dummy(ImVec2(0, 16));

    if (!enabled) {
      ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
      ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.25);
    }
    if (ImGui::Button("Apply", ImVec2(80, 0))) {
      apply_func();
      open_popup = false;
    }
    if (!enabled) {
      ImGui::PopItemFlag();
      ImGui::PopStyleVar();
    }
    ImGui::SameLine();
    if (ImGui::Button("Cancel", ImVec2(80, 0))) {
      open_popup = false;
    }
    if (!open_popup) {
      ImGui::CloseCurrentPopup();
    }
  }

  void drawModalPopups()
  {
    if (open_popup_set_alpha) {
      ImGui::OpenPopup("Set Alpha In Selection");
    }
    if (open_popup_set_invd) {
      ImGui::OpenPopup("Set Depth In Selection");
    }
    if (open_popup_copy_channels_to_frames) {
      ImGui::OpenPopup("Copy Channels To Selected Frames");
    }
    if (open_popup_encode_video) {
      ImGui::OpenPopup("Encode Video");
    }
    if (open_popup_render_vr180_ldi) {
      ImGui::OpenPopup("Render VR180 to LDI3");
    }
    if (open_popup_config_tools) {
      ImGui::OpenPopup("Configure ffmpeg, python");
    }

    if (ImGui::BeginPopupModal(
            "Set Depth In Selection", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
      makeApplyToLayerCheckboxes();

      ImGui::PushItemWidth(60);
      ImGui::InputFloat("Inverse Depth Value", &set_invd_value);
      ImGui::PopItemWidth();
      set_invd_value = math::clamp(set_invd_value, 0.0f, 1.0f);

      makeApplyAndCancelButtons(open_popup_set_invd, true, [&] {
        setInverseDepthInSelection({set_in_layer0, set_in_layer1, set_in_layer2}, set_invd_value);
      });
      ImGui::EndPopup();
    }

    if (ImGui::BeginPopupModal(
            "Set Alpha In Selection", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
      makeApplyToLayerCheckboxes();

      ImGui::PushItemWidth(60);
      ImGui::InputFloat("Alpha Value", &set_alpha_value);
      ImGui::PopItemWidth();
      set_alpha_value = math::clamp(set_alpha_value, 0.0f, 1.0f);

      makeApplyAndCancelButtons(open_popup_set_alpha, true, [&] {
        setAlphaInSelection({set_in_layer0, set_in_layer1, set_in_layer2}, set_alpha_value);
      });
      ImGui::EndPopup();
    }

    if (ImGui::BeginPopupModal(
            "Copy Channels To Selected Frames", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
      ImGui::Text(
          "Copies channels from the current frame to other frames selected in the timeline.");

      ImGui::Dummy(ImVec2(0, 16));

      for (int l = kNumLayers - 1; l >= 0; --l) {
        const std::string sl = std::to_string(l);
        ImGui::Text("%s", std::string("Layer " + sl).c_str());
        ImGui::SameLine();
        ImGui::Dummy(ImVec2(5, 0));
        ImGui::SameLine();
        ImGui::SetCursorPosY(
            ImGui::GetCursorPosY() -
            ImGui::GetTextLineHeight() *
                0.1);  // Fix an inconsistency in ImGui::Text vs ImGui::RadioButton
        ImGui::Checkbox(("RGB##copy" + sl).c_str(), &layer_copy_rgb[l]);
        ImGui::SameLine();
        ImGui::Checkbox(("A##copy" + sl).c_str(), &layer_copy_a[l]);
        ImGui::SameLine();
        ImGui::Checkbox(("Depth##copy" + sl).c_str(), &layer_copy_invd[l]);
      }

      makeApplyAndCancelButtons(
          open_popup_copy_channels_to_frames, true, [&] { copyChannelDataToSelectedFrames(); });
      ImGui::EndPopup();
    }

    if (ImGui::BeginPopupModal("Encode Video", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
      ImGui::Text(
          "Encodes the ldi3_* images in the project directory. ffmpeg must be is installed!");

      ImGui::Dummy(ImVec2(0, 16));

      ImGui::Text("For Web:");
      ImGui::Checkbox("Encode h264 - Full Size 5760x5760", &encode_h264_streaming);
      ImGui::Checkbox("Encode h264 - 3840x3840", &encode_h264_looking_glass);
      ImGui::Checkbox("Encode h264 - Optimized for Mobile 1920x1920", &encode_h264_mobile);
      ImGui::Checkbox("Encode h265 - Full Size Optimized for Quest/Vision Pro 5760x5760", &encode_h265_oculus);
      
      ImGui::Dummy(ImVec2(0, 8));

      ImGui::Text("For Editing / Max Quality");
      ImGui::Checkbox("Encode h264 - High Quality (crf 18)", &encode_h264_hq);

      ImGui::Checkbox("Encode ProRes - 422LT", &encode_prores);

      ImGui::Dummy(ImVec2(0, 8));

      ImGui::PushItemWidth(60 * kTextScale);
      ImGui::InputFloat("Frames Per Second", &encode_video_fps);
      ImGui::PopItemWidth();

      makeApplyAndCancelButtons(
          open_popup_encode_video, true, [&] { encodeVideoFromFramesWithFfmpeg(); });
      ImGui::EndPopup();
    }

    if (ImGui::BeginPopupModal("Render VR180 to LDI3", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
      ImGui::Dummy(ImVec2(1000, 0));
      ImGui::Text("Input a VR180 photo or video, output LDI3 png images for each frame.");
      ImGui::Text(" ");
      ImGui::Text("You may need to encode an mp4 after this step.");
      ImGui::Text("Runs stereo depth estimation, layer decomposition, in-painting, and temporal stabilization to create a layered depth video.");
      ImGui::Text("This may generate a large amount of data in the project directory!");

      ImGui::Dummy(ImVec2(0, 16));

      ImGui::Text("Input VR180 photo or video: ");
      ImGui::SameLine();
      ImGui::PushItemWidth(500 * kTextScale);
      vr180toLdi3_input_file_select.drawAndUpdate();
      ImGui::PopItemWidth();

      ImGui::Dummy(ImVec2(0, 16));

      ImGui::PushItemWidth(50 * kTextScale);
      ImGui::InputFloat("Baseline [Unit: cm] (distance between cameras)", &vr180toLdi3_baseline_cm);
      ImGui::PopItemWidth();

      ImGui::Checkbox("Temporally stabilize inpainting (video only)", &vr180toLdi3_stabilize_inpainting);

      bool enabled = !std::string(vr180toLdi3_input_file_select.path).empty();
      makeApplyAndCancelButtons(open_popup_render_vr180_ldi, enabled, [&] { renderVR180toLDI3(); });
      ImGui::EndPopup();
    }

    if (ImGui::BeginPopupModal("Configure ffmpeg, python", nullptr, ImGuiWindowFlags_AlwaysAutoResize)) {
      ImGui::Text("ffmpeg command or location: ");
      ImGui::PushItemWidth(500 * kTextScale);
      ffmpeg_tool_select.drawAndUpdate();
      ImGui::PopItemWidth();

      ImGui::Text("python3 command or location: ");
      ImGui::PushItemWidth(500 * kTextScale);
      python_tool_select.drawAndUpdate();
      ImGui::PopItemWidth();
      
      if (ImGui::Button("OK", ImVec2(80, 0))) {
        getExternalToolFileSelects();
        prefs["ffmpeg"] = ffmpeg;
        prefs["python3"] = python3;
        preferences::setPrefs(prefs);

        open_popup_config_tools = false;
        ImGui::CloseCurrentPopup();
      }
      ImGui::SameLine();
      if (ImGui::Button("Cancel", ImVec2(80, 0))) {
        python_tool_select.setPath(python3.c_str());
        ffmpeg_tool_select.setPath(ffmpeg.c_str());

        open_popup_config_tools = false;
        ImGui::CloseCurrentPopup();
      }
      ImGui::SameLine();
      if (ImGui::Button("Restore Defaults", ImVec2(200, 0))) {
        setExternalToolDefaultPaths();
        setExternalToolFileSelects();
      }

      ImGui::EndPopup();
    }
  }

  void drawMainWindow()
  {
    ImGuiStyle& style = ImGui::GetStyle();
    ImGuiWindowFlags editor_subwindow_flags = ImGuiWindowFlags_HorizontalScrollbar;
    int editor_width = (ImGui::GetWindowContentRegionWidth() - kToolbarWidth) * 0.5;
    int editor_height = ImGui::GetWindowHeight() - VideoTimelineWidget::kTimelineHeight -
                        style.WindowPadding.y * 2 - ImGui::GetFrameHeight();

    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 4.0);
    ImGui::BeginChild("Toolbar", ImVec2(kToolbarWidth, editor_height - style.WindowPadding.y), false);
    drawToolbar();
    ImGui::EndChild();
    ImGui::PopStyleVar();
    ImGui::SameLine();

    ImGui::BeginChild("Editor", ImVec2(editor_width, editor_height), false, editor_subwindow_flags);
    if (editor_needs_scroll_update) {
      ImGui::SetScrollX(editor_scroll_x);
      ImGui::SetScrollY(editor_scroll_y);
      editor_needs_scroll_update = false;
    } else {
      editor_scroll_x = ImGui::GetScrollX();
      editor_scroll_y = ImGui::GetScrollY();
    }
    drawEditor();
    ImGui::EndChild();
    ImGui::SameLine();

    // Make the background of the 3D view transparent
    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0, 0, 0, 0));
    
    ImGui::BeginChild(
        "3D Preview",
        ImVec2((ImGui::GetWindowContentRegionWidth() - kToolbarWidth) * 0.5, editor_height),
        false);
    drawPreview3D();
    ImGui::EndChild();
    ImGui::PopStyleColor();

    timeline.render();
    timelinePlayVideo();
  }

  void timelinePlayVideo()
  {
    if (toggle_play_video && have_video_data && !load_photo_in_progress) moveTimelineByDelta(1);
  }

  bool commandWindowIsVisible() { return command_runner.isRunning(); }

  void drawCommandWindow()
  {
    if (ImGui::Button("Cancel")) {
      command_runner.kill();
    }

    if (command_runner.waitingForCancel()) {
      ImGui::SameLine();
      ImGui::Text("Cancelling...");
    }

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
  }

  void drawProjectDirRequiredOverlay() {
    // Center horizontally and vertically
    ImVec2 box_size(580 * kTextScale, 120 * kTextScale);
    ImVec2 avail = ImGui::GetContentRegionAvail();
    ImGui::SetNextWindowPos(ImVec2(
      ImGui::GetCursorPosX() + (avail.x - box_size.x) * 0.5,
      ImGui::GetCursorPosY() + (avail.y - box_size.y) * 0.5));
    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 4.0);
    ImGui::BeginChild("##ProjectDirSelectRoundedFrame", box_size, true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoMove);

    ImGui::Text("Select Project Directory");
    ImGui::PushItemWidth(496 * kTextScale);
    project_dir_folder_select.drawAndUpdate();
    ImGui::PopItemWidth();

    if (!std::string(project_dir_folder_select.path).empty()) {
      ImGui::SameLine();
      if (ImGui::Button("OK")) {
        if (file::directoryExists(project_dir_folder_select.path)) {
          project_dir = project_dir_folder_select.path;
          prefs["project_dir"] = project_dir;
          preferences::setPrefs(prefs);
          if (kEnableAutoLogging) xpl::stdoutLogger.attachTextFileLog(project_dir + "/log.txt");

          scanProjectDirForVideoFrames();
        } else {
          tinyfd_messageBox("Invalid Project Directory Path", "The path does not appear to be a directory.", "ok", "error", 0);
        }
      }
    }

    ImGui::Dummy(ImVec2(0.0f, 20.0f));
    ImGui::Text("Files in this directory may be deleted or irreversibly modified.");
    ImGui::Text("Large amounts of data may be generated.");

    ImGui::EndChild();
    ImGui::PopStyleVar();
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

  void drawFrame()
  {
    gl_context_mutex.lock();
    glfwMakeContextCurrent(window);

    if (scan_project_dir_requested) {
      scan_project_dir_requested = false;
      scanProjectDirForVideoFrames();
    }

    if (opengl_textures_need_update) {
      opengl_textures_need_update = false;
      updateEditorImageWithSelectedChannel();
      updateAllLdiTextures();
    }

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    const bool hide_main_menu = project_dir.empty() || commandWindowIsVisible();

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

    std::string window_title = std::string() + "Lifecast Volumetric Video Editor" +
                               (project_dir.empty() ? " - Select Project Directory" : " - ") +
                               project_dir;
    glfwSetWindowTitle(window, window_title.c_str());
    
    if (ImGui::Begin("Main Window", nullptr, window_flags)) {
      if (project_dir.empty()) {
        drawProjectDirRequiredOverlay();
      } else if (commandWindowIsVisible()) {
        drawCommandWindow();
      } else {
        handleKeyboard();
        updateMouse();

        drawMainMenu();

        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
        drawMainWindow();
        ImGui::PopStyleVar();
      }

      drawModalPopups();

      ImGui::End();
    }

    // Here we would normally call finishDrawingImguiAndGl(), but don't because we are mixing in custom opengl
    beginGlDrawingForImgui();
    
    //ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    if (!commandWindowIsVisible() && !isAnyModalPopupOpen() && !project_dir.empty()) {
      drawGlStuff();
    }

    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    updateCursor();
    glfwSwapBuffers(window);

    glfwMakeContextCurrent(nullptr);
    gl_context_mutex.unlock();
  }
} app;  // An instance of LdiStudio named app exists in this namespace. This is used in glfw
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
  if (app.mouse_in_editor_viewport &&
      (app.selected_tool == kSelectBrush || app.selected_tool == kDepthBrushTool)) {
    if (yoffset > 0) app.brush_size *= 1.1;
    if (yoffset < 0) app.brush_size *= 0.9;
  }
  if (app.mouse_in_timeline) {
    if (yoffset > 0) app.timeline.pixels_per_frame *= 1.1;
    if (yoffset < 0) app.timeline.pixels_per_frame *= 0.9;
    app.timeline.pixels_per_frame = math::clamp<float>(app.timeline.pixels_per_frame, 0.1, 10.0);
  }

  // I'm removing this out of superstition. It doesn't do anything useful right now, and segfaults
  // appeared when it was included, although this does not make sense.
  //if (!app.mouse_in_3d_viewport && !app.mouse_in_editor_viewport && !app.mouse_in_timeline)
  //  ImGui_ImplGlfw_ScrollCallback(window, xoffset, yoffset);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
  if (key == GLFW_KEY_H && action == GLFW_PRESS) app.selected_tool = kScroll;
  if (key == GLFW_KEY_B && action == GLFW_PRESS) app.selected_tool = kSelectBrush;
  if (key == GLFW_KEY_G && action == GLFW_PRESS) app.selected_tool = kGetDepthTool;
  if (key == GLFW_KEY_V && action == GLFW_PRESS) app.selected_tool = kDepthBrushTool;

  if (key == GLFW_KEY_SPACE && action == GLFW_RELEASE)
    app.toggle_play_video = !app.toggle_play_video;

  if (app.mouse_in_3d_viewport) {
    if (key == GLFW_KEY_1 && action == GLFW_RELEASE)
      app.toggle_layer_visible[0] = !app.toggle_layer_visible[0];
    if (key == GLFW_KEY_2 && action == GLFW_RELEASE)
      app.toggle_layer_visible[1] = !app.toggle_layer_visible[1];
    if (key == GLFW_KEY_3 && action == GLFW_RELEASE)
      app.toggle_layer_visible[2] = !app.toggle_layer_visible[2];
  }

  //if (key == GLFW_KEY_R && action == GLFW_RELEASE) app.reloadCurrentLDI3Photo();

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

  p11::app.init("Lifecast Volumetric Video Editor", 1280, 720);
  glfwSetMouseButtonCallback(p11::app.window, p11::mouse_button_callback);
  glfwSetScrollCallback(p11::app.window, p11::scroll_callback);
  glfwSetKeyCallback(p11::app.window, p11::key_callback);

  p11::app.initLdiStudio();
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
