// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
on Mac with ARM, to properly use Metal:

PYTORCH_ENABLE_MPS_FALLBACK=1 bazel run -- //source:upscale_video


bazel run -- //source:upscale_video
*/

// Make the application run without a terminal in Windows.
#if defined(windows_hide_console) && defined(_WIN32)
#pragma comment(linker, "/SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
#endif

#include "logger.h"
#include "dear_imgui_app.h"
#include "third_party/dear_imgui/imgui_internal.h" // For PushItemFlag on Windows
#include "imgui_filedialog.h"
#include "imgui_cvmat.h"
#include "util_runfile.h"
#include "util_file.h"
#include "util_time.h"
#include "util_math.h"
#include "util_command.h"
#include "util_opencv.h"
#include "util_browser.h"
#include "util_torch.h"
#include "torch_opencv.h"
#include "video_transcode_lib.h"
#include "preferences.h"
#include "tinysr_lib.h"
#include "rof.h"
#include "view_interpolation.h"
#include "char_traits_uc.h" // Workaround for Xcode 16.3 breaking it, must go before ccprest
#include "cpprest/http_client.h"
#include "cpprest/filestream.h"
#include <regex>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <filesystem>
#include <locale>

#ifdef _WIN32
#include <Windows.h>
#include <io.h>
#include <fcntl.h>
#endif
#ifdef __linux__
#include <GL/gl.h>
#endif

namespace p11 {

constexpr float kSoftwareVersion = 1.16;

//constexpr bool kEnableAutoLogging = true;

constexpr char kPrefNumThreadsKey[] = "pref_num_threads";

enum UpscaleVideoAppState {
  STATE_SPLASH
};

// TinyFileDialog doesn't support quotes in messages on Mac...
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
      ImGui::Text("Select an input video file (.mp4, .mov)");
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
      ImGui::Text("Batch input videos");
      ImGui::SameLine();
      ImGui::Dummy(ImVec2(40, 0));
      ImGui::SameLine();
      if (ImGui::SmallButton("Clear Batch")) {
        paths.clear();
        string::copyBuffer(path, "", kBufferSize);
      }
      ImGui::BeginChild("BatchVideosScrolling", ImVec2(ImGui::CalcItemWidth() + 40, 100), true);
      for (auto& v : paths) {
        ImGui::TextUnformatted(v.c_str());
      }
      ImGui::EndChild();
    }
  }
};

enum OutputScaleSelectRadioButton {
  OUTPUT_SCALE_15X = 0,
  OUTPUT_SCALE_2X = 1,
  OUTPUT_SCALE_CUSTOM = 2,
  OUTPUT_SCALE_1X = 3
};

struct UpscaleVideoApp : public DearImGuiApp {
  // Application preferences
  std::map<std::string, std::string> prefs;
  int pref_num_threads;

  // Commands, workflow config
  CommandRunner command_runner;

  ImguiInputMultiFileSelect input_video_file_select;
  gui::ImguiOutputFileSelect output_video_file_select =
      gui::ImguiOutputFileSelect("Save Output As:", "");
  gui::ImguiFolderSelect batch_output_dir_select = gui::ImguiFolderSelect("Batch Output Folder:");

  // Application state and workflow
  UpscaleVideoAppState app_state;
  int selected_video_width = 0, selected_video_height = 0, selected_video_total_frames = 0;
  double selected_video_fps = 0.0;
  static constexpr int kNumberBufferSize = 8;
  char width_input_buffer[kNumberBufferSize] = "";
  char height_input_buffer[kNumberBufferSize] = "";
  char crf_input_buffer[kNumberBufferSize] = "20";

  torch::DeviceType device;
  std::shared_ptr<enhance::Base_SuperResModel> sr_model = nullptr;
  static constexpr float kSuperResScale = 2.0;
  
  torch::jit::script::Module raft;

  gui::ImguiCvMat preview_image_lr, preview_image_hr;
  float hr_preview_scale_factor = 1; // Updated when we get a preview image, determines the scaling of the right preview panel
  std::mutex preview_image_mutex_lr, preview_image_mutex_hr;

  // Mouse state
  bool left_mouse_down = false;
  bool right_mouse_down = false;
  double prev_mouse_x = 0, prev_mouse_y = 0, curr_mouse_x = 0, curr_mouse_y = 0;
  
  // Mouse-drag based scrolling of preview viewport
  bool mouse_in_preview_viewport = false;
  bool click_started_in_preview_viewport = false;
  ImVec2 preview_viewport_min, preview_viewport_size;
  int preview_scroll_x = 0, preview_scroll_y = 0;
  bool preview_needs_scroll_update = false;
  bool preview_needs_scroll_reset = false;

  bool isAnyModalPopupOpen() {
    return commandWindowIsVisible();// || (app_state != STATE_3D_VIEW);
  }

  void setAppState(UpscaleVideoAppState new_state) {
    app_state = new_state;
    std::string state_str;
    switch (app_state) {
      case STATE_SPLASH: state_str = "STATE_SPLASH"; break;
    }
  }

  void initUpscaleVideoApp()
  {
    device = util_torch::findBestTorchDevice();

    prefs = preferences::getPrefs();

    pref_num_threads = 1;
    if (prefs.count(kPrefNumThreadsKey)) {
      pref_num_threads = std::atoi(prefs.at(kPrefNumThreadsKey).c_str());
      pref_num_threads = math::clamp(pref_num_threads, 1, 32);
    }

    XPLINFO << "Loading RAFT model";
    torch::jit::getProfilingMode() = false;
    p11::optical_flow::getTorchModelRAFT(raft);
    XPLINFO << "Finished loading RAFT model";

    //if (kEnableAutoLogging) xpl::stdoutLogger.attachTextFileLog(project_dir + "/log.txt");
    setAppState(STATE_SPLASH);
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

  bool enhance_ready = false;
  int enhance_mode_select = 1;
  int sr_model_select = 0;
  int output_scale_select = OUTPUT_SCALE_2X;
  int encode_video_format_select = 1; // For radio buttons to select encoding type
  bool encode_video_keep_other_streams = true;
  int encode_video_bitrate_select = 1; // For radio buttons to select encoding type
  int encode_video_crf_input = 20;
  int encode_video_profile_select = 0;
  bool do_stereo_baseline_checkbox = false;
  bool do_ai_restoration_model_checkbox = true;
  bool do_bilateral_filter_checkbox = false;
  float stereo_baseline_adjust_factor = 1.0;
  int bilateral_filter_radius = 7;
  float bilateral_filter_strength = 0.85;
  void splashScreen() {
    float enhance_button_width = 300;
    float row_height = 40;
    float indent_width = 40;
    float screen_width = ImGui::GetContentRegionAvail().x - indent_width;
    float file_select_width = ImGui::GetContentRegionAvail().x - enhance_button_width -
                              200;  // HACK: this sizing magic number

    ImGui::Dummy(ImVec2(0.0f, row_height));
    float button_ypos = ImGui::GetCursorPosY() + row_height - 18;

    ImGui::Indent(indent_width);
    ImGui::PushItemWidth(file_select_width);
    input_video_file_select.on_change_callback = [this] { this->onSelectInputVideoFile(); };
    input_video_file_select.drawAndUpdate();
    ImGui::Dummy(ImVec2(0.0f, row_height));
    ImGui::PopItemWidth();

    enhance_ready = input_video_file_select.paths.size() > 0;

    if (enhance_ready) {
      ImGui::PushItemWidth(file_select_width);
      if (input_video_file_select.paths.size() == 1) {
        output_video_file_select.drawAndUpdate();
      } else {
        batch_output_dir_select.drawAndUpdate();
      }
      ImGui::PopItemWidth();

      ImGui::Dummy(ImVec2(0.0f, 60.0f));

      ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(20, 20));
      
      ImGui::BeginChild("TabBarContainer", ImVec2(screen_width - 40, 0), false);
      if (ImGui::BeginTabBar("OptionsTabBar", ImGuiTabBarFlags_None)) {
    
        if (ImGui::BeginTabItem("  Resize and enhance  ")) {
          ImGui::BeginChild("ScrollChild", ImVec2(screen_width - 40, 0), false);

          ImGui::Dummy(ImVec2(0.0f, 20.0f));

          //ImGui::SameLine();
          //ImGui::Dummy(ImVec2(0.0f, 20.0f));
          ImGui::Text("Output size");
          ImGui::SameLine();
          ImGui::Text((" (Original: " + std::to_string(selected_video_width) + " x " + std::to_string(selected_video_height) + ")").c_str());
        
          if (ImGui::RadioButton(" 1x (Original)", output_scale_select == OUTPUT_SCALE_1X)) {
            output_scale_select = OUTPUT_SCALE_1X;
            string::copyBuffer(width_input_buffer, std::to_string(int(selected_video_width)).c_str(), kNumberBufferSize);
            string::copyBuffer(height_input_buffer, std::to_string(int(selected_video_height)).c_str(), kNumberBufferSize);
            setDefaultOutputFilename();
          }
          ImGui::SameLine();
          if (ImGui::RadioButton(" 1.5x", output_scale_select == OUTPUT_SCALE_15X)) {
            output_scale_select = OUTPUT_SCALE_15X;
            string::copyBuffer(width_input_buffer, std::to_string(int(selected_video_width * 1.5)).c_str(), kNumberBufferSize);
            string::copyBuffer(height_input_buffer, std::to_string(int(selected_video_height * 1.5)).c_str(), kNumberBufferSize);
            setDefaultOutputFilename();
          }
          ImGui::SameLine();
          if (ImGui::RadioButton(" 2x", output_scale_select == OUTPUT_SCALE_2X)) {
            output_scale_select = OUTPUT_SCALE_2X;
            string::copyBuffer(width_input_buffer, std::to_string(selected_video_width * 2).c_str(), kNumberBufferSize);
            string::copyBuffer(height_input_buffer, std::to_string(selected_video_height * 2).c_str(), kNumberBufferSize);
            setDefaultOutputFilename();
          }
          ImGui::SameLine();
          if (ImGui::RadioButton(" Width x Height: ", output_scale_select == OUTPUT_SCALE_CUSTOM)) {
            output_scale_select = OUTPUT_SCALE_CUSTOM;
            string::copyBuffer(width_input_buffer, std::to_string(selected_video_width * 2).c_str(), kNumberBufferSize);
            string::copyBuffer(height_input_buffer, std::to_string(selected_video_height * 2).c_str(), kNumberBufferSize);
            setDefaultOutputFilename();
          }
          ImGui::SameLine();
          ImGui::PushItemWidth(100);
          if (ImGui::InputText("##OutputWidthInput", width_input_buffer, IM_ARRAYSIZE(width_input_buffer))) {
            output_scale_select = OUTPUT_SCALE_CUSTOM;
            setDefaultOutputFilename();
          }
          ImGui::PopItemWidth();
          ImGui::SameLine();
          ImGui::Text("x");
          ImGui::SameLine();
          ImGui::PushItemWidth(100);
          if (ImGui::InputText("##OutputHeightInput", height_input_buffer, IM_ARRAYSIZE(height_input_buffer))) {
            output_scale_select = OUTPUT_SCALE_CUSTOM;
            setDefaultOutputFilename();
          }
          ImGui::PopItemWidth();
          // Turn non-number inputs into numbers
          int output_w = math::clamp(std::atoi(width_input_buffer), 1, 65535);
          int output_h = math::clamp(std::atoi(height_input_buffer), 1, 65535);
          string::copyBuffer(width_input_buffer, std::to_string(output_w).c_str(), kNumberBufferSize);
          string::copyBuffer(height_input_buffer, std::to_string(output_h).c_str(), kNumberBufferSize);

          if (output_w > 8192 || output_h > 8192) {
            ImGui::SameLine();
            ImGui::PushStyleColor(
                ImGuiCol_Text, ImVec4(0.9f, 0.69f, 0.4f, 1.0f));  // Schoolbus yellow
            ImGui::Text("Warning: encoding larger than 8K may run out of RAM");
            ImGui::PopStyleColor();
          }


          ImGui::Dummy(ImVec2(0.0f, 20.0f));

          ImGui::Checkbox("De-noise with bilateral filter", &do_bilateral_filter_checkbox);
          ImGui::Text("Radius");
          ImGui::SameLine();
          ImGui::PushItemWidth(120);
          ImGui::InputInt("##BilateralRadius", &bilateral_filter_radius);
          bilateral_filter_radius = math::clamp(bilateral_filter_radius, 1, 64);
          ImGui::PopItemWidth();
          ImGui::SameLine();
          ImGui::Text("Strength");
          ImGui::SameLine();
          ImGui::PushItemWidth(250);
          ImGui::SliderFloat("##BilateralStrength", &bilateral_filter_strength, 0.0f, 1.0f);
          ImGui::PopItemWidth();

          ImGui::Dummy(ImVec2(0.0f, 20.0f));

          ImGui::Checkbox("AI Enhancement Model", &do_ai_restoration_model_checkbox);

          if (ImGui::RadioButton(" UNet (Faster)", sr_model_select == 0)) {
            sr_model_select = 0;
          }
          ImGui::SameLine();
          if (ImGui::RadioButton(" EDSR (Higher Quality)", sr_model_select == 1)) {
            sr_model_select = 1;
          }

          if (ImGui::RadioButton(
                  " Super resolution, de-noise, de-blur/sharpen", enhance_mode_select == 1)) {
            enhance_mode_select = 1;
          }
          ImGui::SameLine();
          if (ImGui::RadioButton(" Super resolution only", enhance_mode_select == 0)) {
            enhance_mode_select = 0;
          }

          ImGui::Dummy(ImVec2(0.0f, 20.0f));
          ImGui::EndChild(); // end of ScrollChild
          ImGui::EndTabItem();
        }

        if (ImGui::BeginTabItem("  Video encoding  ")) {
          ImGui::BeginChild("ScrollChild", ImVec2(screen_width - 40, 0), false);
          ImGui::Dummy(ImVec2(0.0f, 20.0f));

          ImGui::Text("Output format");
          bool disable_png = input_video_file_select.paths.size() > 1;
          if (disable_png) {
            if (encode_video_format_select == 3) encode_video_format_select = 1; // dont allow PNG to be selected by any means
            ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
          }
          if (ImGui::RadioButton(" PNG", encode_video_format_select == 3)) {
            encode_video_format_select = 3;
            setDefaultOutputFilename();
          }
          ImGui::SameLine();
          if (ImGui::RadioButton(" JPG", encode_video_format_select == 4)) {
            encode_video_format_select = 4;
            setDefaultOutputFilename();
          }
          if (disable_png) {
            ImGui::PopStyleVar();
            ImGui::PopItemFlag();
          }
          ImGui::SameLine();
          if (ImGui::RadioButton(" h264", encode_video_format_select == 0)) {
            encode_video_format_select = 0;
            setDefaultOutputFilename();
          }
          ImGui::SameLine();
          if (ImGui::RadioButton(" h265", encode_video_format_select == 1)) {
            encode_video_format_select = 1;
            setDefaultOutputFilename();
          }
          ImGui::SameLine();
          if (ImGui::RadioButton(" ProRes", encode_video_format_select == 2)) {
            encode_video_format_select = 2;
            setDefaultOutputFilename();
          }
          ImGui::Checkbox(" Keep audio and other streams", &encode_video_keep_other_streams);

          ImGui::Dummy(ImVec2(0.0f, 20.0f));

          if (encode_video_format_select == 2) {
            ImGui::Text("Quality profile");
            ImGui::RadioButton(" 422 LT", &encode_video_profile_select, 0);
            ImGui::SameLine();
            ImGui::RadioButton(" 422 HQ", &encode_video_profile_select, 1);
            ImGui::SameLine();
            ImGui::RadioButton(" 4444", &encode_video_profile_select, 2);
          } else if (encode_video_format_select == 0 || encode_video_format_select == 1) {
            ImGui::Text("Output bitrate");
            ImGui::RadioButton(" Low (CRF=28)", &encode_video_bitrate_select, 0);
            ImGui::SameLine();
            ImGui::RadioButton(" Medium (CRF=23)", &encode_video_bitrate_select, 1);
            ImGui::SameLine();
            ImGui::RadioButton(" High (CRF=18)", &encode_video_bitrate_select, 2);
          } else {
            //ImGui::Text("");  // HACK: prevent scrollbar tweaking
            //ImGui::Text("");
          }

          ImGui::Dummy(ImVec2(0.0f, 20.0f));
          ImGui::EndChild(); // end of ScrollChild
          ImGui::EndTabItem();
        }  // end of Encoding Options
        
        if (ImGui::BeginTabItem("  Experimental: Stereo, 3D, VR  ")) {
          ImGui::BeginChild("ScrollChild", ImVec2(screen_width - 40, 0), false);
          ImGui::Dummy(ImVec2(0.0f, 20.0f));
        
          ImGui::Checkbox(" Adjust stereo baseline (interpupilary distance) with optical flow by", &do_stereo_baseline_checkbox);
          ImGui::SameLine();
          ImGui::PushItemWidth(100);
          
          if (ImGui::InputFloat("##924378", &stereo_baseline_adjust_factor, 0.0f, 0.0f, "%.3f")) {
            do_stereo_baseline_checkbox = true;
          }
          stereo_baseline_adjust_factor = math::clamp(stereo_baseline_adjust_factor, 0.1f, 10.0f);
          ImGui::PopItemWidth();
          ImGui::SameLine();
          ImGui::Text("× original"); // NOTE: multiply symbol, not x
          ImGui::Text("If enabled, super resolution is not applied. Assumes left/right arragnement of stereo.");
          ImGui::Text("Use a scale factor between 0.5 and 1.5 for best results.");
          

          ImGui::Dummy(ImVec2(0.0f, 20.0f));
          ImGui::EndChild(); // end of ScrollChild
          ImGui::EndTabItem();
        } // end of Experimental options
        ImGui::EndTabBar();
      } // End of TabBar
      ImGui::EndChild(); // TabBarContainer
      ImGui::PopStyleVar();

    } else {
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

    // Draw the enhance button, possibly disabled
    if (!enhance_ready) {
      ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2, 0.5, 0.2, 0.2));
      ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0, 1.0, 1.0, 0.2));
      ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
    } else {
      ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2, 0.5, 0.2, 0.8));
    }
    ImGui::SetCursorPos(ImVec2(screen_width - enhance_button_width, button_ypos));
    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f); // Make the button rounded
    if (ImGui::Button(input_video_file_select.paths.size() <= 1 ? "Enhance" : "Enhance Batch", ImVec2(enhance_button_width, row_height))) {
      clickEnhance();
    }
    ImGui::PopStyleVar();
    if (!enhance_ready) {
      ImGui::PopItemFlag();
      ImGui::PopStyleColor(2);
    } else {
      ImGui::PopStyleColor(1);
    }

    ImGui::Unindent();
  }

  std::shared_ptr<std::atomic<bool>> cancel_enhance_command_requested = std::make_shared<std::atomic<bool>>(false);
  std::shared_ptr<time::TimePoint> enhance_timer = nullptr;
  void clickEnhance() {
    // Clean up previous run and get ready for new run
    preview_needs_scroll_reset = true;
    preview_image_lr.reset();
    preview_image_hr.reset();

    // Figure out what folder to open on completion of either a single video or batch
    std::string final_output_dir;
    if (input_video_file_select.paths.size() <= 1) {
      final_output_dir = file::getDirectoryName(std::string(output_video_file_select.path));
    } else {
      final_output_dir = file::getDirectoryName(input_video_file_select.paths[0]);
    }

    std::shared_ptr<std::atomic<bool>> enhance_command_failed = std::make_shared<std::atomic<bool>>(false);

    command_runner.setCompleteCallback([&,final_output_dir,enhance_command_failed]() mutable {
      if (*enhance_command_failed) {
        tinyfd_messageBox("Processing Failed", "There was an error during processing. This may be due to incompatible audio streams.\n\nTry unchecking Keep audio and other streams", "ok", "error", 1);
      } else {
        if (tinyfd_messageBox("Finished Processing",
              sanitizeForTFD("Video is in the output directory: " + final_output_dir + "\nDo you want to open the folder?").c_str(),
              "okcancel", "question", 1)) {
          file::openFileExplorer(final_output_dir);
        }
      }
    });

    command_runner.setKilledCallback([&,enhance_command_failed]() mutable {
      if (*enhance_command_failed) {
        XPLERROR << "Enhance failed!";
      } else {
        XPLINFO << "Enhance aborted!";
      }
    });

    auto render_progress_parser = [this](const std::string& line, CommandProgressDescription& p) {
      std::regex frame_regex("Rendered frame (\\d+) / (\\d+).*");
      std::smatch matches;
      if (std::regex_search(line, matches, frame_regex) && matches.size() == 3) {
        int curr_frame = std::stoi(matches[1].str());
        int total_frames = std::stoi(matches[2].str());
        p.progress_str = "Rendered frame: " + std::to_string(curr_frame + 1) + " / " + std::to_string(total_frames);
        p.frac = static_cast<float>(curr_frame) / total_frames;
        // Estimate time per frame
        if (curr_frame == 0) {
          this->enhance_timer = std::make_shared<time::TimePoint>(time::now());
        } else {
          float time_since_start = time::timeSinceSec(*this->enhance_timer);
          float avg_sec_per_frame = time_since_start / curr_frame;
          p.progress_str += ", Average time/frame (sec): " + std::to_string(avg_sec_per_frame);
        }
      }
    };

    if (input_video_file_select.paths.size() == 1) {
      // Process a single input video
      command_runner.queueThreadCommand(
        cancel_enhance_command_requested,
        [&, enhance_command_failed]() mutable {
          std::string input_video_path = std::string(input_video_file_select.paths[0]);
          std::string output_video_path = std::string(output_video_file_select.path);
          if (!runEnhance(input_video_path, output_video_path)) {
            *enhance_command_failed = true;
          }
        },
        render_progress_parser);
    } else {
      // Process all videos in a batch
      for (auto& v : input_video_file_select.paths) {
        command_runner.queueThreadCommand(
          cancel_enhance_command_requested,
          [&, enhance_command_failed]() mutable {
            std::string output_filename = getDefaultOutputFilename(v);
            std::string output_dir(batch_output_dir_select.path);
            if (!runEnhance(v, output_dir + "/" + output_filename)) {
              *enhance_command_failed = true;
            }
          },
          render_progress_parser);
      }
    }

    command_runner.runCommandQueue();
  }

  void onSelectInputVideoFile() {
    if (input_video_file_select.paths.empty()) return;

    // Get resolution etc of selected video
    std::string input_video_path = input_video_file_select.paths[0];

    // Get the video info so we can populate the width and height output fields at least.
    video::getVideoInfo(input_video_path, selected_video_width, selected_video_height,
      selected_video_fps, selected_video_total_frames);

    // If the input file is a single image, default to output PNG, else output MP4
    encode_video_format_select = video::hasImageExt(input_video_path) ? 3 : 1;

    string::copyBuffer(width_input_buffer, std::to_string(int(selected_video_width * kSuperResScale)).c_str(), kNumberBufferSize);
    string::copyBuffer(height_input_buffer, std::to_string(int(selected_video_height * kSuperResScale)).c_str(), kNumberBufferSize);

    // Set output file path to a default value based on the input format and selected image extension
    if (input_video_file_select.paths.size() <= 1) {
      setDefaultOutputFilename();
    } else {
      setDefaultBatchOutputDir();
    }
  }

  void setDefaultBatchOutputDir() {
    XCHECK_GE(input_video_file_select.paths.size(), 2);
    std::string dir_of_first_video = file::getDirectoryName(input_video_file_select.paths[0]);
    batch_output_dir_select.setPath(dir_of_first_video.c_str());
  }

  std::string getDefaultOutputFilename(std::string input_filename) {    
    std::string output_ext = ".mp4";
    if (encode_video_format_select == 0) {
      output_ext = ".mp4";
    } else if (encode_video_format_select == 1) {
      output_ext = ".mp4";
    } else if (encode_video_format_select == 2) {
      output_ext = ".mov";
    } else if (encode_video_format_select == 3) {
      output_ext = ".png";
    } else if (encode_video_format_select == 4) {
      output_ext = ".jpg";
    }
    // If the output is a video and not jpg or png, default to png
    if (!(output_ext == ".png" || output_ext == ".jpg") && video::hasImageExt(input_filename)) {
      output_ext = ".png";
    }

    std::filesystem::path inputPath(input_filename);
    std::string stem = inputPath.stem().string();

    auto now = std::chrono::system_clock::now();
    std::time_t now_time_t = std::chrono::system_clock::to_time_t(now);

    std::tm tm{};
#if defined(_WIN32) || defined(_WIN64)
    localtime_s(&tm, &now_time_t); // Use localtime_s on Windows
#else
    localtime_r(&now_time_t, &tm); // Use localtime_r on Unix-like systems
#endif

    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M");
    std::string timestamp = oss.str();

    // If PNG output is selected, use a template syntax for multiple-file output
    if ((output_ext == ".png" || output_ext == ".jpg") && !video::hasImageExt(input_filename)) {
      timestamp += "_{FRAME_NUMBER}";
    }

    // FAIL attempts at showing the user what the resolution is in the filename
    //int w = selected_video_width * kSuperResScale;
    //int h = selected_video_height * kSuperResScale;
    //std::string resolution = std::to_string(w) + "x" + std::to_string(h);
    //std::string resolution = std::string(width_input_buffer) + "x" + std::string(height_input_buffer);
    //std::string output_filename = stem + "-" + resolution + "-" + timestamp + output_ext;
    std::string output_filename = stem + "-enhanced-" + timestamp + output_ext;
  
    return output_filename;
  }

  void setDefaultOutputFilename() {
    XCHECK(!input_video_file_select.paths.empty());
    std::string output_filename = getDefaultOutputFilename(input_video_file_select.paths[0]);
    std::filesystem::path output_dir = std::filesystem::path(input_video_file_select.paths[0]).parent_path();
    std::filesystem::path output_path = output_dir / output_filename;
    std::string s = output_path.string();
    output_video_file_select.setPath(s.c_str());
  }

  void handleDragDrop(std::vector<std::string> drop_paths) {
    // Only all drag-drop on the main view of the app
    if (app_state != STATE_SPLASH) return;
    if (isAnyModalPopupOpen()) return;

    input_video_file_select.setPaths(drop_paths);

    onSelectInputVideoFile();
  }

  std::string getSuperResModelNameFromSelect() {
    std::string model_str;
    switch(sr_model_select) {
    case 0: model_str = "unet"; break;
    case 1: model_str = "edsr2"; break;
    default: XCHECK(false) << "Unknown model: " << sr_model_select;
    }
    return model_str;
  }

  // Considers the selected model from the UI to pick the right model file
  std::string getSuperResModelPath() {
    std::string model_str = getSuperResModelNameFromSelect();
    switch(enhance_mode_select) {
    case 0: model_str += "_noaug.pt"; break;
    case 1: model_str += "_mystery.pt"; break;
    default: XCHECK(false) << "Unknown enhance mode: " << enhance_mode_select;
    }
    #if defined(__linux__)
      const std::string os_res_path = "ml_models/" + model_str;
    #elif defined(__APPLE__)
      const std::string os_res_path = "ml_models/" + model_str;
    #elif defined(_WIN32)
      const std::string os_res_path = model_str;
    #else
      XCHECK(false) << "Unknown platform";
    #endif
    return p11::runfile::getRunfileResourcePath(os_res_path);
  }

  bool runEnhance(std::string input_video_path, std::string output_video_path) {
    torch::NoGradGuard no_grad;
    const torch::DeviceType device = util_torch::findBestTorchDevice();

    XPLINFO << "Enhance input: " << input_video_path << " output: " << output_video_path;
    
    // Get video info so we can have a progress bar
    video::getVideoInfo(input_video_path, selected_video_width, selected_video_height,
      selected_video_fps, selected_video_total_frames);
    if (video::hasImageExt(input_video_path)) {
      selected_video_total_frames = 1;
    }
    XPLINFO << "video info (best guess): width=" << selected_video_width << ", height=" << selected_video_height
      << ", total_frames=" << selected_video_total_frames << ", fps=" << selected_video_fps;

    // Setup the super resolution model
    if (sr_model != nullptr) { sr_model.reset(); }
    std::string model_name = getSuperResModelNameFromSelect();
    XPLINFO << "Model name: " << model_name;
    sr_model = enhance::makeSuperResModelByName(model_name);
  
    std::string model_res_path = getSuperResModelPath();
    XPLINFO << "Loading model: " << model_res_path;
    torch::load(sr_model, model_res_path);
    XPLINFO << "Finished loading model";
    sr_model->to(device);

    video::InputVideoStream in_stream(input_video_path);
    if (!in_stream.valid()) {
      tinyfd_messageBox(
        "Error: Could Not Open Video",
        sanitizeForTFD("Error trying to read video file " + input_video_path + "\r\nCheck that the file exists and is valid").c_str(),
        "ok", "error", 1);
      return false;
    }

    std::unique_ptr<video::OutputVideoStream> out_stream = nullptr; // Allocated later when we know the output size

    std::atomic<bool> failed{false};

    auto decode_func = [&](std::shared_ptr<std::atomic<bool>> cancel_requested, int frame_num) -> video::MediaFrame {
      video::MediaFrame src_frame;

      while (true) {
        video::VideoStreamResult result = in_stream.readFrame(src_frame, CV_32FC3);
        if (result == video::VideoStreamResult::ERR) {
          failed = true;
          return video::MediaFrame();
        } else if (result == video::VideoStreamResult::FINISHED) {
          return video::MediaFrame();
        }

        if (src_frame.is_video() || encode_video_keep_other_streams) {
          break;
        }
      }

      if (*cancel_requested) return video::MediaFrame();

      if (src_frame.is_video()) {
        // We need an 8 bit version of the image to show int he GUI
        XCHECK_EQ(src_frame.img.type(), CV_32FC3);
        cv::Mat src_image_8u;
        src_frame.img.convertTo(src_image_8u, CV_8UC3, 255.0);
        preview_image_mutex_lr.lock();
        if (frame_num == 0) { preview_needs_scroll_reset = true; }
        preview_image_lr.setImage(src_image_8u);
        preview_image_mutex_lr.unlock();
      }

      return src_frame;
    };

    auto process_func = [&](std::shared_ptr<std::atomic<bool>> cancel_requested, video::MediaFrame src_frame, int frame_num) -> video::MediaFrame {
      // Non-video frames are no-op
      if (!src_frame.is_video()) return src_frame;

      std::string model_name = getSuperResModelNameFromSelect();
      auto sr_model_copy = enhance::makeSuperResModelByName(model_name); // we make a copy for thread safety
      util_torch::deepCopyModel(sr_model, sr_model_copy);

      if (*cancel_requested) return video::MediaFrame();

      cv::Mat prefiltered_image = src_frame.img;
      if (do_bilateral_filter_checkbox) {
        const float sigma = bilateral_filter_radius * 0.6;
        const float sigma_color = 0.1;
        cv::Mat bilateral = opencv::bilateralDenoise(src_frame.img, bilateral_filter_radius, sigma, sigma_color);

        prefiltered_image = bilateral * bilateral_filter_strength + src_frame.img * (1.0 - bilateral_filter_strength);
      }

      torch::Tensor output_tensor;
      if (do_stereo_baseline_checkbox) {
        auto raft_copy = raft.clone(); // we make a copy for thread safety
        cv::Mat L_image = prefiltered_image(cv::Rect(0, 0, prefiltered_image.cols / 2, prefiltered_image.rows));
        cv::Mat R_image = prefiltered_image(cv::Rect(prefiltered_image.cols / 2, 0, prefiltered_image.cols / 2, prefiltered_image.rows));
      
        float hi = 0.5 + stereo_baseline_adjust_factor * 0.5;
        float lo = 0.5 - stereo_baseline_adjust_factor * 0.5;
        std::vector<float> interps = {lo, hi};
        std::vector<cv::Mat> adjusted_images = optical_flow::generateBetweenFrameWithFlow(L_image, R_image, raft_copy, interps);
        cv::Mat new_stereo_image;
        cv::hconcat(adjusted_images[0], adjusted_images[1], new_stereo_image);

        output_tensor = torch_opencv::cvMat_to_Tensor(device, new_stereo_image).permute({1, 2, 0});
      } else if (do_ai_restoration_model_checkbox) {
        constexpr float kSuperResScale = 2.0;
        output_tensor = superResolutionEnhance(device, kSuperResScale, sr_model_copy, prefiltered_image);
      } else { // Do nothing in this step, but other steps can still be useful
        output_tensor = torch_opencv::cvMat_to_Tensor(device, prefiltered_image).permute({1, 2, 0});
      }

      if (*cancel_requested) return video::MediaFrame();

      video::MediaFrame output_frame;
      torch_opencv::fastTensor_To_CvMat(output_tensor, output_frame.img);
      return output_frame;
    };

    std::vector<video::MediaFrame> frames_to_encode;
    auto encode_func = [&, input_video_path](std::shared_ptr<std::atomic<bool>> cancel_requested, video::MediaFrame output_frame, int frame_num) {
      if (*cancel_requested) return;

      if (!output_frame.is_video()) {
        // Buffer non-video frames until we can create the encoder from the first video frame
        frames_to_encode.push_back(output_frame);

        if (!out_stream) {
          return;
        }
      } else {
        int output_w = math::clamp(std::atoi(width_input_buffer), 1, 65535);
        int output_h = math::clamp(std::atoi(height_input_buffer), 1, 65535);

        XCHECK_EQ(output_frame.img.type(), CV_32FC3);
        cv::Mat output_frame_8u;
        output_frame.img.convertTo(output_frame_8u, CV_8UC3, 255.0);
        preview_image_mutex_hr.lock();
        preview_image_hr.setImage(output_frame_8u);
        hr_preview_scale_factor = 2.0 * float(preview_image_lr.size.width) / float(output_frame_8u.cols);
        preview_image_mutex_hr.unlock();

        // Resize the image to the expected output size (could be a no-op in some cases)
        cv::resize(output_frame.img, output_frame.img, cv::Size(output_w, output_h), 0, 0, cv::INTER_CUBIC);
    
        frames_to_encode.push_back(output_frame);

        // Don't change this log message. It is used to update the UI
        XPLINFO << "Rendered frame " << frame_num << " / " << selected_video_total_frames
          << " size " << in_stream.getWidth() << "x" << in_stream.getHeight()
          << " to output size " << output_frame.img.cols << "x" << output_frame.img.rows;

        // If on the first frame, allocate the output video stream
        if (frame_num == 0) {
          int crf = 0;
          switch(encode_video_bitrate_select) {
          case 0: crf = 28; break;
          case 1: crf = 23; break;
          case 2: crf = 18; break;
          default: XCHECK(false);
          }
          
          std::string video_encoder;
          switch(encode_video_format_select) {
          case 0: video_encoder = "libx264"; break;
          case 1: video_encoder = "libx265"; break;
          case 2: video_encoder = "prores"; break; // HACK
          case 3: video_encoder = "png"; break;
          case 4: video_encoder = "jpg"; break;
          default: XCHECK(false);
          }
          // Always use PNG or JPG encoder for image inputs.
          if ( !(video_encoder == "png" || video_encoder == "jpg") && video::hasImageExt(input_video_path)) {
            video_encoder = "png";
          }

          video::ProResProfile prores_profile{};
          switch(encode_video_profile_select) {
          case 0: prores_profile = video::PRORES_422LT; break;
          case 1: prores_profile = video::PRORES_422HQ; break;
          case 2: prores_profile = video::PRORES_4444; break;
          default: XCHECK(false);
          }
          XCHECK(out_stream == nullptr);
          video::EncoderConfig config{};
          config.crf = crf;
          config.prores_profile = prores_profile;
          if (encode_video_keep_other_streams) {
            // construct using in_stream as first argument so that the streams are preserved
            out_stream = std::make_unique<video::OutputVideoStream>(
              in_stream, output_video_path, output_frame.img.cols, output_frame.img.rows,
              in_stream.guessFrameRate(), video_encoder, config);
          } else {
            // construct just a single output stream
            out_stream = std::make_unique<video::OutputVideoStream>(
              output_video_path, output_frame.img.cols, output_frame.img.rows,
              in_stream.guessFrameRate(), video_encoder, config);
          }
          if (!out_stream->valid()) {
            XPLERROR << "Failed to create output stream";
            *cancel_requested = true; // Stop the other transcode threads
            failed = true; // Notify the ui that it was a failure
            return;
          }
        }
      }

      XCHECK(out_stream);

      for (auto& frame : frames_to_encode) {
        if (!out_stream->valid()) {
          tinyfd_messageBox(
            "Error while writing video",
            sanitizeForTFD("Error trying to write video file: " + output_video_path).c_str(),
            "ok", "error", 1);
          // Using cancel_requested to stop the rest of the pipeline
          // TODO: proper cancel/error/success primitives
          *cancel_requested = true;
          failed = true; // Notify the ui that it was a failure
          return;
        }

        if (!out_stream->writeFrame(frame)) {
          tinyfd_messageBox(
            "Error while writing video",
            sanitizeForTFD("Error writing video frame: " + output_video_path).c_str(),
            "ok", "error", 1);
          // Using cancel_requested to stop the rest of the pipeline
          // TODO: proper cancel/error/success primitives
          *cancel_requested = true;
          failed = true; // Notify the ui that it was a failure
          return;
        }
      }

      frames_to_encode.clear();
    };

    XPLINFO << "# of Super Resolution Threads: " << pref_num_threads;
    video::transcodeWithThreading(
      decode_func,
      process_func,
      encode_func,
      pref_num_threads,
      cancel_enhance_command_requested);

    return !failed;
  }

  ~UpscaleVideoApp() {
    preview_image_lr.freeGlTexture();
    preview_image_hr.freeGlTexture();
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

  void showNumThreadsPrefDialog() {
    std::string default_num_threads = "1";
    if (prefs.count(kPrefNumThreadsKey)) {
      default_num_threads = prefs.at(kPrefNumThreadsKey);
    }
    const char* result_str = tinyfd_inputBox(
      "Set Number of Super Resolution Threads",
      "Enter a number between 1 and 32",
      default_num_threads.c_str());
    if (result_str == nullptr) return; // cancelled
    pref_num_threads = std::atoi(result_str);
    pref_num_threads = math::clamp(pref_num_threads, 1, 32);

    prefs[kPrefNumThreadsKey] = std::to_string(pref_num_threads);
    preferences::setPrefs(prefs);
  }

  void drawMainMenu()
  {
    if (ImGui::BeginMenuBar()) {
      // Add some more padding around the menu items... otherwise it looks wack.
      ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(30, 30));
      ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(10, 10));
      if (ImGui::BeginMenu(" Settings ")) {
        ImGui::Dummy(ImVec2(0, 8));

        if (ImGui::MenuItem("About UpscaleVideo.ai")) {
          showAboutDialog();
        }

        if (ImGui::MenuItem("Number of Super Resolution Threads")) {
          showNumThreadsPrefDialog();
        }

        ImGui::Dummy(ImVec2(0, 8));
        ImGui::Separator();
        ImGui::Dummy(ImVec2(0, 8));

        if (ImGui::MenuItem("Show Console")) {
          consoleSleepCommand();
        }
        //if (ImGui::MenuItem("Debug Test 1")) { testCommand(); }
        //if (ImGui::MenuItem("Debug Test 2")) { testThreadCommand(); }

        ImGui::Dummy(ImVec2(0, 8));
        ImGui::EndMenu();
      }

      ImGui::PopStyleVar(2);

      ImGui::EndMenuBar();
    }
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

  bool commandWindowIsVisible() { return command_runner.isRunning(); }

  bool Splitter(bool split_vertically, float thickness, float* size1, float* size2, float min_size1, float min_size2, float splitter_long_axis_size = -1.0f)
  {
    using namespace ImGui;
    ImGuiContext& g = *GImGui;
    ImGuiWindow* window = g.CurrentWindow;
    ImGuiID id = window->GetID("##Splitter");
    ImRect bb;
    //bb.Min = window->DC.CursorPos + (split_vertically ? ImVec2(*size1, 0.0f) : ImVec2(0.0f, *size1));
    //bb.Max = bb.Min + CalcItemSize(split_vertically ? ImVec2(thickness, splitter_long_axis_size) : ImVec2(splitter_long_axis_size, thickness), 0.0f, 0.0f);
    bb.Min.x = window->DC.CursorPos.x + (split_vertically ? *size1 : 0.0f);
    bb.Min.y = window->DC.CursorPos.y + (split_vertically ? 0.0f : *size1);
    ImVec2 bb_size = CalcItemSize(split_vertically ? ImVec2(thickness, splitter_long_axis_size) : ImVec2(splitter_long_axis_size, thickness), 0.0f, 0.0f);
    bb.Max.x = bb.Min.x + bb_size.x;
    bb.Max.y = bb.Min.y + bb_size.y;

    return SplitterBehavior(bb, id, split_vertically ? ImGuiAxis_X : ImGuiAxis_Y, size1, size2, min_size1, min_size2, 0.0f);
  }

  void updatePreviewScrolling() {
    if (click_started_in_preview_viewport && left_mouse_down) {
      const float dx = prev_mouse_x - curr_mouse_x;
      const float dy = prev_mouse_y - curr_mouse_y;
      preview_scroll_x += dx;
      preview_scroll_y += dy;
    }
  }

  ImVec2 PredictScrollMax(const ImVec2& content_size, const ImVec2& window_size) {
    //return ImVec2(
    //  ImMax(0.0f, content_size.x - window_size.x),
    //  ImMax(0.0f, content_size.y - window_size.y));
    return ImVec2( // HACK: don't clamp, it all works out somehow
        content_size.x - window_size.x,
        content_size.y - window_size.y);
  }

  // No 1 frame delay but probably not compatible with scroll bars
  void SetWindowScrollImmediately(float sx, float sy)
  {
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    window->Scroll.x = sx;
    window->Scroll.y = sy;
  }

  float preview_padding = 1; // HACK: this seems to work with any value other than 0
  void drawPreviewPanel(gui::ImguiCvMat& preview_image, std::mutex& mu, float scale_factor) {
    updatePreviewScrolling(); // TODO: not sure if this is where we should call this
  
    if (preview_image.size.width == 0) {
      ImGui::Text("Processing...");
      return;
    }

    preview_image.scale_to_fit = false;
    preview_image.center_and_expand = false;
    preview_image.scale_to_factor = true;
    preview_image.scale_factor = scale_factor * previewZoomToScaleFactor();
    mu.lock();
    preview_image.makeGlTexture();
    mu.unlock();

    ImGui::SetCursorPos(ImVec2(preview_padding, preview_padding));

    preview_image.drawInImGui();
  }

  int preview_zoom = 2;
  float previewZoomToScaleFactor() {
    switch(preview_zoom) {
    case 0: return 0.25;
    case 1: return 0.5;
    case 2: return 1.0;
    case 3: return 1.5;
    case 4: return 2.0;
    case 5: return 4.0;
    default: XCHECK(false) << "Invalid zoom level: " << preview_zoom;
    }
    return 1.0;
  }

  void drawCommandWindow()
  {
    constexpr float kProgressPanelHeight = 80;
    constexpr float kCancelButtonWidth = 200;
    constexpr float kCancelButtonHeight = 64;
    constexpr float kPadding = 8;

    ImVec2 size = ImGui::GetContentRegionAvail();
    float w = size.x;  // Full width of the content region
    float h = size.y;  // Full height of the content region
    static float sz2 = 200;       // Initial height of the bottom panel
    static float sz1 = h - kProgressPanelHeight - sz2;   // Initial height of the top panel

    // Cancel / progress panel
    ImGui::BeginChild("ChildProgress", ImVec2(w, kProgressPanelHeight), true, ImGuiWindowFlags_NoScrollbar);

    ImGui::PushStyleVar(ImGuiStyleVar_FrameRounding, 6.0f); // Make the button rounded
    if (ImGui::Button("Stop", ImVec2(kCancelButtonWidth, kCancelButtonHeight))) {
      command_runner.kill();
    }
    ImGui::PopStyleVar();

    ImGui::SameLine();
    ImGui::Dummy(ImVec2(kPadding, 0.0)); 
    ImGui::SameLine();

    if (command_runner.waitingForCancel()) {  
      ImVec2 curr_pos = ImGui::GetCursorPos();
      ImGui::SetCursorPos(ImVec2(curr_pos.x + 10, curr_pos.y + 20)); // HACK tweaks to center text in progress bar
      ImGui::Text("Finishing...");
    } else { // Progress bar
      float available_width = ImGui::GetContentRegionAvail().x - kPadding;
      CommandProgressDescription progress_description = command_runner.updateAndGetProgress();

      ImVec2 progressBarPos = ImGui::GetCursorPos();

      // Draw the progress bar
      ImGui::PushItemWidth(available_width);
      ImGui::ProgressBar(progress_description.frac, ImVec2(available_width, kCancelButtonHeight), "");
      ImGui::PopItemWidth();

      // Move the cursor back to overlay the text on the progress bar
      ImGui::SetCursorPos(ImVec2(progressBarPos.x + 10, progressBarPos.y + 20)); // HACK tweaks to center text in progress bar
      ImGui::TextUnformatted(progress_description.progress_str.c_str());
    }

    ImGui::EndChild(); // End ChildProgress
  
    constexpr int kMinSize = 100;
    constexpr int kSplitterDragSize = 16;
    Splitter(false, kSplitterDragSize, &sz1, &sz2, kMinSize, kMinSize, w);


    /// start of preview section ///

    // Calculate the size of the content to be scrolled in one panel of the preview
    ImVec2 preview_content_size;
    if (!preview_image_lr.empty()) {
      float s = previewZoomToScaleFactor() * kSuperResScale;
      preview_content_size = ImVec2(
        preview_image_lr.size.width * s + 2 * preview_padding,
        preview_image_lr.size.height * s + 2 * preview_padding);
      ImGui::SetNextWindowContentSize(preview_content_size);
    }

    float split_width = w / 2.0f;

    // Left side (original image)
    ImGui::BeginChild("ChildPreviewLeft", ImVec2(split_width, sz1), false, ImGuiWindowFlags_NoScrollbar);

    // Save preview viewport rectangle so we can check if user is dragging to scroll
    preview_viewport_size = ImVec2(w, sz1);
    preview_viewport_min = ImGui::GetWindowPos();

    if (preview_needs_scroll_reset && !preview_image_lr.empty()) {
      ImVec2 window_size(split_width, sz1);
      ImVec2 predicted_max_scroll = PredictScrollMax(preview_content_size, window_size);
      preview_scroll_x = predicted_max_scroll.x * 0.5f;
      preview_scroll_y = predicted_max_scroll.y * 0.5f;
      preview_needs_scroll_reset = false;
    }

    SetWindowScrollImmediately(preview_scroll_x, preview_scroll_y);
    drawPreviewPanel(preview_image_lr, preview_image_mutex_lr, kSuperResScale);
    ImGui::EndChild(); // End ChildPreviewLeft

    // Right side (super resolution image, currently a copy of the original)
    ImGui::SameLine();
    ImGui::BeginChild("ChildPreviewRight", ImVec2(split_width, sz1), false, ImGuiWindowFlags_NoScrollbar);
    if (preview_image_hr.empty()) {
      SetWindowScrollImmediately(0, 0); // If the HR image is not ready to draw, scroll to 0 in this panel so we can see the "Processing..." label instead and not have that scrolled like the left panel.
    } else {
      SetWindowScrollImmediately(preview_scroll_x, preview_scroll_y);
    }

    drawPreviewPanel(preview_image_hr, preview_image_mutex_hr, hr_preview_scale_factor); // Scaled based on target output size
    
    ImGui::EndChild(); // End ChildPreviewRight

    /// end of preview section ///

    // Preview zoom combo
    ImGui::Dummy(ImVec2(0.0, kSplitterDragSize)); // Add some vertical padding. Without this the scroll command content ends up on top of the splitter

    static const char* preview_zoom_labels[] = { "50%", "100%", "200%", "300%", "400%", "800%" };
    constexpr float combo_width = 150;
    ImGui::Dummy(ImVec2(ImGui::GetWindowWidth() - combo_width - 42, kSplitterDragSize));
    ImGui::SameLine(); 
    ImGui::PushItemWidth(combo_width);
    if (ImGui::Combo("##PreviewZoomCombo", &preview_zoom, preview_zoom_labels, IM_ARRAYSIZE(preview_zoom_labels))) {
      preview_needs_scroll_reset = true;
    }
    ImGui::PopItemWidth();

    // Command panel
    ImGui::BeginChild("CommandScroll", ImVec2(0, 0), true, ImGuiWindowFlags_None);

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
  
    ImGui::EndChild(); // End CommandScroll
  }

  void handleMouseDown(int button)
  {
    if (button == GLFW_MOUSE_BUTTON_LEFT) left_mouse_down = true;
    if (button == GLFW_MOUSE_BUTTON_RIGHT) right_mouse_down = true;

    click_started_in_preview_viewport = false;

    if (mouse_in_preview_viewport) click_started_in_preview_viewport = true;
  }

  void handleMouseUp(int button)
  {
    if (button == GLFW_MOUSE_BUTTON_LEFT) left_mouse_down = false;
    if (button == GLFW_MOUSE_BUTTON_RIGHT) right_mouse_down = false;
  }

  void updateMouse() {
    prev_mouse_x = curr_mouse_x;
    prev_mouse_y = curr_mouse_y;
    glfwGetCursorPos(window, &curr_mouse_x, &curr_mouse_y);

    mouse_in_preview_viewport = false;
    if (commandWindowIsVisible()) {
      if (curr_mouse_x >= preview_viewport_min.x && curr_mouse_y >= preview_viewport_min.y &&
          curr_mouse_x <= preview_viewport_min.x + preview_viewport_size.x &&
          curr_mouse_y <= preview_viewport_min.y + preview_viewport_size.y) {
        mouse_in_preview_viewport = true;
      }
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
    if (mouse_in_preview_viewport) {
      ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;
      glfwSetCursor(window, glfwCreateStandardCursor(GLFW_HAND_CURSOR));
    }
  }


  float getScaledFontSize() {
    ImGuiIO& io = ImGui::GetIO();
    return io.Fonts->Fonts[0]->FontSize * io.FontGlobalScale;
  }

  void drawFrame()
  {
    //gl_context_mutex.lock();
    //glfwMakeContextCurrent(window);

    // Scale text for high DPI displays
    float xscale, yscale;
    glfwGetWindowContentScale(window, &xscale, &yscale);
    //constexpr float kFontScaleMultiplier = 1.0;
    constexpr float kFontScaleMultiplier = 0.5; // HACK: scale the font down to get a sharper render of the text
    ImGui::GetIO().FontGlobalScale = xscale * kFontScaleMultiplier;

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

    std::string window_title = "UpscaleVideo.ai";
    
    //if (!source_video_path.empty()) window_title += " - " + source_video_path;
    if (!input_video_file_select.paths.empty()) {
      std::string title = input_video_file_select.paths[0];
      if (!title.empty()) window_title += " - " + title;
    }

    glfwSetWindowTitle(window, window_title.c_str());
    
    if (ImGui::Begin("Main Window", nullptr, window_flags)) {
      updateMouse();
      updateCursor();
      if (commandWindowIsVisible()) {
        drawCommandWindow();
      } else {
        switch(app_state) {
        case STATE_SPLASH:
          drawMainMenu();
          splashScreen();
          break;
        }
      }

      ImGui::End();
    }
    finishDrawingImguiAndGl();
  }
};

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
  auto app = static_cast<UpscaleVideoApp*>(glfwGetWindowUserPointer(window));
  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) app->handleMouseDown(button);
  if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) app->handleMouseUp(button);
  if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) app->handleMouseDown(button);
  if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE) app->handleMouseUp(button);
}

void drop_callback(GLFWwindow* window, int count, const char** paths) {
  UpscaleVideoApp* app = static_cast<UpscaleVideoApp*>(glfwGetWindowUserPointer(window));

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
    if (!(video::hasImageExt(f) || video::hasVideoExt(f))) continue;
    filtered_paths.push_back(f);
  }
  if (filtered_paths.empty()) {
    tinyfd_messageBox(
      "Invalid drag drop",
      "Only videos, images, or a single folder containing videos or images can be drag-dropped.", "ok", "info", 1);
  }
  app->handleDragDrop(filtered_paths);
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

  p11::UpscaleVideoApp app;
  app.init("UpscaleVideo.ai", 1280, 720);

  glfwSetWindowUserPointer(app.window, (void*)&app);

  glfwSetMouseButtonCallback(app.window, p11::mouse_button_callback);
  glfwSetDropCallback(app.window, p11::drop_callback);
  app.initUpscaleVideoApp();
  app.setProStyle();

#ifdef _WIN32
  const std::string font_path = "Helvetica.ttf";  // On Windows, the directory structure is flat.
#else
  const std::string font_path = "fonts/Helvetica.ttf";
#endif
  ImGuiIO& io = ImGui::GetIO();
  io.Fonts->AddFontFromFileTTF(p11::runfile::getRunfileResourcePath(font_path).c_str(), 24.0);

  ImGui::GetIO().IniFilename = nullptr;  // Disable layout saving.

  app.guiDrawLoop();
  app.cleanup();

  return EXIT_SUCCESS;
}
