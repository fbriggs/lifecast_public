// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <string>
#include <vector>
#include <functional>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "dear_imgui_app.h"  // include this to prevent include order shenanagins
#include "third_party/tinyfiledialogs.h"
#include "util_opengl.h"
#include "util_string.h"

namespace p11 { namespace gui {
struct ImguiFolderSelect {
  static constexpr int kBufferSize = 1024;
  char path[kBufferSize] = "";
  std::string label;

  std::string hash1, hash2;  // used to prevent button collisions in Imgui
  bool editable = true;
  bool required = true;

  ImguiFolderSelect(const std::string& label = "") : label(label)
  {
    hash1 = std::to_string(rand());
    hash2 = std::to_string(rand());
    // TODO: there is a small chance of hash collisions that would break the GUI
  }

  void setPath(const char* new_path) {
    string::copyBuffer(path, new_path, kBufferSize);
  }

  void setPath(std::string new_path) { setPath(new_path.c_str()); }

  std::string getPath() const { return std::string(path); }

  void drawAndUpdate()
  {
    if (!label.empty()) ImGui::Text("%s", label.c_str());

    if (ImGui::Button(("...##" + hash2).c_str())) {
      const char* result = tinyfd_selectFolderDialog(label.c_str(), nullptr);
      if (result != nullptr) {
        string::copyBuffer(path, result, kBufferSize);
      }
    }
    ImGui::SameLine();
    ImGui::Dummy(ImVec2(4.0f, 0.0f));
    ImGui::SameLine();
    if (editable) {
      ImGui::InputText(("##" + hash1).c_str(), path, IM_ARRAYSIZE(path));
    } else {
      ImGui::Text("%s", path);
    }

    if (required && std::string(path).empty()) {
      ImGui::SameLine();
      ImGui::Dummy(ImVec2(4.0f, 0.0f));
      ImGui::SameLine();
      ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(78, 245, 17, 240));
      ImGui::Text("!");
      ImGui::PopStyleColor();
    }
  }
};

struct ImguiInputFileSelect {
  static constexpr int kBufferSize = 1024;
  char path[kBufferSize] = "";
  std::string label;
  std::string hash1, hash2;  // used to prevent button collisions in Imgui
  bool editable = true;
  bool required = true;
  std::function<void()> on_change_callback;

  ImguiInputFileSelect(const std::string& label = "") : label(label)
  {
    hash1 = std::to_string(rand());
    hash2 = std::to_string(rand());
    // TODO: there is a small chance of hash collisions that would break the GUI
  }

  void setPath(const char* new_path) {
    string::copyBuffer(path, new_path, kBufferSize);
  }

  void drawAndUpdate()
  {
    if (!label.empty()) ImGui::Text("%s", label.c_str());

    if (ImGui::Button(("...##" + hash2).c_str())) {
      const char* result = tinyfd_openFileDialog(label.c_str(), nullptr, 0, nullptr, nullptr, 0);
      if (result != nullptr) {
        string::copyBuffer(path, result, kBufferSize);
        if (on_change_callback) on_change_callback();
      }
    }
    ImGui::SameLine();
    if (editable) {
      ImGui::InputText(("##" + hash1).c_str(), path, IM_ARRAYSIZE(path));
    } else {
      ImGui::Text("%s", path);
    }

    if (required && std::string(path).empty()) {
      ImGui::SameLine();
      ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));
      ImGui::Text("!");
      ImGui::PopStyleColor();
    }
  }
};

struct ImguiOutputFileSelect {
  static constexpr int kBufferSize = 1024;
  char path[kBufferSize] = "";
  std::string label, default_filename;
  std::string hash1, hash2;  // used to prevent button collisions in Imgui
  bool editable = true;
  bool required = true;

  ImguiOutputFileSelect(const std::string& label, const std::string& default_filename)
      : label(label), default_filename(default_filename)
  {
    hash1 = std::to_string(rand());
    hash2 = std::to_string(rand());
    // TODO: there is a small chance of hash collisions that would break the GUI
  }

  void setPath(const char* new_path) {
    string::copyBuffer(path, new_path, kBufferSize);
  }

  void drawAndUpdate()
  {
    if (!label.empty()) ImGui::Text("%s", label.c_str());

    if (ImGui::Button(("...##" + hash2).c_str())) {
      const char* result =
          tinyfd_saveFileDialog(label.c_str(), default_filename.c_str(), 0, nullptr, nullptr);
      if (result != nullptr) {
        string::copyBuffer(path, result, kBufferSize);
      }
    }

    ImGui::SameLine();
    if (editable) {
      ImGui::InputText(("##" + hash1).c_str(), path, IM_ARRAYSIZE(path));
    } else {
      ImGui::Text("%s", path);
    }

    if (required && std::string(path).empty()) {
      ImGui::SameLine();
      ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 0, 0, 255));
      ImGui::Text("!");
      ImGui::PopStyleColor();
    }
  }
};

}}  // namespace p11::gui
