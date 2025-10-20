// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// The code for Run button, Cancel button, and progress bar.
// Get re-used in a couple of places.
#pragma once

#include <string>
#include <vector>
#include "dear_imgui_app.h"  // include this to prevent include order shenanagins
#include "third_party/dear_imgui/imgui_internal.h"

namespace p11 { namespace gui {

template <typename TGuiData>
void makeProgressBar(GLFWwindow* window, const TGuiData& gui_data)
{
  if (gui_data.is_running && (gui_data.progress_max != 0 || gui_data.user_cancelled)) {
    float progress_frac = float(gui_data.progress_curr) / float(gui_data.progress_max);
    std::string progress_str;
    if (gui_data.user_cancelled) {
      progress_frac = 0;
      progress_str = "... CANCELLING ...";
    } else {
      progress_str = gui_data.progress_prefix + std::to_string(gui_data.progress_curr) + "/" +
                     std::to_string(gui_data.progress_max) + " (" +
                     std::to_string(int(100.0 * progress_frac)) + "%)";
    }
    ImGui::Dummy(ImVec2(0.0f, 2.0f));

    int glfw_window_w, glfw_window_h;
    glfwGetWindowSize(window, &glfw_window_w, &glfw_window_h);
    ImGui::PushItemWidth(glfw_window_w - 30);
    ImGui::ProgressBar(progress_frac, ImVec2(0, 0), progress_str.c_str());
    ImGui::PopItemWidth();
  }
}

template <typename TGuiData>
void makeRunCancelButtonPrefix(const TGuiData& gui_data, bool& need_style_pop)
{
  if (gui_data.is_running) {
    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
    need_style_pop = true;
  }
}

template <typename TGuiData>
void makeRunCancelButtonSuffix(TGuiData& gui_data, const bool need_style_pop)
{
  if (need_style_pop) {
    ImGui::PopItemFlag();
    ImGui::PopStyleVar();
  }

  ImGui::SameLine();

  if (!gui_data.is_running) {
    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);
  }
  if (ImGui::Button("Cancel", ImVec2(100, 50))) {
    gui_data.user_cancelled = true;
  }
  if (!gui_data.is_running) {
    ImGui::PopItemFlag();
    ImGui::PopStyleVar();
  }
}

}}  // namespace p11::gui
