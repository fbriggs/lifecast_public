// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "volurama_timeline.h"

#include <limits>

namespace p11 {

void VideoTimelineWidget::render()
{
  ImGui::PushStyleColor(ImGuiCol_ChildBg, IM_COL32(32, 32, 32, 255));
  ImGui::BeginChild("Timeline", ImVec2(0, kTimelineHeight), false, ImGuiWindowFlags_HorizontalScrollbar);

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
        ImVec2(x, child_topleft.y + top_space + (i % tick_interval != 0 ? kHandleHeight / 2 : 0)),
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

    if (keyframes.count(i) != 0) {
      ImGui::GetWindowDrawList()->AddCircleFilled(ImVec2(x, child_topleft.y + 70), 5, IM_COL32(0, 255, 0, 196));
    }
  }

  ImVec2 curr_handle_pos(
      child_topleft.x + kHorizontalPadding + curr_frame * pixels_per_frame,
      child_topleft.y + top_space);

  int curr_frame_before_update = curr_frame;
  updateHandle(curr_handle_pos, curr_frame, is_dragging_curr_handle);

  if (curr_frame_before_update != curr_frame) {
    if (curr_frame_change_callback) curr_frame_change_callback();
  }

  renderHandle(curr_handle_pos, IM_COL32(100, 100, 255, is_dragging_curr_handle ? 255 : 200));

  viewport_min = ImGui::GetWindowPos();
  viewport_size = ImGui::GetContentRegionAvail();  // excludes scrollbars
  viewport_size.y = kTimelineHeight - ImGui::GetStyle().ScrollbarSize;

  ImGui::EndChild();
  ImGui::PopStyleColor();
}

void VideoTimelineWidget::renderHandle(const ImVec2& pos, const ImU32& color)
{
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
}

void VideoTimelineWidget::updateHandle(const ImVec2& pos, int& frame, bool& is_dragging) {
  if (!ImGui::IsMouseDown(0)) {
    is_dragging = false;
  }

  ImVec2 hover_min, hover_max;
  hover_min = ImVec2(std::numeric_limits<float>::lowest(), pos.y);
  hover_max = ImVec2(std::numeric_limits<float>::max(), pos.y + kTimelineHeight);

  if (!is_dragging && ImGui::IsMouseDown(0)) {
    if (ImGui::IsMouseHoveringRect(hover_min, hover_max)) {
      is_dragging = true;
    }
  }

  if (is_dragging) {
    ImVec2 child_topleft = ImGui::GetCursorScreenPos();
    frame = (ImGui::GetIO().MousePos.x - child_topleft.x - kHorizontalPadding) / pixels_per_frame;
    frame = std::clamp(frame, 0, num_frames - 1);
  }
}

} // namespace p11