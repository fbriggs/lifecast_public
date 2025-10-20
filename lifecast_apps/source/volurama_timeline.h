// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <functional>
#include <string>
#include <map>
#include "third_party/dear_imgui/imgui.h"
#include "rectilinear_camera.h"

namespace p11 {

struct VirtualCamKeyframe {
  calibration::RectilinearCamerad cam;
  float tx = 0;
  float ty = 0;
  float tz = 0;
  float rx = 0;
  float ry = 0;
  float rz = 0;
};

class VideoTimelineWidget {
 public:
  static constexpr int kTimelineHeight = 128;
  static constexpr float kHandleHeight = 20.0;
  static constexpr float kHandleHalfWidth = 10.0;
  static constexpr int kHorizontalPadding = 20;

  float pixels_per_frame = 5.0;  // the width of the selection area for 1 frame
  int num_frames = 0;
  int curr_frame = 0;
  bool is_dragging_curr_handle = false;

  ImVec2 viewport_min, viewport_size;

  std::function<void()> curr_frame_change_callback;

  std::map<int, VirtualCamKeyframe> keyframes;

  void render();

 private:
  void renderHandle(const ImVec2& pos, const ImU32& color);
  void updateHandle(const ImVec2& pos, int& frame, bool& is_dragging);
};

} // namespace p11