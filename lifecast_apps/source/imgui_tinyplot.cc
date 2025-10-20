// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "imgui_tinyplot.h"
#include <string>
#include <sstream>
#include <iomanip>
#include <cmath>
#include "third_party/dear_imgui/imgui.h"

namespace p11 { namespace gui {

void plotLineGraph(
  const std::vector<float>& x_data,
  const std::vector<float>& y_data,
  float min_x, float max_x,
  float min_y, float max_y,
  const bool draw_dots,
  const int plot_width,
  const int plot_height
) {
  if (x_data.size() != y_data.size() || x_data.empty()) {
    return;
  }

  // Calculate y-ticks based on max_y in data
  int magnitude = std::floor(std::log10(max_y));
  float tick_size = pow(10, magnitude - (max_y <= pow(10, magnitude) ? 1 : 0));
  int num_ticks = std::ceil(max_y / tick_size);
  std::vector<float> y_ticks;
  for (int t = 0; t <= num_ticks; ++t) {
    y_ticks.push_back(tick_size * t);
  }

  constexpr float kMarginTop = 20.0;
  constexpr float kMarginLeft = 120.0;
  constexpr float kMarginBottom = 60.0;
  constexpr float kMarginRight = 20.0;

  // Calculate the plot area with margins
  ImVec2 p_min_full = ImGui::GetCursorScreenPos();
  ImVec2 p_max_full = (plot_width <= 0 || plot_height <= 0)
                      ? ImVec2(p_min_full.x + ImGui::GetContentRegionAvail().x, p_min_full.y + ImGui::GetContentRegionAvail().y)
                      : ImVec2(p_min_full.x + plot_width, p_min_full.y + plot_height);
  ImVec2 p_min = ImVec2(p_min_full.x + kMarginLeft, p_min_full.y + kMarginTop);
  ImVec2 p_max = ImVec2(p_max_full.x - kMarginRight, p_max_full.y - kMarginBottom);

  // Fill plot background color
  //ImGui::GetWindowDrawList()->AddRectFilled(p_min_full, p_max_full, IM_COL32(255, 255, 255, 20));

  // Draw plot rectangle
  ImGui::GetWindowDrawList()->AddRect(p_min, p_max, IM_COL32(255, 255, 255, 32));

  auto graph2pixcoord = [&](const float x, const float y) {
    float x_scale_prev = (x - min_x) / (max_x - min_x);
    float y_scale_prev = (y - min_y) / (max_y - min_y);
    return ImVec2(p_min.x + x_scale_prev * (p_max.x - p_min.x), p_max.y - y_scale_prev * (p_max.y - p_min.y));
  };

  // Draw the line plot within the plot area
  for (size_t i = 1; i < x_data.size(); ++i) {
    ImVec2 p1 = graph2pixcoord(x_data[i - 1], y_data[i - 1]);
    ImVec2 p2 = graph2pixcoord(x_data[i], y_data[i]);
    ImGui::GetWindowDrawList()->AddLine(p1, p2, IM_COL32(255, 255, 255, 192), 1.0);
  }

  if (draw_dots) {
    for (size_t i = 0; i < x_data.size(); ++i) {
      ImVec2 p = graph2pixcoord(x_data[i], y_data[i]);
      ImGui::GetWindowDrawList()->AddCircleFilled(p, 3.0, IM_COL32(255, 128, 128, 192));
    }
  }

  std::vector<float> intervals = {1, 10, 100, 1000};
  float range_x = max_x - min_x, tick_interval = *std::lower_bound(intervals.begin(), intervals.end(), range_x / 50);
  float label_interval = *std::lower_bound(intervals.begin(), intervals.end(), range_x / 10);

  // Draw x-axis ticks and labels outside the plot area
  for (float x = std::ceil(min_x / tick_interval) * tick_interval; x <= max_x; x += tick_interval) {
    float x_scale = (x - min_x) / range_x;
    float x_pos = p_min.x + x_scale * (p_max.x - p_min.x);
    ImGui::GetWindowDrawList()->AddLine(ImVec2(x_pos, p_max.y), ImVec2(x_pos, p_max.y + (int(x) % int(label_interval) == 0 ? 10 : 5)), IM_COL32(255, 255, 255, 128));

    if (int(x) % int(label_interval) == 0) {
      std::string label = std::to_string(static_cast<int>(x));
      ImVec2 label_size = ImGui::CalcTextSize(label.c_str());
      ImGui::GetWindowDrawList()->AddText(ImVec2(x_pos - label_size.x / 2, p_max.y + 15), IM_COL32(255, 255, 255, 192), label.c_str());
    }
  }

  // Draw y-axis ticks and labels
  for (float y : y_ticks) {
    if (y >= min_y && y <= max_y) {
      float y_scale = (y - min_y) / (max_y - min_y);
      float y_pos = p_max.y - y_scale * (p_max.y - p_min.y);
      ImGui::GetWindowDrawList()->AddLine(ImVec2(p_min.x - 10, y_pos), ImVec2(p_min.x, y_pos), IM_COL32(255, 255, 255, 128));

      std::ostringstream label_stream;
      label_stream << std::fixed << std::setprecision(3) << y;  
      std::string label = label_stream.str();
      ImVec2 label_size = ImGui::CalcTextSize(label.c_str());
      ImGui::GetWindowDrawList()->AddText(ImVec2(p_min.x - label_size.x - 15, y_pos - label_size.y / 2), IM_COL32(255, 255, 255, 192), label.c_str());
    }
  }
} 

}}  // namespace p11::gui
