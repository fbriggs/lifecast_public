// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
This is a minimal library for drawing 2D plots in DearImGui.
ImGui::PlotLines isn't enough, it doesn't have stuff like axis labels.
The standard answer here is to use an extension to ImGui called ImPlot.
I tried that and it didn't compile (probably a version mismatch). 
So instead, I'm just implementing a plotting library that has what we need.

See examples/hello_tinyplot for a demo use.
*/
#pragma once

#include <vector>

namespace p11 { namespace gui {

void plotLineGraph(
  const std::vector<float>& x_data,
  const std::vector<float>& y_data,
  float min_x, float max_x,
  float min_y, float max_y,
  const bool draw_dots = true,
  const int plot_width = 0,
  const int plot_height = 0);  

}}  // namespace p11::gui
