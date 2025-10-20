// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
bazel run //examples:hello_tinyplot
*/

#include "source/logger.h"
#include "source/dear_imgui_app.h"
#include "source/imgui_tinyplot.h"

namespace p11 {

struct TinyPlotApp : public DearImGuiApp {
  bool first_frame = true;
  void drawFrame() {
    // Scale UI for high DPI displays
    float xscale, yscale;
    glfwGetWindowContentScale(window, &xscale, &yscale);
    ImGui::GetIO().FontGlobalScale = xscale;

    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Set the initial window size and position
    if (first_frame) {
      first_frame = false;
      ImGui::SetNextWindowPos(ImVec2(50, 50));
      ImGui::SetNextWindowSize(ImVec2(640, 480));
    }
    ImGui::Begin("Plot Window");
    
    std::vector<float> plot_x = {0.0, 1.0, 2.0, 7.0};
    std::vector<float> plot_y = {0.0, 1.0, 4.0, 7.0};

    gui::plotLineGraph(plot_x, plot_y, 0.0, 10.0, 0.0, 10.0);
    ImGui::End();
 
    finishDrawingImguiAndGl();
  }
};

}  // namespace p11

int main(int argc, char** argv)
{
  p11::TinyPlotApp app;
  app.init("Hello TinyPlot", 1280, 720);
  app.guiDrawLoop();
  app.cleanup();

  return EXIT_SUCCESS;
}
