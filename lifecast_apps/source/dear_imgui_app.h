// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include "opengl_xplatform_includes.h"
#include "logger.h"

#include <functional>
#include <mutex>
#include <string>

namespace p11 {
inline void dearImGuiAppGlfwErrorCallback(int error, const char* description)
{
  XPLINFO << "GLFW Error: " << error << " " << description;
}

struct DearImGuiAppResizeCallback {
  static std::function<void()> callback;  // set this to the draw method of a derived class to
                                          // handle drawing on resize correctly
};

inline void dearImGuiAppWindowSizeCallback(GLFWwindow* window, int width, int height)
{
  if (DearImGuiAppResizeCallback::callback) {
    DearImGuiAppResizeCallback::callback();
  }
}

struct DearImGuiApp {
  void init(const std::string& window_name, const int window_width, const int window_height);

  void cleanup();

  void guiDrawLoop();

  void finishDrawingImguiAndGl();

  void setCharcoalStyle();
  void setProStyle();

  // Separated from main loop so we can draw while resizing the window
  virtual void drawFrame() = 0;

  GLFWwindow* window;

  // If you need to share the opengl context for window with another thread (not the GUI main loop
  // thread), use this mutex.
  std::mutex gl_context_mutex;
};

};  // namespace p11
