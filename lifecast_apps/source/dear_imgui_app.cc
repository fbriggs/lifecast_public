// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "dear_imgui_app.h"

namespace p11 {
std::function<void()> DearImGuiAppResizeCallback::callback;

void DearImGuiApp::init(
    const std::string& window_name, const int window_width, const int window_height)
{
  // Init GLFW.
  glfwSetErrorCallback(dearImGuiAppGlfwErrorCallback);
  XCHECK(glfwInit());

  // Set the OpenGL and glsl version.
  const char* glsl_version = "#version 150";
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_MAXIMIZED, GL_TRUE);

  // Create a GLFW window with graphics context.
  window = glfwCreateWindow(window_width, window_height, window_name.c_str(), NULL, NULL);
  XCHECK(window) << "Glfw failed to create a window";
  glfwMakeContextCurrent(window);
  glfwSetWindowSizeCallback(window, dearImGuiAppWindowSizeCallback);
  
  glfwSwapInterval(1);  // Enable vsync

  // Setup GL extension loader.
  XCHECK_EQ(gl3wInit(), 0) << "Failed to initialize gl3w.";

  // Setup Dear ImGui context.
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  // ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard
  // Controls

  // Setup Dear ImGui style
  ImGui::StyleColorsDark();

  // Setup Platform/Renderer bindings
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init(glsl_version);

  DearImGuiAppResizeCallback::callback = [this] { this->drawFrame(); };
}

void DearImGuiApp::cleanup()
{
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
}

void DearImGuiApp::guiDrawLoop()
{
  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    drawFrame();
  }
}

void DearImGuiApp::finishDrawingImguiAndGl()
{
  ImGui::Render();
  int display_w, display_h;
  glfwGetFramebufferSize(window, &display_w, &display_h);
  glViewport(0, 0, display_w, display_h);
  glClearColor(0.1, 0.1, 0.1, 1.0);
  glClear(GL_COLOR_BUFFER_BIT);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
  glfwSwapBuffers(window);
}

void DearImGuiApp::setCharcoalStyle()
{
  ImGuiStyle& style = ImGui::GetStyle();
  ImVec4* colors = style.Colors;
  colors[ImGuiCol_Text] = ImVec4(0.80f, 0.80f, 0.80f, 1.00f);
  colors[ImGuiCol_TextDisabled] = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
  colors[ImGuiCol_WindowBg] = ImVec4(0.15f, 0.15f, 0.15f, 1.00f);
  colors[ImGuiCol_ChildBg] = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
  colors[ImGuiCol_PopupBg] = ImVec4(0.19f, 0.19f, 0.19f, 0.92f);
  colors[ImGuiCol_Border] = ImVec4(0.19f, 0.19f, 0.19f, 0.29f);
  colors[ImGuiCol_BorderShadow] = ImVec4(0.00f, 0.00f, 0.00f, 0.24f);
  colors[ImGuiCol_FrameBg] = ImVec4(0.3f, 0.3f, 0.3f, 0.54f);
  colors[ImGuiCol_FrameBgHovered] = ImVec4(0.29f, 0.29f, 0.29f, 0.54f);
  colors[ImGuiCol_FrameBgActive] = ImVec4(0.45f, 0.42f, 0.46f, 1.00f);
  colors[ImGuiCol_TitleBg] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
  colors[ImGuiCol_TitleBgActive] = ImVec4(0.06f, 0.06f, 0.06f, 1.00f);
  colors[ImGuiCol_TitleBgCollapsed] = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
  colors[ImGuiCol_MenuBarBg] = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
  colors[ImGuiCol_ScrollbarBg] = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
  colors[ImGuiCol_ScrollbarGrab] = ImVec4(0.34f, 0.34f, 0.34f, 0.54f);
  colors[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.40f, 0.40f, 0.40f, 0.54f);
  colors[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.56f, 0.56f, 0.56f, 0.54f);
  colors[ImGuiCol_CheckMark] = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
  colors[ImGuiCol_SliderGrab] = ImVec4(0.34f, 0.34f, 0.34f, 0.54f);
  colors[ImGuiCol_SliderGrabActive] = ImVec4(0.56f, 0.56f, 0.56f, 0.54f);
  colors[ImGuiCol_Button] = ImVec4(0.25f, 0.25f, 0.25f, 0.54f);
  colors[ImGuiCol_ButtonHovered] = ImVec4(0.39f, 0.39f, 0.39f, 0.54f);
  colors[ImGuiCol_ButtonActive] = ImVec4(0.40f, 0.42f, 0.43f, 1.00f);
  colors[ImGuiCol_Header] = ImVec4(0.20f, 0.20f, 0.20f, 0.52f);
  colors[ImGuiCol_HeaderHovered] = ImVec4(0.30f, 0.30f, 0.30f, 0.36f);
  colors[ImGuiCol_HeaderActive] = ImVec4(0.25f, 0.25f, 0.25f, 0.33f);
  colors[ImGuiCol_Separator] = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
  colors[ImGuiCol_SeparatorHovered] = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);
  colors[ImGuiCol_SeparatorActive] = ImVec4(0.40f, 0.44f, 0.47f, 1.00f);
  colors[ImGuiCol_ResizeGrip] = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
  colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);
  colors[ImGuiCol_ResizeGripActive] = ImVec4(0.40f, 0.44f, 0.47f, 1.00f);
  colors[ImGuiCol_Tab] = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
  colors[ImGuiCol_TabHovered] = ImVec4(0.24f, 0.24f, 0.24f, 1.00f);
  colors[ImGuiCol_TabActive] = ImVec4(0.30f, 0.30f, 0.30f, 0.36f);
  colors[ImGuiCol_TabUnfocused] = ImVec4(0.20f, 0.20f, 0.20f, 0.52f);
  colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
  colors[ImGuiCol_PlotLines] = ImVec4(0.50f, 0.50f, 0.60f, 1.00f);
  colors[ImGuiCol_PlotLinesHovered] = ImVec4(0.50f, 0.50f, 0.60f, 1.00f);
  colors[ImGuiCol_PlotHistogram] = ImVec4(0.50f, 0.50f, 0.60f, 1.00f);
  colors[ImGuiCol_PlotHistogramHovered] = ImVec4(0.50f, 0.50f, 0.60f, 1.00f);
  colors[ImGuiCol_TextSelectedBg] = ImVec4(0.20f, 0.22f, 0.53f, 1.00f);
  colors[ImGuiCol_DragDropTarget] = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
  colors[ImGuiCol_NavHighlight] = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
  colors[ImGuiCol_NavWindowingHighlight] = ImVec4(0.50f, 0.50f, 0.60f, 0.70f);
  colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.50f, 0.50f, 0.60f, 0.20f);
  colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.50f, 0.50f, 0.60f, 0.35f);

  style.ChildRounding = 4.0f;
  style.FrameBorderSize = 0.0f;
  style.FrameRounding = 2.0f;
  style.GrabMinSize = 7.0f;
  style.PopupRounding = 2.0f;
  style.ScrollbarRounding = 12.0f;
  style.ScrollbarSize = 13.0f;
  style.TabBorderSize = 0.0f;
  style.TabRounding = 0.0f;
  style.WindowRounding = 4.0f;
}

void DearImGuiApp::setProStyle()
{
  ImGuiStyle& style = ImGui::GetStyle();
  ImVec4* colors = style.Colors;

  colors[ImGuiCol_Text] = ImVec4(0.80f, 0.80f, 0.80f, 1.00f);
  colors[ImGuiCol_Button] = ImVec4(0.25f, 0.25f, 0.25f, 0.54f);
  colors[ImGuiCol_ButtonHovered] = ImVec4(0.39f, 0.39f, 0.39f, 0.54f);
  colors[ImGuiCol_ButtonActive] = ImVec4(0.40f, 0.42f, 0.43f, 1.00f);

  colors[ImGuiCol_CheckMark] = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
  colors[ImGuiCol_SliderGrab] = ImVec4(0.64f, 0.64f, 0.64f, 0.54f);
  colors[ImGuiCol_SliderGrabActive] = ImVec4(0.76f, 0.76f, 0.76f, 0.54f);

  ImVec4 dark_grey(0.2, 0.2, 0.2, 1.0);
  colors[ImGuiCol_FrameBg] = dark_grey;
  colors[ImGuiCol_FrameBgHovered] = dark_grey;
  colors[ImGuiCol_FrameBgActive] = dark_grey;
  colors[ImGuiCol_TitleBg] = dark_grey;
  colors[ImGuiCol_TitleBgActive] = dark_grey;
  colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.1, 0.1, 0.1, 0.95);
  colors[ImGuiCol_Border] = ImVec4(0.2, 0.2, 0.2, 0.8);
  colors[ImGuiCol_PlotHistogram] = ImVec4(0.3, 0.7, 0.3, 0.5); // Progress bar color

  colors[ImGuiCol_Header] = ImVec4(0.20f, 0.20f, 0.20f, 0.52f);
  colors[ImGuiCol_HeaderHovered] = ImVec4(0.30f, 0.30f, 0.30f, 0.36f);
  colors[ImGuiCol_HeaderActive] = ImVec4(0.25f, 0.25f, 0.25f, 0.33f);

  colors[ImGuiCol_Tab] = ImVec4(0.20f, 0.20f, 0.20f, 0.8);
  colors[ImGuiCol_TabHovered] = ImVec4(0.30f, 0.30f, 0.30f, 0.8);
  colors[ImGuiCol_TabActive] = ImVec4(0.25f, 0.25f, 0.4f, 0.8);
  colors[ImGuiCol_TabUnfocused] = ImVec4(0.15f, 0.15f, 0.15f, 0.8);
  colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.25f, 0.25f, 0.25f, 0.8);
}

};  // namespace p11
