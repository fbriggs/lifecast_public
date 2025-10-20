// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "imgui_cvmat.h"

#include "logger.h"
#include "util_opengl.h"
#include "third_party/dear_imgui/imgui.h"

namespace p11 { namespace gui {

void ImguiCvMat::setImage(const cv::Mat& image)
{
  cv_image = image.clone();
  size = image.size();
  needs_free = true;
}

void ImguiCvMat::reset()
{
  freeGlTexture();
  cv_image = cv::Mat();
  size = cv::Size(0, 0);
  name = "";
}

void ImguiCvMat::makeGlTexture()
{
  if (needs_free) freeGlTexture();
  if (size.width == 0 || size.height == 0) return;

  if (!has_texture) {
    texture_id = opengl::makeGlTextureFromCvMat(cv_image);
    cv_image.release();
    has_texture = true;
  }
}

void ImguiCvMat::freeGlTexture()
{
  if (has_texture) {
    glDeleteTextures(1, &texture_id);
    has_texture = false;
  }
  needs_free = false;
}

void ImguiCvMat::drawInImGui() const
{
  if (!name.empty()) {
    ImGui::Text("%s", name.c_str());
  }

  if (size.width == 0 || size.height == 0) return;

  // Scale image width down to fit in window.
  float draw_width = size.width;
  float draw_height = size.height;
  if (scale_to_fit) {
    const float container_width = ImGui::GetContentRegionAvailWidth();
    if (draw_width > container_width) {
      const float ratio = container_width / draw_width;
      draw_width *= ratio;
      draw_height *= ratio;
    }
  } else if (scale_to_factor) {
    draw_width *= scale_factor;
    draw_height *= scale_factor;
  } else if (center_and_expand) {
    ImVec2 container_size(
      ImGui::GetContentRegionMax().x,
      ImGui::GetContentRegionMax().y);

    // Calculate scale factors for width and height
    const float scale_w = container_size.x / size.width;
    const float scale_h = container_size.y / size.height;

    // Choose the larger scale factor that fits the entire image
    const float scale_factor = std::min(scale_w, scale_h);

    // Calculate new dimensions
    draw_width = size.width * scale_factor;
    draw_height = size.height * scale_factor;

    // Center the image if it's smaller than the container in any dimension
    const float offset_x = (draw_width < container_size.x) ? (container_size.x - draw_width) / 2 : 0.0f;
    float offset_y = (draw_height < container_size.y) ? (container_size.y - draw_height) / 2 : 0.0f;
    
    offset_y += 20; // HACK for menu bar, might not scale with text size. This is not correct, just less wrong.

    ImGui::SetCursorPos(ImVec2(offset_x, offset_y));
  }

  ImGui::Image(
    ImTextureID(texture_id),
    ImVec2(draw_width, draw_height),
    ImVec2(0, 0),
    ImVec2(1, 1),
    ImColor(255, 255, 255, 255),
    ImColor(0, 0, 0, 0));  
}

}}  // namespace p11::gui
