// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#if defined(_WIN32) && !defined(GLOG_NO_ABBREVIATED_SEVERITIES)  // Prevent an issue where glog redefines the ERROR macro (only relevant on Windows)
#define GLOG_NO_ABBREVIATED_SEVERITIES
#endif

#include <string>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

namespace p11 { namespace gui {
// We construct cv::Mat images in one thread, then need to display them in the OpenGL GUI thread.
// The GL textures can only be constructed from the OpenGL thread, so we have to do this in a lazy
// way.
struct ImguiCvMat {
  bool has_texture;
  uint32_t texture_id;
  cv::Size size;
  std::string name;
  cv::Mat cv_image;
  bool scale_to_fit, center_and_expand, scale_to_factor;
  float scale_factor;
  bool needs_free;

  ImguiCvMat() : has_texture(false), size(cv::Size(0, 0)), scale_to_fit(true), center_and_expand(false), needs_free(false) {}

  ImguiCvMat(const std::string& name, const cv::Mat& image)
      : has_texture(false),
        size(image.size()),
        name(name),
        cv_image(image.clone()),
        scale_to_fit(true),
        center_and_expand(false),
        scale_to_factor(false), // If true, use scale_factor
        scale_factor(1.0),
        needs_free(false)
  {}

  void setImage(const cv::Mat& image);

  void reset();

  void makeGlTexture();

  void freeGlTexture();

  void drawInImGui() const;

  bool empty() const { return size.width == 0 || size.height == 0; }
};

inline void makeTexturesForImguiCvMats(std::vector<ImguiCvMat>& images)
{
  for (auto& image : images) {
    image.makeGlTexture();
  }
}

// Deleting the GL textures in ~ImguiCvMat() causes problems when std::vector resizes due to
// push_backs, so instead we will delete from the owner of the image array.
inline void freeTexturesForImguiCvMats(std::vector<ImguiCvMat>& images)
{
  for (auto& image : images) {
    image.freeGlTexture();
  }
}

inline void clearImguiCvMats(std::vector<ImguiCvMat>& images)
{
  freeTexturesForImguiCvMats(images);
  images.clear();
}

}}  // namespace p11::gui
