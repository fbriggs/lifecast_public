// MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include "vignette.h"

#include "logger.h"
#include "fisheye_camera.h"
#include "util_math.h"
#include "util_opencv.h"

namespace p11 { namespace projection {

cv::Mat makeVignetteMap(
    const calibration::FisheyeCamerad& cam,
    const cv::Size& image_size,
    float r0,
    float r1,
    float e0,
    float e1)
{
  const float fade_start = cam.radius_at_90 * r0;
  const float fade_end = cam.radius_at_90 * r1;
  const int edge_fade_end = cam.radius_at_90 * e0;
  const int edge_fade_start = cam.radius_at_90 * e1;

  cv::Mat alpha_mask(image_size, CV_32F);

  for (int y = 0; y < alpha_mask.rows; ++y) {
    for (int x = 0; x < alpha_mask.cols; ++x) {
      const Eigen::Vector2d pixel(x, y);
      const float r = (pixel - cam.optical_center).norm();

      float a = 1.0;
      if (r > fade_start) {
        a = 1.0f - math::clamp<float>((r - fade_start) / (fade_end - fade_start), 0.0f, 1.0f);
      }

      int closest_edge = std::max(alpha_mask.cols, alpha_mask.cols);
      closest_edge = std::min(closest_edge, x);
      closest_edge = std::min(closest_edge, y);
      closest_edge = std::min(closest_edge, std::abs(x - alpha_mask.cols + 1));
      closest_edge = std::min(closest_edge, std::abs(y - alpha_mask.rows + 1));
      if (closest_edge < edge_fade_start) {
        a *= std::min(
            1.0f, float(closest_edge - edge_fade_end) / float(edge_fade_start - edge_fade_end));
      }

      alpha_mask.at<float>(y, x) = a;
    }
  }

  GaussianBlur(alpha_mask, alpha_mask, cv::Size(21, 21), 11.0);
  return alpha_mask;
}

}}  // namespace p11::projection
