// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "view_interpolation.h"
#include "source/logger.h"
#include "source/rof.h"
#include "source/projection.h"

namespace p11 { namespace optical_flow {

void splatIntoWarp(
  cv::Mat warp,
  cv::Mat weight,
  const cv::Vec2f& splat_pos,
  const cv::Vec2f& splat_value,
  const int splat_radius,
  const float sigma
) {
  int ix = splat_pos[0];
  int iy = splat_pos[1];
  const float sigma2 = sigma * sigma;
  for (int i = -splat_radius; i <= splat_radius; ++i) {
    for (int j = -splat_radius; j <= splat_radius; ++j) {
      int x = ix + i;
      int y = iy + j;
      if (x < 0 || y < 0 || x >= warp.cols || y >= warp.rows) continue;

      float dx = x - splat_pos[0];
      float dy = y - splat_pos[1];

      cv::Vec2f prev = warp.at<cv::Vec2f>(y, x);
      float mag0 = prev[0] * prev[0] + prev[1] * prev[1];
      float mag1 = splat_value[0] * splat_value[0] + splat_value[1] * splat_value[1];
      const float steepness = 0.33;
      float prefer_larger = std::tanh(steepness * (mag1 - mag0)) * 0.5 + 0.5; // prefer large values of displacement
      float w = prefer_larger * std::exp(-0.5 * (dx * dx + dy * dy) / sigma2);
      warp.at<cv::Vec2f>(y, x) = splat_value * w + warp.at<cv::Vec2f>(y, x) * (1.0 - w);

      weight.at<float>(y, x) = math::clamp<float>(weight.at<float>(y, x) + w, 0.0f, 1.0f);
    }
  }
}

std::vector<cv::Mat> generateBetweenFrameWithFlow(
  cv::Mat image0,
  cv::Mat image1,
  torch::jit::script::Module& raft,
  std::vector<float> interps,
  std::string debug_dir
) {
  XCHECK_EQ(image0.type(), CV_32FC3);
  XCHECK_EQ(image1.type(), CV_32FC3);

  // Convert full-res images to 8-bit and downscale to kSizeRAFT.
  cv::Mat image0_8u, image1_8u;
  image0.convertTo(image0_8u, CV_8UC3, 255.0);
  image1.convertTo(image1_8u, CV_8UC3, 255.0);
  constexpr int kSizeRAFT = 1024;
  cv::resize(image0_8u, image0_8u, cv::Size(kSizeRAFT, kSizeRAFT), 0, 0, cv::INTER_AREA);
  cv::resize(image1_8u, image1_8u, cv::Size(kSizeRAFT, kSizeRAFT), 0, 0, cv::INTER_AREA);

  // Compute optical flow on the low-res images.
  cv::Mat flow01_x, flow01_y, flow10_x, flow10_y;
  computeOpticalFlowRAFT(raft, image0_8u, image1_8u, flow01_x, flow01_y);
  computeOpticalFlowRAFT(raft, image1_8u, image0_8u, flow10_x, flow10_y);

  if (!debug_dir.empty()) {
    cv::imwrite(debug_dir + "/flow01_x.jpg", flow01_x * -0.3f);
    cv::imwrite(debug_dir + "/flow10_x.jpg", flow10_x * +0.3f);
    cv::imwrite(debug_dir + "/image0.jpg", image0 * 255.0);
    cv::imwrite(debug_dir + "/image1.jpg", image1 * 255.0);
  }

  // Full resolution dimensions.
  const int full_height = image0.rows;
  const int full_width = image0.cols;
  // Low resolution dimensions (from RAFT).
  const int low_height = flow01_x.rows;
  const int low_width = flow01_x.cols;

  constexpr int splat_radius = 3; // TODO: this is much faster with 2 but has some holes
  constexpr float splat_sigma = 0.75f;

  std::vector<cv::Mat> output_frames;
  for (float t : interps) {
    // Create low-res warp fields and weight maps
    cv::Mat warp0_low(cv::Size(low_width, low_height), CV_32FC2, cv::Scalar(0, 0));
    cv::Mat warp1_low(cv::Size(low_width, low_height), CV_32FC2, cv::Scalar(0, 0));
    cv::Mat weight0_low(cv::Size(low_width, low_height), CV_32FC1, cv::Scalar(0));
    cv::Mat weight1_low(cv::Size(low_width, low_height), CV_32FC1, cv::Scalar(0));

    // Scale the low-res flows by time t
    cv::Mat t_flow01_x = flow01_x * t;
    cv::Mat t_flow01_y = flow01_y * t;
    cv::Mat t_flow10_x = flow10_x * (1.0f - t);
    cv::Mat t_flow10_y = flow10_y * (1.0f - t);

    // Loop over the low-res grid and do forward splatting
    for (int y = 0; y < low_height; ++y) {
      for (int x = 0; x < low_width; ++x) {
        float f01_x = t_flow01_x.at<float>(y, x);
        float f01_y = t_flow01_y.at<float>(y, x);
        cv::Vec2f splat_flow01(-f01_x, -f01_y);
        cv::Vec2f splat_pos01(x + f01_x, y + f01_y);
        splatIntoWarp(warp0_low, weight0_low, splat_pos01, splat_flow01, splat_radius, splat_sigma);

        float f10_x = t_flow10_x.at<float>(y, x);
        float f10_y = t_flow10_y.at<float>(y, x);
        cv::Vec2f splat_flow10(-f10_x, -f10_y);
        cv::Vec2f splat_pos10(x + f10_x, y + f10_y);
        splatIntoWarp(warp1_low, weight1_low, splat_pos10, splat_flow10, splat_radius, splat_sigma);
      }
    }

    // Clean up the weight maps
    cv::erode(weight0_low, weight0_low, cv::Mat(), cv::Point(-1, -1), splat_radius);
    cv::erode(weight1_low, weight1_low, cv::Mat(), cv::Point(-1, -1), splat_radius);
    cv::GaussianBlur(weight0_low, weight0_low, cv::Size(3, 3), 0.5, 0.5);
    cv::GaussianBlur(weight1_low, weight1_low, cv::Size(3, 3), 0.5, 0.5);

    // Convert the warp fields from displacements to absolute (low-res) coordinates
    for (int y = 0; y < low_height; ++y) {
      for (int x = 0; x < low_width; ++x) {
        warp0_low.at<cv::Vec2f>(y, x) += cv::Vec2f(x, y);
        warp1_low.at<cv::Vec2f>(y, x) += cv::Vec2f(x, y);
      }
    }

    // Upscale the low-res warp fields and weight maps to full resolution
    cv::Mat warp0, warp1, weight0, weight1;
    cv::resize(warp0_low, warp0, cv::Size(full_width, full_height), 0, 0, cv::INTER_LINEAR);
    cv::resize(warp1_low, warp1, cv::Size(full_width, full_height), 0, 0, cv::INTER_LINEAR);
    cv::resize(weight0_low, weight0, cv::Size(full_width, full_height), 0, 0, cv::INTER_LINEAR);
    cv::resize(weight1_low, weight1, cv::Size(full_width, full_height), 0, 0, cv::INTER_LINEAR);

    // Because warp0 and warp1 are in low-res pixel units, rescale them into full-res coordinates
    float scale_x = float(full_width) / float(low_width);
    float scale_y = float(full_height) / float(low_height);
    std::vector<cv::Mat> warp0_uv, warp1_uv;
    cv::split(warp0, warp0_uv);
    cv::split(warp1, warp1_uv);
    warp0_uv[0] *= scale_x;
    warp0_uv[1] *= scale_y;
    warp1_uv[0] *= scale_x;
    warp1_uv[1] *= scale_y;
    cv::Mat warped0 = projection::warp(image0, warp0_uv, cv::INTER_CUBIC);
    cv::Mat warped1 = projection::warp(image1, warp1_uv, cv::INTER_CUBIC);

    if (!debug_dir.empty()) {
      cv::imwrite(debug_dir + "/warped0_" + std::to_string(t) + ".jpg", warped0 * 255.0);
      cv::imwrite(debug_dir + "/warped1_" + std::to_string(t) + ".jpg", warped1 * 255.0);
    }

    // Blend the two warped images using the weight maps and time parameter.
    cv::Mat blended_frame(image0.size(), CV_32FC3);
    for (int y = 0; y < full_height; ++y) {
      for (int x = 0; x < full_width; ++x) {
        cv::Vec3f color0 = warped0.at<cv::Vec3f>(y, x);
        cv::Vec3f color1 = warped1.at<cv::Vec3f>(y, x);
        float w0 = 1e-6f + weight0.at<float>(y, x) * (1.0f - t);
        float w1 = 1e-6f + weight1.at<float>(y, x) * t;
        float sum = w0 + w1;
        blended_frame.at<cv::Vec3f>(y, x) = color0 * (w0 / sum) + color1 * (w1 / sum);
      }
    }
    output_frames.push_back(blended_frame);
  }
  return output_frames;
}

}} // namespace p11::optical_flow