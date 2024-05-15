// MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include "ldi_common.h"

#include "opencv2/photo.hpp"
#include "logger.h"
#include "check.h"
#include "util_math.h"
#include "util_time.h"
#include "util_opencv.h"
#include "vignette.h"

namespace p11 { namespace ldi {

cv::Mat encodeDepthWithSplit12(const cv::Mat& src)
{
  static constexpr int kNumBits = 12;

  cv::Mat result = src.clone();
  cv::Mat lo_image(result.size(), CV_8U);
  cv::Mat hi_image(result.size(), CV_8U);

  for (int y = 0; y < result.rows; ++y) {
    for (int x = 0; x < result.cols; ++x) {
      const float v = math::clamp(result.at<float>(y, x), 0.0f, 1.0f);
      const int iv = v * ((1 << kNumBits) - 1);

      int high4 = iv >> 8;
      int low8 = iv & ((1 << 8) - 1);
      low8 = (iv & (1 << 8)) == 0 ? low8 : 255 - low8;  // fold (to improve compression)
      high4 = high4 * 16 + 8;                           // encode error correcting code

      lo_image.at<uint8_t>(y, x) = low8;
      hi_image.at<uint8_t>(y, x) = high4;
    }
  }

  // In addition to the split 12 encoding we have some extra space to store an 8bit encoding
  // as well.
  cv::Mat half8;
  result.convertTo(half8, CV_8U, 255);
  cv::Mat empty(result.size(), CV_8U, cv::Scalar(0));

  cv::Mat lohi;
  cv::hconcat(lo_image, hi_image, lohi);
  cv::Mat src_half_and_empty;
  cv::hconcat(half8, empty, src_half_and_empty);
  cv::Mat stacked;
  cv::vconcat(lohi, src_half_and_empty, stacked);
  return stacked;
}

cv::Mat make6DofGrid(
    const std::vector<cv::Mat>& layer_bgra,
    const std::vector<cv::Mat>& layer_invd,
    const std::string format,
    const std::vector<cv::Mat>& warp_ftheta_to_inflated,
    const bool dilate_invd)
{
  cv::Mat stacked6dof;
  for (int l = 0; l < layer_bgra.size(); ++l) {
    cv::Mat bgra_inflated =
        warp_ftheta_to_inflated.empty()
            ? layer_bgra[l]
            : opencv::warp(layer_bgra[l], warp_ftheta_to_inflated, cv::INTER_CUBIC);
    cv::Mat invd_inflated =
        warp_ftheta_to_inflated.empty()
            ? layer_invd[l]
            : opencv::warp(layer_invd[l], warp_ftheta_to_inflated, cv::INTER_AREA);

    std::vector<cv::Mat> bgra_components;
    cv::split(bgra_inflated, bgra_components);
    std::vector<cv::Mat> bgr_components = {
        bgra_components[0], bgra_components[1], bgra_components[2]};
    cv::Mat bgr_image, alpha_image;
    cv::merge(bgr_components, bgr_image);
    cv::cvtColor(bgra_components[3], alpha_image, cv::COLOR_GRAY2BGR);

    // Pre-dilate depth for better sillouettes in the real-time renderer.
    // TODO: this is destroying detail in the depthmap away from edges. could be improved.
    constexpr int kDilateSize = 6;
    if (dilate_invd && l > 0)
      cv::dilate(invd_inflated, invd_inflated, cv::Mat(), cv::Point(-1, -1), kDilateSize);

    if (format == "split12") {
      cv::Size half_size(bgr_image.cols / 2, bgr_image.rows / 2);
      if (invd_inflated.size() != half_size) {
        cv::resize(invd_inflated, invd_inflated, half_size, 0.0, 0.0, cv::INTER_AREA);
      }
      invd_inflated = encodeDepthWithSplit12(invd_inflated);
      if (bgr_image.type() == CV_32FC3) bgr_image.convertTo(bgr_image, CV_8UC3, 255.0);
      if (alpha_image.type() == CV_32FC3) alpha_image.convertTo(alpha_image, CV_8UC3, 255.0);
    }

    cv::cvtColor(invd_inflated, invd_inflated, cv::COLOR_GRAY2BGR);
    cv::Mat img_depth_alpha;

    cv::hconcat(bgr_image, invd_inflated, img_depth_alpha);
    cv::hconcat(img_depth_alpha, alpha_image, img_depth_alpha);
    if (l == 0)
      stacked6dof = img_depth_alpha;
    else
      cv::vconcat(img_depth_alpha, stacked6dof, stacked6dof);
  }
  if (format == "8bit") {
    stacked6dof.convertTo(stacked6dof, CV_8UC3, 255.0);
  }
  if (format == "16bit") {
    stacked6dof.convertTo(stacked6dof, CV_16UC3, 65535.0);
  }
  return stacked6dof;
}

void fuseLayers(
    const std::vector<cv::Mat>& layer_bgra,
    const std::vector<cv::Mat>& layer_invd,
    cv::Mat& bgra,
    cv::Mat& invd)
{
  bgra = layer_bgra[0].clone();
  invd = layer_invd[0].clone();
  XCHECK_EQ(bgra.type(), CV_32FC4);
  XCHECK_EQ(invd.type(), CV_32F);

  for (int l = 1; l < layer_bgra.size(); ++l) {
    // Merge layer l down onto fused
    for (int y = 0; y < bgra.rows; ++y) {
      for (int x = 0; x < bgra.cols; ++x) {
        const cv::Vec4f bottom_color = bgra.at<cv::Vec4f>(y, x);
        const cv::Vec4f top_color = layer_bgra[l].at<cv::Vec4f>(y, x);
        const float bottom_a = bottom_color[3];
        const float top_a = top_color[3];

        const float new_a = 1.0 - (1.0 - bottom_a) * (1.0 - top_a);

        const float top_frac = math::clamp(top_a / (new_a + 1e-6), 0.0, 1.0);

        cv::Vec4f fused_color = bottom_color * (1.0 - top_frac) + top_color * top_frac;
        fused_color[3] = new_a;

        bgra.at<cv::Vec4f>(y, x) = fused_color;

        const float bottom_invd = invd.at<float>(y, x);
        const float top_invd = layer_invd[l].at<float>(y, x);

        // NOTE: this is blending with top_a instead of top_frac. This
        // seems to work better but doesn't make sense / may be a hack.
        invd.at<float>(y, x) = bottom_invd * (1.0 - top_a) + top_invd * top_a;
      }
    }
  }
}

cv::Mat unpackLDI3_12BitDepth(
    const cv::Mat& ldi3_8bit, const cv::Rect rect_lo, const cv::Rect rect_hi)
{
  cv::Mat depth_lo = ldi3_8bit(rect_lo);
  cv::Mat depth_hi = ldi3_8bit(rect_hi);
  cv::cvtColor(depth_lo, depth_lo, cv::COLOR_BGR2GRAY);
  cv::cvtColor(depth_hi, depth_hi, cv::COLOR_BGR2GRAY);
  cv::Mat depth(depth_lo.size(), CV_32F, cv::Scalar(0));

  for (int y = 0; y < depth.rows; ++y) {
    for (int x = 0; x < depth.cols; ++x) {
      int lo = depth_lo.at<uint8_t>(y, x);
      int hi = depth_hi.at<uint8_t>(y, x);
      hi = hi / 16;                        // decode error correcting code
      lo = (hi & 1) == 0 ? lo : 255 - lo;  // unfold
      int i12 = (lo & 255) | ((hi & 15) << 8);
      float f12 = float(i12) / float((1 << 12) - 1);
      depth.at<float>(y, x) = f12;
    }
  }

  return depth;
}

void unpackLDI3(
    const cv::Mat& ldi3_8bit, std::vector<cv::Mat>& layer_bgra, std::vector<cv::Mat>& layer_invd)
{
  static constexpr int kNumLayers = 3;
  for (int layer = 0; layer < kNumLayers; ++layer) {
    int y_offset;
    switch (layer) {
      case 0:
        y_offset = ldi3_8bit.rows * 2 / 3;
        break;
      case 1:
        y_offset = ldi3_8bit.rows * 1 / 3;
        break;
      case 2:
        y_offset = 0;
        break;
    }
    cv::Mat bgr = ldi3_8bit(cv::Rect(0, y_offset, ldi3_8bit.cols / 3, ldi3_8bit.rows / 3));
    cv::Mat alpha = ldi3_8bit(
        cv::Rect(ldi3_8bit.cols * 2 / 3, y_offset, ldi3_8bit.cols / 3, ldi3_8bit.rows / 3));
    cv::cvtColor(alpha, alpha, cv::COLOR_BGR2GRAY);

    cv::Mat bgra;
    std::vector<cv::Mat> channels = {bgr, alpha};
    cv::merge(channels, bgra);
    layer_bgra.push_back(bgra);

    cv::Mat depth = unpackLDI3_12BitDepth(
        ldi3_8bit,
        cv::Rect(ldi3_8bit.cols / 3, y_offset, ldi3_8bit.cols / 6, ldi3_8bit.rows / 6),
        cv::Rect(
            ldi3_8bit.cols / 3 + ldi3_8bit.cols / 6,
            y_offset,
            ldi3_8bit.cols / 6,
            ldi3_8bit.rows / 6));
    layer_invd.push_back(depth);
  }
}

}}  // namespace p11::ldi
