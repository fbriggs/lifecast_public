// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ldi_common.h"
#include "opencv2/photo.hpp"
#include "logger.h"
#include "check.h"
#include "util_math.h"
#include "util_time.h"
#include "util_opencv.h"
#include "ldi_segmentation.h"
#include "ldi_segmentation2.h"
#include "inpaint_ceres.h"
#include "projection.h"
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
            : projection::warp(layer_bgra[l], warp_ftheta_to_inflated, cv::INTER_CUBIC);
    cv::Mat invd_inflated =
        warp_ftheta_to_inflated.empty()
            ? layer_invd[l]
            : projection::warp(layer_invd[l], warp_ftheta_to_inflated, cv::INTER_AREA);

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

void reduceSegAlphaTransitionArtifacts(
    cv::Mat& segmentation,
    cv::Mat& l1_alpha,
    cv::Mat& l2_alpha,
    cv::Mat& l0_alpha_blend,
    cv::Mat& l1_alpha_blend)
{
  // alpha_transition_map detects where a smooth transition in alpha occurs (not an object edge)
  cv::Mat l1_alpha_transition(l1_alpha.size(), CV_32F);
  cv::Mat l2_alpha_transition(l1_alpha.size(), CV_32F);
  cv::Mat segmentation_purity(l1_alpha.size(), CV_32F);
  for (int y = 0; y < l1_alpha.rows; ++y) {
    for (int x = 0; x < l1_alpha.cols; ++x) {
      static constexpr float kS = 10.0;
      l1_alpha_transition.at<float>(y, x) =
          std::tanh(kS * (0.5 - std::fabs(l1_alpha.at<float>(y, x) - 0.5)));
      l2_alpha_transition.at<float>(y, x) =
          std::tanh(kS * (0.5 - std::fabs(l2_alpha.at<float>(y, x) - 0.5)));

      const float s0 = segmentation.at<cv::Vec3f>(y, x)[0];
      const float s1 = segmentation.at<cv::Vec3f>(y, x)[1];
      const float s2 = segmentation.at<cv::Vec3f>(y, x)[2];
      const float s_max = std::max(std::max(s0, s1), s2);
      const float s_min = std::min(std::min(s0, s1), s2);
      static constexpr float kSegPurityBias = 0.7;
      static constexpr float kSegPuritySteepness = 7.0;
      const float seg_purity =
          std::max(0.0f, std::tanh(kSegPuritySteepness * (s_max - s_min - kSegPurityBias)));
      segmentation_purity.at<float>(y, x) = seg_purity;
    }
  }

  static constexpr int kTransitionErode = 1;
  static constexpr int kTransitionDilate = 17;
  static constexpr int kTransitionBlur = 3;
  cv::erode(
      l1_alpha_transition, l1_alpha_transition, cv::Mat(), cv::Point(-1, -1), kTransitionErode);
  cv::dilate(
      l1_alpha_transition, l1_alpha_transition, cv::Mat(), cv::Point(-1, -1), kTransitionDilate);
  cv::GaussianBlur(
      l1_alpha_transition,
      l1_alpha_transition,
      cv::Size(kTransitionBlur, kTransitionBlur),
      kTransitionBlur / 2,
      kTransitionBlur / 2);
  cv::erode(
      l2_alpha_transition, l2_alpha_transition, cv::Mat(), cv::Point(-1, -1), kTransitionErode);
  cv::dilate(
      l2_alpha_transition, l2_alpha_transition, cv::Mat(), cv::Point(-1, -1), kTransitionDilate);
  cv::GaussianBlur(
      l2_alpha_transition,
      l2_alpha_transition,
      cv::Size(kTransitionBlur, kTransitionBlur),
      kTransitionBlur / 2,
      kTransitionBlur / 2);

  static constexpr int kPurityDilate = 3;
  cv::dilate(segmentation_purity, segmentation_purity, cv::Mat(), cv::Point(-1, -1), kPurityDilate);

  for (int y = 0; y < l1_alpha.rows; ++y) {
    for (int x = 0; x < l1_alpha.cols; ++x) {
      l1_alpha_transition.at<float>(y, x) = std::max(
          0.0f, l1_alpha_transition.at<float>(y, x) - 1.1f * segmentation_purity.at<float>(y, x));
      l2_alpha_transition.at<float>(y, x) = std::max(
          0.0f, l2_alpha_transition.at<float>(y, x) - 1.1f * segmentation_purity.at<float>(y, x));
    }
  }
  // cv::imwrite(debug_dir + "/segmentation_purity.png", segmentation_purity * 255.0);
  // cv::imwrite(debug_dir + "/l1_alpha_transition.png", l1_alpha_transition * 255.0);
  // cv::imwrite(debug_dir + "/l2_alpha_transition.png", l2_alpha_transition * 255.0);
  // cv::imwrite(debug_dir + "/topb_nosup.png", l1_alpha_blend * 255.0);
  // cv::imwrite(debug_dir + "/midb_nosup.png", l0_alpha_blend * 255.0);

  // To fix gap artifacs in alpha transitions, suppress l0_alpha_blend where alpha_transition_map
  // is high.
  static constexpr int kTransitionSupressionStrength = 3.0;
  for (int y = 0; y < l0_alpha_blend.rows; ++y) {
    for (int x = 0; x < l0_alpha_blend.cols; ++x) {
      l1_alpha_blend.at<float>(y, x) = std::max(
          0.0f,
          l1_alpha_blend.at<float>(y, x) -
              kTransitionSupressionStrength * l2_alpha_transition.at<float>(y, x));
      l0_alpha_blend.at<float>(y, x) = std::max(
          0.0f,
          l0_alpha_blend.at<float>(y, x) -
              kTransitionSupressionStrength * l1_alpha_transition.at<float>(y, x));
    }
  }
}

void assembleLayersAndChannels(
    const cv::Mat color_vignette,
    const cv::Mat depth_vignette,
    const cv::Mat& R_ftheta,
    const cv::Mat& R_inv_depth_ftheta,
    const cv::Mat& l0_alpha_blend,
    const cv::Mat& l1_alpha_blend,
    const cv::Mat& l1_alpha,
    const cv::Mat& l2_alpha,
    cv::Mat& l0_depth,
    cv::Mat& l1_depth,
    cv::Mat& inpainted_bottom,
    cv::Mat& inpainted_mid,
    std::vector<cv::Mat>& layer_bgra,
    std::vector<cv::Mat>& layer_invd)
{
  XCHECK_EQ(R_ftheta.type(), CV_8UC3);
  XCHECK_EQ(R_inv_depth_ftheta.type(), CV_32F);
  XCHECK_EQ(l0_alpha_blend.type(), CV_32F);
  XCHECK_EQ(l1_alpha_blend.type(), CV_32F);
  XCHECK_EQ(l1_depth.type(), CV_32F);
  XCHECK_EQ(l0_depth.type(), CV_32F);
  XCHECK_EQ(inpainted_mid.type(), CV_8UC3);
  XCHECK_EQ(inpainted_bottom.type(), CV_8UC3);
  const int num_layers = 3;
  static constexpr int kBottom = 0;
  static constexpr int kMid = 1;
  static constexpr int kTop = 2;

  // Resize all of the inpainting results up (they might be small)
  cv::resize(l1_depth, l1_depth, R_inv_depth_ftheta.size(), 0, 0, cv::INTER_LINEAR);
  cv::resize(l0_depth, l0_depth, R_inv_depth_ftheta.size(), 0, 0, cv::INTER_LINEAR);
  cv::resize(inpainted_mid, inpainted_mid, R_ftheta.size(), 0, 0, cv::INTER_LINEAR);
  cv::resize(inpainted_bottom, inpainted_bottom, R_ftheta.size(), 0, 0, cv::INTER_LINEAR);

  // Blend the upscaled inpainted image with the original (restore pixels that weren't inpainted to
  // full res, and create a fade between real and inpainted pixels). Also blend pre-inpainting
  // depth.
  for (int y = 0; y < R_ftheta.rows; ++y) {
    for (int x = 0; x < R_ftheta.cols; ++x) {
      const float a_bot = l0_alpha_blend.at<float>(y, x);
      const float a_mid = l1_alpha_blend.at<float>(y, x);
      inpainted_mid.at<cv::Vec3b>(y, x) = cv::Vec3f(inpainted_mid.at<cv::Vec3b>(y, x)) * a_mid +
                                          cv::Vec3f(R_ftheta.at<cv::Vec3b>(y, x)) * (1.0 - a_mid);
      inpainted_bottom.at<cv::Vec3b>(y, x) =
          cv::Vec3f(inpainted_bottom.at<cv::Vec3b>(y, x)) * a_bot +
          cv::Vec3f(R_ftheta.at<cv::Vec3b>(y, x)) * (1.0 - a_bot);
    }
  }

  cv::Mat l0_alpha_blend_half = opencv::halfSize(l0_alpha_blend);
  cv::Mat l1_alpha_blend_half = opencv::halfSize(l1_alpha_blend);
  for (int y = 0; y < R_inv_depth_ftheta.rows; ++y) {
    for (int x = 0; x < R_inv_depth_ftheta.cols; ++x) {
      const float a_mid = l1_alpha_blend_half.at<float>(y, x);
      const float a_bot = l0_alpha_blend_half.at<float>(y, x);
      l1_depth.at<float>(y, x) =
          l1_depth.at<float>(y, x) * a_mid + R_inv_depth_ftheta.at<float>(y, x) * (1.0 - a_mid);
      l0_depth.at<float>(y, x) =
          l0_depth.at<float>(y, x) * a_bot + R_inv_depth_ftheta.at<float>(y, x) * (1.0 - a_bot);
    }
  }

  // Enforce layer depth ordering.
  constexpr float kEps = 1.0f / 1024.0f;
  for (int y = 0; y < R_inv_depth_ftheta.rows; ++y) {
    for (int x = 0; x < R_inv_depth_ftheta.cols; ++x) {
      l1_depth.at<float>(y, x) = std::max(
          0.0f, std::min(l1_depth.at<float>(y, x), R_inv_depth_ftheta.at<float>(y, x) - kEps));
      l0_depth.at<float>(y, x) =
          std::max(0.0f, std::min(l0_depth.at<float>(y, x), l1_depth.at<float>(y, x) - kEps));
    }
  }

  // From here on we will need a full-size depth map
  cv::Mat R_inv_depth_ftheta_fullsize;
  cv::resize(
      R_inv_depth_ftheta, R_inv_depth_ftheta_fullsize, R_ftheta.size(), 0.0, 0.0, cv::INTER_LINEAR);
  cv::resize(l1_depth, l1_depth, R_ftheta.size(), 0.0, 0.0, cv::INTER_LINEAR);
  cv::resize(l0_depth, l0_depth, R_ftheta.size(), 0.0, 0.0, cv::INTER_LINEAR);

  // Copy results into the corresponding channels of the LDI
  layer_bgra = std::vector<cv::Mat>(num_layers);
  layer_invd = std::vector<cv::Mat>(num_layers);
  layer_invd[kTop] = R_inv_depth_ftheta_fullsize;
  layer_invd[kMid] = l1_depth;
  layer_invd[kBottom] = l0_depth;
  for (int l = 0; l < num_layers; ++l) {
    layer_bgra[l] = cv::Mat(R_ftheta.size(), CV_32FC4, cv::Scalar(0, 0, 0, 0));
  }
  for (int y = 0; y < R_ftheta.rows; ++y) {
    for (int x = 0; x < R_ftheta.cols; ++x) {
      auto top_bgr = cv::Vec3f(R_ftheta.at<cv::Vec3b>(y, x)) / 255.0f;
      auto inpainted_mid_bgr = cv::Vec3f(inpainted_mid.at<cv::Vec3b>(y, x)) / 255.0f;
      auto inpainted_bottom_bgr = cv::Vec3f(inpainted_bottom.at<cv::Vec3b>(y, x)) / 255.0f;
      layer_bgra[kTop].at<cv::Vec4f>(y, x) =
          cv::Vec4f(top_bgr[0], top_bgr[1], top_bgr[2], l2_alpha.at<float>(y, x));
      layer_bgra[kBottom].at<cv::Vec4f>(y, x) =
          cv::Vec4f(inpainted_bottom_bgr[0], inpainted_bottom_bgr[1], inpainted_bottom_bgr[2], 1.0);
      layer_bgra[kMid].at<cv::Vec4f>(y, x) = cv::Vec4f(
          inpainted_mid_bgr[0],
          inpainted_mid_bgr[1],
          inpainted_mid_bgr[2],
          l1_alpha.at<float>(y, x));
    }
  }

  // Apply vignettes
  for (int l = 0; l < num_layers; ++l) {
    layer_bgra[l] = projection::applyVignette<cv::Vec4f>(layer_bgra[l], color_vignette);
    layer_invd[l] = projection::applyVignette<float>(layer_invd[l], depth_vignette);
  }
  // Remove vignette from bottom layer alpha channel
  for (int y = 0; y < R_ftheta.rows; ++y) {
    for (int x = 0; x < R_ftheta.cols; ++x) {
      layer_bgra[kBottom].at<cv::Vec4f>(y, x)[3] = 1.0;
    }
  }
}

void makeLdiHeuristic(
    const std::string& curr_working_dir,
    const std::string& debug_dir,
    const std::string& inpaint_method,
    const std::string& seg_method,
    const std::string& sd_ver,
    const int num_layers,
    const calibration::FisheyeCamerad cam_R,
    const cv::Mat& R_ftheta,
    const cv::Mat& R_inv_depth_ftheta,
    const cv::Mat color_vignette,
    const cv::Mat depth_vignette,
    std::vector<cv::Mat>& layer_bgra,
    std::vector<cv::Mat>& layer_invd,
    const bool write_inpainting_stabilization_files,
    const bool assemble_ldi,
    const int inpaint_dilate_radius,
    const bool run_seg_only,
    const bool write_seg_image,
    cv::Mat cached_seg,
    const std::string& frame_num)
{
  //// Foreground-background segmentation optimization method based on high and low edges ////
  cv::Mat segmentation;
  if (!cached_seg.empty()) {
    segmentation = cached_seg;
  } else {
    if (seg_method == "heuristic") {
      segmentation = segment3layerWithHeuristic(debug_dir, R_inv_depth_ftheta);
    } else if (seg_method == "neural") {
      const auto& [edge_hi, edge_lo] = getEdgesHiAndLow(cam_R, R_inv_depth_ftheta);
      // cv::imwrite(debug_dir + "/edge_hi.png", edge_hi * 255.0f);
      // cv::imwrite(debug_dir + "/edge_lo.png", edge_lo * 255.0f);
      calibration::FisheyeCamerad cam_R_half(cam_R);
      cam_R_half.optical_center /= 2.0;
      cam_R_half.radius_at_90 /= 2.0;
      cam_R_half.width /= 2;
      cam_R_half.height /= 2;
      XCHECK_EQ(R_ftheta.cols, cam_R.width);
      XCHECK_EQ(R_ftheta.rows, cam_R.height);
      XCHECK_EQ(R_inv_depth_ftheta.cols, cam_R_half.width);
      XCHECK_EQ(R_inv_depth_ftheta.rows, cam_R_half.height);

      segmentation = segmentFgBgWithMultiresolutionHashmap(
          edge_hi, edge_lo, cam_R, cam_R_half, R_inv_depth_ftheta);
    }
  }

  if (write_seg_image) {
    cv::imwrite(debug_dir + "/seg_" + frame_num + ".png", segmentation * 255.0);
  }

  if (run_seg_only) return;

  // Do the first part of computing the alpha channels for the middle and top layers
  // here. Well use an result for something else, then resume work on this below.
  cv::Mat l2_alpha(R_ftheta.size(), CV_32F, cv::Scalar(0));
  cv::Mat l1_alpha(R_ftheta.size(), CV_32F, cv::Scalar(0));
  for (int y = 0; y < R_ftheta.rows; ++y) {
    for (int x = 0; x < R_ftheta.cols; ++x) {
      float s_top = segmentation.at<cv::Vec3f>(y, x)[0];
      float s_mid = segmentation.at<cv::Vec3f>(y, x)[1];
      float s_bot = segmentation.at<cv::Vec3f>(y, x)[2];

      static constexpr float kSteepness = 10.0;
      float b_top = s_mid + s_bot;
      s_top = (std::tanh(kSteepness * (s_top - b_top)) + 1.0) * 0.5;

      float a_mid = 1.0 - s_bot;
      float b_mid = 1.0 - (s_top + s_mid);
      s_mid = (std::tanh(kSteepness * (a_mid - b_mid)) + 1.0) * 0.5;

      l2_alpha.at<float>(y, x) = s_top;
      l1_alpha.at<float>(y, x) = s_mid;
    }
  }

  // Remove speckles in the alpha channel
  cv::medianBlur(l2_alpha, l2_alpha, 3);
  cv::medianBlur(l1_alpha, l1_alpha, 3);
  // cv::imwrite(debug_dir + "/l1_alpha.png", l1_alpha * 255.0);

  // Dilate the masks in preparation for inpainting
  static constexpr float kSegmentationThreshold = 0.999;

  cv::Mat l2_alpha_thresholded, l1_alpha_thresholded;
  cv::threshold(l2_alpha, l2_alpha_thresholded, kSegmentationThreshold, 1.0, cv::THRESH_BINARY);
  cv::threshold(l1_alpha, l1_alpha_thresholded, kSegmentationThreshold, 1.0, cv::THRESH_BINARY);

  cv::Mat l1_inpaint_mask, l0_inpaint_mask;
  cv::dilate(l2_alpha_thresholded, l1_inpaint_mask, cv::Mat(), cv::Point(-1, -1), inpaint_dilate_radius);
  cv::dilate(l1_alpha_thresholded, l0_inpaint_mask, cv::Mat(), cv::Point(-1, -1), inpaint_dilate_radius);

  cv::imwrite(debug_dir + "/l0_inpaint_mask_" + frame_num + ".png", l0_inpaint_mask * 255.0);
  cv::imwrite(debug_dir + "/l1_inpaint_mask_" + frame_num + ".png", l1_inpaint_mask * 255.0);

  // We dilate the mask to get better context for inpainting, but we don't actually need to
  // replace all of the pixels in the dilated mask. We want to smoothly blend from real
  // to inpainted and get as close to the outline of the occluding object as possible.
  cv::Mat l1_alpha_blend, l0_alpha_blend;
  cv::Size blend_kernel(inpaint_dilate_radius * 2 + 1, inpaint_dilate_radius * 2 + 1);
  cv::GaussianBlur(l1_inpaint_mask, l1_alpha_blend, blend_kernel, inpaint_dilate_radius, inpaint_dilate_radius);
  cv::GaussianBlur(l0_inpaint_mask, l0_alpha_blend, blend_kernel, inpaint_dilate_radius, inpaint_dilate_radius);

  // Do a hack to reduce artifacts.
  reduceSegAlphaTransitionArtifacts(
      segmentation, l1_alpha, l2_alpha, l0_alpha_blend, l1_alpha_blend);

  // Set the alpha channels for top and middle by feathering and eroding (now, after we
  // thresholded them above), to avoid crap stuck to the edges / funky sillouettes.
  cv::erode(l2_alpha, l2_alpha, cv::Mat(), cv::Point(-1, -1), 1);
  cv::erode(l1_alpha, l1_alpha, cv::Mat(), cv::Point(-1, -1), 1);
  cv::GaussianBlur(l2_alpha, l2_alpha, cv::Size(3, 3), 1.0, 1.0);
  cv::GaussianBlur(l1_alpha, l1_alpha, cv::Size(3, 3), 1.0, 1.0);

  // Do inpainting with various algorithms
  l1_inpaint_mask.convertTo(l1_inpaint_mask, CV_8U, 255.0);
  l0_inpaint_mask.convertTo(l0_inpaint_mask, CV_8U, 255.0);

  cv::Mat inpainted_mid, inpainted_bottom;

  if (inpaint_method == "aot") {
    XCHECK(false) << "aot inpainting disabled";
    //inpainted_mid = inpaint::inpaintLazyModelLoad(R_ftheta, l1_inpaint_mask);
    //inpainted_bottom = inpaint::inpaintLazyModelLoad(R_ftheta, l0_inpaint_mask);
  }

  // TODO: pre-inpainting with ceres for stable diffusion could be an option (it takes more time but
  // produces better results in some cases)
  // if (inpaint_method == "ceres" || inpaint_method == "sd") {
  static constexpr int kCeresInpaintScale = 10;
  if (inpaint_method == "ceres") {
    cv::Mat float_R_ftheta;
    R_ftheta.convertTo(float_R_ftheta, CV_32FC3, 1.0 / 255.0);

    inpainted_mid =
        inpaint::inpaintWithCeresSmallSize<3>(kCeresInpaintScale, float_R_ftheta, l1_inpaint_mask);
    inpainted_bottom =
        inpaint::inpaintWithCeresSmallSize<3>(kCeresInpaintScale, float_R_ftheta, l0_inpaint_mask);

    inpainted_mid.convertTo(inpainted_mid, CV_8UC3, 255.0);
    inpainted_bottom.convertTo(inpainted_bottom, CV_8UC3, 255.0);

    // Use the ceres inpainting result as a seed for stable diffusion.
    // if (inpaint_method == "sd") {
    //  cv::imwrite(debug_dir + "/ceres_mid.png", inpainted_mid);
    //  cv::imwrite(debug_dir + "/ceres_bottom.png", inpainted_bottom);
    //}
  }

  // Depth inpainting
  cv::Mat l1_depth = inpaint::inpaintWithCeresSmallSize<1>(
      kCeresInpaintScale / 2,
      R_inv_depth_ftheta,
      opencv::halfSize(l1_inpaint_mask),
      R_inv_depth_ftheta);
  cv::Mat l0_depth = inpaint::inpaintWithCeresSmallSize<1>(
      kCeresInpaintScale / 2, R_inv_depth_ftheta, opencv::halfSize(l0_inpaint_mask), l1_depth);

  // Save data we need to temporally stabilize inpainting.
  if (write_inpainting_stabilization_files) {
    cv::Mat l1_depth_16bit, l0_depth_16bit;
    l1_depth.convertTo(l1_depth_16bit, CV_16UC1, 65535.0f);
    l0_depth.convertTo(l0_depth_16bit, CV_16UC1, 65535.0f);
    cv::imwrite(debug_dir + "/l1_alpha_" + frame_num + ".png", l1_alpha * 255.0);
    cv::imwrite(debug_dir + "/l2_alpha_" + frame_num + ".png", l2_alpha * 255.0);
    cv::imwrite(debug_dir + "/l0_blend_" + frame_num + ".png", l0_alpha_blend * 255.0);
    cv::imwrite(debug_dir + "/l1_blend_" + frame_num + ".png", l1_alpha_blend * 255.0);
    cv::imwrite(debug_dir + "/l0_inpainted_" + frame_num + ".png", inpainted_bottom);
    cv::imwrite(debug_dir + "/l1_inpainted_" + frame_num + ".png", inpainted_mid);
    cv::imwrite(debug_dir + "/l0_inpainted_depth_" + frame_num + ".png", l0_depth_16bit);
    cv::imwrite(debug_dir + "/l1_inpainted_depth_" + frame_num + ".png", l1_depth_16bit);
  }

  if (assemble_ldi) {
    assembleLayersAndChannels(
        color_vignette,
        depth_vignette,
        R_ftheta,
        R_inv_depth_ftheta,
        l0_alpha_blend,
        l1_alpha_blend,
        l1_alpha,
        l2_alpha,
        l0_depth,
        l1_depth,
        inpainted_bottom,
        inpainted_mid,
        layer_bgra,
        layer_invd);
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
