// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <atomic>
#include <string>
#include <map>
#include <fstream>
#include <type_traits>
#include <regex>
#include "logger.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/photo.hpp"
#include "fisheye_camera.h"
#include "util_math.h"
#include "util_time.h"
#include "util_string.h"
#include "util_opencv.h"
#include "util_file.h"
#include "torch/script.h"
#include "torch/torch.h"
#include "rof.h"
#include "depth_estimation.h"
#include "ldi_common.h"
#include "ldi_segmentation.h"

namespace p11 { namespace ldi {

struct LdiPipelineConfig {
  std::shared_ptr<std::atomic<bool>> cancel_requested;
  std::string cwd;
  std::string src_vr180;
  std::string src_ftheta_image; // alternative to src_vr180, provide image and depthmap already
  std::string src_ftheta_depth;
  std::string dest_dir;
  std::string output_filename; // only used for photo mode
  bool rm_dest_dir;
  int ftheta_size;
  int inflated_ftheta_size;
  int rectified_size_for_depth;
  double disparity_bias;
  double baseline_m;
  double inv_depth_coef;
  double ftheta_scale;
  std::string inpaint_method;
  std::string seg_method;
  std::string sd_ver;
  int first_frame;
  int last_frame;
  std::string phase;
  bool stabilize_inpainting;
  bool run_seg_only;
  bool write_seg;
  bool use_cached_seg;
  std::string output_encoding;
  bool make_fused_image;
  int inpaint_dilate_radius;
  bool skip_every_other_frame;
};

inline void printConfig(const LdiPipelineConfig& cfg) {
  XPLINFO << "cancel_requested=" << cfg.cancel_requested;
  XPLINFO << "cwd=" << cfg.cwd;
  XPLINFO << "src_vr180=" << cfg.src_vr180;
  XPLINFO << "src_ftheta_image=" << cfg.src_ftheta_image;
  XPLINFO << "src_ftheta_depth=" << cfg.src_ftheta_depth;
  XPLINFO << "dest_dir=" << cfg.dest_dir;
  XPLINFO << "output_filename=" << cfg.output_filename;
  XPLINFO << "rm_dest_dir=" << cfg.rm_dest_dir;
  XPLINFO << "ftheta_size=" << cfg.ftheta_size;
  XPLINFO << "inflated_ftheta_size=" << cfg.inflated_ftheta_size;
  XPLINFO << "rectified_size_for_depth=" << cfg.rectified_size_for_depth;
  XPLINFO << "disparity_bias=" << cfg.disparity_bias;
  XPLINFO << "baseline_m=" << cfg.baseline_m;
  XPLINFO << "inv_depth_coef=" << cfg.inv_depth_coef;
  XPLINFO << "ftheta_scale=" << cfg.ftheta_scale;
  XPLINFO << "inpaint_method=" << cfg.inpaint_method;
  XPLINFO << "seg_method=" << cfg.seg_method;
  XPLINFO << "sd_ver=" << cfg.sd_ver;
  XPLINFO << "first_frame=" << cfg.first_frame;
  XPLINFO << "last_frame=" << cfg.last_frame;
  XPLINFO << "phase=" << cfg.phase;
  XPLINFO << "stabilize_inpainting=" << cfg.stabilize_inpainting;
  XPLINFO << "run_seg_only=" << cfg.run_seg_only;
  XPLINFO << "write_seg=" << cfg.write_seg;
  XPLINFO << "use_cached_seg=" << cfg.use_cached_seg;
  XPLINFO << "output_encoding=" << cfg.output_encoding;
  XPLINFO << "make_fused_image=" << cfg.make_fused_image;
  XPLINFO << "inpaint_dilate_radius=" << cfg.inpaint_dilate_radius;
  XPLINFO << "skip_every_other_frame=" << cfg.skip_every_other_frame;
}

// Stabilized video pipeline
void runVR180toLdi3VideoPipelineAllPhases(const LdiPipelineConfig& cfg);

void videoDepthPhase(const LdiPipelineConfig& cfg);
void temporallyStabilizeDepth(const LdiPipelineConfig& cfg);
void inpaintPhase(const LdiPipelineConfig& cfg);
void stabilizeInpaintingPhase(const LdiPipelineConfig& cfg);
void cleanup(const LdiPipelineConfig& cfg);

// Similar pipeline for VR180 photo to LDI3 (no stabilization, all in memory instead of lots of file-IO)
void runVR180PhototoLdiPipeline(const LdiPipelineConfig& cfg);

// Helpers (must be in header due to templating)
template <typename TImage, typename TAccum>
void accumulateWeightedSum(
    const cv::Mat& nei_image,
    const float scale,
    const cv::Mat& mask,
    cv::Mat& accumulator,
    cv::Mat& sum_weight)
{
  XCHECK_EQ(accumulator.size(), nei_image.size());
  XCHECK_EQ(accumulator.size(), sum_weight.size());
  if (std::is_same<TImage, cv::Vec3b>::value) XCHECK_EQ(nei_image.type(), CV_8UC3);
  if (std::is_same<TImage, float>::value) XCHECK_EQ(nei_image.type(), CV_32FC1);
  if (std::is_same<TAccum, cv::Vec3f>::value) XCHECK_EQ(accumulator.type(), CV_32FC3);
  if (std::is_same<TAccum, float>::value) XCHECK_EQ(accumulator.type(), CV_32FC1);

  cv::Mat mask_resized;
  cv::resize(mask, mask_resized, accumulator.size(), 0.0, 0.0, cv::INTER_AREA);
  for (int y = 0; y < accumulator.rows; ++y) {
    for (int x = 0; x < accumulator.cols; ++x) {
      float w = mask_resized.at<uint8_t>(y, x) / 255.0;
      auto val = TAccum(nei_image.at<TImage>(y, x)) * scale;
      accumulator.at<TAccum>(y, x) += val * w;
      sum_weight.at<float>(y, x) += w;
    }
  }
}

}}  // namespace p11::ldi
