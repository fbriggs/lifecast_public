// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "torch/script.h"
#include "torch/torch.h"
#include "fisheye_camera.h"

namespace p11 { namespace ldi {

struct NeuralHashmapsForegroundSegmentationModel : public torch::nn::Module {
  static constexpr int kNumLayers = 3;

  // Multi-scale neural hashmap params
  static constexpr int kNumFeaturesPerLevel = 2;
  static constexpr int kCourseResolution = 64;
  static constexpr float kLevelScale = 1.26;
  static constexpr int kNumLevels = 16;
  static constexpr int kHashFeatureDim = kNumLevels * kNumFeaturesPerLevel;

  torch::DeviceType device;

  std::vector<int> level_to_resolution;
  std::vector<int> level_to_hash_offset;
  torch::Tensor level_resolution_tensor;
  torch::Tensor level_hash_offset_tensor;
  torch::Tensor hashmap;

  torch::nn::Linear fc1, fc2, fc_final;
  static constexpr int H = 64;

  NeuralHashmapsForegroundSegmentationModel(const torch::DeviceType device)
      : device(device),
        fc1(torch::nn::Linear(kHashFeatureDim, H)),
        fc2(torch::nn::Linear(H, H)),
        fc_final(torch::nn::Linear(H, kNumLayers))
  {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc_final", fc_final);

    initHashmap();
  }

  void initHashmap();

  torch::Tensor hashTensor(torch::Tensor xi, int dx, int dy);

  torch::Tensor batchPointsToHashCodes(const torch::DeviceType device, torch::Tensor batch_points);

  torch::Tensor pointToSeg(torch::Tensor xy);
};

std::pair<cv::Mat, cv::Mat> getEdgesHiAndLow(
    const calibration::FisheyeCamerad cam_R, const cv::Mat& R_inv_depth_ftheta);

// Returns a 32F mask in [0, 1], which needs to be thresholded/dilated.
cv::Mat segmentFgBgWithMultiresolutionHashmap(
    const cv::Mat& edge_hi,
    const cv::Mat& edge_lo,
    const calibration::FisheyeCamerad cam_R,
    const calibration::FisheyeCamerad cam_R_half,
    const cv::Mat& R_inv_depth_ftheta,
    const cv::Mat prior_segmentation = cv::Mat(),
    const cv::Mat prior_weight = cv::Mat());

}}  // namespace p11::ldi
