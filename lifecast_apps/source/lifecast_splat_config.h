// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <string>
#include "torch/torch.h"
#include "logger.h"

namespace p11 { namespace splat {

namespace {
constexpr float kZNear = 0.01; // Anything closer than this is culled.
constexpr float kZFar = 1000.0; // Distance to the sky
constexpr float kInvDepthCoef = 0.1; // Smaller value to handle cases where camera is close up
constexpr float kScaleBias = 1.0;
constexpr float kMaxScaleExponent = 7.0; // We store a normalized encoding for scale exponents up to this value.
constexpr float kDeadSplatAlphaThreshold = 0.05;
constexpr float kMaxFracInitFromSfm = 0.2;
constexpr float kFracInitRandom = 0.05;
}

struct SplatConfig {
  std::string train_images_dir;
  std::string train_json;
  std::string sfm_pointcloud;
  std::string output_dir;
  bool save_steps;
  bool calc_psnr;
  int resize_max_dim;
  int max_num_splats;
  int num_itrs;
  int first_frame_warmup_itrs;
  int images_per_batch;
  int train_vis_interval;
  int population_update_interval;
  double learning_rate;
  bool init_with_monodepth;
  bool use_depth_loss;
  // video-only properties
  std::string vid_dir;
  bool is_video;
};

struct SplatModel {
  torch::Tensor splat_pos;    // [N, 3]
  torch::Tensor splat_color;  // [N, 3], pre-sigmoid activation
  torch::Tensor splat_alpha;  // [N, 1], pre-sigmoid activation
  torch::Tensor splat_scale;  // [N, 3], pre-activation
  torch::Tensor splat_quat;   // [N, 4]. May not be normalized. Must normalize before passing to renderSplatImage. wxyz format. identity is (1,0,0,0)
  torch::Tensor splat_alive;  // [N, 1]. If false, splat is skipped

  torch::Tensor per_image_depth_scale; // [M] - One per image, if depth loss is used
  torch::Tensor per_image_depth_bias;  // [M]

  void copyFrom(const std::shared_ptr<SplatModel>& src) {
    splat_pos = src->splat_pos.clone();
    splat_color = src->splat_color.clone();
    splat_alpha = src->splat_alpha.clone();
    splat_scale = src->splat_scale.clone();
    splat_quat = src->splat_quat.clone();
    splat_alive = src->splat_alive.clone();

    if (src->per_image_depth_scale.defined()) {
      per_image_depth_scale = src->per_image_depth_scale.clone();
    }
    if (src->per_image_depth_bias.defined()) {
      per_image_depth_bias = src->per_image_depth_bias.clone();
    }
  }
};

}}  // end namespace p11::splat
