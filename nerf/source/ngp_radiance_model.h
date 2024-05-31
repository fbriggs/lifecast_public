// MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#pragma once

#include <memory>

#include "torch/torch.h"

namespace p11 { namespace nerf {

constexpr int kSphericalHarmonicDim = 25;
constexpr int kRGBDim = 3;

constexpr const char* kHashmapParamsName = "hashmap";

torch::Tensor contractUnbounded(torch::Tensor x1);

torch::Tensor sphericalHarmonicEncoding(torch::Tensor dir);

/*
class TruncExp : public Function<TruncExp> {
 public:
  static torch::Tensor forward(AutogradContext *ctx, torch::Tensor x)
  {
    ctx->save_for_backward({x});
    return torch::exp(x);
  }

  static tensor_list backward(AutogradContext *ctx, tensor_list grads)
  {
    auto saved = ctx->get_saved_variables();
    auto x = saved[0];
    torch::Tensor g = grads[0];
    return {g * torch::exp(torch::clamp(x, -11, 11))};
  }
};
*/

// Similar to TrunExp but funkier (more numerically stable)
// This activation funtion is designed to be used for the part of a
// radiance model that computes density.
// Adding epsilon prevents zero density from ever occurring!
inline torch::Tensor funkExp(torch::Tensor x)
{
  return torch::where(x < 0, torch::exp(x), (x + 1) * (x + 1)) + 1e-6;
}

// Using PIMPL to swap out tiny-cuda-nn vs libtorch implementations
struct NeuralHashmapImpl;

class NeuralHashmap : public torch::nn::Module {
 public:
  NeuralHashmap(
      torch::DeviceType device,
      int num_levels,
      int num_features_per_level,
      int log2_hashmap_size,
      int coarse_resolution,
      float level_scale);

  torch::Tensor multiResolutionHashEncoding(torch::Tensor cube_points);
  int numOutputDims() const;

 private:
   // Using shared_ptr since both Impls are themselves torch::nn::Modules, which
   // need to be shared_ptrs.
   std::shared_ptr<NeuralHashmapImpl> impl;
};

struct NeoNerfModel : public torch::nn::Module {
  static constexpr int kHiddenDim = 64;
  static constexpr int kGeoFeatureDim = 16;

  static constexpr int kNumLevels = 16;
  static constexpr int kNumFeaturesPerLevel = 2;
  static constexpr int kNumFeatures = kNumLevels * kNumFeaturesPerLevel;
  static constexpr int kCoarseResolution = 16;
  static constexpr float kLevelScale = 1.382;

  static constexpr int kLog2HashtableSizePerLevel = 19;

  NeoNerfModel(const torch::DeviceType device, int image_code_dim);

  std::tuple<torch::Tensor, torch::Tensor> pointAndDirToRadiance(
      torch::Tensor x, torch::Tensor ray_dir, torch::Tensor image_code);

  // Only run the part of the model that gets density
  torch::Tensor pointToDensity(torch::Tensor x);

 private:
  std::shared_ptr<NeuralHashmap> encoder;
  torch::nn::Linear sigma_layer1, sigma_layer2;
  torch::nn::Linear color_layer1, color_layer2, color_layer3;
};

struct ProposalDensityModel : public torch::nn::Module {
  static constexpr int kHiddenDim = 32;
  static constexpr int kNumLevels = 5;
  static constexpr int kNumFeaturesPerLevel = 2;
  static constexpr int kNumFeatures = kNumLevels * kNumFeaturesPerLevel;

  static constexpr int kCoarseResolution = 16;
  static constexpr float kLevelScale = 2.0;

  static constexpr int kLog2HashtableSizePerLevel = 18;

  ProposalDensityModel(const torch::DeviceType device);

  torch::Tensor pointToDensity(torch::Tensor x);

 private:
  std::shared_ptr<NeuralHashmap> encoder;
  torch::nn::Linear sigma_layer1, sigma_layer2;
};

}}  // end namespace p11::nerf
