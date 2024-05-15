// MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#pragma once

#include <cstdint>
#include <vector>

#include "torch/torch.h"

namespace p11 { namespace nerf {

struct ModelWithNeuralHashmap : public torch::nn::Module {
  torch::DeviceType device;

  int hashtable_size_per_level;
  int num_features_per_level;
  int course_resolution;
  int num_Levels;
  float level_scale;
  int hash_feature_dim;

  std::vector<float> level_to_resolution;
  std::vector<int32_t> level_to_hash_offset;
  torch::Tensor level_resolution_tensor;
  torch::Tensor level_hash_offset_tensor;
  torch::Tensor hashmap;

  ModelWithNeuralHashmap(
      const torch::DeviceType device,
      const int hashtable_size_per_level,
      const int num_features_per_level,
      const int course_resolution,
      const int num_Levels,
      const float level_scale);

  torch::Tensor hashTensor(torch::Tensor xi, int dx, int dy, int dz);
  torch::Tensor multiResolutionHashEncoding(
      const torch::DeviceType device, torch::Tensor cube_points);
};

struct NeoNerfModel : public ModelWithNeuralHashmap {
  static constexpr int kHiddenDim = 64;
  static constexpr int kGeoFeatureDim = 16;

  torch::nn::Linear sigma_layer1, sigma_layer2;
  torch::nn::Linear color_layer1, color_layer2, color_layer3;

  NeoNerfModel(const torch::DeviceType device, int image_code_dim);

  std::tuple<torch::Tensor, torch::Tensor> pointAndDirToRadiance(
      torch::Tensor x, torch::Tensor ray_dir, torch::Tensor image_code);

  // Only run the part of the model that gets density
  torch::Tensor pointToDensity(torch::Tensor x);
};

struct ProposalDensityModel : public ModelWithNeuralHashmap {
  static constexpr int kHiddenDim = 32;
  torch::nn::Linear sigma_layer1, sigma_layer2;

  ProposalDensityModel(const torch::DeviceType device);

  torch::Tensor pointToDensity(torch::Tensor x);
};

}}  // end namespace p11::nerf
