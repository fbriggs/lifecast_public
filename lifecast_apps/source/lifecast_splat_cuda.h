// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include "torch/torch.h"

namespace p11 { namespace splat {

constexpr int kTileSize = 1;
constexpr int kSplatsPerTileLimit = 16;
constexpr float kMaxAxisLenPixels = 512.0f; // NOTE: splats wont be axis aligned so the AABB can be larger
constexpr int kBigSplatTileThreshold = 5; // Splats that are greater than this many tiles wide or tall are "big"

std::tuple<torch::Tensor, torch::Tensor>
makeSplatTileIndexTensorCUDA(
  torch::DeviceType device,
  const int tiles_h,
  const int tiles_w,
  torch::Tensor uv, // projected splat centers
  torch::Tensor cov_inv,
  torch::Tensor eigenvector1, // unit vector in the direction of the elipse major axis
  torch::Tensor eigenvector2,
  torch::Tensor axis_len1,
  torch::Tensor axis_len2,
  torch::Tensor splat_alpha,
  torch::Tensor dist_sq
);

}} // namespace p11::splat
