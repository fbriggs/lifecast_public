// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include "util_torch.h"
#include "third_party/gsplat/gsplat/cuda/include/bindings.h"

#include <tuple>

namespace p11 { namespace gsplat {

enum RenderMode {
  RGB,
  RGBD,

  // TODO (or not):
  // D
  // ED
  // RGBED
};

enum RasterizeMode {
  CLASSIC,
  ANTIALIASED,
};

struct RasterizationConfig {
  RenderMode render_mode = RGBD;
  float near_plane = 0.01;
  float far_plane = 1e10;
  float radius_clip = 0.0;
  float eps2d = 0.3;
  // Sigma level has a fudge factor that was empirically determined by observing undesired tile clipping given the alpha threshold. TODO: calculate these automatically from the bit depths
  float sigma_level = 3.f * 1.33333;  // For radius calculation and tile intersection; increase if you see clipping
  // Alpha threshold allows for 10-bit depth, converted to srgb (0.003... is the linearToSrgb scalar) (TODO: are we using sRGB?)
  float alpha_threshold = 0.0031308 * 1.f / float(1 << 10);
  c10::optional<int> sh_degree;
  bool packed = false; // TODO: default was true
  int tile_size = 16;
  c10::optional<torch::Tensor> backgrounds;
  bool sparse_grad = false;
  bool absgrad = false;
  RasterizeMode rasterize_mode = CLASSIC;
  int channel_chunk = 32;
  bool distributed = false;
  ::gsplat::CameraModelType camera_model = ::gsplat::PINHOLE;
};

struct RasterizationMetas {
  // Only used for packed
  //torch::Tensor camera_ids;
  //torch::Tensor gaussian_ids;

  torch::Tensor radii;
  torch::Tensor means2d;
  torch::Tensor means2d_absgrad;
  torch::Tensor depths;
  torch::Tensor conics;
  torch::Tensor opacities;

  int width;
  int height;
  int tile_size;
  int tile_width;
  int tile_height;
  torch::Tensor tiles_per_gauss;
  int n_cameras;

  torch::Tensor isect_ids;
  torch::Tensor flatten_ids;
  torch::Tensor isect_offsets;

};

std::tuple<torch::Tensor, torch::Tensor, RasterizationMetas> rasterization(
  torch::Tensor means,
  torch::Tensor quats,
  torch::Tensor scales,
  //at::optional<torch::Tensor> covars,
  torch::Tensor opacities,
  torch::Tensor colors,
  torch::Tensor viewmats,
  torch::Tensor Ks,
  int width,
  int height,
  const RasterizationConfig& cfg
);

}}  // namespace p11::gsplat
