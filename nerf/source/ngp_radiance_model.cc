// MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include "ngp_radiance_model.h"

#include <fstream>

#include "logger.h"

namespace p11 { namespace nerf {
using namespace torch::autograd;

torch::Tensor contractUnbounded(torch::Tensor x1)
{
  static constexpr double kContractCoef = 0.3;
  static constexpr double kAvoidBoundary = 0.999;

  torch::Tensor x = x1 * kContractCoef;
  torch::Tensor mag_sq =
      torch::maximum(torch::ones_like(x) * 1e-7, torch::sum(torch::square(x), -1, true));
  torch::Tensor mag = torch::sqrt(mag_sq);
  torch::Tensor contracted =
      torch::where(mag_sq <= 1, x, (2.0 - 1.0 / mag) * (x / mag)) * kAvoidBoundary;
  return (contracted + 2.0) / 4.0;
}

torch::Tensor sphericalHarmonicEncoding(torch::Tensor dir)
{
  torch::Tensor x = torch::select(dir, 1, 0);
  torch::Tensor y = torch::select(dir, 1, 1);
  torch::Tensor z = torch::select(dir, 1, 2);
  torch::Tensor xx = x * x;
  torch::Tensor yy = y * y;
  torch::Tensor zz = z * z;
  torch::Tensor xy = x * y;
  torch::Tensor yz = y * z;
  torch::Tensor xz = x * z;

  std::vector<torch::Tensor> result(kSphericalHarmonicDim);

  // degree 1
  result[0] = 0.4886025119029199 * y;
  result[1] = 0.4886025119029199 * z;
  result[2] = 0.4886025119029199 * x;
  // degree 2
  result[3] = 1.0925484305920792 * x * y;
  result[4] = 1.0925484305920792 * y * z;
  result[5] = 0.9461746957575601 * zz - 0.31539156525251999;
  result[6] = 1.0925484305920792 * x * z;
  result[7] = 0.5462742152960396 * (xx - yy);
  // degree 3
  result[8] = 0.5900435899266435 * y * (3 * xx - yy);
  result[9] = 2.890611442640554 * x * y * z;
  result[10] = 0.4570457994644658 * y * (5 * zz - 1);
  result[11] = 0.3731763325901154 * z * (5 * zz - 3);
  result[12] = 0.4570457994644658 * x * (5 * zz - 1);
  result[13] = 1.445305721320277 * z * (xx - yy);
  result[14] = 0.5900435899266435 * x * (xx - 3 * yy);
  // degree 4
  result[15] = 2.5033429417967046 * x * y * (xx - yy);
  result[16] = 1.7701307697799304 * y * z * (3 * xx - yy);
  result[17] = 0.9461746957575601 * x * y * (7 * zz - 1);
  result[18] = 0.6690465435572892 * y * (7 * zz - 3);
  result[19] = 0.10578554691520431 * (35 * zz * zz - 30 * zz + 3);
  result[20] = 0.6690465435572892 * x * z * (7 * zz - 3);
  result[21] = 0.47308734787878004 * (xx - yy) * (7 * zz - 1);
  result[22] = 1.7701307697799304 * x * z * (xx - 3 * yy);
  result[23] = 0.4425326924449826 * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
  // degree 0.. NOTE: not clear why we want this, it seems useless.. unless its like a canonical
  // bias feature or something.
  result[24] = torch::zeros_like(x) + 0.28209479177387814;

  return torch::stack(result, -1);
}

NeoNerfModel::NeoNerfModel(const torch::DeviceType device, int image_code_dim)
    : encoder(std::make_shared<NeuralHashmap>(
          device,
          kNumLevels,
          kNumFeaturesPerLevel,
          kLog2HashtableSizePerLevel,
          kCoarseResolution,
          kLevelScale)),
      sigma_layer1(encoder->numOutputDims(), kHiddenDim),
      sigma_layer2(kHiddenDim, NeoNerfModel::kGeoFeatureDim),
      color_layer1(
          kGeoFeatureDim + kSphericalHarmonicDim + image_code_dim,
          kHiddenDim),
      color_layer2(kHiddenDim, NeoNerfModel::kHiddenDim),
      color_layer3(kHiddenDim, kRGBDim)
{
  register_module("encoder", encoder);
  register_module("sigma_layer1", sigma_layer1);
  register_module("sigma_layer2", sigma_layer2);
  register_module("color_layer1", color_layer1);
  register_module("color_layer2", color_layer2);
  register_module("color_layer3", color_layer3);
  to(device);
}

std::tuple<torch::Tensor, torch::Tensor> NeoNerfModel::pointAndDirToRadiance(
    torch::Tensor x, torch::Tensor ray_dir, torch::Tensor image_code)
{
  torch::Tensor contracted_points = contractUnbounded(x);

  // auto start_timer = time::now();
  torch::Tensor hash_codes = encoder->multiResolutionHashEncoding(contracted_points);
  // torch::cuda::synchronize();
  // XPLINFO << "hash time(sec):\t\t\t" << time::timeSinceSec(start_timer);
  torch::Tensor h = torch::relu(sigma_layer1->forward(hash_codes));
  torch::Tensor geo = sigma_layer2->forward(h);
  torch::Tensor sh = sphericalHarmonicEncoding(ray_dir);
  torch::Tensor color_input = torch::cat({geo, sh, image_code}, -1);
  h = torch::relu(color_layer1->forward(color_input));
  h = torch::relu(color_layer2->forward(h));

  //torch::Tensor sigma = TruncExp::apply(geo.index({"...", 0}));  // instead of torch::relu
  torch::Tensor sigma = funkExp(geo.index({"...", 0}));  // instead of torch::relu
  torch::Tensor bgr = torch::sigmoid(color_layer3->forward(h));

  return {bgr, sigma};
}

// Only run the part of the model that gets density
torch::Tensor NeoNerfModel::pointToDensity(torch::Tensor x)
{
  torch::Tensor contracted_points = contractUnbounded(x);
  torch::Tensor hash_codes = encoder->multiResolutionHashEncoding(contracted_points);
  torch::Tensor h = torch::relu(sigma_layer1->forward(hash_codes));
  torch::Tensor geo = sigma_layer2->forward(h);
  //torch::Tensor sigma = TruncExp::apply(geo.index({"...", 0}));  // instead of torch::relu
  torch::Tensor sigma = funkExp(geo.index({"...", 0}));  // instead of torch::relu

  return sigma;
}

////////

ProposalDensityModel::ProposalDensityModel(const torch::DeviceType device)
    : encoder(std::make_shared<NeuralHashmap>(
          device,
          kNumLevels,
          kNumFeaturesPerLevel,
          kLog2HashtableSizePerLevel,  // log2(65536 * 4)
          kCoarseResolution,
          kLevelScale)),
      sigma_layer1(encoder->numOutputDims(), kHiddenDim),
      sigma_layer2(kHiddenDim, 1)
{
  register_module("proposal_encoder", encoder);
  register_module("sigma_layer1", sigma_layer1);
  register_module("sigma_layer2", sigma_layer2);
  to(device);
}

torch::Tensor ProposalDensityModel::pointToDensity(torch::Tensor x)
{
  torch::Tensor contracted_points = contractUnbounded(x);
  torch::Tensor hash_codes = encoder->multiResolutionHashEncoding(contracted_points);

  torch::Tensor h = torch::relu(sigma_layer1->forward(hash_codes));
  //torch::Tensor sigma = TruncExp::apply(sigma_layer2->forward(h));
  torch::Tensor sigma = funkExp(sigma_layer2->forward(h));
  return sigma;
}

}}  // end namespace p11::nerf
