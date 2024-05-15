// MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include "ngp_radiance_model.h"

#include <fstream>

#include "logger.h"

namespace p11 { namespace nerf {
using namespace torch::autograd;

static constexpr int k3DSpace = 3;
static constexpr int kRGBDim = 3;
static constexpr int kSphericalHarmonicDim = 25;

static torch::Tensor contractUnbounded(torch::Tensor x1)
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
torch::Tensor funkExp(torch::Tensor x) {
  return torch::where(x < 0, torch::exp(x), (x+1) * (x+1)) + 1e-6;
}

static torch::Tensor sphericalHarmonicEncoding(torch::Tensor dir)
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

ModelWithNeuralHashmap::ModelWithNeuralHashmap(
    const torch::DeviceType device,
    const int hashtable_size_per_level,
    const int num_features_per_level,
    const int course_resolution,
    const int num_Levels,
    const float level_scale)
    : device(device),
      hashtable_size_per_level(hashtable_size_per_level),
      num_features_per_level(num_features_per_level),
      course_resolution(course_resolution),
      num_Levels(num_Levels),
      level_scale(level_scale)
{
  hash_feature_dim = num_Levels * num_features_per_level;

  XPLINFO << "num_Levels=" << num_Levels;
  float res = course_resolution;
  for (int level = 0; level < num_Levels; ++level) {
    XPLINFO << level << "\t" << res;
    level_to_resolution.push_back(res);
    res *= level_scale;

    level_to_hash_offset.push_back(level * hashtable_size_per_level);
  }

  // NOTE: we add some extra singleton dimensions for convenience later
  level_resolution_tensor =
      torch::from_blob(level_to_resolution.data(), {1, 1, num_Levels}, {torch::kFloat32})
          .to(device);
  level_hash_offset_tensor =
      torch::from_blob(level_to_hash_offset.data(), {1, num_Levels}, {torch::kInt32}).to(device);
  // No cloneIfOnCPU necessary on these from_blob's because the data isn't going out of scope.

  hashmap = torch::zeros({hashtable_size_per_level * num_Levels, num_features_per_level}, torch::kFloat32).to(device);

  register_parameter("hashmap", hashmap);
}

torch::Tensor ModelWithNeuralHashmap::hashTensor(torch::Tensor xi, int dx, int dy, int dz)
{
  return (torch::abs(
              ((torch::select(xi, 1, 0) + dx) * static_cast<int32_t>(2165219737U)) ^
              ((torch::select(xi, 1, 1) + dy) * static_cast<int32_t>(2654435761U)) ^
              (torch::select(xi, 1, 2) + dz))  // Don't need a prime on one dimension
          & (hashtable_size_per_level - 1))    // hashtable_size_per_level must be a power of 2.
                                               // equivalent to % hashtable_size_per_level
         + level_hash_offset_tensor;           // add an offset into the hashmap for each level
}

torch::Tensor ModelWithNeuralHashmap::multiResolutionHashEncoding(
    const torch::DeviceType device, torch::Tensor cube_points)
{
  const int num_points = cube_points.sizes()[0];

  torch::Tensor xl = torch::unsqueeze(cube_points, 2) * level_resolution_tensor;
  torch::Tensor xi = xl.toType(torch::kInt32);
  torch::Tensor r = xl - xi;

  torch::Tensor rx = torch::unsqueeze(torch::select(r, 1, 0), 2);
  torch::Tensor ry = torch::unsqueeze(torch::select(r, 1, 1), 2);
  torch::Tensor rz = torch::unsqueeze(torch::select(r, 1, 2), 2);

  std::vector<torch::Tensor> hash_index_corner(8);
  hash_index_corner[0] = hashTensor(xi, 0, 0, 0);
  hash_index_corner[1] = hashTensor(xi, 1, 0, 0);
  hash_index_corner[2] = hashTensor(xi, 0, 1, 0);
  hash_index_corner[3] = hashTensor(xi, 1, 1, 0);
  hash_index_corner[4] = hashTensor(xi, 0, 0, 1);
  hash_index_corner[5] = hashTensor(xi, 1, 0, 1);
  hash_index_corner[6] = hashTensor(xi, 0, 1, 1);
  hash_index_corner[7] = hashTensor(xi, 1, 1, 1);
  torch::Tensor hash_index = torch::stack(hash_index_corner, 2);

  torch::Tensor flat_hash_index = hash_index.reshape({num_points * num_Levels * 8});
  torch::Tensor hashed_vals = torch::index_select(hashmap, 0, flat_hash_index);
  torch::Tensor codes = hashed_vals.reshape({num_points, num_Levels, 8, num_features_per_level});

  torch::Tensor c000 = torch::select(codes, 2, 0);
  torch::Tensor c100 = torch::select(codes, 2, 1);
  torch::Tensor c010 = torch::select(codes, 2, 2);
  torch::Tensor c110 = torch::select(codes, 2, 3);
  torch::Tensor c001 = torch::select(codes, 2, 4);
  torch::Tensor c101 = torch::select(codes, 2, 5);
  torch::Tensor c011 = torch::select(codes, 2, 6);
  torch::Tensor c111 = torch::select(codes, 2, 7);

  torch::Tensor c00 = c000 * (1.0 - rx) + c100 * rx;
  torch::Tensor c01 = c001 * (1.0 - rx) + c101 * rx;
  torch::Tensor c10 = c010 * (1.0 - rx) + c110 * rx;
  torch::Tensor c11 = c011 * (1.0 - rx) + c111 * rx;

  torch::Tensor c0 = c00 * (1.0 - ry) + c10 * ry;
  torch::Tensor c1 = c01 * (1.0 - ry) + c11 * ry;
  torch::Tensor c = c0 * (1.0 - rz) + c1 * rz;

  torch::Tensor batch_features = c.reshape({num_points, num_features_per_level * num_Levels});

  return batch_features;
}

NeoNerfModel::NeoNerfModel(const torch::DeviceType device, int image_code_dim)
    : ModelWithNeuralHashmap(
          device,
          65536 * 8,  // hashtable_size_per_level
          2,          // num_features_per_level
          16,         // course_resolution
          16,         // num_Levels
          1.382       // level_scale
          ),
      sigma_layer1(hash_feature_dim, kHiddenDim),
      sigma_layer2(kHiddenDim, kGeoFeatureDim),
      color_layer1(kGeoFeatureDim + kSphericalHarmonicDim + image_code_dim, kHiddenDim),
      color_layer2(kHiddenDim, kHiddenDim),
      color_layer3(kHiddenDim, kRGBDim)
{
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
  torch::Tensor hash_codes = multiResolutionHashEncoding(device, contracted_points);
  // torch::cuda::synchronize();
  // XPLINFO << "hash time(sec):\t\t\t" << time::timeSinceSec(start_timer);

  torch::Tensor h = torch::relu(sigma_layer1->forward(hash_codes));
  torch::Tensor geo = sigma_layer2->forward(h);
  torch::Tensor sh = sphericalHarmonicEncoding(ray_dir);
  torch::Tensor color_input = torch::cat({geo, sh, image_code}, -1);
  h = torch::relu(color_layer1->forward(color_input));
  h = torch::relu(color_layer2->forward(h));
  //torch::Tensor sigma = TruncExp::apply(geo.index({"...", 0}));
  torch::Tensor sigma = funkExp(geo.index({"...", 0}));  
  torch::Tensor bgr = torch::sigmoid(color_layer3->forward(h));
  return {bgr, sigma};
}

// Only run the part of the model that gets density
torch::Tensor NeoNerfModel::pointToDensity(torch::Tensor x)
{
  torch::Tensor contracted_points = contractUnbounded(x);
  torch::Tensor hash_codes = multiResolutionHashEncoding(device, contracted_points);
  torch::Tensor h = torch::relu(sigma_layer1->forward(hash_codes));
  torch::Tensor geo = sigma_layer2->forward(h);
  //torch::Tensor sigma = TruncExp::apply(geo.index({"...", 0})); 
  torch::Tensor sigma = funkExp(geo.index({"...", 0}));

  return sigma;
}

////////

ProposalDensityModel::ProposalDensityModel(const torch::DeviceType device)
    : ModelWithNeuralHashmap(
          device,
          65536 * 4,  // hashtable_size_per_level
          2,          // num_features_per_level
          16,         // course_resolution
          5,          // num_Levels
          2           // level_scale
          ),
      sigma_layer1(hash_feature_dim, kHiddenDim),
      sigma_layer2(kHiddenDim, 1)
{
  register_module("sigma_layer1", sigma_layer1);
  register_module("sigma_layer2", sigma_layer2);
  to(device);
}

torch::Tensor ProposalDensityModel::pointToDensity(torch::Tensor x)
{
  torch::Tensor contracted_points = contractUnbounded(x);
  torch::Tensor hash_codes = multiResolutionHashEncoding(device, contracted_points);
  torch::Tensor h = torch::relu(sigma_layer1->forward(hash_codes));
  //torch::Tensor sigma = TruncExp::apply(sigma_layer2->forward(h));
  torch::Tensor sigma = funkExp(sigma_layer2->forward(h));
  return sigma;
}

}}  // end namespace p11::nerf
