// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ngp_radiance_model.h"

#include "logger.h"
#include "source/util_time.h"

namespace p11 { namespace nerf {

struct NeuralHashmapImpl : public torch::nn::Module {
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

  NeuralHashmapImpl(
      const torch::DeviceType device,
      const int num_Levels,
      const int num_features_per_level,
      const int log2_hashmap_size,
      const int course_resolution,
      const float level_scale)
      : hashtable_size_per_level(1 << log2_hashmap_size),
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

    // hashmap = ((torch::rand({hashtable_size_per_level * num_Levels, num_features_per_level},
    // torch::kFloat16) - 0.5) * 2e-4).to(device);

    // hashmap = ((torch::rand({hashtable_size_per_level * num_Levels, num_features_per_level},
    // torch::kFloat32) -0.5) * 2e-4).to(device);

    // Initializing with zeros produces slightly higher error solutions than random, but is (in
    // thoery) more temporally stable.
    hashmap = torch::zeros(
                  {hashtable_size_per_level * num_Levels, num_features_per_level}, torch::kFloat32)
                  .to(device);

    register_parameter(kHashmapParamsName, hashmap);
  }

  torch::Tensor hashTensor(torch::Tensor xi, int dx, int dy, int dz)
  {
    return (torch::abs(
                ((torch::select(xi, 1, 0) + dx) * static_cast<int32_t>(2165219737U)) ^
                ((torch::select(xi, 1, 1) + dy) * static_cast<int32_t>(2654435761U)) ^
                (torch::select(xi, 1, 2) + dz))  // Don't need a prime on one dimension
            & (hashtable_size_per_level - 1))    // hashtable_size_per_level must be a power of 2.
                                                 // equivalent to % hashtable_size_per_level
           + level_hash_offset_tensor;           // add an offset into the hashmap for each level
  }

  torch::Tensor multiResolutionHashEncoding(torch::Tensor cube_points)
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
};

NeuralHashmap::NeuralHashmap(
    torch::DeviceType device,
    int num_levels,
    int num_features_per_level,
    int log2_hashmap_size,
    int coarse_resolution,
    float level_scale)
    : impl(std::make_shared<NeuralHashmapImpl>(
          device,
          num_levels,
          num_features_per_level,
          log2_hashmap_size,
          coarse_resolution,
          level_scale))
{
  // Necessary for the hashmap parameters to be added to the named params
  register_module("torch_encoder", impl);
}

int NeuralHashmap::numOutputDims() const { return impl->hash_feature_dim; }

torch::Tensor NeuralHashmap::multiResolutionHashEncoding(torch::Tensor cube_points)
{
  return impl->multiResolutionHashEncoding(cube_points);
}

}}  // end namespace p11::nerf
