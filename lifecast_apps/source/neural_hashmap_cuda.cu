// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ngp_radiance_model.h"

#include "source/tcnn_module.h"
#include "check.h"

namespace p11 { namespace nerf {

// Use TcnnEncoding as the impl
struct NeuralHashmapImpl : public TcnnEncoding {
  using TcnnEncoding::TcnnEncoding;
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
          3,
          nlohmann::json{
              {"otype", "HashGrid"},
              {"n_levels", num_levels},
              {"n_features_per_level", num_features_per_level},
              {"log2_hashmap_size", log2_hashmap_size},
              {"base_resolution", coarse_resolution},
              {"per_level_scale", level_scale},
          },
          kHashmapParamsName))
{
  XCHECK_EQ(device, torch::DeviceType::CUDA);
  // Necessary for the hashmap parameters to be added to the named params
  register_module("tcnn_encoder", impl);
}

int NeuralHashmap::numOutputDims() const { return impl->n_output_dims(); }

torch::Tensor NeuralHashmap::multiResolutionHashEncoding(torch::Tensor cube_points)
{
  return impl->forward(cube_points);
}

}}  // end namespace p11::nerf
