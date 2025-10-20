// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

// This is a port of
// https://github.com/NVlabs/tiny-cuda-nn/blob/master/bindings/torch/tinycudann/modules.py#L116
//
// in combination with techniques from
// https://github.com/NVlabs/tiny-cuda-nn/blob/master/bindings/torch/tinycudann/bindings.cpp

#include <tiny-cuda-nn/common.h>
#include <tiny-cuda-nn/cpp_api.h>
#include <torch/torch.h>

namespace p11 { namespace nerf {

inline const char* kDefaultParamsName = "params";

class TcnnModule : public torch::nn::Module {
 public:
  TcnnModule(
      const torch::DeviceType device,
      tcnn::cpp::Module* module,
      const char* params_name = kDefaultParamsName);

  virtual torch::Tensor forward(torch::Tensor input);

  virtual unsigned int n_output_dims() const { return module->n_output_dims(); }

 private:
  std::unique_ptr<tcnn::cpp::Module> module;
  torch::Tensor params;
  float loss_scale;
};

class TcnnEncoding : public TcnnModule {
 public:
  TcnnEncoding(
      const torch::DeviceType device,
      int input_dims,
      const nlohmann::json& config,
      const char* params_name = kDefaultParamsName);
};

#if !defined(TCNN_NO_NETWORKS)
class TcnnNetwork : public TcnnModule {
 public:
  TcnnNetwork(
      const torch::DeviceType device,
      int input_dims,
      int output_dims,
      const nlohmann::json& config,
      const char* params_name = kDefaultParamsName);

  virtual unsigned int n_output_dims() const override { return output_dims; }

 private:
  int output_dims;  // Need to store this ourselves because the tcnn module pads its output
};
#endif // !defined(TCNN_NO_NETWORKS)

}}  // end namespace p11::nerf
