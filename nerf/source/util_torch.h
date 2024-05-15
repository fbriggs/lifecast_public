// MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#pragma once

#include <fstream>
#include "torch/torch.h"
#include "check.h"

namespace p11 { namespace util_torch {

// We aim to write code that runs on multiple torch devices including CPU.
// from_blob works differently on CPU vs CUDA. With CUDA, the data is copied
// to the GPU, so it is OK if it goes out of scope. With the CPU device,
// if the data goes out of scope, the tensor contains garbage. To fix this
// without causing undue overhead for CUDA users, call this on tensors constructed
// with from_blob if the backed data goes out of scope.
static void cloneIfOnCPU(torch::Tensor& x) {
  if (x.device().type() == torch::kCPU) {
    x = x.clone();
  }
}

static torch::DeviceType findBestTorchDevice()
{
  torch::DeviceType device = torch::kCPU;
  if (torch::cuda::is_available()) {
    XPLINFO << "Using Torch device: CUDA";
    device = torch::kCUDA;
#ifdef __APPLE__
  } else if (torch::mps::is_available()) {
    // NOTE: 'aten::searchsorted.Tensor' is not currently implemented for the MPS device
    // This can be worked around with PYTORCH_ENABLE_MPS_FALLBACK=1 (and some bazel shenanagins)
    XPLINFO << "Using Torch device: MPS";
    device = torch::kMPS;
#endif
  } else {
    XPLINFO << "Using Torch device: CPU";
  }
  return device;
}

static std::string deviceTypeToString(torch::DeviceType device) {
  switch(device) {
  case torch::kCPU: return "CPU";
  case torch::kCUDA: return "CUDA";
  case torch::kMPS: return "Metal";
  default: return "Unknown Torch Device";
  }
}

// Ever want to copy a torch model's parameters without a lot of boilerplate?
// Well now you can...
template<typename T>
void deepCopyModel(std::shared_ptr<T>& src, std::shared_ptr<T>& dest) {
  torch::serialize::OutputArchive output_archive;
  src->save(output_archive);

  std::stringstream ss;
  output_archive.save_to(ss);

  torch::serialize::InputArchive input_archive;
  input_archive.load_from(ss);
  dest->load(input_archive);
}

template<typename T>
void saveOutputArchive(std::shared_ptr<T>& model, const std::string& path) {
  torch::serialize::OutputArchive output_archive;
  model->save(output_archive);
  output_archive.save_to(path);
}

template<typename T>
void loadModelArchive(std::shared_ptr<T>& model, const torch::DeviceType device, const std::string& path) {
  XCHECK(std::ifstream(path).good()) << "Model file not found: " << path;
  torch::serialize::InputArchive input_archive;
  input_archive.load_from(path);
  model->load(input_archive);
  model->to(device);
}

}}  // namespace p11::util_torch
