// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
bazel run -- //examples:hello_torch
*/
#include <iostream>
#include "torch/script.h"
#include "torch/torch.h"

#ifdef _WIN32
#include <Windows.h>
#endif

int main()
{
// Workaround a bug in libtorch for windows where it links against the wrong dll and doesn't support
// CUDA. See https://github.com/pytorch/pytorch/issues/72396
#ifdef _WIN32
  LoadLibraryA("torch_cuda.dll");
#endif

  torch::DeviceType device = torch::kCPU;
  if (torch::cuda::is_available()) {
    std::cout << "CUDA is available!" << std::endl;
    device = torch::kCUDA;
  }
  if (torch::mps::is_available()) {
    std::cout << "MPS is available!" << std::endl;
    device = torch::kMPS;
  }

  torch::Tensor tensor = torch::rand({2, 3}).to(device);
  std::cout << tensor << std::endl;
}
