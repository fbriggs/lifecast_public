// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/* Demonstrates running a CUDA kernel

Example:

    bazel run --@rules_cuda//cuda:enable_cuda //examples:hello_cuda

*/

#include "hello_cuda.h"

#include <array>
#include <cuda_runtime_api.h>
#include <gflags/gflags.h>

#include "source/check.h"
#include "source/logger.h"

namespace p11 {
void helloCuda()
{
  constexpr size_t kNumElems = 3;
  const std::vector<float> in1_cpu{2.f, 3.f, 5.f};
  constexpr std::array<float, kNumElems> in2_cpu{7.f, 11.f, 13.f};
  XCHECK_EQ(in1_cpu.size(), kNumElems);
  XCHECK_EQ(in2_cpu.size(), kNumElems);

  auto in1_gpu = CudaBuffer(in1_cpu);
  auto in2_gpu = CudaBuffer(in2_cpu);
  auto out_gpu = CudaBuffer<float>(kNumElems);

  XCHECK_CUDA(hello_cuda::sum(out_gpu, in1_gpu, in2_gpu))
      << "CUDA launch failed. Make sure you are compiling for the correct GPU architecture.";

  std::array<float, kNumElems> out_cpu;
  out_gpu.copyToCpu(out_cpu);

  for (size_t i = 0; i < kNumElems; ++i) {
    XPLINFO << in1_cpu[i] << " + " << in2_cpu[i] << " = " << out_cpu[i];
  }
}
}  // namespace p11

int main(int argc, char** argv) { p11::helloCuda(); }
