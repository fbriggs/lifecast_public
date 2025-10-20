// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "hello_cuda.h"

#include "source/check.h"

__global__ void sumKernel(float* out, const float* in1, const float* in2)
{
  int i = threadIdx.x;
  out[i] = in1[i] + in2[i];
}

namespace p11 { namespace hello_cuda {
cudaError_t sum(CudaBuffer<float>& out, const CudaBuffer<float>& in1, const CudaBuffer<float>& in2)
{
  XCHECK_EQ(out.size(), in1.size());
  XCHECK_EQ(in2.size(), in1.size());
  // Launch kernel with 1 block per grid, and one thread per element
  sumKernel<<<1, out.size()>>>(out.get(), in1.get(), in2.get());

  return cudaGetLastError();
}
}}  // namespace p11::hello_cuda
