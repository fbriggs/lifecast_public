// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <cstddef>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>

#include "check.h"

#define XCHECK_CUDA(expr) p11::CudaCheck(cudaError_t(expr), #expr, __FILE__, __LINE__)

namespace p11 {
struct CudaCheck : public CheckBase {
  CudaCheck(const cudaError_t result, const char* expr, const char* filename, const int line)
      : CheckBase(result == cudaSuccess, filename, line)
  {
    if (!condition) {
      message << "XCHECK_CUDA(" << expr << ") FAILED: [" << int(result) << "] "
              << cudaGetErrorString(result) << ", ";
    }
  }
};

template <typename T>
class CudaBuffer final {
 public:
  explicit CudaBuffer(size_t size) : device_ptr_(nullptr), size_(size)
  {
    if (size > 0) {
      XCHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_ptr_), bytes()));
    }
  }

  template <typename Alloc = std::allocator<T>>
  explicit CudaBuffer(const std::vector<T, Alloc>& vec) : CudaBuffer(std::size(vec))
  {
    copyFromCpu(vec);
  }

  template <size_t N>
  explicit CudaBuffer(const std::array<T, N>& arr) : CudaBuffer(std::size(arr))
  {
    copyFromCpu(arr);
  }

  CudaBuffer(const CudaBuffer& other) = delete;
  CudaBuffer(CudaBuffer&& other) = default;

  ~CudaBuffer() { cudaFree(device_ptr_); }

  size_t size() const { return size_; }
  size_t bytes() const { return size_ * sizeof(T); }

  const T* get() const { return device_ptr_; }
  T* get() { return device_ptr_; }

  const void* get_void() const { return reinterpret_cast<const void*>(device_ptr_); }
  void* get_void() { return reinterpret_cast<void*>(device_ptr_); }

  // TODO: enforce array-like storage for container
  template <typename C>
  void copyToCpu(C& container) const
  {
    copyToCpu(&*std::begin(container), 0, std::size(container));
  }

  void copyToCpu(T* dest, size_t start, size_t count) const
  {
    const auto source = device_ptr_ + start;
    XCHECK_LE(start + count, size());

    const auto bytes = count * sizeof(T);

    XCHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void*>(dest),
        reinterpret_cast<const void*>(source),
        bytes,
        cudaMemcpyDeviceToHost));
  }

  // TODO: enforce array-like storage for container
  template <typename C>
  void copyFromCpu(const C& container)
  {
    copyFromCpu(&*std::begin(container), 0, std::size(container));
  }

  void copyFromCpu(const T* data, size_t start, size_t count)
  {
    const auto dest = device_ptr_ + start;
    XCHECK_LE(start + count, size());

    const auto bytes = count * sizeof(T);

    XCHECK_CUDA(cudaMemcpy(
        reinterpret_cast<void*>(dest),
        reinterpret_cast<const void*>(data),
        bytes,
        cudaMemcpyHostToDevice));
  }

  std::vector<T> toVector() const
  {
    std::vector<T> result(size());
    copyToCpu(result);
    return result;
  }

 private:
  T* device_ptr_;
  size_t size_;
};

class CudaStream final {
 public:
  CudaStream() { XCHECK_CUDA(cudaStreamCreate(&stream_)); }
  CudaStream(const CudaStream&) = delete;
  CudaStream(CudaStream&&) = default;

  ~CudaStream() { XCHECK_CUDA(cudaStreamDestroy(stream_)); }

  cudaStream_t get() { return stream_; }

 private:
  cudaStream_t stream_;
};
}  // namespace p11
