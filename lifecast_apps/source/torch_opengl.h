// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// Functions for transfering from torch (CUDA) tensors to OpenGL textures
#pragma once

#include "check.h"
#include "logger.h"
#include "opencv2/core.hpp"
#include "torch/torch.h"
#include "opengl_xplatform_includes.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

namespace p11 { namespace torch_opengl {

// Class to manage direct CUDA-OpenGL interoperability for efficient texture updates
class CudaGLTexture {
public:
  CudaGLTexture();
  ~CudaGLTexture();

  // Initialize or reinitialize the texture with new dimensions
  bool init(int width, int height);

  // Update texture directly from CUDA tensor
  bool updateFromTensor(const torch::Tensor& tensor);

  // Clean up resources
  void cleanup();

  // Getters
  GLuint getTextureId() const { return texture_id; }
  bool isInitialized() const { return initialized; }
  int getWidth() const { return width; }
  int getHeight() const { return height; }

  // Convert dimensions to cv::Size for compatibility with existing code
  cv::Size getSize() const { return cv::Size(width, height); }

private:
  GLuint texture_id;
  cudaGraphicsResource* cuda_resource; // Forward declared struct pointer
  bool initialized;
  int width, height;
};

}}  // namespace p11::torch_opengl
