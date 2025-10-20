// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "torch_opengl.h"

namespace p11 { namespace torch_opengl {

CudaGLTexture::CudaGLTexture() 
  : texture_id(0), cuda_resource(nullptr), initialized(false), width(0), height(0) {}

CudaGLTexture::~CudaGLTexture() { cleanup(); }

bool CudaGLTexture::init(int w, int h) {
  if (initialized) cleanup();
  
  width = w;
  height = h;

  // Create OpenGL texture
  glGenTextures(1, &texture_id);
  glBindTexture(GL_TEXTURE_2D, texture_id);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  
  // Use RGBA8 format for standard rendering (8 bits per channel)
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glBindTexture(GL_TEXTURE_2D, 0);
  
  // Register texture with CUDA
  cudaError_t err = cudaGraphicsGLRegisterImage(
    &cuda_resource, 
    texture_id, 
    GL_TEXTURE_2D, 
    cudaGraphicsRegisterFlagsWriteDiscard);
  
  if (err != cudaSuccess) {
    XPLERROR << "Failed to register OpenGL texture with CUDA: " << cudaGetErrorString(err);
    glDeleteTextures(1, &texture_id);
    texture_id = 0;
    return false;
  }
  
  initialized = true;
  return true;
}

bool CudaGLTexture::updateFromTensor(const torch::Tensor& tensor) {
  if (!initialized) {
    XPLERROR << "CudaGLTexture not initialized";
    return false;
  }
  
  // Verify tensor is on CUDA and has correct dimensions
  if (!tensor.is_cuda()) {
    XPLERROR << "Tensor is not on CUDA device";
    return false;
  }
  
  XCHECK_EQ(tensor.dim(), 3) << "Expected 3D tensor [height, width, channels]";

  if (tensor.size(0) != height || tensor.size(1) != width) {
    XPLWARN << "Tensor dimensions don't match texture: tensor(" 
            << tensor.size(0) << "," << tensor.size(1) 
            << ") vs texture(" << height << "," << width << ")";
    // Reinitialize with new dimensions
    cleanup();
    if (!init(tensor.size(1), tensor.size(0))) {
      return false;
    }
  }
  
  int channels = tensor.size(2);
  XCHECK(channels == 3 || channels == 4) << "Tensor must have 3 or 4 channels";
  
  // Map resource
  cudaError_t err = cudaGraphicsMapResources(1, &cuda_resource);
  if (err != cudaSuccess) {
    XPLERROR << "Failed to map CUDA resource: " << cudaGetErrorString(err);
    return false;
  }
  
  // Get mapped array
  cudaArray_t array;
  err = cudaGraphicsSubResourceGetMappedArray(&array, cuda_resource, 0, 0);
  if (err != cudaSuccess) {
    XPLERROR << "Failed to get mapped array: " << cudaGetErrorString(err);
    cudaGraphicsUnmapResources(1, &cuda_resource);
    return false;
  }
  
  // Since we're using RGBA8 format (unsigned byte), but our tensor contains floats,
  // we need to convert the tensor values from [0.0, 1.0] to [0, 255]
  torch::Tensor normalized_tensor;
  
  if (channels == 3) {
    // Create normalized RGBA tensor (byte) from float RGB tensor
    torch::Tensor rgba_tensor = torch::ones({height, width, 4}, 
                                           tensor.options().dtype(torch::kUInt8));
    
    // Scale and convert to bytes
    torch::Tensor rgb_bytes = (tensor * 255.0).clamp(0, 255).to(torch::kUInt8);
    rgba_tensor.slice(2, 0, 3) = rgb_bytes;
    rgba_tensor.slice(2, 3, 4) = 255; // Full alpha
    
    normalized_tensor = rgba_tensor;
  } else {
    // Scale and convert to bytes for RGBA
    normalized_tensor = (tensor * 255.0).clamp(0, 255).to(torch::kUInt8);
  }
  
  // Copy from tensor to array
  err = cudaMemcpy2DToArray(
    array, 0, 0,
    normalized_tensor.data_ptr<uint8_t>(),
    width * sizeof(uint8_t) * 4,
    width * sizeof(uint8_t) * 4,
    height,
    cudaMemcpyDeviceToDevice);
  
  if (err != cudaSuccess) {
    XPLERROR << "Failed to copy data to CUDA array: " << cudaGetErrorString(err);
    cudaGraphicsUnmapResources(1, &cuda_resource);
    return false;
  }
  
  // Unmap resources
  err = cudaGraphicsUnmapResources(1, &cuda_resource);
  if (err != cudaSuccess) {
    XPLERROR << "Failed to unmap CUDA resource: " << cudaGetErrorString(err);
    return false;
  }

  return true;
}

void CudaGLTexture::cleanup() {
  if (initialized) {
    if (cuda_resource) {
      cudaGraphicsUnregisterResource(cuda_resource);
      cuda_resource = nullptr;
    }
    
    if (texture_id) {
      glDeleteTextures(1, &texture_id);
      texture_id = 0;
    }
    
    initialized = false;
    width = 0;
    height = 0;
  }
}

}}  // namespace p11::torch_opengl