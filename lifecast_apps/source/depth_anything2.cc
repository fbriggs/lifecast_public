// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "depth_anything2.h"

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "torch/torch.h"
#include "logger.h"
#include "util_runfile.h"
#include "util_time.h"
#include "util_torch.h"
#include "Eigen/Core"
#include "Eigen/Geometry"

namespace p11 { namespace depth_estimation {

void getTorchModelDepthAnything2(torch::jit::script::Module& module, std::string model_path) {
  torch::NoGradGuard no_grad;

  if (model_path.empty()) {
#if defined(__linux__)
    const std::string default_model = "ml_models/depth_anything3_cuda.pt";
#elif defined(_WIN32)
    const std::string default_model = "depth_anything3_cuda.pt";  // On windows the directory structure is flat
#elif defined(__APPLE__)
    const std::string default_model = "ml_models/depth_anything3_cuda.pt";
#else
    const std::string default_model = "ml_models/depth_anything3_cuda.pt";
    XCHECK(false) << "midas unsupported on CPU";
#endif
    model_path = p11::runfile::getRunfileResourcePath(default_model);
  }
  XPLINFO << "model_path: " << model_path;

  try {
#ifdef _WIN32
    torch::DeviceType device = util_torch::findBestTorchDevice();
    XCHECK_EQ(device, torch::kCUDA) << "CUDA not avilable, but required for Windows";
    std::ifstream inp(model_path, std::ios::binary);
    XCHECK(inp.is_open()) << "Failed to open ML model file: " << model_path;
    module = torch::jit::load(inp, device);
#else
    module = torch::jit::load(model_path);
#endif
  } catch (const c10::Error& e) {
    XCHECK(false) << "Error loading torch module: " << e.what() << "\n" << e.msg();
  }
}

torch::Tensor preprocessImage(const cv::Mat& image, torch::DeviceType device) {
  constexpr int h = 1036;
  constexpr int w = 1036;

  cv::Mat image_rgb, resized_image;

  cv::cvtColor(image, image_rgb, cv::COLOR_BGR2RGB);
  cv::resize(image_rgb, resized_image, cv::Size(w, h), 0, 0, cv::INTER_CUBIC);
  resized_image.convertTo(resized_image, CV_32FC3, 1.0f / 255.0f);

  // Wrap the image in a tensor and permute to [1, 3, H, W]
  torch::Tensor input_tensor = torch::from_blob(resized_image.data, {1, h, w, 3}, torch::kFloat);
  input_tensor = input_tensor.permute({0, 3, 1, 2}).clone();  // clone to ensure contiguous memory

  // Normalize using ImageNet mean and std
  input_tensor[0][0] = input_tensor[0][0].sub(0.485).div(0.229);
  input_tensor[0][1] = input_tensor[0][1].sub(0.456).div(0.224);
  input_tensor[0][2] = input_tensor[0][2].sub(0.406).div(0.225);

  return input_tensor.to(device);
}

cv::Mat estimateMonoDepthWithDepthAnything2(torch::jit::script::Module& module,
  const cv::Mat& image,
  const bool normalize_output,
  const bool resize_output
) {
  torch::NoGradGuard no_grad;
  torch::DeviceType device = util_torch::findBestTorchDevice();
  torch::Tensor input_tensor = preprocessImage(image, device);

  auto output = module.forward({input_tensor});
  auto tup = output.toTuple();
  //auto features = tup->elements()[0].toTensor(); // TODO: unused, but interesting!
  auto depth = tup->elements()[1].toTensor();

  depth = depth.squeeze(0).squeeze(0);
  depth = depth.detach().cpu();
  cv::Mat depth_map(cv::Size(depth.size(1), depth.size(0)), CV_32FC1, depth.contiguous().data_ptr<float>());
  if (resize_output) { cv::resize(depth_map, depth_map, image.size(), 0, 0, cv::INTER_LINEAR); }
  if (normalize_output) cv::normalize(depth_map, depth_map, 0, 1, cv::NORM_MINMAX);

  // Convert disparity to proper depth
  for (int y = 0; y < depth_map.rows; ++y) {
    for (int x = 0; x < depth_map.cols; ++x) {
      // NOTE: magic number here is hand tuned for da2 trying to get meters 
      depth_map.at<float>(y, x) = math::clamp(10.0f / depth_map.at<float>(y, x), 0.1f, 50.0f);
    }
  }

  return depth_map.clone();
}

}}  // namespace p11::depth_estimation
