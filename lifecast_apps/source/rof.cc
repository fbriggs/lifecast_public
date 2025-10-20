// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <fstream>
#include <filesystem>
#include <regex>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include "rof.h"
#include "torch/torch.h"
#include "logger.h"
#include "util_runfile.h"
#include "util_time.h"
#include "util_torch.h"

namespace p11 { namespace optical_flow {

void getTorchModelRAFT(torch::jit::script::Module& module, std::string model_path)
{
  torch::NoGradGuard no_grad;

  if (model_path.empty()) {
#if defined(__linux__)
    const std::string default_model = "ml_models/rof_cuda.pt";
#elif defined(_WIN32)
    const std::string default_model = "rof_cuda.pt";  // On windows the directory structure is flat
#elif defined(__APPLE__)
    //const std::string default_model = "ml_models/rof_mps.pt";
    const std::string default_model = "ml_models/rof_cpu.pt";
#else
    const std::string default_model = "ml_models/rof_cpu.pt";
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

void computeOpticalFlowRAFT(
    torch::jit::script::Module& module,
    const cv::Mat& image1_8u,
    const cv::Mat& image2_8u,
    cv::Mat& flow_x,
    cv::Mat& flow_y)
{
  torch::NoGradGuard no_grad;

  XCHECK_EQ(image1_8u.size(), image2_8u.size());
  XCHECK(image1_8u.type() == image2_8u.type());
  XCHECK(image1_8u.type() == CV_8UC3 || image1_8u.type() == CV_8UC4);
  XCHECK(image2_8u.type() == CV_8UC3 || image2_8u.type() == CV_8UC4);
  XCHECK_EQ(image1_8u.rows % 8, 0);
  XCHECK_EQ(image1_8u.cols % 8, 0);
  XCHECK_EQ(image2_8u.rows % 8, 0);
  XCHECK_EQ(image2_8u.cols % 8, 0);

  cv::Mat image1, image2;
  // The original python implementation expects RGB, not BGR
  if (image1_8u.type() == CV_8UC3) {
    cv::cvtColor(image1_8u, image1, cv::COLOR_BGR2RGB);
    cv::cvtColor(image2_8u, image2, cv::COLOR_BGR2RGB);
  } else if (image1_8u.type() == CV_8UC4) {
    cv::cvtColor(image1_8u, image1, cv::COLOR_BGRA2RGB);
    cv::cvtColor(image2_8u, image2, cv::COLOR_BGRA2RGB);
  } else {
    XCHECK(false) << "unexpected image type";
  }
  image1.convertTo(image1, CV_32FC3);
  image2.convertTo(image2, CV_32FC3);

  // Pack the image data into a tensor with appropriate shape.
  at::Tensor tensor_image1 =
      torch::from_blob(image1.data, {image1.rows, image1.cols, 3}, at::kFloat);
  at::Tensor tensor_image2 =
      torch::from_blob(image2.data, {image2.rows, image2.cols, 3}, at::kFloat);

  torch::DeviceType device = util_torch::findBestTorchDevice();
#if defined(__APPLE__)
  device = torch::kCPU; // HACK: metal seems to be broken on mac in this version
#endif

  tensor_image1.unsqueeze_(0);
  tensor_image2.unsqueeze_(0);
  auto tensor_image1_perm = tensor_image1.permute({0, 3, 1, 2});
  auto tensor_image2_perm = tensor_image2.permute({0, 3, 1, 2});

  std::vector<torch::jit::IValue> inputs;
  auto inp1 = tensor_image1_perm.to(device, 0);
  auto inp2 = tensor_image2_perm.to(device, 0);
  inputs.push_back(inp1);
  inputs.push_back(inp2);

  module.to(device);

  // auto t0 = time::now();
  //#ifdef __APPLE__
  // auto outputs = module.forward(inputs).toTuple();
  auto rawOutputs = module.forward(inputs);
  auto outputs = rawOutputs.toTuple();
  auto flow_up = outputs->elements()[1].toTensor();
  //#endif
  //#ifdef __linux__
  //  auto outputs = module.forward(inputs);
  //  auto flow_up = outputs.toList().get(1).toTensor();
  //#endif
  // XPLINFO << "time forward: " << time::timeSinceSec(t0);

  // auto t3 = time::now();
  auto flow_up_perm = flow_up.permute({0, 2, 3, 1});
  // XPLINFO << "time permute: " << time::timeSinceSec(t3);

  const int w = flow_up_perm.sizes()[2];
  const int h = flow_up_perm.sizes()[1];

  // auto t1 = time::now();
  auto flow_up_perm_cpu = flow_up_perm.to(torch::kCPU);
  // XPLINFO << "time GPU->CPU: " << time::timeSinceSec(t1);

  // auto t2 = time::now();
  flow_x = cv::Mat(cv::Size(w, h), CV_32F, cv::Scalar(0));
  flow_y = cv::Mat(cv::Size(w, h), CV_32F, cv::Scalar(0));
  auto accessor = flow_up_perm_cpu.accessor<float, 4>();
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      // Slow way (not using accessor):
      // flow_x.at<float>(y, x) = flow_up_perm_cpu.index({0, y, x, 0}).item<float>();
      // flow_y.at<float>(y, x) = flow_up_perm_cpu.index({0, y, x, 1}).item<float>();
      flow_x.at<float>(y, x) = accessor[0][y][x][0];
      flow_y.at<float>(y, x) = accessor[0][y][x][1];
    }
  }
  // XPLINFO << "time for cv::Mat: " << time::timeSinceSec(t2);
}

cv::Mat computeDisparityRAFT(
    torch::jit::script::Module& module,
    const cv::Mat& image1_8u,
    const cv::Mat& image2_8u,
    const float bias)
{
  cv::Mat flow_x, flow_y;
  computeOpticalFlowRAFT(module, image1_8u, image2_8u, flow_x, flow_y);

  for (int y = 0; y < flow_x.rows; ++y) {
    for (int x = 0; x < flow_x.cols; ++x) {
      flow_x.at<float>(y, x) = std::max(0.0f, flow_x.at<float>(y, x) + bias);
    }
  }
  return flow_x;
}

void computeDisparityRAFTBothWays(
    torch::jit::script::Module& module,
    const cv::Mat& R_image,
    const cv::Mat& L_image,
    const float bias,
    cv::Mat& R_disparity,
    cv::Mat& L_disparity,
    cv::Mat& R_error,
    cv::Mat& L_error)
{
  XCHECK_EQ(R_image.size(), L_image.size());
  const int w = R_image.cols;
  const int h = R_image.rows;

  cv::Mat R_flow_x, R_flow_y;
  cv::Mat L_flow_x, L_flow_y;

  computeOpticalFlowRAFT(module, R_image, L_image, R_flow_x, R_flow_y);
  computeOpticalFlowRAFT(module, L_image, R_image, L_flow_x, L_flow_y);

  static constexpr float kVerticalDisparityErrorCoef =
      30.0;  // weight of vertical disparity in overall error
  static constexpr float kDisparityConsistencyErrorCoef = 100.0;
  R_error = cv::Mat(cv::Size(w, h), CV_32F);
  L_error = cv::Mat(cv::Size(w, h), CV_32F);

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      R_flow_x.at<float>(y, x) = std::max(0.0f, R_flow_x.at<float>(y, x) + bias);
      L_flow_x.at<float>(y, x) = std::max(0.0f, -L_flow_x.at<float>(y, x) + bias);

      // Any vertical flow suggests depth estimation / calibration error (or at least uncertainty in
      // depth).
      R_error.at<float>(y, x) =
          std::abs(R_flow_y.at<float>(y, x)) * (kVerticalDisparityErrorCoef / h);
      L_error.at<float>(y, x) =
          std::abs(L_flow_y.at<float>(y, x)) * (kVerticalDisparityErrorCoef / h);
    }
  }
  R_disparity = R_flow_x;
  L_disparity = L_flow_x;

  // Compute error estimate using loop consistency by warping, and from vertical disparity
  // TODO: idea- propage high uncertainty outward to better cover edges

  cv::Mat warp_R_from_L(cv::Size(w, h), CV_32FC2);
  cv::Mat warp_L_from_R(cv::Size(w, h), CV_32FC2);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      warp_R_from_L.at<cv::Vec2f>(y, x) =
          cv::Vec2f(x, y) + cv::Vec2f(R_disparity.at<float>(y, x), 0.0);
      warp_L_from_R.at<cv::Vec2f>(y, x) =
          cv::Vec2f(x, y) - cv::Vec2f(L_disparity.at<float>(y, x), 0.0);
    }
  }
  std::vector<cv::Mat> warp_R_from_L_uv, warp_L_from_R_uv;
  cv::split(warp_R_from_L, warp_R_from_L_uv);
  cv::split(warp_L_from_R, warp_L_from_R_uv);

  cv::Mat R_reconstructed_from_L, L_reconstructed_from_R;
  cv::remap(
      L_disparity,
      R_reconstructed_from_L,
      warp_R_from_L_uv[0],
      warp_R_from_L_uv[1],
      cv::INTER_LINEAR,
      cv::BORDER_CONSTANT,
      cv::Scalar(0, 0, 0, 0));
  cv::remap(
      R_disparity,
      L_reconstructed_from_R,
      warp_L_from_R_uv[0],
      warp_L_from_R_uv[1],
      cv::INTER_LINEAR,
      cv::BORDER_CONSTANT,
      cv::Scalar(0, 0, 0, 0));

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      const float R_err =
          std::abs(R_reconstructed_from_L.at<float>(y, x) - R_disparity.at<float>(y, x)) *
          (kDisparityConsistencyErrorCoef / w);
      const float L_err =
          std::abs(L_reconstructed_from_R.at<float>(y, x) - L_disparity.at<float>(y, x)) *
          (kDisparityConsistencyErrorCoef / w);

      R_error.at<float>(y, x) = math::clamp(R_error.at<float>(y, x) + R_err, 0.0f, 1.0f);
      L_error.at<float>(y, x) = math::clamp(L_error.at<float>(y, x) + L_err, 0.0f, 1.0f);
    }
  }
}

}}  // namespace p11::optical_flow
