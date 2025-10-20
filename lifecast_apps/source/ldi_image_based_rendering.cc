// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ldi_image_based_rendering.h"

#include "torch/script.h"
#include "torch/torch.h"
#include "util_torch.h"
#include "logger.h"
#include "check.h"

namespace p11 { namespace ldi {

torch::Tensor cvMatToTensor(const cv::Mat& mat, const torch::DeviceType device) {
  auto tensor = torch::from_blob(mat.data, {mat.rows, mat.cols, mat.channels()}, torch::kFloat32).to(device);
  util_torch::cloneIfOnCPU(tensor);
  return tensor.permute({2, 0, 1}); // HWC to CHW
}

struct LdiModel : torch::nn::Module {
  torch::Tensor layer0_rgb, layer1_rgb, layer2_rgb; 
  torch::Tensor init_layer0_rgb, init_layer1_rgb, init_layer2_rgb;
  torch::Tensor fixed_alpha1, fixed_alpha2;

  LdiModel(
    torch::Tensor init_layer0_rgb,
    torch::Tensor init_layer1_rgb,
    torch::Tensor init_layer2_rgb
  ) :
    layer0_rgb(init_layer0_rgb.clone()),
    layer1_rgb(init_layer1_rgb.clone()),
    layer2_rgb(init_layer2_rgb.clone()),
    init_layer0_rgb(init_layer0_rgb),
    init_layer1_rgb(init_layer1_rgb),
    init_layer2_rgb(init_layer2_rgb)
  {
    register_parameter("layer0_rgb", layer0_rgb);
    register_parameter("layer1_rgb", layer1_rgb);
    register_parameter("layer2_rgb", layer2_rgb);
  }

  torch::Tensor loss(torch::Tensor target_rgb) {
    auto blend1 = layer0_rgb * (1 - fixed_alpha1) + layer1_rgb * fixed_alpha1;
    auto blend_final = blend1 * (1 - fixed_alpha2) + layer2_rgb * fixed_alpha2;

    auto target_err = torch::mse_loss(blend_final, target_rgb);
    
    auto self_diff =
      torch::mse_loss(layer0_rgb, init_layer0_rgb) +
      torch::mse_loss(layer1_rgb, init_layer1_rgb) +
      torch::mse_loss(layer2_rgb, init_layer2_rgb);

    return target_err + self_diff * 0.1;
  }

  void clamp() {
    layer0_rgb.data().clamp_(init_layer0_rgb - 0.1, init_layer0_rgb + 0.1).clamp_(0, 1);
    layer1_rgb.data().clamp_(init_layer1_rgb - 0.1, init_layer1_rgb + 0.1).clamp_(0, 1);
    layer2_rgb.data().clamp_(init_layer2_rgb - 0.1, init_layer2_rgb + 0.1).clamp_(0, 1);
  }
};

cv::Mat featherAlphaChannel(const cv::Mat& image, int K) {
  std::vector<cv::Mat> channels(4);
  cv::split(image, channels);
  cv::Mat alpha = channels[3];
  cv::erode(alpha, alpha, cv::Mat(), cv::Point(-1, -1), K);
  cv::GaussianBlur(alpha, alpha, cv::Size(K/2*2+1, K/2*2+1), K/2);
  channels[3] = alpha;

  cv::Mat feathered;
  cv::merge(channels, feathered);
  return feathered;
}

void refineLdiWithPrimaryImageProjection(
  std::vector<cv::Mat>& layer_bgra,
  std::vector<cv::Mat>& layer_invd,
  const cv::Mat& primary_image_projection
) {
  XCHECK_EQ(layer_bgra.size(), 3);
  XCHECK_EQ(layer_bgra[0].type(), CV_32FC4);
  XCHECK_EQ(primary_image_projection.type(), CV_8UC4);
  
  const int w = layer_bgra[0].cols;
  const int h = layer_bgra[0].rows;

  const torch::DeviceType device = util_torch::findBestTorchDevice();

  auto layer0_tensor = cvMatToTensor(layer_bgra[0], device);
  auto layer1_tensor = cvMatToTensor(layer_bgra[1], device);
  auto layer2_tensor = cvMatToTensor(layer_bgra[2], device);

  const int feather_size = primary_image_projection.cols / 32;
  cv::Mat primary_image_feather_alpha = featherAlphaChannel(primary_image_projection, feather_size);
  primary_image_feather_alpha.convertTo(primary_image_feather_alpha, CV_32FC4, 1.0/255.0);
  auto target_tensor = cvMatToTensor(primary_image_feather_alpha, device);

  auto layer0_rgb = layer0_tensor.slice(0, 0, 3);
  auto layer1_rgb = layer1_tensor.slice(0, 0, 3);
  auto layer2_rgb = layer2_tensor.slice(0, 0, 3);
  auto target_rgb = target_tensor.slice(0, 0, 3);

  LdiModel model(layer0_rgb, layer1_rgb, layer2_rgb);
  model.to(device);

  model.fixed_alpha1 = layer1_tensor.select(0, 3);
  model.fixed_alpha2 = layer2_tensor.select(0, 3);

  auto optimizer = torch::optim::Adam(model.parameters(), torch::optim::AdamOptions(1e-2));

  model.train();
  for (size_t itr = 0; itr < 100; itr++) {
    optimizer.zero_grad();
    auto loss = model.loss(target_rgb);
    loss.backward();
    optimizer.step();
    model.clamp();

    XPLINFO << itr << "\t" << loss.item<float>();
  }

  // Unpack the solution and blend it with the original based on the projection's alpha chanel
  auto cpu_layer0_rgb = model.layer0_rgb.to(torch::kCPU);
  auto acc_layer0_rgb = cpu_layer0_rgb.accessor<float, 3>();
  auto cpu_layer1_rgb = model.layer1_rgb.to(torch::kCPU);
  auto acc_layer1_rgb = cpu_layer1_rgb.accessor<float, 3>();
  auto cpu_layer2_rgb = model.layer2_rgb.to(torch::kCPU);
  auto acc_layer2_rgb = cpu_layer2_rgb.accessor<float, 3>();

  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      for (int c = 0; c < 3; ++c) {
        const float a = primary_image_feather_alpha.at<cv::Vec4f>(y, x)[3];
        layer_bgra[0].at<cv::Vec4f>(y, x)[c] = a * acc_layer0_rgb[c][y][x] + (1.0 - a) * layer_bgra[0].at<cv::Vec4f>(y, x)[c];
        layer_bgra[1].at<cv::Vec4f>(y, x)[c] = a * acc_layer1_rgb[c][y][x] + (1.0 - a) * layer_bgra[1].at<cv::Vec4f>(y, x)[c];
        layer_bgra[2].at<cv::Vec4f>(y, x)[c] = a * acc_layer2_rgb[c][y][x] + (1.0 - a) * layer_bgra[2].at<cv::Vec4f>(y, x)[c];
      }
    }
  }
}

}}  // namespace p11::ldi
