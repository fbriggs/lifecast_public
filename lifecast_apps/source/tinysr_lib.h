// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include "tinysr_zoo.h"

#include <string>
#include "torch/torch.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "logger.h"

namespace p11 { namespace enhance {

struct TinySuperResConfig {
  int rng_seed;
  std::string mode;
  std::string model_name;
  double scale;
  int batch_size;
  double lr;
  double lr_decay;
  // Training parameters
  std::string train_images_dir;
  std::string test_images_dir;
  std::string dest_dir;
  std::string augment;
  int num_itrs;
  // Inference (mode="upscale")
  std::string model_file;
  std::string src_image;
  std::string dest_image;
};

// run the full super resolution training pipeline
void trainSuperResModel(const TinySuperResConfig& cfg);

// load and apply the model to a single image
void testSuperResModel(const TinySuperResConfig& cfg);

 // evaluate model PSNR on a set of test images. this is run during training loop
double evalSuperResModel(const TinySuperResConfig& cfg, std::shared_ptr<Base_SuperResModel> model);

 // run evalSuperResModel standalone to evaluate pre-trained models
double loadAndEvalSuperResModel(const TinySuperResConfig& cfg);

torch::Tensor superResolutionEnhance(
  const torch::DeviceType device,
  float scale,
  std::shared_ptr<Base_SuperResModel> model,
  cv::Mat input_image);

void superResolutionEnhance(
  const torch::DeviceType device,
  float scale,
  std::shared_ptr<Base_SuperResModel> model,
  cv::Mat input_image,
  cv::Mat& output_image,
  cv::Mat& image_bicubic_upscaled,
  int cv_output_type = CV_8UC3);

double applyLearningRateSchedule(
  torch::optim::Optimizer& optimizer,
  double initial_lr,
  int current_iter,
  const std::vector<int>& milestones,
  double gamma);

}}  // end namespace p11::enhance
