// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include "torch/torch.h"
#include "lifecast_splat_config.h"
#include "multicamera_dataset.h"
#include "util_torch.h"

namespace p11 { namespace splat {

std::shared_ptr<SplatModel> initSplatPopulation(
  const torch::DeviceType& device,
  const SplatConfig& cfg,
  calibration::MultiCameraDataset train_dataset); // TODO: can we pass train_dataset by const ref?

std::shared_ptr<SplatModel> initSplatPopulationWithSingleImageRGBD(
  const torch::DeviceType& device,
  int max_num_splats,
  const calibration::RectilinearCamerad& cam,
  const int num_initial_monodepth_splats,
  cv::Mat& image, // CV_8UC3
  cv::Mat& depthmap,// CV_32F
  cv::Mat& mask); // CV_8U

void addNewSplatsToModelFromSingleImageRGBD(
  const torch::DeviceType& device,
  const SplatConfig& cfg,
  const calibration::RectilinearCamerad& cam,
  const int target_num_splats_to_add,
  cv::Mat& image,
  cv::Mat& depthmap,
  cv::Mat& mask,
  std::shared_ptr<SplatModel>& model);

std::tuple<torch::Tensor, torch::Tensor> getLargestProjectedEllipseOverAllImages(
  const torch::DeviceType device,
  const calibration::MultiCameraDataset& dataset,
  std::shared_ptr<SplatModel> model);

void splatPopulationDynamics(
  const torch::DeviceType& device,
  const SplatConfig& cfg,
  const calibration::MultiCameraDataset& dataset,
  std::shared_ptr<SplatModel> model,
  torch::Tensor& should_stabilize,
  int target_num_alive,
  torch::Tensor grad2d_norm // accumulated norms of means2d gradients
  );

}}  // end namespace p11::splat
