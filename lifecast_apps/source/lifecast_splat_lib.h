// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <string>
#include <memory>
#include <mutex>
#include "logger.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/photo.hpp"
#include "source/gsplat_lib.h"
#include "torch/torch.h"
#include "rectilinear_camera.h"
#include "fisheye_camera.h"
#include "nerf_kludge_camera.h"
#include "util_time.h"
#include "util_torch.h"
#include "lifecast_splat_config.h"
#include "multicamera_dataset.h"

namespace p11 { namespace splat {

struct GaussianSplatGuiData {
  std::mutex mutex;
  std::shared_ptr<SplatModel> current_model = nullptr;
  std::atomic<bool> model_needs_update = false;
  // TODO: put cancel requested in here?
};

void runSplatPipelineStatic(SplatConfig& cfg);

void runSplatPipelineVideo(SplatConfig& cfg);

void printPopulationStats();

std::tuple<
  torch::Tensor,
  torch::Tensor,
  torch::Tensor,
  torch::Tensor,
  torch::Tensor,
  torch::Tensor,
  gsplat::RasterizationMetas
> renderSplatImageGsplat(
  const torch::DeviceType device,
  const calibration::RectilinearCamerad& cam,
  std::shared_ptr<SplatModel> model,
  c10::optional<torch::Tensor> background_colors = c10::nullopt,
  const Eigen::Matrix4d world_transform = Eigen::Matrix4d::Identity());

std::tuple<
  torch::Tensor,
  torch::Tensor,
  torch::Tensor,
  torch::Tensor,
  torch::Tensor,
  torch::Tensor,
  gsplat::RasterizationMetas
> renderSplatImageGsplat(
  const torch::DeviceType device,
  const calibration::RectilinearCamerad& cam,
  torch::Tensor splat_alive,
  torch::Tensor splat_pos,
  torch::Tensor splat_color,
  torch::Tensor splat_alpha,
  torch::Tensor splat_scale,
  torch::Tensor splat_quat,
  c10::optional<torch::Tensor> background_colors = c10::nullopt,
  const Eigen::Matrix4d world_transform = Eigen::Matrix4d::Identity());

void estimateMonoDepthmaps(
  const torch::DeviceType device,
  torch::jit::script::Module midas,
  calibration::MultiCameraDataset& dataset);

calibration::RectilinearCamerad precomputeFisheyeToRectilinearWarp(
  const calibration::FisheyeCamerad& src_cam,
  std::vector<cv::Mat>& warp_uv,
  const double focal_multiplier,
  const int resize_max_dim // Only used if not zero
);

void trainSplatModel(
  SplatConfig& cfg,
  const torch::DeviceType device,
  calibration::MultiCameraDataset& dataset,
  std::shared_ptr<SplatModel> model,
  std::shared_ptr<SplatModel> prev_model = nullptr,
  GaussianSplatGuiData* gui_data = nullptr,
  std::shared_ptr<std::atomic<bool>> cancel_requested = nullptr);

void calculatePsnrForDataset(
  SplatConfig& cfg,
  torch::DeviceType device,
  calibration::MultiCameraDataset& dataset,
  std::shared_ptr<SplatModel> model);

}}  // end namespace p11::splat
