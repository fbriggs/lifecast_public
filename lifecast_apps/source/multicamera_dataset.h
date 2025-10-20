// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "torch/script.h"
#include "torch/torch.h"
#include "rectilinear_camera.h"
#include "fisheye_camera.h"
#include "nerf_kludge_camera.h"

namespace p11 { namespace calibration {

struct MultiCameraDataset {
  // Required
  std::vector<calibration::NerfKludgeCamera> cameras;
  std::vector<cv::Mat> images;
  // Optional
  std::vector<cv::Mat> depthmaps;
  std::vector<torch::Tensor> depthmap_tensors;
  std::vector<torch::Tensor> image_tensors;
  std::vector<torch::Tensor> ignore_masks;
  std::vector<std::string> image_filenames;
};

std::vector<calibration::NerfKludgeCamera> readDatasetCameraJson(const std::string& json_path);

void createEmptyTimeOffsetJson(const std::string& path, const std::vector<std::string> camera_names);

void readTimeOffsetJson(
  const std::string& time_offset_json_path,
  std::vector<calibration::NerfKludgeCamera>& cameras);

std::map<std::string, int> readTimeOffsetJsonAsMap(const std::string& time_offset_json_path);

MultiCameraDataset readDataset(
  const std::string& images_dir,
  const std::string& json_path,
  const torch::DeviceType device,
  const int resize_max_dim, // if 0, don't resize
  bool load_depthmaps = true
);

torch::Tensor cvMat8UC3_to_Tensor(const torch::DeviceType device, cv::Mat image);

}}  // end namespace p11::calibration
