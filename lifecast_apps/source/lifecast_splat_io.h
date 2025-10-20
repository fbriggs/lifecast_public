// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <memory>
#include <vector>
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/photo.hpp"
#include "torch/torch.h"
#include "logger.h"
#include "util_math.h"
#include "util_torch.h"
#include "lifecast_splat_config.h"

namespace p11 { namespace splat {

struct SerializableSplat {
  Eigen::Vector3f pos, scale;
  Eigen::Vector4f color, quat;
};

cv::Mat encodeSplatsInImage(
  const std::vector<SerializableSplat>& splats,
  const int w,
  const int h);

cv::Mat encodeSerializableSplatsWithSizzleZ(std::vector<SerializableSplat>& splats, int w, int h);
cv::Mat encodeSplatModelWithSizzleZ(std::shared_ptr<SplatModel> model, int w = 4096, int h = 2048);
void saveEncodedSplatFileWithSizzleZ(const std::string& png_filename, std::shared_ptr<SplatModel> model, int w = 4096, int h = 2048);


std::vector<SerializableSplat> loadSplatPLYFile(const std::string& filename);

// load the format created by encodeSplatsInImage
std::vector<SerializableSplat> decodeSplatImage(const cv::Mat& image);
std::vector<SerializableSplat> loadSplatImageFile(const std::string& filename);

std::shared_ptr<SplatModel> serializableSplatsToModel(
  torch::DeviceType device,
  const std::vector<SerializableSplat>& splats);

std::vector<SerializableSplat> splatModelToSerializable(std::shared_ptr<SplatModel> model);


}}  // end namespace p11::splat
