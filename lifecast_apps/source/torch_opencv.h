// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// Functions for going back and forth between torch and opencv
#pragma once

#include "check.h"
#include "logger.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "torch/torch.h"

namespace p11 { namespace torch_opencv {

torch::Tensor cvMat8UC3_to_Tensor(const torch::DeviceType device, cv::Mat image);
torch::Tensor cvMat8UC1_to_Tensor(const torch::DeviceType device, cv::Mat image);
torch::Tensor cvMat_to_Tensor(const torch::DeviceType device, const cv::Mat& image);

void fastTensor_To_CvMat(torch::Tensor image_tensor, cv::Mat& dest);

}}  // namespace p11::torch_opencv
