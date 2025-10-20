// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
RAFT optical flow wrapper. See:

"RAFT: Recurrent All-Pairs Field Transforms for Optical Flow"
https://arxiv.org/abs/2003.12039

https://github.com/princeton-vl/RAFT
*/
#pragma once

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "torch/script.h"
#include <string>
#include "util_math.h"

namespace p11 { namespace optical_flow {

// Loads the weights/traced graph for the RAFT model.
void getTorchModelRAFT(torch::jit::script::Module& module, std::string model_path = "");

// Run RAFT optical flow estimation.
void computeOpticalFlowRAFT(
    torch::jit::script::Module& module,
    const cv::Mat& image1_8u,
    const cv::Mat& image2_8u,
    cv::Mat& flow_x,
    cv::Mat& flow_y);

// Use optical flow to compute disparity (just discard y flow and clamp x flow to positive values).
// bias can be added to artificatially adjust the disparity values by a constant (this might be
// useful for converting VR180 to 6DOF, when the input video doesn't have great calibration).
cv::Mat computeDisparityRAFT(
    torch::jit::script::Module& module,
    const cv::Mat& image1_8u,
    const cv::Mat& image2_8u,
    const float bias = 0);

// Compute flow from left to right and right to left. R_error and L_error are scaled to [0, 1] with
// 0 being high confidence and 1 low confidence
void computeDisparityRAFTBothWays(
    torch::jit::script::Module& module,
    const cv::Mat& R_image,
    const cv::Mat& L_image,
    const float bias,
    cv::Mat& R_disparity,
    cv::Mat& L_disparity,
    cv::Mat& R_error,
    cv::Mat& L_error);

}}  // namespace p11::optical_flow
