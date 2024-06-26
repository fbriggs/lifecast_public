// MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#pragma once

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "fisheye_camera.h"

namespace p11 { namespace ldi {

// format can be "8bit", "16bit", or "split12".
// If warp_ftheta_to_inflated is empty, we assume the inputs are already in inflated projection.
cv::Mat make6DofGrid(
    const std::vector<cv::Mat>& layer_bgra,
    const std::vector<cv::Mat>& layer_invd,
    const std::string format,
    const std::vector<cv::Mat>& warp_ftheta_to_inflated,
    const bool dilate_invd = true);

// Mostly a tool for debugging. Merge all of the layers down to compare this with the original
// image, to detect some kinds of artifacts.
void fuseLayers(
    const std::vector<cv::Mat>& layer_bgra,
    const std::vector<cv::Mat>& layer_invd,
    cv::Mat& bgra,
    cv::Mat& invd);

// Decode the 12 bit format into a 32_F depthmap.
cv::Mat unpackLDI3_12BitDepth(
    const cv::Mat& ldi3_8bit, const cv::Rect rect_lo, const cv::Rect rect_hi);

// Unpack an 8 bit LDI3 format image into its channels (32F format).
// Decode the 12 bit depth code as needed.
void unpackLDI3(
    const cv::Mat& ldi3_8bit, std::vector<cv::Mat>& layer_bgra, std::vector<cv::Mat>& layer_invd);

}}  // namespace p11::ldi
