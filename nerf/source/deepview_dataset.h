// MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#pragma once

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "rectilinear_camera.h"

namespace p11 { namespace nerf {

// Rectify images in the format of the DeepView dataset 
// https://github.com/augmentedperception/deepview_video_dataset
// to remove radial distortion and put it into a cannonical rectilinear projection.
// Returns a camera model for the rectified image. This isn't the same as the OpenCV
// distortion model, its really just for the DeepView dataset!
calibration::RectilinearCamerad precomputeDeepViewRectifyWarp(
  const calibration::RectilinearCamerad& src_cam,
  std::vector<cv::Mat>& warp_uv,
  const float focal_length_multiplier = 1.0 // avoid sampling missing pixels when remapping
);


}}  // end namespace p11::nerf
