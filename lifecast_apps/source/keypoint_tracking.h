// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <vector>
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "third_party/json.h"

namespace p11 { namespace keypoint_tracking {
// uses FAST algorithm to find keypoints. we dont care about the descriptors for now, just the pixel
// detection coordinate.
std::vector<cv::Point2f> findKeypointsFAST(const cv::Mat& image, const int threshold = 15);

// track keypoints in source image to their corresponding position in destination image
// a result is returned for every input point, but some might not ve valid. this is encoded in
// valid_mask.
void trackKeyPointsFromSrcToDest(
    const cv::Mat& src_gray,
    const cv::Mat& dest_gray,
    const std::vector<cv::Point2f>& points_in_src,
    std::vector<cv::Point2f>& points_in_dest,
    std::vector<bool>& valid_mask);

nlohmann::json matchesToJson(
  const std::vector<std::string>& image_filenames,
  const std::vector<cv::Mat>& images,  // Added images parameter
  const std::vector<std::vector<cv::KeyPoint>>& image_to_keypoints,
  const std::vector<std::vector<cv::DMatch>>& image_to_matches);

void matchesFromJson(
  const nlohmann::json& j,
  std::vector<std::string>& image_filenames,
  std::vector<std::vector<cv::KeyPoint>>& image_to_keypoints,
  std::vector<std::vector<cv::DMatch>>& image_to_matches);

}}  // namespace p11::keypoint_tracking
