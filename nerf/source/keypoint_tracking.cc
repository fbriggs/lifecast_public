// MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include "keypoint_tracking.h"

#include "logger.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/optflow.hpp"
#include "opencv2/video/tracking.hpp"

namespace p11 { namespace keypoint_tracking {
std::vector<cv::Point2f> findKeypointsFAST(const cv::Mat& image, const int threshold)
{
  constexpr bool kNonMaxSupress = true;

  std::vector<cv::KeyPoint> keypoints;
  std::vector<cv::Point2f> detection_coords;
  cv::FAST(image, keypoints, threshold, kNonMaxSupress);
  cv::KeyPoint::convert(keypoints, detection_coords, std::vector<int>());
  return detection_coords;
}

void trackKeyPointsFromSrcToDest(
    const cv::Mat& src_gray,
    const cv::Mat& dest_gray,
    const std::vector<cv::Point2f>& points_in_src,
    std::vector<cv::Point2f>& points_in_dest,
    std::vector<bool>& valid_mask)
{
  if (points_in_src.empty()) {
    return;
  }  // flow will crash with 0 input

  std::vector<uchar> status;
  std::vector<float> err;
  calcOpticalFlowPyrLK(
      src_gray,
      dest_gray,
      points_in_src,
      points_in_dest,
      status,
      err,
      cv::Size(21, 21),
      3,
      cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30.0, 0.01),
      0,
      0.001);

  XCHECK_EQ(points_in_src.size(), status.size());
  valid_mask.resize(status.size());
  for (int i = 0; i < status.size(); ++i) {
    valid_mask[i] = bool(status[i]);
    // TODO: check err as well to get better filtering?
  }
}

}}  // namespace p11::keypoint_tracking
