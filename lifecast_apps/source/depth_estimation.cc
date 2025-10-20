// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "depth_estimation.h"

#include "logger.h"
#include "opencv2/optflow.hpp"
#include "third_party/turbo_colormap.h"

#include "util_opencv.h"

namespace p11 { namespace depth_estimation {
cv::Mat colorizeInvDepth(const cv::Mat& inv_depth)
{
  cv::Mat colorized(inv_depth.size(), CV_32FC3);
  for (int y = 0; y < colorized.rows; ++y) {
    for (int x = 0; x < colorized.cols; ++x) {
      const float id = inv_depth.at<float>(y, x);
      const Eigen::Vector3f color = p11::turbo_colormap::float01ToColor(id);
      colorized.at<cv::Vec3f>(y, x) = cv::Vec3f(color.x(), color.y(), color.z());
    }
  }
  return colorized;
}

cv::Mat disparityFromOpticalFlow(const cv::Mat& L_image, const cv::Mat& R_image)
{
  cv::Mat L_grey, R_grey;
  cv::cvtColor(L_image, L_grey, cv::COLOR_BGRA2GRAY);
  cv::cvtColor(R_image, R_grey, cv::COLOR_BGRA2GRAY);
  L_grey.convertTo(L_grey, CV_8U);
  R_grey.convertTo(R_grey, CV_8U);

  cv::Ptr<cv::DenseOpticalFlow> algorithm = cv::optflow::createOptFlow_DeepFlow();

  cv::Mat flow;
  algorithm->calc(R_grey, L_grey, flow);
  std::vector<cv::Mat> flow_components;
  cv::split(flow, flow_components);
  return flow_components[0];
}

}}  // namespace p11::depth_estimation
