// MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include "deepview_dataset.h"

#include "Eigen/Core"
#include "Eigen/Geometry"

namespace p11 { namespace nerf {

Eigen::Vector2d deepViewFisheyeToPerspective(
  const calibration::RectilinearCamerad& src_cam,
  const Eigen::Vector3d& point
) {
  const float r = std::sqrt(point.x() * point.x() + point.y() * point.y());
  const float theta = std::atan2(r, point.z());
  const float r2 = theta * theta;
  const float distortion = 1.0 + r2 * (src_cam.k1 + r2 * src_cam.k2);
  const Eigen::Vector2d undistorted(
    theta / r * point.x() * distortion,
    theta / r * point.y() * distortion);
  const Eigen::Vector2d projected_pixel(
    undistorted.x() * src_cam.focal_length.x() + src_cam.optical_center.x(),
    undistorted.y() * src_cam.focal_length.y() + src_cam.optical_center.y());
  return projected_pixel;
}

calibration::RectilinearCamerad precomputeDeepViewRectifyWarp(
  const calibration::RectilinearCamerad& src_cam,
  std::vector<cv::Mat>& warp_uv,
  const float focal_length_multiplier
) {
  calibration::RectilinearCamerad new_cam(src_cam);
  new_cam.focal_length *= focal_length_multiplier;
  new_cam.k1 = 0;
  new_cam.k2 = 0;
  new_cam.optical_center = Eigen::Vector2d(new_cam.width/2, new_cam.height/2);

  cv::Mat warp(cv::Size(new_cam.width, new_cam.height), CV_32FC2);
  for (int y = 0; y < new_cam.height; ++y) {
    for (int x = 0; x < new_cam.width; ++x) {
      Eigen::Vector3d ray_dir(
        (x - new_cam.optical_center.x()) / new_cam.focal_length.x(),
        (y - new_cam.optical_center.y()) / new_cam.focal_length.y(),
        1);
      ray_dir.normalize();

      const Eigen::Vector2d projected_pixel = deepViewFisheyeToPerspective(src_cam, ray_dir);
      warp.at<cv::Vec2f>(y, x) = cv::Vec2f(projected_pixel.x(), projected_pixel.y());
    }
  }
  cv::split(warp, warp_uv);
  return new_cam;
}

}}  // end namespace p11::nerf

