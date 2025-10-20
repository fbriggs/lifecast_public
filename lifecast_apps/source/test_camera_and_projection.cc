// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "fisheye_camera.h"
#include "logger.h"
#include "gtest/gtest.h"
#include "projection.h"

namespace p11 { namespace calibration { namespace {

constexpr double kPixelTolerance = 0.01;

FisheyeCamerad getTestCamera()
{
  FisheyeCamerad cam;
  cam.width = 301;
  cam.height = 101;
  cam.radius_at_90 = 50;
  cam.optical_center = Eigen::Vector2d(150, 50);
  cam.k1 = 0;
  cam.tilt = 0;
  cam.cam_from_world = Eigen::Isometry3d();
  return cam;
}

TEST(FisheyeCamera, pixelFromCam)
{
  const FisheyeCamerad cam = getTestCamera();

  const Eigen::Vector3d right(1, 0, 0);
  const Eigen::Vector3d up(0, 1, 0);
  const Eigen::Vector3d forward(0, 0, 1);

  const Eigen::Vector2d fwd_pixel = cam.pixelFromCam(forward);
  EXPECT_LT((fwd_pixel - cam.optical_center).norm(), kPixelTolerance);

  const Eigen::Vector2d right_pixel = cam.pixelFromCam(right);
  const Eigen::Vector2d expected_right_pixel =
      cam.optical_center + Eigen::Vector2d(cam.radius_at_90, 0);
  EXPECT_LT((right_pixel - expected_right_pixel).norm(), kPixelTolerance);

  const Eigen::Vector2d up_pixel = cam.pixelFromCam(up);
  const Eigen::Vector2d expected_up_pixel =
      cam.optical_center + Eigen::Vector2d(0, -cam.radius_at_90);
  EXPECT_LT((up_pixel - expected_up_pixel).norm(), kPixelTolerance);
}

TEST(FisheyeCamera, pixelFromCam_is_rayDirFromPixel_inverse)
{
  FisheyeCamerad cam = getTestCamera();
  cam.k1 = 0.05;
  cam.tilt = -0.01;

  const Eigen::Vector2d test_pixel1(162, 35);
  const Eigen::Vector2d test_pixel2(150, 50);
  const Eigen::Vector2d test_pixel3(200, 50);
  const Eigen::Vector2d test_pixel4(150, 0);
  EXPECT_LT(
      (test_pixel1 - cam.pixelFromCam(cam.rayDirFromPixel(test_pixel1))).norm(), kPixelTolerance);
  EXPECT_LT(
      (test_pixel2 - cam.pixelFromCam(cam.rayDirFromPixel(test_pixel2))).norm(), kPixelTolerance);
  EXPECT_LT(
      (test_pixel3 - cam.pixelFromCam(cam.rayDirFromPixel(test_pixel3))).norm(), kPixelTolerance);
  EXPECT_LT(
      (test_pixel4 - cam.pixelFromCam(cam.rayDirFromPixel(test_pixel4))).norm(), kPixelTolerance);
}

}}}  // namespace p11::calibration::