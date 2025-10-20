// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "projection.h"

#include "logger.h"
#include "fisheye_camera.h"
#include "util_math.h"
#include "util_opencv.h"

namespace p11 { namespace projection {

cv::Mat warp(
    const cv::Mat& src, const std::vector<cv::Mat>& warp_uv, cv::InterpolationFlags interp_mode)
{
  cv::Mat output_image;
  cv::remap(
      src,
      output_image,
      warp_uv[0],
      warp_uv[1],
      interp_mode,
      cv::BORDER_CONSTANT,
      cv::Scalar(0, 0, 0, 0));
  return output_image;
}

std::vector<cv::Mat> composeWarps(
    const std::vector<cv::Mat>& warp1,
    const std::vector<cv::Mat>& warp2
) {
  cv::Mat map1, map2;
  cv::remap(warp2[0], map1, warp1[0], warp1[1], cv::INTER_LINEAR, cv::BORDER_REPLICATE);
  cv::remap(warp2[1], map2, warp1[0], warp1[1], cv::INTER_LINEAR, cv::BORDER_REPLICATE);

  std::vector<cv::Mat> composed_warp = {map1, map2};
  return composed_warp;
}

Eigen::Vector2d camPoint3dToRectifiedPixel(
    const Eigen::Vector3d& cam_point, const int rectified_width, const int rectified_height)
{
  const double psi =
      atan2(cam_point.x(), sqrt(cam_point.z() * cam_point.z() + cam_point.y() * cam_point.y()));
  const double beta = atan2(cam_point.z(), cam_point.y());

  const double x = psi * rectified_width / M_PI + float(rectified_width) / 2.0;
  const double y = beta * rectified_height / M_PI;
  return Eigen::Vector2d(x, y);
}

Eigen::Vector3d rectifiedPixelToRayDirection(
    const Eigen::Vector2d& pixel, const int rectified_width, const int rectified_height)
{
  const float x_prime = M_PI * (pixel.x() - float(rectified_width) / 2.0) / rectified_width;
  const float y_prime = -M_PI * (pixel.y() - float(rectified_height) / 2.0) / rectified_height;
  const float px = sin(x_prime);
  const float py = cos(x_prime) * sin(y_prime);
  const float pz = cos(x_prime) * cos(y_prime);
  return Eigen::Vector3d(px, py, pz);
}

Eigen::Vector2d rectifiedPixelToCamPixel(
    const int rectified_width,
    const int rectified_height,
    const calibration::FisheyeCamerad& cam,
    const float x,
    const float y)
{
  const Eigen::Vector3d ray_dir =
      rectifiedPixelToRayDirection(Eigen::Vector2d(x, y), rectified_width, rectified_height);
  const Eigen::Vector3d p_cam = cam.camFromWorld(ray_dir);
  return cam.pixelFromCam(p_cam);
}

void precomputeFisheyeToRectifyWarp(
    const int rectified_width,
    const int rectified_height,
    const calibration::FisheyeCamerad& cam,
    std::vector<cv::Mat>& warp_uv)
{
  cv::Mat warp(cv::Size(rectified_width, rectified_height), CV_32FC2);
  for (int y = 0; y < rectified_height; ++y) {
    for (int x = 0; x < rectified_width; ++x) {
      const Eigen::Vector2d cam_pixel =
          rectifiedPixelToCamPixel(rectified_width, rectified_height, cam, x, y);
      warp.at<cv::Vec2f>(y, x) = cv::Vec2f(cam_pixel.x(), cam_pixel.y());
    }
  }
  cv::split(warp, warp_uv);
}

void precomputeRectifiedToFisheyeWarp(
    const int rectified_width,
    const int rectified_height,
    const cv::Size& fisheye_size,
    const calibration::FisheyeCamerad& cam,
    std::vector<cv::Mat>& warp_uv,
    float ftheta_scale)
{
  calibration::FisheyeCamerad cam_scaled(cam);
  cam_scaled.radius_at_90 *= ftheta_scale;
  cv::Mat warp(fisheye_size, CV_32FC2);
  for (int y = 0; y < fisheye_size.height; ++y) {
    for (int x = 0; x < fisheye_size.width; ++x) {
      // inverted distortion is not well behaved far outside the image circle, so skip these pixels
      // TODO: this fudge margin might be OK for low distortion, but if k1 is large such that
      // radius_at_90 isn't very close to the truth, this could break it.
      constexpr double kFudgeMargin = 1.1;
      if ((Eigen::Vector2d(x, y) - cam_scaled.optical_center).norm() >
          cam_scaled.radius_at_90 * kFudgeMargin) {
        warp.at<cv::Vec2f>(y, x) = cv::Vec2f(-123, -456);
      } else {
        const Eigen::Vector3d dir_in_cam = cam_scaled.rayDirFromPixel(Eigen::Vector2d(x, y));
        const Eigen::Vector3d dir_in_rectified = cam_scaled.worldFromCam(dir_in_cam);
        const Eigen::Vector2d point_in_rectified =
            camPoint3dToRectifiedPixel(dir_in_rectified, rectified_width, rectified_height);
        warp.at<cv::Vec2f>(y, x) = cv::Vec2f(point_in_rectified.x(), point_in_rectified.y());
      }
    }
  }
  cv::split(warp, warp_uv);
}

void precomputeFisheyeToFisheyeWarp(
    const calibration::FisheyeCamerad& src_cam,
    const calibration::FisheyeCamerad& dest_cam,
    std::vector<cv::Mat>& warp_uv,
    float dest_ftheta_scale)
{
  calibration::FisheyeCamerad dest_cam_scaled(dest_cam);
  dest_cam_scaled.radius_at_90 *= dest_ftheta_scale;
  cv::Mat warp(cv::Size(dest_cam.width, dest_cam.height), CV_32FC2);
  for (int y = 0; y < warp.rows; ++y) {
    for (int x = 0; x < warp.cols; ++x) {
      // inverted distortion is not well behaved far outside the image circle, so skip these pixels
      // TODO: this fudge margin might be OK for low distortion, but if k1 is large such that
      // radius_at_90 isn't very close to the truth
      constexpr double kFudgeMargin = 1.1;
      if ((Eigen::Vector2d(x, y) - dest_cam_scaled.optical_center).norm() >
          dest_cam_scaled.radius_at_90 * kFudgeMargin) {
        warp.at<cv::Vec2f>(y, x) = cv::Vec2f(-123, -456);
      } else {
        const Eigen::Vector3d dir_in_dest = dest_cam_scaled.rayDirFromPixel(Eigen::Vector2d(x, y));
        const Eigen::Vector2d point_in_src = src_cam.pixelFromWorld(dir_in_dest);
        warp.at<cv::Vec2f>(y, x) = cv::Vec2f(point_in_src.x(), point_in_src.y());
      }
    }
  }
  cv::split(warp, warp_uv);
}

void precomputeFisheyeToInflatedWarp(
    calibration::FisheyeCamerad& src_cam,
    calibration::FisheyeCamerad& dest_cam,
    std::vector<cv::Mat>& warp_uv,
    double inflate_exponent)
{
  cv::Mat warp(cv::Size(dest_cam.width, dest_cam.height), CV_32FC2);

  for (int y = 0; y < warp.rows; ++y) {
    for (int x = 0; x < warp.cols; ++x) {
      const Eigen::Vector3d dir_in_dest = dest_cam.rayDirFromPixelInflated(Eigen::Vector2d(x, y), inflate_exponent);
      const Eigen::Vector2d point_in_src = src_cam.pixelFromWorld(dir_in_dest);
      warp.at<cv::Vec2f>(y, x) = cv::Vec2f(point_in_src.x(), point_in_src.y());
    }
  }

  cv::split(warp, warp_uv);
}

double disparityToDepth(
    const int depthmap_size,
    const double baseline,
    const int R_x,
    const int R_y,
    const float disparity)
{
  const float L_x = R_x + disparity;
  const float L_phi = (M_PI - M_PI * float(L_x) / float(depthmap_size)) - M_PI / 2.0;
  const float R_phi = (M_PI - M_PI * float(R_x) / float(depthmap_size)) - M_PI / 2.0;
  return baseline * (cos(L_phi) / sin(R_phi - L_phi));
}

cv::Mat disparityToInvDepth(const cv::Mat& disparity, const double baseline)
{
  XCHECK_EQ(disparity.rows, disparity.cols);
  const int depthmap_size = disparity.cols;

  cv::Mat inv_depth(depthmap_size, depthmap_size, CV_32F);
  for (int R_y = 0; R_y < depthmap_size; ++R_y) {
    for (int R_x = 0; R_x < depthmap_size; ++R_x) {
      const double depth =
          disparityToDepth(depthmap_size, baseline, R_x, R_y, disparity.at<float>(R_y, R_x));
      inv_depth.at<float>(R_y, R_x) = p11::math::clamp<float>(float(1.0 / depth), 0.001f, 10000.0f);
    }
  }
  return inv_depth;
}

void makePointCloudFromDepthmap(
    const cv::Mat& inv_depthmap,
    const cv::Mat& image,
    std::vector<Eigen::Vector3f>& point_cloud,
    std::vector<Eigen::Vector3f>& point_cloud_colors)
{
  const int depthmap_size = inv_depthmap.cols;
  XCHECK_EQ(depthmap_size, inv_depthmap.rows);
  XCHECK_EQ(depthmap_size, image.cols);
  XCHECK_EQ(depthmap_size, image.rows);

  for (int y = 0; y < depthmap_size; ++y) {
    for (int x = 0; x < depthmap_size; ++x) {
      const cv::Vec4f bgra = image.at<cv::Vec4f>(y, x);
      if (bgra[3] > 0.99) {
        const Eigen::Vector3d ray_dir =
            rectifiedPixelToRayDirection(Eigen::Vector2d(x, y), depthmap_size, depthmap_size);

        Eigen::Vector3f point3d = ray_dir.cast<float>() / inv_depthmap.at<float>(y, x);

        const Eigen::Vector3f color(bgra[2], bgra[1], bgra[0]);

        point_cloud.push_back(point3d);
        point_cloud_colors.push_back(color);
      }
    }
  }
}

void precomputeremapFisheyeToEquirectWarp(
    const int eqr_width,
    const int eqr_height,
    const calibration::FisheyeCamerad& cam,
    std::vector<cv::Mat>& warp_uv)
{
  cv::Mat warp(cv::Size(eqr_width, eqr_height), CV_32FC2);
  for (int y = 0; y < eqr_height; ++y) {
    for (int x = 0; x < eqr_width; ++x) {
      const float lon = -M_PI * (float(x) / float(eqr_width) - 0.5) + M_PI / 2;
      const float lat = M_PI * (float(y) / float(eqr_height) - 0.5);
      const float px = cos(lat) * cos(lon);
      const float py = -sin(lat);
      const float pz = cos(lat) * sin(lon);

      const Eigen::Vector3d p_cam = cam.camFromWorld(Eigen::Vector3d(px, py, pz));
      const Eigen::Vector2d pixel = cam.pixelFromCam(p_cam);
      warp.at<cv::Vec2f>(y, x) = cv::Vec2f(pixel.x(), pixel.y());
    }
  }
  cv::split(warp, warp_uv);
}

void computeFisheyeTo360EquirectWarp(
    const int eqr_width,
    const int eqr_height,
    const calibration::FisheyeCamerad& cam,
    std::vector<cv::Mat>& warp_uv)
{
  cv::Mat warp(cv::Size(eqr_width, eqr_height), CV_32FC2);
  for (int y = 0; y < eqr_height; ++y) {
    for (int x = 0; x < eqr_width; ++x) {
      const float lon = -2.0 * M_PI * float(x) / float(eqr_width) - M_PI / 2.0;
      const float lat = M_PI * (float(y) / float(eqr_height) - 0.5);
      const float px = cos(lat) * cos(lon);
      const float py = -sin(lat);
      const float pz = cos(lat) * sin(lon);
      const Eigen::Vector3d p_cam = cam.camFromWorld(Eigen::Vector3d(px, py, pz));
      const Eigen::Vector2d pixel = cam.pixelFromCam(p_cam);
      warp.at<cv::Vec2f>(y, x) = cv::Vec2f(pixel.x(), pixel.y());
    }
  }
  cv::split(warp, warp_uv);
}

void precomputeVR180toFthetaWarp(
    const calibration::FisheyeCamerad& cam_perfect_ftheta,
    const int ftheta_size,
    const int eqr_size,
    std::vector<cv::Mat>& warp_uv,
    const double ftheta_scale)
{
  calibration::FisheyeCamerad cam_scaled(cam_perfect_ftheta);
  cam_scaled.radius_at_90 *= ftheta_scale;

  float rmax = ftheta_scale * ftheta_size / 2.0;
  cv::Mat warp(cv::Size(ftheta_size, ftheta_size), CV_32FC2);
  for (int y = 0; y < ftheta_size; ++y) {
    for (int x = 0; x < ftheta_size; ++x) {
      int dx = x - ftheta_size / 2;
      int dy = y - ftheta_size / 2;
      if (dx * dx + dy * dy > rmax * rmax) {
        warp.at<cv::Vec2f>(y, x) = cv::Vec2f(-1, -1);
        continue;
      }
      Eigen::Vector3d r = cam_scaled.rayDirFromPixel(Eigen::Vector2d(x, y));
      Eigen::Vector3d ray_dir(r.z(), r.x(), -r.y());
      const float u = 0.5 + atan2(ray_dir.y(), ray_dir.x()) / M_PI;
      const float v =
          0.5 +
          atan2(ray_dir.z(), sqrt(ray_dir.x() * ray_dir.x() + ray_dir.y() * ray_dir.y())) / M_PI;
      warp.at<cv::Vec2f>(y, x) = cv::Vec2f(u * eqr_size, v * eqr_size);
    }
  }
  cv::split(warp, warp_uv);
}

calibration::FisheyeCamerad makePerfectFthetaCamera(const int image_size)
{
  calibration::FisheyeCamerad cam_perfect_ftheta;
  cam_perfect_ftheta.k1 = 0;  // For perfect f-theta, there is no distortion.
  cam_perfect_ftheta.width = image_size;
  cam_perfect_ftheta.height = image_size;
  cam_perfect_ftheta.radius_at_90 = image_size / 2.0;
  cam_perfect_ftheta.optical_center = Eigen::Vector2d(image_size / 2.0, image_size / 2.0);
  return cam_perfect_ftheta;
}

}}  // namespace p11::projection
