// MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#pragma once

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "logger.h"
#include "util_math.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace p11 { namespace calibration {

static constexpr int kIntrinsicDim = 4;

template <typename T>
struct FisheyeCamera {
  using TVector2d = Eigen::Matrix<T, 2, 1>;
  using TVector3d = Eigen::Matrix<T, 3, 1>;
  using TMatrix3d = Eigen::Matrix<T, 3, 3>;
  using TIsometry3d = Eigen::Transform<T, 3, Eigen::Isometry>;

  int width, height;  // resolution
  T radius_at_90;  // analogous to focal length, what number of pixels must we go away from center
                   // to be at 90 degree rotation away
  T k1, k2, k3;    // distortion
  bool is_inflated; // a hack for inflated equiangular projection. changes rayDirFromPixel to use rayDirFromPixelInflated

  TVector2d optical_center;    // w/2, h/2 for perfect lens
  TIsometry3d cam_from_world;  // [R|t], transforms from world coordinates to camera coordinates

  FisheyeCamera()
  {
    cam_from_world.linear() = TMatrix3d::Identity();
    cam_from_world.translation() = TVector3d(0, 0, 0);
    k1 = T(0.0);
    k2 = T(0.0);  // NOTE: k2 and k3 are not supported for inverse distortion. TODO: remove
    k3 = T(0.0);
    is_inflated = false;
  }

  // copy construct for converting from double to T
  FisheyeCamera(const FisheyeCamera<double>& other)
  {
    width = other.width;
    height = other.height;
    radius_at_90 = T(other.radius_at_90);
    k1 = T(other.k1);
    k2 = T(other.k2);
    k3 = T(other.k3);
    is_inflated = other.is_inflated;

    optical_center = TVector2d(T(other.optical_center.x()), T(other.optical_center.y()));

    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        cam_from_world.linear()(i, j) = T(other.cam_from_world.linear()(i, j));
      }
    }
    cam_from_world.translation() = TVector3d(
        T(other.cam_from_world.translation().x()),
        T(other.cam_from_world.translation().y()),
        T(other.cam_from_world.translation().z()));
  }

  TVector3d right() const { return cam_from_world.linear().row(0); }
  TVector3d up() const { return cam_from_world.linear().row(1); }
  TVector3d forward() const { return cam_from_world.linear().row(2); }

  void setPositionInWorld(const TVector3d& p_world)
  {
    cam_from_world.translation() = -cam_from_world.linear() * p_world;
  }

  TVector3d getPositionInWorld() const
  {
    return -cam_from_world.linear().transpose() * cam_from_world.translation();
  }

  TIsometry3d worldFromCam() const
  {
    TIsometry3d w_from_c;
    w_from_c.linear() = cam_from_world.linear().transpose();
    w_from_c.translation() = -cam_from_world.linear().transpose() * cam_from_world.translation();
    return w_from_c;
  }

  inline TVector3d worldFromCam(const TVector3d& p_cam) const { return worldFromCam() * p_cam; }

  inline TVector3d camFromWorld(const TVector3d& p_world) const { return cam_from_world * p_world; }

  inline TVector2d pixelFromCam(const TVector3d& p_cam) const
  {
    const T kEpsilon = T(1E-9);  // ceres doesn't like sqrt(0)
    const T phi = ceres::atan2(
        ceres::sqrt(kEpsilon + p_cam.x() * p_cam.x() + p_cam.y() * p_cam.y()), p_cam.z());
    const T phi2 = phi * phi;
    const T phi_dist = phi * (1.0 + phi2 * k1);
    const T theta = ceres::atan2(-p_cam.y(), p_cam.x());
    const T r = phi_dist * radius_at_90 / (T(M_PI) / 2.0);
    return r * TVector2d(ceres::cos(theta), ceres::sin(theta)) + optical_center;
  }

  // returns a ray direction in the camera's local coordinate frame.
  // references:
  // https://math.stackexchange.com/questions/692762/how-to-calculate-the-inverse-of-a-known-optical-distortion-function
  // http://paulbourke.net/dome/dualfish2sphere/
  inline TVector3d rayDirFromPixel(const TVector2d& pixel) const
  {
    if (is_inflated) return rayDirFromPixelInflated(pixel);

    const T theta = ceres::atan2(pixel.y() - optical_center.y(), pixel.x() - optical_center.x());

    const T phi = (pixel - optical_center).norm() * T(M_PI / 2.0) / radius_at_90;
    const T phi2 = phi * phi;
    const T phi4 = phi2 * phi2;
    const T phi6 = phi4 * phi2;

    // for an initial guess, use a series expansion
    T phi0 = phi * (1 - k1 * phi2 + 3.0 * k1 * k1 * phi4 - 12.0 * k1 * k1 * k1 * phi6);

    // do a few iterations of newton's method
    phi0 = phi0 - (phi0 * (1.0 + k1 * phi0 * phi0) - phi) / (1.0 + 3 * k1 * phi0 * phi0);
    phi0 = phi0 - (phi0 * (1.0 + k1 * phi0 * phi0) - phi) / (1.0 + 3 * k1 * phi0 * phi0);
    phi0 = phi0 - (phi0 * (1.0 + k1 * phi0 * phi0) - phi) / (1.0 + 3 * k1 * phi0 * phi0);

    return TVector3d(
        ceres::cos(theta) * ceres::sin(phi0),
        -ceres::sin(theta) * ceres::sin(phi0),
        ceres::cos(phi0));
  }

  // HACK: returns the ray direction assuming this fisheye camera is "inflated".
  Eigen::Vector3d rayDirFromPixelInflated(const Eigen::Vector2d& pixel) const
  {
    const double theta = std::atan2(pixel.y() - optical_center.y(), pixel.x() - optical_center.x());
    const double r = (pixel - optical_center).norm() / radius_at_90;
    const double r_inflated =
        0.5 * r + 0.5 * r * r * r;  // NOTE: r can be raised to different powers to control this.

    const double phi = r_inflated * (M_PI / 2.0);
    return TVector3d(
        std::cos(theta) * std::sin(phi), -std::sin(theta) * std::sin(phi), std::cos(phi));
  }

  inline TVector2d pixelFromWorld(const TVector3d& p_world) const
  {
    return pixelFromCam(camFromWorld(p_world));
  }

  std::vector<T> getIntrinsicParamVec() const
  {
    std::vector<T> params(kIntrinsicDim);
    params[0] = optical_center.x();
    params[1] = optical_center.y();
    params[2] = radius_at_90;
    params[3] = k1 * 100.0;  // TODO: hack for preconditioning
    // params[4] = k2 * 100.0;  // TODO: hack for preconditioning
    // params[5] = k3 * 100.0;  // TODO: hack for preconditioning
    return params;
  }

  void applyIntrinsicParamVec(const std::vector<T>& param)
  {
    XCHECK_EQ(param.size(), kIntrinsicDim);
    optical_center.x() = param[0];
    optical_center.y() = param[1];
    radius_at_90 = param[2];
    k1 = param[3] / 100.0;  // TODO: hack for preconditioning
    // k2 = param[4] / 100.0;  // TODO: hack for preconditioning
    // k3 = param[5] / 100.0;  // TODO: hack for preconditioning
  }
};

using FisheyeCamerad = FisheyeCamera<double>;

}}  // namespace p11::calibration
