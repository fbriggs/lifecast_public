// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "logger.h"
#include "util_math.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace p11 { namespace calibration {

template <typename T>
struct FisheyeCamera {
  static constexpr int kIntrinsicDim = 5;
  
  using TVector2d = Eigen::Matrix<T, 2, 1>;
  using TVector3d = Eigen::Matrix<T, 3, 1>;
  using TMatrix3d = Eigen::Matrix<T, 3, 3>;
  using TIsometry3d = Eigen::Transform<T, 3, Eigen::Isometry>;

  std::string name;
  int width, height;  // resolution
  T radius_at_90;  // analogous to focal length, what number of pixels must we go away from center
                   // to be at 90 degree rotation away
  T k1;   // distortion
  T tilt; // how much like an oval it is (0 = perfect circle). this can happen if the lens is tilted relative to the image sensor
  bool is_inflated; // a hack for inflated equiangular projection. changes rayDirFromPixel to use rayDirFromPixelInflated

  TVector2d optical_center;    // w/2, h/2 for perfect lens
  TIsometry3d cam_from_world;  // [R|t], transforms from world coordinates to camera coordinates

  double useable_radius;

  // Setup for nested templatization over Camera type and T.
  template <typename U> struct rebind { using type = FisheyeCamera<U>; };

  FisheyeCamera()
  {
    cam_from_world.linear() = TMatrix3d::Identity();
    cam_from_world.translation() = TVector3d(0, 0, 0);
    k1 = T(0.0);
    tilt = T(0.0);
    is_inflated = false;
    useable_radius = 0;
  }

  // copy construct for converting from double to T
  FisheyeCamera(const FisheyeCamera<double>& other)
  {
    name = other.name;
    width = other.width;
    height = other.height;
    radius_at_90 = T(other.radius_at_90);
    k1 = T(other.k1);
    tilt = T(other.tilt);
    is_inflated = other.is_inflated;
    useable_radius = other.useable_radius;

    optical_center = TVector2d(T(other.optical_center.x()), T(other.optical_center.y()));

    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        cam_from_world.matrix()(i, j) = T(other.cam_from_world.matrix()(i, j));
      }
    }
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
    const T r = phi_dist * radius_at_90 / (T(M_PI) / 2.0);
    
    // Version 1: use atan2
    //const T theta = ceres::atan2(-p_cam.y(), p_cam.x());
    //return r * TVector2d(ceres::cos(theta), ceres::sin(theta)) + optical_center;

    // Version 2: no atan2, no tilt
    //const T norm = ceres::sqrt(kEpsilon + p_cam.x() * p_cam.x() + p_cam.y() * p_cam.y());
    //return r * TVector2d(p_cam.x() / norm, -p_cam.y() / norm) + optical_center;

    // Version 3: no atan2, with tilt
    const T norm = ceres::sqrt(kEpsilon + p_cam.x() * p_cam.x() + p_cam.y() * p_cam.y());
    return r * TVector2d((1.0 + tilt) * p_cam.x() / norm, -p_cam.y() / norm) + optical_center;
  }

  // returns a ray direction in the camera's local coordinate frame.
  // references:
  // https://math.stackexchange.com/questions/692762/how-to-calculate-the-inverse-of-a-known-optical-distortion-function
  // http://paulbourke.net/dome/dualfish2sphere/
  inline TVector3d rayDirFromPixel(const TVector2d& pixel) const
  {
    if (is_inflated) return rayDirFromPixelInflated(pixel);

    // Calculation of theta and phi without tilt
    //const T theta = ceres::atan2(pixel.y() - optical_center.y(), pixel.x() - optical_center.x());
    //const T phi = (pixel - optical_center).norm() * T(M_PI / 2.0) / radius_at_90;

    // With tilt:
    const T dx = (pixel.x() - optical_center.x()) / (1.0 + tilt);
    const T dy = pixel.y() - optical_center.y();
    const T theta = ceres::atan2(dy, dx);
    const T kEpsilon = T(1E-9);  // ceres doesn't like sqrt(0)
    const T norm = ceres::sqrt(dx * dx + dy * dy + kEpsilon);
    const T phi = norm * T(M_PI / 2.0) / radius_at_90;

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
  Eigen::Vector3d rayDirFromPixelInflated(const Eigen::Vector2d& pixel, double inflate_exponent = 3.0) const
  {
    XCHECK_EQ(tilt, 0.0) << "Inflated cameras are symmetrical only";

    const double theta = std::atan2(pixel.y() - optical_center.y(), pixel.x() - optical_center.x());
    const double r = (pixel - optical_center).norm() / radius_at_90;
    const double r_inflated =
        //0.5 * r + 0.5 * r * r * r;
        0.5 * r + 0.5 * std::pow(r, inflate_exponent);

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
    params[4] = tilt;
    return params;
  }

  void applyIntrinsicParamVec(const std::vector<T>& param)
  {
    XCHECK_EQ(param.size(), kIntrinsicDim);
    optical_center.x() = param[0];
    optical_center.y() = param[1];
    radius_at_90 = param[2];
    k1 = param[3] / 100.0;  // TODO: hack for preconditioning
    tilt = param[4];
  }

  void resizeToWidth(const int new_width) {
    const float ratio = float(new_width) / float(width);
    width = new_width;
    height = float(height) * ratio;
    radius_at_90 *= ratio;
    optical_center *= ratio;
    useable_radius *= ratio;
  }

  void resizeToHeight(const int new_height) {
    const float ratio = float(new_height) / float(height);
    height = new_height;
    width = float(width) * ratio;
    radius_at_90 *= ratio;
    optical_center *= ratio;
    useable_radius *= ratio;
  }

  void resizeToMaxDim(const int new_max) {
    if (width > height) {
      resizeToWidth(new_max);
    } else {
      resizeToHeight(new_max);
    }
  }

  // Getters: these are mostly just for compatibility with NerfKludgeCamera
  int getWidth() const { return width; }
  int getHeight() const { return height; }
  TIsometry3d camFromWorld() const { return cam_from_world; }
};

// Traits for matching all types of FisheyeCamera regardless of T
template<typename T> struct is_fisheye_camera : std::false_type {};
template<typename T> struct is_fisheye_camera<FisheyeCamera<T>> : std::true_type {};
template<typename T> inline constexpr bool is_fisheye_camera_v = is_fisheye_camera<T>::value;

using FisheyeCamerad = FisheyeCamera<double>;

static calibration::FisheyeCamerad guessGoProIntrinsics(int w, int h) {
  // Values determined from calibration
  calibration::FisheyeCamerad guess_intrinsics;
  guess_intrinsics.width = 3840;
  guess_intrinsics.height = 3360;
  guess_intrinsics.k1 = 0.06001369441859313;
  guess_intrinsics.tilt = 0.0010783222019413348;
  guess_intrinsics.radius_at_90 = 2662.8540299512697;
  guess_intrinsics.useable_radius = 100000;
  guess_intrinsics.resizeToWidth(w);
  guess_intrinsics.height = h;
  guess_intrinsics.optical_center = Eigen::Vector2d(w/2.0, h/2.0);
  return guess_intrinsics;
}

}}  // namespace p11::calibration
