// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <string>
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "logger.h"
#include "util_math.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace p11 { namespace calibration {

template <typename T>
struct RectilinearCamera {
  static constexpr int kIntrinsicDim = 1;

  using TVector2d = Eigen::Matrix<T, 2, 1>;
  using TVector3d = Eigen::Matrix<T, 3, 1>;
  using TMatrix3d = Eigen::Matrix<T, 3, 3>;
  using TIsometry3d = Eigen::Transform<T, 3, Eigen::Isometry>;

  std::string name; // used to keep track of which images/videos are associated with this camera model.
  int width, height;  // resolution
  TVector2d focal_length; // focal length can have different values for x and y but they are usually pretty close. In units of pixels (not mm).
  TVector2d optical_center;    // w/2, h/2 for perfect lens
  TIsometry3d cam_from_world;  // [R|t], transforms from world coordinates to camera coordinates
  float k1, k2; // TODO: these aren't fully supported, just a storage space for now

  // Setup for nested templatization over Camera type and T.
  template <typename U> struct rebind { using type = RectilinearCamera<U>; };

  RectilinearCamera()
  {
    cam_from_world.linear() = TMatrix3d::Identity();
    cam_from_world.translation() = TVector3d(0, 0, 0);
    k1 = 0;
    k2 = 0;
  }

  // copy construct for converting from double to T
  RectilinearCamera(const RectilinearCamera<double>& other)
  {
    name = other.name;
    width = other.width;
    height = other.height;
    focal_length = TVector2d(T(other.focal_length.x()), T(other.focal_length.y()));
    optical_center = TVector2d(T(other.optical_center.x()), T(other.optical_center.y()));
    k1 = other.k1;
    k2 = other.k2;
  
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
    // NOTE: it is -p_cam.y() here because pixel coordinates increase down the screen,
    // and we are using XYZ=RUF.
    TVector2d p_perspective(p_cam.x() / p_cam.z(), -p_cam.y() / p_cam.z());
    return p_perspective.cwiseProduct(focal_length) + optical_center;
  }

  // returns a ray direction in the camera's local coordinate frame.
  inline TVector3d rayDirFromPixel(const TVector2d& pixel) const
  {
    T x = (pixel.x() - optical_center.x()) / focal_length.x();
    //T y = (height - 1 - pixel.y() - optical_center.y()) / focal_length.y();
    T y = (height - pixel.y() - optical_center.y()) / focal_length.y();
    TVector3d dir(x, y, 1);
    return dir.normalized();
  }

  inline TVector2d pixelFromWorld(const TVector3d& p_world) const
  {
    return pixelFromCam(camFromWorld(p_world));
  }

  void resizeToWidth(const int new_width) {
    const float ratio = float(new_width) / float(width);
    width = new_width;
    height = float(height) * ratio;
    focal_length *= ratio;
    optical_center *= ratio;
  }

  std::vector<T> getIntrinsicParamVec() const
  {
    std::vector<T> params(kIntrinsicDim);
    params[0] = focal_length.x();
    // TODO: allow optimize more parameters
    return params;
  }

  void applyIntrinsicParamVec(const std::vector<T>& param)
  {
    XCHECK_EQ(param.size(), kIntrinsicDim);
    focal_length.x() = param[0];
    focal_length.y() = param[0];
    k1 = 0;
    k2 = 0;
  }

  void resizeToHeight(const int new_height) {
    const float ratio = float(new_height) / float(height);
    height = new_height;
    width = float(width) * ratio;
    focal_length *= ratio;
    optical_center *= ratio;
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

// Traits for matching all types of RectilinearCamera regardless of T
template<typename T> struct is_rectilinear_camera : std::false_type {};
template<typename T> struct is_rectilinear_camera<RectilinearCamera<T>> : std::true_type {};
template<typename T> inline constexpr bool is_rectilinear_camera_v = is_rectilinear_camera<T>::value;

using RectilinearCamerad = RectilinearCamera<double>;

static calibration::RectilinearCamerad guessRectilinearIntrinsics(int w, int h, float fov_deg = 120.0f) {
  calibration::RectilinearCamerad guess_intrinsics;
  guess_intrinsics.width = w;
  guess_intrinsics.height = h;
  guess_intrinsics.optical_center = Eigen::Vector2d(w/2.0, h/2.0);
  guess_intrinsics.k1 = 0;
  guess_intrinsics.k2 = 0;

  double fov_radians = fov_deg * M_PI / 180.0;
  double max_dim = std::max(w, h);
  double focal_length = max_dim / (2.0 * std::tan(fov_radians / 2.0));
  guess_intrinsics.focal_length.x() = focal_length;
  guess_intrinsics.focal_length.y() = focal_length;

  XPLINFO << "Guess focal_length [pixels]: " << focal_length;
  return guess_intrinsics;
}

}}  // namespace p11::calibration
