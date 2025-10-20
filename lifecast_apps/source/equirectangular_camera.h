// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "logger.h"
#include "util_math.h"

namespace p11 { namespace calibration {

struct EquirectangularCamera {
  int width, height;  // resolution
  bool is_180; // 180 degree FOV if true, otherwise 360

  Eigen::Isometry3d cam_from_world;  // [R|t], transforms from world coordinates to camera coordinates

  EquirectangularCamera()
  {
    is_180 = false;
    cam_from_world.linear() = Eigen::Matrix3d::Identity();
    cam_from_world.translation() = Eigen::Vector3d(0, 0, 0);
  }

  Eigen::Vector3d right() const { return cam_from_world.linear().row(0); }
  Eigen::Vector3d up() const { return cam_from_world.linear().row(1); }
  Eigen::Vector3d forward() const { return cam_from_world.linear().row(2); }

  void setPositionInWorld(const Eigen::Vector3d& p_world)
  {
    cam_from_world.translation() = -cam_from_world.linear() * p_world;
  }

  Eigen::Vector3d getPositionInWorld() const
  {
    return -cam_from_world.linear().transpose() * cam_from_world.translation();
  }

  Eigen::Isometry3d worldFromCam() const
  {
    Eigen::Isometry3d w_from_c;
    w_from_c.linear() = cam_from_world.linear().transpose();
    w_from_c.translation() = -cam_from_world.linear().transpose() * cam_from_world.translation();
    return w_from_c;
  }

  inline Eigen::Vector3d worldFromCam(const Eigen::Vector3d& p_cam) const { return worldFromCam() * p_cam; }

  inline Eigen::Vector3d camFromWorld(const Eigen::Vector3d& p_world) const { return cam_from_world * p_world; }

  inline Eigen::Vector2d pixelFromCam(const Eigen::Vector3d& p_cam) const
  {
    XCHECK(false) << "Not implemented";
    return Eigen::Vector2d(0, 0);
  }

  // returns a ray direction in the camera's local coordinate frame.
  inline Eigen::Vector3d rayDirFromPixel(const Eigen::Vector2d& pixel) const
  {
    // TODO: could have 180 or 360 degree FOV easily here.
    const double theta = 
      is_180 
      ? -M_PI * pixel.x() / width - M_PI
      : -2.0 * M_PI * pixel.x() / width - M_PI / 2.0;
    const double phi = M_PI * (pixel.y() / height - 0.5);
    return Eigen::Vector3d(
      cos(phi) * cos(theta), 
      -sin(phi),
      cos(phi) * sin(theta)
    );
  }

  inline Eigen::Vector2d pixelFromWorld(const Eigen::Vector3d& p_world) const
  {
    return pixelFromCam(camFromWorld(p_world));
  }

  // Getters: these are mostly just for compatibility with NerfKludgeCamera
  int getWidth() const { return width; }
  int getHeight() const { return height; }
  Eigen::Isometry3d camFromWorld() const { return cam_from_world; }
};

}}  // namespace p11::calibration
