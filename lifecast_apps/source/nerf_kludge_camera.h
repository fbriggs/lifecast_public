// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include "opencv2/core.hpp"
#include "rectilinear_camera.h"
#include "fisheye_camera.h"

namespace p11 { namespace calibration {

// Kind of a polymorphic wrapper around different types of camera +
// some extra junk like time offsets and ignore masks
struct NerfKludgeCamera {
  // Extra stuff for convience building a NeRF video
  int time_offset_frames; // HACK- not great that this is here
  cv::Mat ignore_mask;

  // Wrapper around different camera types
  bool is_rectilinear, is_fisheye;
  RectilinearCamerad rectilinear;
  FisheyeCamerad fisheye;

  NerfKludgeCamera() {}

  NerfKludgeCamera(const RectilinearCamerad& src) {
    is_rectilinear = true;
    is_fisheye = false;
    rectilinear = src;
  }

  NerfKludgeCamera(const FisheyeCamerad& src) {
    is_rectilinear = false;
    is_fisheye = true;
    fisheye = src;
  }

  std::string name() const {
    if (is_rectilinear) return rectilinear.name;
    else                return fisheye.name;
  }

  int getWidth() const {
    if (is_rectilinear) return rectilinear.width;
    else                return fisheye.width;
  }

  int getHeight() const {
    if (is_rectilinear) return rectilinear.height;
    else                return fisheye.height;
  }

  Eigen::Isometry3d camFromWorld() const {
    if (is_rectilinear) return rectilinear.cam_from_world;
    else                return fisheye.cam_from_world;
  }

  void setCamFromWorld(const Eigen::Isometry3d& c_f_w) {
    if (is_rectilinear) rectilinear.cam_from_world = c_f_w;
    else                fisheye.cam_from_world = c_f_w;
  }

  Eigen::Vector3d rayDirFromPixel(const Eigen::Vector2d& pixel) const {
    if (is_rectilinear) return rectilinear.rayDirFromPixel(pixel);
    else                return fisheye.rayDirFromPixel(pixel);
  }

  Eigen::Vector2d pixelFromCam(const Eigen::Vector3d& p_cam) const {
    if (is_rectilinear) return rectilinear.pixelFromCam(p_cam);
    else                return fisheye.pixelFromCam(p_cam);
  }

  Eigen::Vector3d camFromWorld(const Eigen::Vector3d& p_world) const {
    if (is_rectilinear) return rectilinear.camFromWorld(p_world);
    else                return fisheye.camFromWorld(p_world);
  }
  
  Eigen::Vector3d getPositionInWorld() const {
    if (is_rectilinear) return rectilinear.getPositionInWorld();
    else                return fisheye.getPositionInWorld();
  }

  void setPositionInWorld(const Eigen::Vector3d& p) {
    if (is_rectilinear) rectilinear.setPositionInWorld(p);
    else                fisheye.setPositionInWorld(p);
  }

  void resizeToWidth(const int new_width) {
    if (is_rectilinear) rectilinear.resizeToWidth(new_width);
    else                fisheye.resizeToWidth(new_width);
  }

  void resizeToHeight(const int new_height) {
    if (is_rectilinear) rectilinear.resizeToHeight(new_height);
    else                fisheye.resizeToHeight(new_height);
  }

  void resizeToMaxDim(const int new_max) {
    if (is_rectilinear) rectilinear.resizeToMaxDim(new_max);
    else                fisheye.resizeToMaxDim(new_max);
  }
};


}}  // end namespace p11::calibration
