// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <string>
#include <memory>
#include "logger.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/photo.hpp"
#include "torch/script.h"
#include "torch/torch.h"
#include "rectilinear_camera.h"
#include "equirectangular_camera.h"
#include "ngp_radiance_model.h"
#include "lifecast_nerf_lib.h"
#include "util_time.h"

namespace p11 {

enum VirtualCameraType {
  CAM_TYPE_RECTILINEAR              = 0,
  CAM_TYPE_RECTILINEAR_STEREO       = 1,
  CAM_TYPE_EQUIRECTANGULAR          = 2,
  CAM_TYPE_VR180                    = 3,
  CAM_TYPE_LOOKING_GLASS_PORTRAIT   = 4
};

struct PolymorphicCameraOptions {
  VirtualCameraType cam_type;

  // Depending on the the output type specified, different fields in this
  // struct may or not be used to construct different cameras.
  
  // If the output camera is equirectangular, this is the size.
  int eqr_width, eqr_height;

  // If the output camra is VR180, this is the size (square) per eye.
  int vr180_size;
  
  // For virtual stereoscopic cameras, this is the baseline.
  // Note this might not be in the units a user would expect, the whole 
  // scene may be unitless.
  float virtual_stereo_baseline;

  // When rendering for Looking Glass, this fov is used (otherwise it is implied 
  // by the focal length of base_cam)
  float looking_glass_hfov;

  // Preview downscaling is quirky for looking glass. This is part of kludgey special case.
  // This is because Looking Glass Portrait has a specific resolution, so the user can't specify that.
  int looking_glass_downscale;
};

// This is a enum-based polymorphic interface to renderImageWithNerf (which is templated).
// It takes a rectilinear camera (base cam), and modifies it as necessary
// to instead render with whatever type of camera is requested by the VirtualCameraType enum.
cv::Mat renderImageWithNerfPolymorphicCamera(
  const PolymorphicCameraOptions& opts,
  const torch::DeviceType device,
  std::shared_ptr<nerf::NeoNerfModel>& radiance_model,
  std::shared_ptr<nerf::ProposalDensityModel>& proposal_model,
  const calibration::RectilinearCamerad& base_cam, // This may get transformed into a different type of camera!
  torch::Tensor image_code,
  const int num_basic_samples,
  const int num_importance_samples,
  const Eigen::Matrix4d& world_transform = Eigen::Matrix4d::Identity(),
  std::shared_ptr<std::atomic<bool>> cancel_requested = nullptr);

} // namespace p11
