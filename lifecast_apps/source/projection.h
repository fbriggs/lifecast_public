// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
This file contains functions related to a rectified projection for fisheye cameras, which is useful
for doing stereo depth estimation with fisheye lenses. The projection geometry is based on
"Fish-Eye-Stereo Calibration and Epipolar Rectification" (Steffen Abraham, Wolfgang Forstner).
Most importantly, this projection has purely horizontal epipolar lines, enabling efficient disparity
estimation as a 1D search. Refer to equations (7) and (8) for projection and inverse projection
equations.

It also contains functions for equirectangular projection.
*/
#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "fisheye_camera.h"

namespace p11 { namespace projection {

// This is just a wrapper around cv::remap that makes it more concise.
cv::Mat warp(
    const cv::Mat& src,
    const std::vector<cv::Mat>& warp_uv,
    cv::InterpolationFlags interp_mode = cv::INTER_CUBIC);

std::vector<cv::Mat> composeWarps(
    const std::vector<cv::Mat>& warp1,
    const std::vector<cv::Mat>& warp2);

// input a pixel coordinate in the rectified projection, output a pixel coordinate in the orignal
// camera fisheye image.
Eigen::Vector2d rectifiedPixelToCamPixel(
    const int rectified_width,
    const int rectified_height,
    const calibration::FisheyeCamerad& cam,
    const float x,
    const float y);

// maps a pixel in rectified projection to a ray direction
Eigen::Vector3d rectifiedPixelToRayDirection(
    const Eigen::Vector2d& pixel, const int rectified_width, const int rectified_height);

// inverse of rectifiedPixelToRayDirection
Eigen::Vector2d camPoint3dToRectifiedPixel(
    const Eigen::Vector3d& cam_point, const int rectified_width, const int rectified_height);

// precompute a warp matrix suitable to pass to cv::remap, which projects a real image from the
// camera to a "fisheye rectified" projection.
void precomputeFisheyeToRectifyWarp(
    const int rectified_width,
    const int rectified_height,
    const calibration::FisheyeCamerad& cam,
    std::vector<cv::Mat>& warp_uv);

void precomputeRectifiedToFisheyeWarp(
    const int rectified_width,
    const int rectified_height,
    const cv::Size& fisheye_size,
    const calibration::FisheyeCamerad& cam,
    std::vector<cv::Mat>& warp_uv,
    float ftheta_scale = 1);

// This is used to compute a warp from a "RAW" fisheye image (with distortion and imperfect optical
// center), to a "perfect" f-theta projection with no distortion and optical center at w/2, h/2.
void precomputeFisheyeToFisheyeWarp(
    const calibration::FisheyeCamerad& src_cam,
    const calibration::FisheyeCamerad& dest_cam,
    std::vector<cv::Mat>& warp_uv,
    float dest_ftheta_scale = 1.0f);

// Compute a warp from f-theta to inflated f-theta
void precomputeFisheyeToInflatedWarp(
    calibration::FisheyeCamerad& src_cam,
    calibration::FisheyeCamerad& dest_cam,
    std::vector<cv::Mat>& warp_uv,
    double inflate_exponent = 3.0);

// convert a disparity in rectified projection to a depth.
// note that disparities are positive, and
// Lx = Rx + disparity
// Lx - Rx = disparity
// also, if we want to associate an image with the disparity/depth map, the it is correct to use
// the right image (not the left).
double disparityToDepth(
    const int depthmap_size,
    const double baseline,
    const int R_x,
    const int R_y,
    const float disparity);

// converts a disparity map to an inverse depth map.
cv::Mat disparityToInvDepth(const cv::Mat& disparity, const double baseline);

// generates a point cloud from an inverse depth map and image.
// it is assumed that inv_depthmap and image are in rectified projection.
void makePointCloudFromDepthmap(
    const cv::Mat& inv_depthmap,
    const cv::Mat& image,
    std::vector<Eigen::Vector3f>& point_cloud,
    std::vector<Eigen::Vector3f>& point_cloud_colors);

// NOTE: this is for 180 equirect
void precomputeremapFisheyeToEquirectWarp(
    const int eqr_width,
    const int eqr_height,
    const calibration::FisheyeCamerad& cam,
    std::vector<cv::Mat>& warp_uv);

// Use this to render fisheye cameras into a mono 360 equirect.
void computeFisheyeTo360EquirectWarp(
    const int eqr_width,
    const int eqr_height,
    const calibration::FisheyeCamerad& cam,
    std::vector<cv::Mat>& warp_uv);

// This warp is used to convert VR180 to ftheta projection.
void precomputeVR180toFthetaWarp(
    const calibration::FisheyeCamerad& cam_perfect_ftheta,
    const int ftheta_size,
    const int eqr_size,
    std::vector<cv::Mat>& warp_uv,
    const double ftheta_scale = 1.0);

calibration::FisheyeCamerad makePerfectFthetaCamera(const int image_size);

}}  // namespace p11::projection
