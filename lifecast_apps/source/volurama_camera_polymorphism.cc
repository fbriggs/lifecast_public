// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "volurama_camera_polymorphism.h"

namespace p11 {

cv::Mat renderImageWithNerfPolymorphicCamera(
  const PolymorphicCameraOptions& opts,
  const torch::DeviceType device,
  std::shared_ptr<nerf::NeoNerfModel>& radiance_model,
  std::shared_ptr<nerf::ProposalDensityModel>& proposal_model,
  const calibration::RectilinearCamerad& base_cam,
  torch::Tensor image_code,
  const int num_basic_samples,
  const int num_importance_samples,
  const Eigen::Matrix4d& world_transform,
  std::shared_ptr<std::atomic<bool>> cancel_requested
) {
  cv::Mat image_mat;

  if (opts.cam_type == CAM_TYPE_RECTILINEAR) {
    torch::Tensor image_tensor = nerf::renderImageWithNerf<calibration::RectilinearCamerad>(
      device,
      radiance_model,
      proposal_model,
      base_cam,
      image_code,
      num_basic_samples,
      num_importance_samples,
      world_transform,
      cancel_requested);
    if (image_tensor.dim() == 1) return cv::Mat(); // We get this if it was cancelled
    image_mat = nerf::imageTensorToCvMat(image_tensor);
  }

  if (opts.cam_type == CAM_TYPE_RECTILINEAR_STEREO) {
    calibration::RectilinearCamerad L_cam(base_cam);
    calibration::RectilinearCamerad R_cam(base_cam);
    L_cam.setPositionInWorld(base_cam.getPositionInWorld() - base_cam.right() * opts.virtual_stereo_baseline * 0.5);
    R_cam.setPositionInWorld(base_cam.getPositionInWorld() + base_cam.right() * opts.virtual_stereo_baseline * 0.5);

    torch::Tensor L_image_tensor = nerf::renderImageWithNerf<calibration::RectilinearCamerad>(
      device,
      radiance_model,
      proposal_model,
      L_cam,
      image_code,
      num_basic_samples,
      num_importance_samples,
      world_transform,
      cancel_requested);
    if (L_image_tensor.dim() == 1) return cv::Mat(); // We get this if it was cancelled

    torch::Tensor R_image_tensor = nerf::renderImageWithNerf<calibration::RectilinearCamerad>(
      device,
      radiance_model,
      proposal_model,
      R_cam,
      image_code,
      num_basic_samples,
      num_importance_samples,
      world_transform,
      cancel_requested);
    if (R_image_tensor.dim() == 1) return cv::Mat(); // We get this if it was cancelled

    cv::Mat L_image_mat = nerf::imageTensorToCvMat(L_image_tensor);
    cv::Mat R_image_mat = nerf::imageTensorToCvMat(R_image_tensor);
    cv::hconcat(L_image_mat, R_image_mat, image_mat);
  }

  if (opts.cam_type == CAM_TYPE_EQUIRECTANGULAR) {
    calibration::EquirectangularCamera eqr_cam;
    eqr_cam.cam_from_world = base_cam.cam_from_world;
    eqr_cam.width = opts.eqr_width;
    eqr_cam.height = opts.eqr_height;
    
    torch::Tensor image_tensor = nerf::renderImageWithNerf<calibration::EquirectangularCamera>(
      device,
      radiance_model,
      proposal_model,
      eqr_cam,
      image_code,
      num_basic_samples,
      num_importance_samples,
      world_transform,
      cancel_requested);
    if (image_tensor.dim() == 1) return cv::Mat(); // We get this if it was cancelled
    image_mat = nerf::imageTensorToCvMat(image_tensor);
  }

  if (opts.cam_type == CAM_TYPE_VR180) {
    calibration::EquirectangularCamera eqr_cam;
    eqr_cam.cam_from_world = base_cam.cam_from_world;
    eqr_cam.width = opts.vr180_size;
    eqr_cam.height = opts.vr180_size;
    eqr_cam.is_180 = true;
    
    calibration::EquirectangularCamera L_cam(eqr_cam);
    calibration::EquirectangularCamera R_cam(eqr_cam);
    L_cam.setPositionInWorld(eqr_cam.getPositionInWorld() - eqr_cam.right() * opts.virtual_stereo_baseline * 0.5);
    R_cam.setPositionInWorld(eqr_cam.getPositionInWorld() + eqr_cam.right() * opts.virtual_stereo_baseline * 0.5);

    torch::Tensor L_image_tensor = nerf::renderImageWithNerf<calibration::EquirectangularCamera>(
      device,
      radiance_model,
      proposal_model,
      L_cam,
      image_code,
      num_basic_samples,
      num_importance_samples,
      world_transform,
      cancel_requested);
    if (L_image_tensor.dim() == 1) return cv::Mat(); // We get this if it was cancelled

    torch::Tensor R_image_tensor = nerf::renderImageWithNerf<calibration::EquirectangularCamera>(
      device,
      radiance_model,
      proposal_model,
      R_cam,
      image_code,
      num_basic_samples,
      num_importance_samples,
      world_transform,
      cancel_requested);
    if (R_image_tensor.dim() == 1) return cv::Mat(); // We get this if it was cancelled

    cv::Mat L_image_mat = nerf::imageTensorToCvMat(L_image_tensor);
    cv::Mat R_image_mat = nerf::imageTensorToCvMat(R_image_tensor);
    cv::hconcat(L_image_mat, R_image_mat, image_mat);
  }

  if (opts.cam_type == CAM_TYPE_LOOKING_GLASS_PORTRAIT) {
    constexpr int kNumRows = 6;
    constexpr int kNumCols = 8;
    constexpr int kTotalTiles = kNumRows * kNumCols;
    std::vector<cv::Mat> row_images;
    std::vector<cv::Mat> images;

    for (int row = 0; row < kNumRows; ++row) {
      for (int col = 0; col < kNumCols; ++col) {
        // Compute the index in the quilt
        int tile = (kNumRows - 1 - row) * kNumCols + col;
        XPLINFO << "Quilt Row: " << row << " Quilt Column: " << col;

        calibration::RectilinearCamerad quilt_cam(base_cam);
        quilt_cam.width = (3360 / opts.looking_glass_downscale) / kNumCols;
        quilt_cam.height = (3360 / opts.looking_glass_downscale) / kNumRows;
        quilt_cam.optical_center = Eigen::Vector2d(quilt_cam.width / 2, quilt_cam.height / 2);
        double f = quilt_cam.width / (2.0 * tan(opts.looking_glass_hfov * M_PI / 360.0));
        quilt_cam.focal_length = Eigen::Vector2d(f, f);

        // Calculate beta, ensuring the bottom-left tile is the leftmost view
        const float beta = float(tile) / (kTotalTiles - 1) - 0.5;
        const Eigen::Vector3d cam_pos =
          base_cam.getPositionInWorld() + 
          base_cam.right() * opts.virtual_stereo_baseline * beta;
        quilt_cam.setPositionInWorld(cam_pos);

        torch::Tensor image_tensor = nerf::renderImageWithNerf<calibration::RectilinearCamerad>(
          device,
          radiance_model,
          proposal_model,
          quilt_cam,
          image_code,
          num_basic_samples,
          num_importance_samples,
          world_transform,
          cancel_requested);
        if (image_tensor.dim() == 1) return cv::Mat(); // We get this if it was cancelled
        images.push_back(nerf::imageTensorToCvMat(image_tensor));
      }
      cv::Mat row_image;
      cv::hconcat(images, row_image);
      row_images.push_back(row_image);
      images.clear();
    }
    cv::vconcat(row_images, image_mat);
  }

  image_mat.convertTo(image_mat, CV_8UC3, 255.0);
  return image_mat;
}

} // namespace p11
