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
#include "fisheye_camera.h"
#include "nerf_kludge_camera.h"
#include "ngp_radiance_model.h"
#include "util_time.h"
#include "util_torch.h"

namespace p11 { namespace nerf {

constexpr int kImageCodeDim = 16;
constexpr double kInvDepthCoef = 0.3;

struct NeoNerfConfig {
  std::shared_ptr<std::atomic<bool>> cancel_requested;
  std::string train_images_dir;
  std::string train_json;
  std::string output_dir;
  std::string load_model_dir;
  int num_training_itrs;
  int rays_per_batch;
  int num_basic_samples;
  int num_importance_samples;
  int warmup_itrs;
  int num_novel_views;
  bool compute_train_psnr;
  double radiance_lr;
  double radiance_decay;
  double image_code_lr;
  double image_code_decay;
  double prop_lr;
  double prop_decay;
  double adam_eps;
  double floater_min_dist;
  double floater_weight;
  double gini_weight;
  double distortion_weight;
  double density_weight;
  double far_away_weight;
  double visibility_weight;
  int num_visibility_points;
  double prev_density_weight;
  int prev_reg_num_samples;
  int resize_max_dim;
  bool show_epipolar_viz; // warning: requiers user input, uses cv::imshow (not compatible with ImGui)!
};

// Options for video pipeline
struct NerfVideoConfig {
  std::string vid_dir;
  int ldi_resolution;
  bool transparent_bg;
  bool use_ibr;
  bool render_looking_glass;
};

struct NeoNerfGuiData {
  std::mutex mutex;
  std::vector<float> plot_data_x, plot_data_y;
};

inline void printConfig(const NeoNerfConfig& cfg) {
  XPLINFO << "cancel_requested=" << cfg.cancel_requested;
  XPLINFO << "train_images_dir=" << cfg.train_images_dir;
  XPLINFO << "train_json=" << cfg.train_json;
  XPLINFO << "output_dir=" << cfg.output_dir;
  XPLINFO << "load_model_dir=" << cfg.load_model_dir;
  XPLINFO << "num_training_itrs=" << cfg.num_training_itrs;
  XPLINFO << "rays_per_batch=" << cfg.rays_per_batch;
  XPLINFO << "num_basic_samples=" << cfg.num_basic_samples;
  XPLINFO << "num_importance_samples=" << cfg.num_importance_samples;
  XPLINFO << "warmup_itrs=" << cfg.warmup_itrs;
  XPLINFO << "num_novel_views=" << cfg.num_novel_views;
  XPLINFO << "radiance_lr=" << cfg.radiance_lr;
  XPLINFO << "radiance_decay=" << cfg.radiance_decay;
  XPLINFO << "image_code_lr=" << cfg.image_code_lr;
  XPLINFO << "image_code_decay=" << cfg.image_code_decay;
  XPLINFO << "prop_lr=" << cfg.prop_lr;
  XPLINFO << "prop_decay=" << cfg.prop_decay;
  XPLINFO << "adam_eps=" << cfg.adam_eps;
  XPLINFO << "floater_min_dist=" << cfg.floater_min_dist;
  XPLINFO << "floater_weight=" << cfg.floater_weight;
  XPLINFO << "gini_weight=" << cfg.gini_weight;
  XPLINFO << "distortion_weight=" << cfg.distortion_weight;
  XPLINFO << "density_weight=" << cfg.density_weight;
  XPLINFO << "far_away_weight=" << cfg.far_away_weight;
  XPLINFO << "prev_density_weight=" << cfg.prev_density_weight;
  XPLINFO << "prev_reg_num_samples=" << cfg.prev_reg_num_samples;
  XPLINFO << "resize_max_dim=" << cfg.resize_max_dim;
  XPLINFO << "show_epipolar_viz=" << cfg.show_epipolar_viz;
}

void runNerfPipeline(NeoNerfConfig& cfg, NeoNerfGuiData* gui_data = nullptr);

void runVideoNerfPipeline(NerfVideoConfig& vid_cfg, NeoNerfConfig& cfg, NeoNerfGuiData* gui_data = nullptr);

cv::Mat imageTensorToCvMat(const torch::Tensor& tensor);

void samplePointCloudFromRadianceField(
    const torch::DeviceType device,
    std::shared_ptr<NeoNerfModel>& radiance_model,
    const torch::Tensor& image_code,
    const int target_num_points,
    std::vector<Eigen::Vector3f>& point_cloud,
    std::vector<Eigen::Vector4f>& point_cloud_colors,
    std::shared_ptr<std::atomic<bool>> cancel_requested = nullptr);


std::tuple<torch::Tensor, torch::Tensor> computeQueryPointsFromRays(
  torch::Tensor ray_origins,
  torch::Tensor ray_dirs,
  const int num_samples);

std::tuple<torch::Tensor, torch::Tensor> importanceSampling(
  bool warmup,
  std::shared_ptr<ProposalDensityModel>& proposal_model,
  torch::Tensor ray_origins,
  torch::Tensor ray_dirs,
  torch::Tensor query_points,
  torch::Tensor depth_values,
  const int num_importance_samples,
  const bool include_original_samples);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>renderVolumeDensity(
  torch::Tensor rgb,
  torch::Tensor sigma,
  torch::Tensor depth_values,
  const bool transparent_background = false); // This should be false during training

torch::Tensor renderVolumeDensityInverseDepthOnly(
  torch::Tensor sigma,
  torch::Tensor depth_values);

void loadNerfAndProposalModels(
  const std::string& model_dir, 
  const torch::DeviceType device,
  std::shared_ptr<nerf::NeoNerfModel>& radiance_model,
  std::shared_ptr<nerf::ProposalDensityModel>& proposal_model);

void testNerfToLdi3Distillation(
  const std::string& model_dir, // path to folder containing raidiance model and proposal model files
  const std::string& dest_dir, // path to write outputs
  const int ldi_resolution,
  const bool transparent_bg);

void testImageBasedProjectionRefinement(const std::string& data_dir);

cv::Mat distillNerfToLdi3(
    const std::string& debug_dir,
    const torch::DeviceType device,
    std::shared_ptr<NeoNerfModel>& radiance_model,
    std::shared_ptr<ProposalDensityModel>& proposal_model,
    const calibration::FisheyeCamerad& cam,
    const cv::Mat& vignette,
    torch::Tensor image_code,
    const int num_basic_samples,
    const int num_importance_samples,
    const bool transparent_bg,
    cv::Mat& out_fused_bgra, // mostly for debugging, might not be filled in but you have to pass a container anyway. 
    const cv::Mat primary_image_projection = cv::Mat(), // if not empty, well use this for IBR
    const Eigen::Matrix4d& world_transform = Eigen::Matrix4d::Identity(),
    std::shared_ptr<std::atomic<bool>> cancel_requested = nullptr);


// This version fills in layer_bgra and layer_invd
void distillNerfToLdi3(
  const std::string& debug_dir,
  const torch::DeviceType device,
  std::shared_ptr<NeoNerfModel>& radiance_model,
  std::shared_ptr<ProposalDensityModel>& proposal_model,
  const calibration::FisheyeCamerad& cam,
  const cv::Mat& vignette,
  torch::Tensor image_code,
  const int num_basic_samples,
  const int num_importance_samples,
  const bool transparent_bg,
  cv::Mat& out_fused_bgra, // mostly for debugging, might not be filled in but you have to pass a container anyway. 
  const Eigen::Matrix4d& world_transform,
  std::shared_ptr<std::atomic<bool>> cancel_requested,
  std::vector<cv::Mat>& layer_bgra,
  std::vector<cv::Mat>& layer_invd);

template<typename TCamera>
inline torch::Tensor renderImageWithNerf(
    const torch::DeviceType device,
    std::shared_ptr<NeoNerfModel>& radiance_model,
    std::shared_ptr<ProposalDensityModel>& proposal_model,
    const TCamera& cam,
    torch::Tensor image_code,
    const int num_basic_samples,
    const int num_importance_samples,
    const Eigen::Matrix4d& world_transform = Eigen::Matrix4d::Identity(),
    std::shared_ptr<std::atomic<bool>> cancel_requested = nullptr
){
  auto start_timer = time::now();
  
  constexpr int MAX_RAYS_PER_BATCH = 16384;
  
  torch::NoGradGuard no_grad;

  torch::Tensor final_image = torch::zeros({cam.getHeight(), cam.getWidth(), 3}, device);

  Eigen::Transform<double, 3, Eigen::Affine> world_transform_inv(world_transform.inverse());
  Eigen::Matrix3d world_rotation_inv = world_transform_inv.linear();

  const Eigen::Vector3d ray_origin_in_world = world_transform_inv * cam.getPositionInWorld();

  int i = 0;
  while (i < cam.getHeight()) {
    if (cancel_requested != nullptr && (*cancel_requested)) return torch::tensor({});

    int rows_in_batch = std::min(MAX_RAYS_PER_BATCH / cam.getWidth(), cam.getHeight() - i);
    
    std::vector<float> ray_origin_data, ray_dir_data;
    for (int y = i; y < i + rows_in_batch; ++y) {
      for (int x = 0; x < cam.getWidth(); ++x) {
        const Eigen::Vector3d ray_dir_in_cam = cam.rayDirFromPixel(Eigen::Vector2d(x, y));
        Eigen::Vector3d ray_dir_in_world = world_rotation_inv * cam.camFromWorld().linear().transpose() * ray_dir_in_cam;

        ray_dir_in_world.normalize();

        ray_origin_data.push_back(ray_origin_in_world.x());
        ray_origin_data.push_back(ray_origin_in_world.y());
        ray_origin_data.push_back(ray_origin_in_world.z());
        ray_dir_data.push_back(ray_dir_in_world.x());
        ray_dir_data.push_back(ray_dir_in_world.y());
        ray_dir_data.push_back(ray_dir_in_world.z());
      }
    }

    const int num_rays_in_batch = ray_origin_data.size() / 3;
    torch::Tensor ray_origins = torch::from_blob(ray_origin_data.data(), {num_rays_in_batch, 3}, torch::kFloat32).to(device);
    torch::Tensor ray_dirs = torch::from_blob(ray_dir_data.data(), {num_rays_in_batch, 3}, torch::kFloat32).to(device);
    util_torch::cloneIfOnCPU(ray_origins);
    util_torch::cloneIfOnCPU(ray_dirs);

    const auto [basic_query_points, basic_depth_values] = computeQueryPointsFromRays(ray_origins, ray_dirs, num_basic_samples);
    constexpr bool kWarmup = false;
    constexpr bool kIncludeOriginalSamples = false;
    const auto [query_points, depth_values] = importanceSampling(kWarmup, proposal_model, ray_origins, ray_dirs, basic_query_points, basic_depth_values, num_importance_samples, kIncludeOriginalSamples);
    const int num_samples_per_ray = query_points.size(1);

    torch::Tensor ray_image_codes = image_code.repeat({num_rays_in_batch, 1});

    torch::Tensor flat_query_points = query_points.reshape({-1, 3});
    torch::Tensor flat_ray_dirs = ray_dirs.unsqueeze(1).repeat_interleave(num_samples_per_ray, 1).reshape({-1, 3});
    torch::Tensor flat_image_codes = ray_image_codes.unsqueeze(1).repeat_interleave(num_samples_per_ray, 1).reshape({-1, kImageCodeDim});

    const auto& [rgb, sigma] = radiance_model->pointAndDirToRadiance(flat_query_points, flat_ray_dirs, flat_image_codes);
    torch::Tensor unflat_rgb = rgb.reshape({num_rays_in_batch, num_samples_per_ray, 3});
    torch::Tensor unflat_sigma = sigma.reshape({num_rays_in_batch, num_samples_per_ray});
    const auto& [predicted_rgb, _1, _2, _3, _4] = renderVolumeDensity(unflat_rgb, unflat_sigma, depth_values);

    final_image.slice(0, i, i + rows_in_batch) = predicted_rgb.reshape({rows_in_batch, cam.getWidth(), 3});
    i += rows_in_batch;
  }

  XPLINFO << "render time(sec):\t" << time::timeSinceSec(start_timer);
  return final_image;
}

template<typename TCamera>
inline cv::Mat renderInverseDepthmapAsCvMat(
    const torch::DeviceType device,
    std::shared_ptr<NeoNerfModel>& radiance_model,
    std::shared_ptr<ProposalDensityModel>& proposal_model,
    const TCamera& cam,
    const int num_basic_samples,
    const int num_importance_samples,
    const Eigen::Matrix4d& world_transform = Eigen::Matrix4d::Identity(),
    std::shared_ptr<std::atomic<bool>> cancel_requested = nullptr
){
  auto timer = time::now();

  constexpr int MAX_RAYS_PER_BATCH = 16384;
  torch::NoGradGuard no_grad;
  cv::Size image_size(cam.width, cam.height);
  cv::Mat final_depthmap = cv::Mat::zeros(image_size, CV_32F);

  Eigen::Transform<double, 3, Eigen::Affine> world_transform_inv(world_transform.inverse());
  Eigen::Matrix3d world_rotation_inv = world_transform_inv.linear();
  const Eigen::Vector3d ray_origin_in_world = world_transform_inv * cam.getPositionInWorld();

  std::vector<float> ray_origin_data, ray_dir_data;
  std::vector<std::pair<int, int>> valid_pixels;
  int pixel_count = 0;

  for (int y = 0; y < cam.height; ++y) {
    for (int x = 0; x < cam.width; ++x) {
      if (cancel_requested != nullptr && (*cancel_requested)) return cv::Mat();
      if ((cam.optical_center - Eigen::Vector2d(x, y)).norm() > cam.radius_at_90) {
        continue;
      }

      valid_pixels.emplace_back(x, y);
      const Eigen::Vector3d ray_dir_in_cam = cam.rayDirFromPixel(Eigen::Vector2d(x, y));
      Eigen::Vector3d ray_dir_in_world = world_rotation_inv * cam.cam_from_world.linear().transpose() * ray_dir_in_cam;
      ray_dir_in_world.normalize();

      ray_origin_data.push_back(ray_origin_in_world.x());
      ray_origin_data.push_back(ray_origin_in_world.y());
      ray_origin_data.push_back(ray_origin_in_world.z());
      ray_dir_data.push_back(ray_dir_in_world.x());
      ray_dir_data.push_back(ray_dir_in_world.y());
      ray_dir_data.push_back(ray_dir_in_world.z());

      if (++pixel_count == MAX_RAYS_PER_BATCH || (y == cam.height - 1 && x == cam.width - 1)) {
        const int num_rays_in_batch = ray_origin_data.size() / 3;

        torch::Tensor ray_origins = torch::from_blob(ray_origin_data.data(), {num_rays_in_batch, 3}, torch::kFloat32).to(device);
        torch::Tensor ray_dirs = torch::from_blob(ray_dir_data.data(), {num_rays_in_batch, 3}, torch::kFloat32).to(device);
        util_torch::cloneIfOnCPU(ray_origins);
        util_torch::cloneIfOnCPU(ray_dirs);

        auto [basic_query_points, basic_depth_values] = computeQueryPointsFromRays(ray_origins, ray_dirs, num_basic_samples);
        constexpr bool warmup = false;
        constexpr bool include_original_samples = false;
        auto [query_points, depth_values] = importanceSampling(warmup, proposal_model, ray_origins, ray_dirs, basic_query_points, basic_depth_values, num_importance_samples, include_original_samples);

        const int num_samples_per_ray = query_points.size(1);
        torch::Tensor flat_query_points = query_points.reshape({-1, 3});
        const auto sigma = radiance_model->pointToDensity(flat_query_points);
        torch::Tensor unflat_sigma = sigma.reshape({num_rays_in_batch, num_samples_per_ray});
        const auto predicted_depth = renderVolumeDensityInverseDepthOnly(unflat_sigma, depth_values);
        
        auto cpu_depth = predicted_depth.to(torch::kCPU);
        auto acc_depth = cpu_depth.accessor<float, 1>();
        
        int i = 0;
        for (const auto& [x, y] : valid_pixels) {
          final_depthmap.at<float>(y, x) = acc_depth[i++];
        }
        
        ray_origin_data.clear();
        ray_dir_data.clear();
        valid_pixels.clear();
        pixel_count = 0;
      }
    }
  }

  XPLINFO << "render depthmap time: " << time::timeSinceSec(timer);
  return final_depthmap;
}


}}  // end namespace p11::nerf
