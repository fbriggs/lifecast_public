// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "lifecast_splat_population.h"

#include <random>
#include <algorithm>
#include "opencv2/flann.hpp"
#include "lifecast_splat_lib.h"
#include "lifecast_splat_io.h"
#include "lifecast_splat_math.h"
#include "util_opencv.h"
#include "point_cloud.h"

namespace p11 { namespace splat {

std::vector<float> calcAvgDistToNearestNeighbors(const std::vector<Eigen::Vector3f>& points) {
  XCHECK(!points.empty());

  constexpr int kDimensions = 3;        // x, y, z
  constexpr int kNearestNeighbors = 3;  // number of actual neighbors we want
  constexpr int kNeighbors = 1 + kNearestNeighbors; // point itself + 3 nearest neighbors
  constexpr int kKDTreeBranches = 4;    // number of branches in KD-tree
  constexpr int kMaxLeafChecks = 32;    // max leaf checks in search

  constexpr float kMinAvgDistHack = 0.003; // Clamp distances here for initialization purposes 
  constexpr float kMaxAvgDistHack = 0.1; // Prevent huge initial gaussians

  cv::Mat data(points.size(), kDimensions, CV_32F);
  for (size_t i = 0; i < points.size(); ++i) {
    data.at<float>(i, 0) = points[i].x();
    data.at<float>(i, 1) = points[i].y();
    data.at<float>(i, 2) = points[i].z();
  }

  cv::Mat indices, dists;
  cv::flann::Index flann_index(data, cv::flann::KDTreeIndexParams(kKDTreeBranches));
  flann_index.knnSearch(data, indices, dists, kNeighbors, cv::flann::SearchParams(kMaxLeafChecks));

  std::vector<float> avg_distances(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    avg_distances[i] = (std::sqrt(dists.at<float>(i, 1)) + std::sqrt(dists.at<float>(i, 2))) / kNearestNeighbors;
    // TODO: it looks like some points are exactly on top of each other.. maybe this is causing problems like Nan?
    // The line below will hide this issue.
    avg_distances[i] = math::clamp(avg_distances[i], kMinAvgDistHack, kMaxAvgDistHack);
  }

  return avg_distances;
}

std::shared_ptr<SplatModel> initSplatTensorsFromPointCloud(
  const torch::DeviceType& device,
  const std::vector<Eigen::Vector3f> initial_pointcloud,
  const std::vector<Eigen::Vector3f> initial_pointcloud_colors,
  int max_num_splats
) {
  const std::vector<float> avg_dist_to_neighbors = calcAvgDistToNearestNeighbors(initial_pointcloud);

  std::vector<float> splat_pos_data, splat_color_data, splat_alpha_data, splat_scale_data, splat_quat_data;
  std::vector<uint8_t> splat_alive_data;
  constexpr float kDefaultAlpha = -1.0; // 0 -> 0.5 out of sigmoid
  for (int i = 0; i < initial_pointcloud.size(); ++i) {
    splat_pos_data.push_back(initial_pointcloud[i].x());
    splat_pos_data.push_back(initial_pointcloud[i].y());
    splat_pos_data.push_back(initial_pointcloud[i].z());
    // TODO: do linear<->srgb conversions upon load/save for all image, video, and point cloud data
    splat_color_data.push_back(sigmoidInverse(initial_pointcloud_colors[i].x()));
    splat_color_data.push_back(sigmoidInverse(initial_pointcloud_colors[i].y()));
    splat_color_data.push_back(sigmoidInverse(initial_pointcloud_colors[i].z()));
    splat_alpha_data.push_back(kDefaultAlpha);

    const float s = math::clamp<float>(std::log(avg_dist_to_neighbors[i]), -kMaxScaleExponent, 0.0f);
    splat_scale_data.push_back(inverseScaleActivation(s));
    splat_scale_data.push_back(inverseScaleActivation(s));
    splat_scale_data.push_back(inverseScaleActivation(s));
    splat_quat_data.push_back(0);
    splat_quat_data.push_back(0);
    splat_quat_data.push_back(0);
    splat_quat_data.push_back(1);
    splat_alive_data.push_back(1);
  }
  int num_splats = splat_pos_data.size() / 3;
  XPLINFO << "num_splats=" << num_splats;

  // Fill the rest with dead splats
  while(num_splats < max_num_splats) {
    Eigen::Vector3f random_point = Eigen::Vector3f(
      math::randUnif() - 0.5,
      math::randUnif() - 0.5,
      math::randUnif() - 0.5).normalized() * kZFar;
    splat_pos_data.push_back(random_point.x()); // NOTE: even those these splats are dead, it is not OK to initialize them on top of a camera or at (0, 0, 0)
    splat_pos_data.push_back(random_point.y());
    splat_pos_data.push_back(random_point.z());
    splat_color_data.push_back(0.0);
    splat_color_data.push_back(0.0);
    splat_color_data.push_back(0.0);
    splat_alpha_data.push_back(-10); // sigmoid(-10) --> 0
    splat_scale_data.push_back(0);
    splat_scale_data.push_back(0);
    splat_scale_data.push_back(0);
    splat_quat_data.push_back(0);
    splat_quat_data.push_back(0);
    splat_quat_data.push_back(0);
    splat_quat_data.push_back(1);
    num_splats = splat_pos_data.size() / 3;
    splat_alive_data.push_back(0);
  }
  XPLINFO << "num_splats=" << num_splats;

  auto model = std::make_shared<SplatModel>();
  model->splat_pos   = torch::from_blob(splat_pos_data.data(),   {num_splats, 3}, {torch::kFloat32}).to(device);
  model->splat_color = torch::from_blob(splat_color_data.data(),   {num_splats, 3}, {torch::kFloat32}).to(device);
  model->splat_alpha = torch::from_blob(splat_alpha_data.data(), {num_splats, 1}, {torch::kFloat32}).to(device);
  model->splat_scale = torch::from_blob(splat_scale_data.data(), {num_splats, 3}, {torch::kFloat32}).to(device);
  model->splat_quat  = torch::from_blob(splat_quat_data.data(),  {num_splats, 4}, {torch::kFloat32}).to(device);
  model->splat_alive  = torch::from_blob(splat_alive_data.data(),  {num_splats, 1}, {torch::kUInt8}).to(device).to(torch::kBool); // we cant just go direct from std::vector<bool> here.
  model->splat_pos = splatPosInverseActivation(model->splat_pos);
  XCHECK(!torch::isnan(model->splat_pos).any().item<bool>());
  XCHECK(!torch::isinf(model->splat_pos).any().item<bool>());
  util_torch::cloneIfOnCPU(model->splat_pos);
  util_torch::cloneIfOnCPU(model->splat_color);
  util_torch::cloneIfOnCPU(model->splat_alpha);
  util_torch::cloneIfOnCPU(model->splat_scale);
  util_torch::cloneIfOnCPU(model->splat_quat);
  util_torch::cloneIfOnCPU(model->splat_alive);
  return model;
}

void subsamplePointcloud(
  std::vector<Eigen::Vector3f>& points, 
  std::vector<Eigen::Vector3f>& colors,
  const int target_num_points
) {
  if (points.size() <= target_num_points) return;

  std::vector<int> idxs(points.size());
  for (int i = 0; i < points.size(); ++i) { idxs[i] = i; }
 
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(idxs.begin(), idxs.end(), g);
 
  std::vector<Eigen::Vector3f> new_points(target_num_points);
  std::vector<Eigen::Vector3f> new_colors(target_num_points);
  for (int i = 0; i < target_num_points; ++i) {
    new_points[i] = points[idxs[i]];
    new_colors[i] = colors[idxs[i]];
  }

  std::swap(new_points, points);
  std::swap(new_colors, colors);
}

void addPointsUsingMonoDepth(
  calibration::MultiCameraDataset& dataset,
  std::vector<Eigen::Vector3f>& pointcloud,
  std::vector<Eigen::Vector3f>& pointcloud_colors,
  int num_points_to_add
) {
  int points_per_camera = num_points_to_add / dataset.cameras.size();
  for (int cam_idx = 0; cam_idx < dataset.cameras.size(); ++cam_idx) {
    calibration::NerfKludgeCamera& cam = dataset.cameras[cam_idx];
    cv::Mat& mono_depth = dataset.depthmaps[cam_idx];
    float margin_x = cam.getWidth() * 0.1; // Crop to avoid boundary issues in the mono depthmap
    float margin_y = cam.getHeight() * 0.1; 
  
    Eigen::Matrix3d world_R_cam = cam.camFromWorld().linear().inverse().matrix();

    for (int i = 0; i < points_per_camera; ++i) {
      float rand_x = margin_x + math::randUnif() * (cam.getWidth() - 2.0 * margin_x);
      float rand_y = margin_y + math::randUnif() * (cam.getHeight() - 2.0 * margin_y);

      const Eigen::Vector3d ray_dir_in_cam = cam.rayDirFromPixel(Eigen::Vector2d(rand_x, rand_y));

      float d = opencv::getPixelBilinear<float>(mono_depth, rand_x, rand_y);
      
      if (d < kZNear) { continue; }

      Eigen::Vector3d adjusted_ray = Eigen::Vector3d(
        ray_dir_in_cam.x() * d / ray_dir_in_cam.z(), 
        ray_dir_in_cam.y() * d / ray_dir_in_cam.z(), 
        d);
      const Eigen::Vector3d point_in_world = cam.getPositionInWorld() + world_R_cam * adjusted_ray;

      cv::Vec3f color = cv::Vec3f(opencv::getPixelBilinear<cv::Vec3b>(
        dataset.images[cam_idx], rand_x, rand_y)) / 255.0f;
      pointcloud.push_back(point_in_world.cast<float>());
      pointcloud_colors.emplace_back(color[0], color[1], color[2]);
    }
  }
  XPLINFO << "Added " << points_per_camera << " points using mono depth estimation";
}

// Some points in the initial point cloud may be right on top of a camera, which causes 
// NaNs and other bad stuff, so we'll make sure there are no such points at the start.
void removePointsTooCloseOrTooFar(
  calibration::MultiCameraDataset dataset,
  std::vector<Eigen::Vector3f>& pointcloud,
  std::vector<Eigen::Vector3f>& pointcloud_colors
) {
  std::vector<Eigen::Vector3f> new_pointcloud;
  std::vector<Eigen::Vector3f> new_pointcloud_colors;
  int num_too_close = 0;
  int num_too_far = 0;
  for (int i = 0; i < pointcloud.size(); ++i) {
    bool point_is_ok = true;
  
    // Project points that are very far away back onto a sphere with radius kZFar
    float norm = pointcloud[i].norm();
    if (norm > kZFar) {
      pointcloud[i] = kZFar * pointcloud[i] / norm;
      ++num_too_far;
    }

    for (int j = 0; j < dataset.cameras.size(); ++j) {
      const float dist = (pointcloud[i] - dataset.cameras[j].getPositionInWorld().cast<float>()).norm();
      if (dist < kZNear) {
        point_is_ok = false;
        ++num_too_close;
        break;
      }
    }
    if (point_is_ok) {
      new_pointcloud.push_back(pointcloud[i]);
      new_pointcloud_colors.push_back(pointcloud_colors[i]);
    }
  }
  std::swap(new_pointcloud, pointcloud);
  std::swap(new_pointcloud_colors, pointcloud_colors);

  XPLINFO << "# removed points too close: " << num_too_close << " too far: " << num_too_far;
}

void estimateInitialColorsByProjection(
  calibration::MultiCameraDataset dataset,
  std::vector<Eigen::Vector3f>& pointcloud,
  std::vector<Eigen::Vector3f>& pointcloud_colors
) {
  std::vector<Eigen::Vector3f> new_pointcloud;
  std::vector<Eigen::Vector3f> new_pointcloud_colors;

  int num_projections_found = 0;
  for (int i = 0; i < pointcloud.size(); ++i) {
    bool found_color = false;
    for (int j = 0; j < dataset.cameras.size(); ++j) {
      const auto& cam = dataset.cameras[j];
      const Eigen::Vector3d point_in_cam = cam.camFromWorld(pointcloud[i].cast<double>());
      if (point_in_cam.z() < kZNear) continue;

      const Eigen::Vector2d projected_pixel = cam.pixelFromCam(point_in_cam);
      if (projected_pixel.x() < 0 ||
          projected_pixel.y() < 0 ||
          projected_pixel.x() >= cam.getWidth() - 1 ||
          projected_pixel.y() >= cam.getHeight() - 1) continue;
      XCHECK(dataset.images[j].type() == CV_8UC3);
      const cv::Vec3f color = cv::Vec3f(opencv::getPixelBilinear<cv::Vec3b>(
        dataset.images[j], projected_pixel.x(), projected_pixel.y())) / 255.0f;
      pointcloud_colors[i] = Eigen::Vector3f(color[0], color[1], color[2]);
      ++num_projections_found;
      found_color = true;
      break; // Stop once we find one valid projection. TODO: average multiple projections?
    }
    // Only keep the point if we found a color for it
    if (found_color) {
      new_pointcloud.push_back(pointcloud[i]);
      new_pointcloud_colors.push_back(pointcloud_colors[i]);
    }
  }
  std::swap(new_pointcloud, pointcloud);
  std::swap(new_pointcloud_colors, pointcloud_colors);
  XPLINFO << "projected color for " << num_projections_found << " points out of " << pointcloud.size();
}

std::shared_ptr<SplatModel> initSplatPopulation(
  const torch::DeviceType& device,
  const SplatConfig& cfg,
  calibration::MultiCameraDataset train_dataset
) {
  std::vector<Eigen::Vector3f> sfm_pointcloud, sfm_pointcloud_colors;
  std::vector<Eigen::Vector3f> initial_pointcloud, initial_pointcloud_colors;
  XCHECK(!cfg.sfm_pointcloud.empty()) << "TODO: init without SFM pointcloud";
  point_cloud::loadPointCloudBinary(cfg.sfm_pointcloud, sfm_pointcloud, sfm_pointcloud_colors);
  XPLINFO << "# sfm points: " << sfm_pointcloud.size();
  
  if(!sfm_pointcloud.empty() && !cfg.init_with_monodepth) {
    initial_pointcloud = sfm_pointcloud;
    initial_pointcloud_colors = sfm_pointcloud_colors;

    subsamplePointcloud(initial_pointcloud, initial_pointcloud_colors, kMaxFracInitFromSfm * cfg.max_num_splats);
    estimateInitialColorsByProjection(train_dataset, initial_pointcloud, initial_pointcloud_colors);
    XPLINFO << "# sfm points after subsample: " << initial_pointcloud.size();
  }
  
  // Keep the SFM points (if they exist) when initializing with mono depth
  if (cfg.init_with_monodepth) {
    const int num_initial_monodepth_splats = std::min(
      int(kFracInitRandom * cfg.max_num_splats),
      int(cfg.max_num_splats - initial_pointcloud.size()));
    XPLINFO << "# initial mono depth splats: " << num_initial_monodepth_splats;
    addPointsUsingMonoDepth(
      train_dataset, initial_pointcloud, initial_pointcloud_colors, num_initial_monodepth_splats);
  }
  
  removePointsTooCloseOrTooFar(train_dataset, initial_pointcloud, initial_pointcloud_colors);
  
  return initSplatTensorsFromPointCloud(
    device, initial_pointcloud, initial_pointcloud_colors, cfg.max_num_splats);
}

std::shared_ptr<SplatModel> initSplatPopulationWithSingleImageRGBD(
  const torch::DeviceType& device,
  int max_num_splats,
  const calibration::RectilinearCamerad& cam,
  const int num_initial_monodepth_splats,
  cv::Mat& image,
  cv::Mat& depthmap,
  cv::Mat& mask
) {
  Eigen::Matrix3d world_R_cam = cam.camFromWorld().linear().inverse().matrix();

  float margin_x = cam.getWidth() * 0.01; // Crop to avoid boundary issues in the mono depthmap
  float margin_y = cam.getHeight() * 0.01; 

  cv::Mat dilated_mask;
  cv::dilate(mask, dilated_mask, cv::Mat(), cv::Point(-1, -1), 1);

  std::vector<Eigen::Vector3f> pointcloud, pointcloud_colors;
  while(pointcloud.size() < num_initial_monodepth_splats) {
    float rand_x = margin_x + math::randUnif() * (cam.getWidth() - 2.0 * margin_x);
    float rand_y = margin_y + math::randUnif() * (cam.getHeight() - 2.0 * margin_y);

    if (dilated_mask.at<uint8_t>(rand_y, rand_x) > 0) continue;

    const Eigen::Vector3d ray_dir_in_cam = cam.rayDirFromPixel(Eigen::Vector2d(rand_x, rand_y));

    float d = opencv::getPixelBilinear<float>(depthmap, rand_x, rand_y);
    
    if (d < kZNear) { continue; }

    Eigen::Vector3d adjusted_ray = Eigen::Vector3d(
      ray_dir_in_cam.x() * d / ray_dir_in_cam.z(), 
      ray_dir_in_cam.y() * d / ray_dir_in_cam.z(), 
      d);
    const Eigen::Vector3d point_in_world = cam.getPositionInWorld() + world_R_cam * adjusted_ray;
    cv::Vec3f color = cv::Vec3f(opencv::getPixelBilinear<cv::Vec3b>(
      image, rand_x, rand_y)) / 255.0f;
    pointcloud.push_back(point_in_world.cast<float>());
    pointcloud_colors.emplace_back(color[0], color[1], color[2]);
  }
  XPLINFO << "Created " << pointcloud.size() << " points using mono depth estimation";

  return initSplatTensorsFromPointCloud(
    device, pointcloud, pointcloud_colors, max_num_splats);
}

// Only populate un-alive splats
void addNewSplatsToModelFromSingleImageRGBD(
  const torch::DeviceType& device,
  const SplatConfig& cfg,
  const calibration::RectilinearCamerad& cam,
  const int target_num_splats_to_add,
  cv::Mat& image,
  cv::Mat& depthmap,
  cv::Mat& mask,
  std::shared_ptr<SplatModel>& model
) {
  torch::NoGradGuard no_grad;

  int curr_num_alive = model->splat_alive.sum().item<int>();
  int num_splats_to_add = std::min(target_num_splats_to_add, int(model->splat_alive.size(0) - curr_num_alive));
  XPLINFO << "Adding splats to existing model, # to add: " << num_splats_to_add;

  Eigen::Matrix3d world_R_cam = cam.camFromWorld().linear().inverse().matrix();

  float margin_x = cam.getWidth() * 0.01; // Crop to avoid boundary issues in the mono depthmap
  float margin_y = cam.getHeight() * 0.01;
  XCHECK_EQ(mask.type(), CV_32F);
  cv::Mat dilated_mask;
  cv::dilate(mask, dilated_mask, cv::Mat(), cv::Point(-1, -1), 1);

  std::vector<Eigen::Vector3f> pointcloud, pointcloud_colors;
  while(pointcloud.size() < num_splats_to_add) {
    float rand_x = margin_x + math::randUnif() * (cam.getWidth() - 2.0 * margin_x);
    float rand_y = margin_y + math::randUnif() * (cam.getHeight() - 2.0 * margin_y);

    if (dilated_mask.at<float>(rand_y, rand_x) > 0.5) continue;

    const Eigen::Vector3d ray_dir_in_cam = cam.rayDirFromPixel(Eigen::Vector2d(rand_x, rand_y));

    float d = opencv::getPixelBilinear<float>(depthmap, rand_x, rand_y);
    
    if (d < kZNear) { continue; }

    Eigen::Vector3d adjusted_ray = Eigen::Vector3d(
      ray_dir_in_cam.x() * d / ray_dir_in_cam.z(), 
      ray_dir_in_cam.y() * d / ray_dir_in_cam.z(), 
      d);
    const Eigen::Vector3d point_in_world = cam.getPositionInWorld() + world_R_cam * adjusted_ray;
    cv::Vec3f color = cv::Vec3f(opencv::getPixelBilinear<cv::Vec3b>(
      image, rand_x, rand_y)) / 255.0f;
    pointcloud.push_back(point_in_world.cast<float>());
    pointcloud_colors.emplace_back(color[0], color[1], color[2]);
  }

  // Compute the average distance to neighbors for the new points
  const std::vector<float> avg_dist_to_neighbors = calcAvgDistToNearestNeighbors(pointcloud);


  // Identify dead splat indices using batch operations.
  torch::Tensor dead_mask = model->splat_alive.logical_not(); // shape {N,1}
  // Use select to extract the first column, resulting in a 1D tensor.
  torch::Tensor dead_indices = torch::nonzero(dead_mask).select(1, 0);
  // Only take as many indices as needed.
  dead_indices = dead_indices.slice(0, 0, num_splats_to_add);

  // Prepare new splat data.
  constexpr float kDefaultAlpha = -1.0f;
  std::vector<float> new_pos_data, new_color_data, new_alpha_data, new_scale_data, new_quat_data;
  new_pos_data.reserve(num_splats_to_add * 3);
  new_color_data.reserve(num_splats_to_add * 3);
  new_alpha_data.reserve(num_splats_to_add);
  new_scale_data.reserve(num_splats_to_add * 3);
  new_quat_data.reserve(num_splats_to_add * 4);

  for (int i = 0; i < num_splats_to_add; ++i) {
      // Position.
      const Eigen::Vector3f& pos = pointcloud[i];
      new_pos_data.push_back(pos.x());
      new_pos_data.push_back(pos.y());
      new_pos_data.push_back(pos.z());
      // Color (apply inverse sigmoid conversion).
      const Eigen::Vector3f& col = pointcloud_colors[i];
      //const Eigen::Vector3f col(0.5, 0, 0);
      new_color_data.push_back(sigmoidInverse(col.x()));
      new_color_data.push_back(sigmoidInverse(col.y()));
      new_color_data.push_back(sigmoidInverse(col.z()));
      // Alpha.
      new_alpha_data.push_back(kDefaultAlpha);
      // Scale from neighbor distances.
      float log_val = std::log(avg_dist_to_neighbors[i]);
      float s = math::clamp<float>(log_val, -kMaxScaleExponent, 0.0f);
      float scale = inverseScaleActivation(s);
      new_scale_data.push_back(scale);
      new_scale_data.push_back(scale);
      new_scale_data.push_back(scale);
      // Default quaternion (no rotation).
      new_quat_data.push_back(0.0f);
      new_quat_data.push_back(0.0f);
      new_quat_data.push_back(0.0f);
      new_quat_data.push_back(1.0f);
  }

  // Create new tensors from the vectors.
  auto options = torch::TensorOptions().dtype(torch::kFloat32);
  auto new_pos_tensor = torch::from_blob(new_pos_data.data(), {num_splats_to_add, 3}, options).clone().to(device);
  auto new_color_tensor = torch::from_blob(new_color_data.data(), {num_splats_to_add, 3}, options).clone().to(device);
  auto new_alpha_tensor = torch::from_blob(new_alpha_data.data(), {num_splats_to_add, 1}, options).clone().to(device);
  auto new_scale_tensor = torch::from_blob(new_scale_data.data(), {num_splats_to_add, 3}, options).clone().to(device);
  auto new_quat_tensor = torch::from_blob(new_quat_data.data(), {num_splats_to_add, 4}, options).clone().to(device);

  // Batch update the model's tensors.
  // Apply inverse activation on the new positions.
  auto updated_pos = splatPosInverseActivation(new_pos_tensor);
  model->splat_pos.index_copy_(0, dead_indices, updated_pos);
  model->splat_color.index_copy_(0, dead_indices, new_color_tensor);
  model->splat_alpha.index_copy_(0, dead_indices, new_alpha_tensor);
  model->splat_scale.index_copy_(0, dead_indices, new_scale_tensor);
  model->splat_quat.index_copy_(0, dead_indices, new_quat_tensor);

  // Mark the updated splats as alive.
  model->splat_alive.index_fill_(0, dead_indices, true);

  XPLINFO << "Updated " << num_splats_to_add << " splats in the model";
}

torch::Tensor getVisibileInAnyCameraAndNotTooClose(
  const torch::DeviceType device,
  const calibration::MultiCameraDataset& dataset,
  std::shared_ptr<SplatModel> model)
{
  torch::NoGradGuard no_grad;

  constexpr float kTooCloseZ = 0.01;

  torch::Tensor visible_in_any = torch::zeros({model->splat_pos.size(0)},
      torch::TensorOptions().dtype(torch::kBool).device(device));
  torch::Tensor too_close_in_any = torch::zeros({model->splat_pos.size(0)},
      torch::TensorOptions().dtype(torch::kBool).device(device));

  for (int i = 0; i < dataset.cameras.size(); ++i) {
    XCHECK(dataset.cameras[i].is_rectilinear);
    const auto& cam = dataset.cameras[i].rectilinear;
    
    auto splat_pos = model->splat_pos;

    torch::Tensor pos_linear = splatPosActivation(splat_pos); // [N, 3]
    torch::Tensor ones = torch::ones({pos_linear.size(0), 1}, pos_linear.options());
    torch::Tensor homogeneous_splat_pos = torch::cat({pos_linear, ones}, 1); // [N, 4]
  
    torch::Tensor cam_from_world_transpose = isometryToTensor(device, cam.cam_from_world);
    torch::Tensor splat_in_cam = torch::matmul(homogeneous_splat_pos, cam_from_world_transpose);
    torch::Tensor x = splat_in_cam.select(1, 0);
    torch::Tensor y = splat_in_cam.select(1, 1);
    torch::Tensor z = splat_in_cam.select(1, 2);
    torch::Tensor u = cam.focal_length.x() * x / z + cam.optical_center.x();
    torch::Tensor v = -cam.focal_length.y() * y / z + cam.optical_center.y();

    // Dont just check screen coordinates, include some margin
    torch::Tensor valid_mask = (z > kZNear) &
      (u >= -cam.width) & (u < cam.width * 2.0) &
      (v >= -cam.height) & (v < cam.height * 2.0);
    visible_in_any = visible_in_any | valid_mask;

    torch::Tensor too_close_mask = (z < kTooCloseZ) & (z > 0);
    too_close_in_any = too_close_in_any | too_close_mask;
  }

  torch::Tensor final_mask = visible_in_any & (~too_close_in_any);
  return final_mask.unsqueeze(-1);
}

void splatPopulationDynamics(
  const torch::DeviceType& device,
  const SplatConfig& cfg,
  const calibration::MultiCameraDataset& dataset,
  std::shared_ptr<SplatModel> model,
  torch::Tensor& should_stabilize,
  int target_num_alive,
  torch::Tensor grad2d_norm
) {
  torch::NoGradGuard no_grad;
  const int num_splats = model->splat_pos.size(0);

  auto sigmoid_alpha = torch::sigmoid(model->splat_alpha).squeeze(-1);
  auto exp_scale    = torch::exp(scaleActivation(model->splat_scale));
  auto normalized_quat = model->splat_quat
    / (torch::norm(model->splat_quat, 2, 1, true) + 1e-6);
  auto splat_R = quaternionTo3x3(normalized_quat);

  // Check for zero-gradient splats (not contributing to optimization)
  constexpr float kMinGradientThreshold = 1e-9f;
  auto has_gradient = (grad2d_norm > kMinGradientThreshold).unsqueeze(-1);
  auto alive_before = model->splat_alive.squeeze(-1);
  auto zero_grad_alive = alive_before & (~has_gradient.squeeze(-1));
  int zero_grad_count = zero_grad_alive.sum().item<int>();
  XPLINFO << "Found " << zero_grad_count << " alive splats with zero gradients - culling them";

  // kill logic
  auto is_not_nan    = ~torch::isnan(model->splat_pos).any(1, true);
  auto alpha_valid   = sigmoid_alpha.unsqueeze(-1) > kDeadSplatAlphaThreshold;
  auto not_too_small = torch::all(
    exp_scale > std::exp(-kMaxScaleExponent * 0.95), 1
  ).unsqueeze(-1);
  auto valid_vis = getVisibileInAnyCameraAndNotTooClose(
    device, dataset, model
  );
  auto preserve_alive = is_not_nan & alpha_valid & not_too_small & valid_vis & has_gradient;
  model->splat_alive &= preserve_alive;

  // prevent cloning dead/static
  grad2d_norm *= model->splat_alive.squeeze(-1);

  // reset stabilize for killed splats
  should_stabilize = torch::where(
    model->splat_alive,
    should_stabilize,
    torch::zeros_like(should_stabilize, torch::kBool)
  );

  // spawn exactly to target_num_alive
  int curr = model->splat_alive.sum().item<int>();
  XPLINFO << "curr_num_alive: " << curr;

  int max_growth_rate = curr * 2.0;
  target_num_alive = std::min(target_num_alive, max_growth_rate);
  XPLINFO << "growth rate limited target_num_alive: " << target_num_alive;

  int to_spawn = target_num_alive - curr;
  to_spawn = std::max(0, std::min(to_spawn, num_splats - curr));
  if (to_spawn <= 0) return;
  XPLINFO << "num_to_spawn: " << to_spawn;

  // pick parents (repeat if needed) - ONLY from alive splats
  auto alive_mask = model->splat_alive.squeeze(-1);
  auto alive_indices = torch::nonzero(alive_mask).squeeze(-1);
  auto alive_grad2d_norm = grad2d_norm.index_select(0, alive_indices);
  auto sorted_alive = torch::argsort(alive_grad2d_norm, 0, true).squeeze(-1);
  int P = alive_indices.size(0), R = (to_spawn + P - 1) / P;
  auto source_indices_in_alive = sorted_alive.repeat({R}).slice(0, 0, to_spawn);
  auto source_indices = alive_indices.index_select(0, source_indices_in_alive);

  // pick targets
  auto low_imp = model->splat_alive.squeeze(-1) * sigmoid_alpha;
  auto target_indices = torch::argsort(low_imp, 0, false)
                          .slice(0, 0, to_spawn);

  // compute offsets
  constexpr float kScaleFactor = 0.1f;
  auto rand1 = (torch::rand({to_spawn, 3}, device) - 0.5) * kScaleFactor;
  auto rand2 = (torch::rand({to_spawn, 3}, device) - 0.5) * kScaleFactor;
  auto src_scales = torch::exp(
    scaleActivation(model->splat_scale.index_select(0, source_indices))
  );
  auto v1 = rand1 * src_scales;
  auto v2 = rand2 * src_scales;
  auto Rsrc = splat_R
    .index_select(0, source_indices)
    .transpose(1, 2);
  auto offset1 = torch::bmm(Rsrc, v1.unsqueeze(-1)).squeeze(-1);
  auto offset2 = torch::bmm(Rsrc, v2.unsqueeze(-1)).squeeze(-1);

  // shrink sources
  model->splat_scale.index_copy_(
    0, source_indices,
    inverseScaleActivation(
      scaleActivation(
        model->splat_scale.index_select(0, source_indices)
      ) - std::log(1.1)
    )
  );

  // clone into targets
  model->splat_pos.index_copy_(
    0, target_indices,
    model->splat_pos.index_select(0, source_indices) + offset1
  );
  model->splat_color.index_copy_(0, target_indices,
    model->splat_color.index_select(0, source_indices)
  );
  model->splat_alpha.index_copy_(0, target_indices,
    model->splat_alpha.index_select(0, source_indices)
  );
  model->splat_scale.index_copy_(0, target_indices,
    model->splat_scale.index_select(0, source_indices)
  );
  model->splat_quat.index_copy_(0, target_indices,
    model->splat_quat.index_select(0, source_indices)
  );
  model->splat_alive.index_fill_(0, target_indices, true);

  // reset age & stabilize for new splats
  should_stabilize.index_fill_(0, source_indices, false);
  should_stabilize.index_fill_(0, target_indices, false);

  // push sources outward
  model->splat_pos.index_add_(0, source_indices, offset2);
}



}}  // end namespace p11::splat
