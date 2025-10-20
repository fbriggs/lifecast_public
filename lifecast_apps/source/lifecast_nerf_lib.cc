// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "lifecast_nerf_lib.h"

#include <fstream>
#include <random>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "ngp_radiance_model.h"
#include "util_math.h"
#include "util_string.h"
#include "util_opencv.h"
#include "third_party/json.h"
#include "util_torch.h"
#include "util_math.h"
#include "util_file.h"
#include "ldi_common.h"
#include "vignette.h"
#include "nerf_heuristic_seg.h"
#include "ldi_image_based_rendering.h"
#include "multicamera_dataset.h"

namespace p11 { namespace nerf {

namespace {
std::mt19937 rng(123); // Global RNG
};

// TODO: expose these as configurable parameters?
static constexpr float kZNear = 0.2;
static constexpr float kZMid = 10.0; // Here we switch from linear to inverse sampling
static constexpr float kZFar = 1000.0;

// Based on: https://github.com/SteEsp/TinyNeRF-LibTorch-Renderer/blob/main/main.cpp
torch::Tensor cumprodExclusive(torch::Tensor tensor)
{
  auto cumprod = torch::cumprod(tensor, -1);
  cumprod = torch::roll(cumprod, 1, -1);
  cumprod.index({"...", 0}) = 1.0;
  return cumprod;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> generateBatchRays(
  const torch::DeviceType device,
  const calibration::MultiCameraDataset& dataset,
  const int rays_per_batch
) {
  const int num_frames = dataset.images.size();
  const int width = dataset.cameras[0].getWidth(); // TODO: support mix of different size images
  const int height = dataset.cameras[0].getHeight();
  XCHECK_EQ(dataset.images[0].type(), CV_8UC3);

  std::uniform_int_distribution<int> rand_image_dist(0, num_frames - 1);
  std::uniform_real_distribution<float> rand_x_dist(0, width);
  std::uniform_real_distribution<float> rand_y_dist(0, height);

  std::vector<float> ray_origin_data, ray_dir_data, target_color_data;
  std::vector<int32_t> image_idx_data;
  int i = 0;
  while(i < rays_per_batch) {
    const int rand_image = rand_image_dist(rng);
    image_idx_data.push_back(rand_image);
    const float x = rand_x_dist(rng);
    const float y = rand_y_dist(rng);

    if (!dataset.cameras[rand_image].ignore_mask.empty()) {
      bool is_masked_out = (255 == opencv::getPixelExtend<uint8_t>(dataset.cameras[rand_image].ignore_mask, x, y));
      if (is_masked_out) continue;
    }

    if (dataset.cameras[rand_image].is_fisheye) {
      const double dist_from_center =
        (Eigen::Vector2d(x, y) - dataset.cameras[rand_image].fisheye.optical_center).norm();
      if (dist_from_center > dataset.cameras[rand_image].fisheye.useable_radius) continue;
    }

    const Eigen::Vector3d ray_dir_in_cam = dataset.cameras[rand_image].rayDirFromPixel(Eigen::Vector2d(x, y));
    const Eigen::Vector3d ray_dir_in_world = dataset.cameras[rand_image].camFromWorld().linear().transpose() * ray_dir_in_cam; // TODO: precompute worldFromCam!
    const Eigen::Vector3d ray_origin_in_world = dataset.cameras[rand_image].getPositionInWorld(); // TODO: precompute
    const cv::Vec3f target_color =  cv::Vec3f(opencv::getPixelBilinear<cv::Vec3b>(dataset.images[rand_image], x, y)) / 255.0f;
    // TODO: try bicubic interpolation

    ray_origin_data.push_back(ray_origin_in_world.x());
    ray_origin_data.push_back(ray_origin_in_world.y());
    ray_origin_data.push_back(ray_origin_in_world.z());
    ray_dir_data.push_back(ray_dir_in_world.x());
    ray_dir_data.push_back(ray_dir_in_world.y());
    ray_dir_data.push_back(ray_dir_in_world.z());
    target_color_data.push_back(target_color[0]);
    target_color_data.push_back(target_color[1]);
    target_color_data.push_back(target_color[2]);

    ++i;
  }

  torch::Tensor image_idxs = torch::from_blob(image_idx_data.data(), {rays_per_batch}, {torch::kInt32}).to(device);
  torch::Tensor ray_origins = torch::from_blob(ray_origin_data.data(), {rays_per_batch, 3}, {torch::kFloat32}).to(device);
  torch::Tensor ray_dirs = torch::from_blob(ray_dir_data.data(), {rays_per_batch, 3}, {torch::kFloat32}).to(device);
  torch::Tensor target_colors = torch::from_blob(target_color_data.data(), {rays_per_batch, 3}, {torch::kFloat32}).to(device);
  util_torch::cloneIfOnCPU(image_idxs);
  util_torch::cloneIfOnCPU(ray_origins);
  util_torch::cloneIfOnCPU(ray_dirs);
  util_torch::cloneIfOnCPU(target_colors);

  return {ray_origins, ray_dirs, target_colors, image_idxs}; 
}

// Uniform sampling from kZNear to kZMid, then inverse distance sampling from kZMid to kZFar.
std::tuple<torch::Tensor, torch::Tensor> computeQueryPointsFromRays(
  torch::Tensor ray_origins,
  torch::Tensor ray_dirs,
  const int num_samples
) {
  const int half_samples = num_samples / 2;

  // First half: regular uniform sampling from kZNear to kZMid
  torch::Tensor depth_values_near = torch::linspace(kZNear, kZMid, half_samples).unsqueeze(0).to(ray_origins.device());
  depth_values_near = depth_values_near.repeat({ray_origins.size(0), 1});
  depth_values_near += torch::rand_like(depth_values_near) * (kZMid - kZNear) / (half_samples-1);

  // Second half: inverse depth sampling from kZMid to kZFar
  constexpr double kZMidInv = 1.0 / kZMid;
  constexpr double kZFarInv = 1.0 / kZFar;
  torch::Tensor inverse_depth_values = torch::linspace(kZFarInv, kZMidInv, half_samples).unsqueeze(0).to(ray_origins.device());
  inverse_depth_values = inverse_depth_values.repeat({ray_origins.size(0), 1});
  inverse_depth_values += torch::rand_like(inverse_depth_values) * (kZMidInv - kZFarInv) / (half_samples-1);
  torch::Tensor depth_values_far = 1.0 / inverse_depth_values;
  depth_values_far = depth_values_far.flip({1});

  // Combine the depth values from both strategies
  torch::Tensor depth_values = torch::cat({depth_values_near, depth_values_far}, 1);
  depth_values = std::get<0>(torch::sort(depth_values, 1)); // TODO HACK: slow, but fixes a bug where there are unsorted values at the transition between strategies!
  
  torch::Tensor query_points = ray_origins.unsqueeze(1) + ray_dirs.unsqueeze(1) * depth_values.unsqueeze(-1);

  return {query_points, depth_values};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> renderVolumeDensity(
  torch::Tensor rgb,
  torch::Tensor sigma,
  torch::Tensor depth_values,
  const bool transparent_background
) {
  using namespace torch::indexing;

  auto one_e_10 = torch::full({1}, 1e10, depth_values.options());
  auto dists = torch::cat({
        depth_values.index({"...", Slice(1, None)}) - 
        depth_values.index({"...", Slice(None, -1)}),
        one_e_10.expand(depth_values.index({"...", Slice(0, 1)}).sizes())
    }, -1);

  auto alpha = 1.0 - torch::exp(-sigma * dists);
  if (transparent_background) {
    alpha = torch::where(dists < 1e10, alpha, 0.0);
  }
  auto weights = alpha * cumprodExclusive(1.0 - alpha + 1e-10);

  auto rgb_map = (weights.unsqueeze(-1) * rgb).sum(-2);
  auto inv_depth = torch::clamp(kInvDepthCoef / (depth_values + 1e-6), 0.0, 1.0);
  auto depth_map = (weights * inv_depth).sum(-1);

  auto acc_map = weights.sum(-1);

  return {rgb_map, depth_map, acc_map, alpha, weights};
}


torch::Tensor renderVolumeDensityInverseDepthOnly(
  torch::Tensor sigma,
  torch::Tensor depth_values
) {
  using namespace torch::indexing;

  auto one_e_10 = torch::full({1}, 1e10, depth_values.options());
  auto dists = torch::cat({
        depth_values.index({"...", Slice(1, None)}) - 
        depth_values.index({"...", Slice(None, -1)}),
        one_e_10.expand(depth_values.index({"...", Slice(0, 1)}).sizes())
    }, -1);

  auto alpha = 1.0 - torch::exp(-sigma * dists);
  auto weights = alpha * cumprodExclusive(1.0 - alpha + 1e-10);
  auto inv_depth = torch::clamp(kInvDepthCoef / (depth_values + 1e-6), 0.0, 1.0);
  auto depth_map = (weights * inv_depth).sum(-1);
  return depth_map;
}

// Any sample outside [min_dist, max_dist] will get 0 weight.
// the returned acc_map is essentially the alpha channel for the part of the 
// volume that fits within the bounds.
// TODO: a more advanced version could use different bounds per-ray
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
renderVolumeDensityWithDistanceBounds(
  torch::Tensor rgb,
  torch::Tensor sigma,
  torch::Tensor depth_values,
  torch::Tensor min_dist,
  torch::Tensor max_dist,
  bool transparent_bg
) {
  using namespace torch::indexing;

  auto one_e_10 = torch::full({1}, 1e10, depth_values.options());

  auto dists = torch::cat({
        depth_values.index({"...", Slice(1, None)}) - 
        depth_values.index({"...", Slice(None, -1)}),
        one_e_10.expand(depth_values.index({"...", Slice(0, 1)}).sizes())
    }, -1);

  auto alpha = 1.0 - torch::exp(-sigma * dists);
  auto outside_bounds = (depth_values < min_dist) | (depth_values > max_dist);
  alpha = torch::where(
    transparent_bg ? outside_bounds | (dists >= 1e10) : outside_bounds,
    torch::zeros_like(alpha),
    alpha);

  auto weights = alpha * cumprodExclusive(1.0 - alpha + 1e-10);

  auto rgb_map = (weights.unsqueeze(-1) * rgb).sum(-2);
  auto acc_map = weights.sum(-1);

  // BAD:
  //auto depth_map = (weights * depth_values).sum(-1);
  
  // GOOD: blending inverse depth instead of depth is super important to not make silloutte artifacts
  // This is because when the blending weights sum to less than 0, the depthmap is biased toward 0.
  // 0 in inverse depth is good (far away). 1 in regular depth is bad (very close).
  auto inv_depth = torch::clamp(kInvDepthCoef / (depth_values + 1e-6), 0.0, 1.0);
  auto depth_map = (weights * inv_depth).sum(-1);

  // Normalize the RGB map by the accumulated alpha map to adjust the color intensity
  // Avoid division by zero by adding a small epsilon where acc_map is zero
  auto epsilon = 1e-10;
  auto normalized_rgb_map = rgb_map / (acc_map.unsqueeze(-1) + epsilon);

  // TODO: this is better than not normalizing depth, but when alpha (acc_map) is very low,
  // it pushes the depth too close.
  auto normalized_depth_map = torch::clamp(depth_map / (acc_map + epsilon), 0.0, 1.0);

  return {normalized_rgb_map, normalized_depth_map, acc_map, alpha, weights};
}

// Factored out the core part of importanceSampling to make it easier to test.
// Tensor shapes use N = # of rays, and S = # of proposal samples per ray.
std::tuple<torch::Tensor, torch::Tensor> importanceSamplingHelper(
  torch::Tensor ray_origins,      // [N, 3]
  torch::Tensor ray_dirs,         // [N, 3]
  torch::Tensor depth_values,     // [N, S]
  torch::Tensor sigma,            // [N, S]
  const int num_importance_samples, // How many output samples to generate, not the same as to S
  const bool include_original_samples
) {
  using namespace torch::indexing;
  torch::NoGradGuard no_grad;

  // Add 1e-5 to all elements of sigma except the last one. This prevents NaN, but we 
  // must avoid putting too much weight on the last sample which can mess up importance sampling.
  sigma = torch::cat({
      sigma.index({"...", Slice(None, -1)}) + 1e-5,
      sigma.index({"...", -1}).unsqueeze(-1)  // Add an extra dimension to the last element
  }, -1);

  // Compute distances between sample points along the ray
  auto one_e_10 = torch::full({1}, 1e10, ray_origins.options());
  auto dists = torch::cat({
        depth_values.index({"...", Slice(1, None)}) - 
        depth_values.index({"...", Slice(None, -1)}),
        one_e_10.expand(depth_values.index({"...", Slice(0, 1)}).sizes())
    }, -1);

  // Compute importance weights. This is similar to renderVolumeDensity,
  // but normalized to ensure a PDF that sums to 1.
  auto alpha = 1.0 - torch::exp(-sigma * dists);
  auto weights = alpha * cumprodExclusive(1.0 - alpha + 1e-10);
  auto pdf = weights / weights.sum(-1, /*keepdim=*/true);
  auto cdf = torch::cumsum(pdf, -1);

  // Make sure the CDF starts with 0
  cdf = torch::cat({torch::zeros({cdf.size(0), 1}, cdf.options()), cdf}, -1);

  auto bin_start = torch::cat({torch::full({ray_origins.size(0), 1}, kZNear, depth_values.options()), 
                               (depth_values.index({"...", Slice(None, -1)}) + depth_values.index({"...", Slice(1, None)})) / 2}, -1);
  auto bin_end = torch::cat({(depth_values.index({"...", Slice(None, -1)}) + depth_values.index({"...", Slice(1, None)})) / 2, 
                             torch::full({ray_origins.size(0), 1}, kZFar, depth_values.options())}, -1);

  // Inverse transform sampling using new bin boundaries
  torch::Tensor uniform_samples = torch::rand({ray_origins.size(0), num_importance_samples}, ray_origins.options());
  torch::Tensor indices = torch::searchsorted(cdf, uniform_samples, /*int32=*/false, /*right=*/false);
  indices = torch::clamp(indices - 1, 0, bin_start.size(1) - 2);

  // Interpolation within new bins
  torch::Tensor bins_start = bin_start.gather(-1, indices);
  torch::Tensor bins_end = bin_end.gather(-1, indices);
  torch::Tensor cdf_g_start = cdf.gather(-1, indices);
  torch::Tensor cdf_g_end = cdf.index({"...", Slice(1, None)}).gather(-1, indices);
  torch::Tensor denom = cdf_g_end - cdf_g_start;
  denom = torch::where(denom < 1e-5, torch::ones_like(denom), denom);
  torch::Tensor t = (uniform_samples - cdf_g_start) / denom;
  torch::Tensor new_depth_values = bins_start + t * (bins_end - bins_start);

  // Ensure sorted depth values
  torch::Tensor to_sort;
  if (include_original_samples) {
    to_sort = torch::cat({new_depth_values, depth_values}, 1);
  } else {
    to_sort = new_depth_values;
  }
  auto sorted = torch::sort(to_sort, 1);
  torch::Tensor sorted_depth_values = std::get<0>(sorted);

  // Calculate new query points
  torch::Tensor new_query_points = ray_origins.unsqueeze(1) + ray_dirs.unsqueeze(1) * sorted_depth_values.unsqueeze(-1);

  return {new_query_points, sorted_depth_values};
}

std::tuple<torch::Tensor, torch::Tensor> importanceSampling(
  bool warmup,
  std::shared_ptr<ProposalDensityModel>& proposal_model,
  torch::Tensor ray_origins,
  torch::Tensor ray_dirs,
  torch::Tensor query_points,
  torch::Tensor depth_values,
  const int num_importance_samples,
  const bool include_original_samples
) {
  torch::NoGradGuard no_grad;

  if (warmup) return {query_points, depth_values};

  // Compute sigma from density model
  torch::Tensor flat_query_points = query_points.reshape({-1, 3});
  torch::Tensor sigma = proposal_model->pointToDensity(flat_query_points).reshape(depth_values.sizes());
  
  return importanceSamplingHelper(ray_origins, ray_dirs, depth_values, sigma, num_importance_samples, include_original_samples);
}

double applyLearningRateSchedule(
  torch::optim::Optimizer& optimizer,
  double initial_lr,
  int current_iter,
  const std::vector<int>& milestones,
  double gamma,               // Learning rate decays by this factor each time we reach a milestone
  double warmup_scale_factor, // At the first iteration the learning rate is lr_at_itr0, 
  int warmup_itrs             // then it ramps up to initial_lr over this many iterations.
){
  double lr_at_itr0 = initial_lr * warmup_scale_factor;
  double curr_lr;

  // Linear ramp up during warmup phase.
  if (current_iter < warmup_itrs) {
    curr_lr = lr_at_itr0 + (initial_lr - lr_at_itr0) * float(current_iter) / float(warmup_itrs);
  } else {
    // If not in warmup, use the initial_lr as base
    curr_lr = initial_lr;

    // Apply multi-step LR decay if a milestone is reached
    for (int milestone : milestones) {
      if (current_iter >= milestone) {
        curr_lr *= gamma;
      }
    }
  }

  // Update the learning rate for each parameter group
  for (auto& group : optimizer.param_groups()) {
    auto& options = static_cast<torch::optim::AdamOptions&>(group.options());
    options.lr(curr_lr);
  }

  return curr_lr;
}


// Returns the proposal network training loss
float updateProposalModel(
  NeoNerfConfig& cfg,
  const torch::DeviceType device,
  const calibration::MultiCameraDataset& dataset,
  std::shared_ptr<NeoNerfModel>& radiance_model,
  std::shared_ptr<ProposalDensityModel>& proposal_model,
  torch::optim::Adam& proposal_model_optimizer
) {
  radiance_model->eval();
  proposal_model->train();

  const auto [ray_origins, ray_dirs, _, __] = generateBatchRays(device, dataset, cfg.rays_per_batch);
  const auto [query_points, depths] = computeQueryPointsFromRays(ray_origins, ray_dirs, cfg.num_basic_samples);
  torch::Tensor flat_query_points = query_points.reshape({-1, 3});

  auto nerf_sigma = radiance_model->pointToDensity(flat_query_points);
  auto prop_sigma = proposal_model->pointToDensity(flat_query_points).squeeze(-1);

  torch::Tensor prop_loss = torch::smooth_l1_loss(nerf_sigma, prop_sigma);

  proposal_model_optimizer.zero_grad();
  prop_loss.backward();
  proposal_model_optimizer.step();

  return prop_loss.item<float>();
}

// Generate a batch of random points, and for each one, check if it is in any
// camera's frustum. 
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> generateVisibilityRegularizationPoints(
  const torch::DeviceType device,
  const calibration::MultiCameraDataset& dataset,
  const int num_points
) {
  constexpr double kMaxRegularizationDist = 100;
  constexpr int kMinCamerasToNotRegularize = 1;
  std::vector<float> points_data, dir_data, visibility_data;
  for (int i = 0; i < num_points; ++i) {
    const double dist = kMaxRegularizationDist * math::randUnif();
    const Eigen::Vector3d dir = math::randomUnitVec();
    const Eigen::Vector3d point = dist * dir;
  
    int visibility_count = 0; // how many cameras jave this point in their frustum
    for (const auto& cam : dataset.cameras) {
      const Eigen::Vector3d point_in_cam = cam.camFromWorld(point);
      if (point_in_cam.z() < 0) continue; // point is behind camera

      const Eigen::Vector2d projected_pixel = cam.pixelFromCam(point_in_cam);
      // does the point project into the image?
      if (projected_pixel.x() < 0 ||
          projected_pixel.y() < 0 ||
          int(projected_pixel.x()) > cam.getWidth() - 1 ||
          int(projected_pixel.y()) > cam.getHeight() - 1) continue;
      // If we got this far, point is in cam's frustum.
      ++visibility_count;
    }

    points_data.push_back(point.x());
    points_data.push_back(point.y());
    points_data.push_back(point.z());
    dir_data.push_back(dir.x());
    dir_data.push_back(dir.y());
    dir_data.push_back(dir.z());
    visibility_data.push_back(visibility_count >= kMinCamerasToNotRegularize ? 0.0 : 1.0);
  }

  torch::Tensor points = torch::from_blob(points_data.data(), {num_points, 3}, torch::kFloat32).to(device);
  torch::Tensor dirs = torch::from_blob(dir_data.data(), {num_points, 3}, torch::kFloat32).to(device);
  torch::Tensor visibility = torch::from_blob(visibility_data.data(), {num_points, 1}, torch::kFloat32).to(device);
  util_torch::cloneIfOnCPU(points);
  util_torch::cloneIfOnCPU(dirs);
  util_torch::cloneIfOnCPU(visibility);

  return {points, dirs, visibility};
}

torch::Tensor distortionLoss(const torch::Tensor& s, const torch::Tensor& w) {
  // Calculate the midpoints of s
  torch::Tensor s_mid = 0.5 * (s.slice(1, 0, -1) + s.slice(1, 1));

  // Calculate the size of each interval
  torch::Tensor s_size = s.slice(1, 1) - s.slice(1, 0, -1);

  // Weighted size term
  torch::Tensor size_term = torch::mean(w.slice(1, 0, -1).pow(2) * s_size / 3.0);

  // Prepare for broadcasting by adding extra dimensions
  torch::Tensor s_mid_i = s_mid.unsqueeze(2); // Shape: [4096, 127, 1]
  torch::Tensor s_mid_j = s_mid.unsqueeze(1); // Shape: [4096, 1, 127]
  torch::Tensor w_i = w.slice(1, 0, -1).unsqueeze(2); // Shape: [4096, 127, 1]
  torch::Tensor w_j = w.slice(1, 0, -1).unsqueeze(1); // Shape: [4096, 1, 127]

  // Compute distances between all pairs of midpoints
  torch::Tensor distances = torch::abs(s_mid_i - s_mid_j); // Shape: [4096, 127, 127]

  // Weighted distance term
  torch::Tensor distance_term = torch::mean(w_i * w_j * distances);

  // Combine terms
  return size_term + distance_term;
}

void trainRadianceModel(
    NeoNerfConfig& cfg,
    const torch::DeviceType device,
    const calibration::MultiCameraDataset& dataset,
    std::shared_ptr<NeoNerfModel>& radiance_model,
    std::shared_ptr<ProposalDensityModel>& proposal_model,
    std::shared_ptr<NeoNerfModel>& prev_radiance_model, // May be nullptr. If not, this is a model that we regularize toward
    torch::Tensor& image_codes,
    NeoNerfGuiData* gui_data = nullptr
){
  using namespace torch::optim;
  auto train_timer = time::now();

  image_codes.set_requires_grad(true);

  // Setup optimizers and learning rate schedule
  auto image_code_optimizer = Adam(
    {image_codes},
    AdamOptions(cfg.image_code_lr)
    .weight_decay(cfg.image_code_decay)
    .eps(cfg.adam_eps));

  // Split the radiance model parameters up into hashmap and other (MLP).
  // hashmap doesn't get weight decay.

  auto is_hashmap_key = [](const std::string& key) {
    // Nested torch modules end in ".hashmap". Others are just "hashmap"
    static const std::string hashmap_param_suffix = std::string(".") + kHashmapParamsName;
    return key == kHashmapParamsName || string::endsWith(key, hashmap_param_suffix);
  };

  std::vector<torch::Tensor> radiance_hashmap_params, radiance_mlp_params;
  for (const auto& param : radiance_model->named_parameters()) {
    auto& dest = is_hashmap_key(param.key()) ? radiance_hashmap_params : radiance_mlp_params;
    dest.push_back(param.value());
  }
  XCHECK(!radiance_hashmap_params.empty()) << "Radiance hashmap params not found. Make sure all submodules are registered.";

  auto radiance_model_optimizer = Adam({
    OptimizerParamGroup(
      radiance_mlp_params,
      std::make_unique<AdamOptions>(
        AdamOptions(cfg.radiance_lr)
        .weight_decay(cfg.radiance_decay)
        .eps(cfg.adam_eps))),
    OptimizerParamGroup(
      radiance_hashmap_params,
      std::make_unique<AdamOptions>(
        AdamOptions(cfg.radiance_lr)
        .weight_decay(0)
        .eps(cfg.adam_eps)))});

  // Same for the proposal model - split the parameters and only decay non-hashmap.
  std::vector<torch::Tensor> proposal_hashmap_params, proposal_mlp_params;
  for (const auto& param : proposal_model->named_parameters()) {
    auto& dest = is_hashmap_key(param.key()) ? proposal_hashmap_params : proposal_mlp_params;
    dest.push_back(param.value());
  }
  XCHECK(!proposal_hashmap_params.empty()) << "Proposal hashmap params not found. Make sure all submodules are registered.";

  auto proposal_model_optimizer = Adam({
    OptimizerParamGroup(
      proposal_mlp_params,
      std::make_unique<AdamOptions>(
        AdamOptions(cfg.prop_lr)
        .weight_decay(cfg.prop_decay)
        .eps(cfg.adam_eps))),
    OptimizerParamGroup(
      proposal_hashmap_params,
      std::make_unique<AdamOptions>(
        AdamOptions(cfg.prop_lr)
        .weight_decay(0)
        .eps(cfg.adam_eps)))});

  std::vector<int> milestones = {
    int(cfg.num_training_itrs / 2),
    int(cfg.num_training_itrs * 3 / 4),
    int(cfg.num_training_itrs * 9 / 10)
  };
  XPLINFO << "Learning schedule milestones: " << milestones[0] << " " << milestones[1] << " " << milestones[2];

  constexpr double kLRMilestoneDecay = 0.333;
  constexpr double kLRWarmupFactor = 0.01;
  constexpr int kLRWarmupItrs = 100;

  XPLINFO << "Phase: Train neural radiance field";
  float proposal_loss = 0;
  for (int itr = 0; itr < cfg.num_training_itrs; ++itr) {
    //const double curr_lr = //
    applyLearningRateSchedule(radiance_model_optimizer, cfg.radiance_lr, itr, milestones, kLRMilestoneDecay, kLRWarmupFactor, kLRWarmupItrs);
    applyLearningRateSchedule(image_code_optimizer, cfg.image_code_lr, itr, milestones,   kLRMilestoneDecay, kLRWarmupFactor, kLRWarmupItrs);
    applyLearningRateSchedule(proposal_model_optimizer, cfg.prop_lr, itr, milestones,     kLRMilestoneDecay, kLRWarmupFactor, kLRWarmupItrs);

    // Optimize the radiance model while keeping the proposal model fixed
    radiance_model->train();
    proposal_model->eval();

    const bool warmup = itr < cfg.warmup_itrs;
    const auto [ray_origins, ray_dirs, target_colors, image_idxs] = generateBatchRays(device, dataset, cfg.rays_per_batch);
    const auto [basic_query_points, basic_depth_values] = computeQueryPointsFromRays(ray_origins, ray_dirs, cfg.num_basic_samples);
    constexpr bool kIncludeOriginalSamples = false;
    const auto [query_points, depth_values] = importanceSampling(warmup, proposal_model, ray_origins, ray_dirs, basic_query_points, basic_depth_values, cfg.num_importance_samples, kIncludeOriginalSamples);
    const int num_samples_per_ray = query_points.size(1);

    torch::Tensor ray_image_codes = torch::index_select(image_codes, 0, image_idxs);

    torch::Tensor flat_query_points = query_points.reshape({-1, 3});
    torch::Tensor flat_ray_dirs = ray_dirs.unsqueeze(1).repeat_interleave(num_samples_per_ray, 1).reshape({-1, 3});
    torch::Tensor flat_image_codes = ray_image_codes.unsqueeze(1).repeat_interleave(num_samples_per_ray, 1).reshape({-1, kImageCodeDim});
    
    const auto& [rgb, sigma] = radiance_model->pointAndDirToRadiance(flat_query_points, flat_ray_dirs, flat_image_codes);
    torch::Tensor unflat_rgb = rgb.reshape({cfg.rays_per_batch, num_samples_per_ray, 3});
    torch::Tensor unflat_sigma = sigma.reshape({cfg.rays_per_batch, num_samples_per_ray});   

    // Predict colors for training pixels and compute the most important loss- color.
    const auto& [predicted_rgb, predicted_inv_depth, _2, alpha, weights] = renderVolumeDensity(unflat_rgb, unflat_sigma, depth_values);
    
    // PHotometric loss (predicted colors should match image colors)
    torch::Tensor color_loss = torch::smooth_l1_loss(predicted_rgb, target_colors);

    // All other things being equal, we prefer predicted depths to be far away instead of close
    // which means smaller values of inverse depth.
    torch::Tensor far_away_loss = torch::mean(predicted_inv_depth);

    // Distortion loss, as in MipNerf360
    torch::Tensor distortion_loss = distortionLoss(depth_values, weights);

    // Floater regularization (see FreeNerf)
    torch::Tensor floater_indicator = torch::where(depth_values < cfg.floater_min_dist, 1.0, 0.0);
    torch::Tensor reg_floater = (unflat_sigma * floater_indicator).sum(-1).mean(); // TODO: is the sum necessary with mean?

    // Info regularization (see InfoNerf) but with a Lifecast twist- use Gini index instead of information
    auto p_alpha = alpha / (alpha.sum(-1, true) + 1e-10);
    auto gini = 1 - torch::pow(p_alpha, 2).sum(-1, true);
    auto reg_gini = gini.mean();

    // Visibility regularization- encourage zero density in points that are not in at least K camera frustums
    const auto& [visreg_points, visreg_raydirs, not_visible_enough] = generateVisibilityRegularizationPoints(device, dataset, cfg.num_visibility_points);
    torch::Tensor visreg_image_codes = torch::zeros({cfg.num_visibility_points,kImageCodeDim }, {torch::kFloat32}).to(device);
    const auto& [visreg_rgb, visreg_sigma] = radiance_model->pointAndDirToRadiance(visreg_points, visreg_raydirs, visreg_image_codes);
    torch::Tensor not_visible_density_loss = (visreg_sigma * not_visible_enough).mean();
    torch::Tensor not_visible_color_loss = (visreg_rgb * not_visible_enough).mean(); // learn a black background color
    torch::Tensor not_visible_loss = not_visible_density_loss + not_visible_color_loss;

    // For temporal stability in video, compute a loss between the current and previous frame's density field
    torch::Tensor prev_density_loss = torch::tensor(0.0, torch::dtype(torch::kFloat).device(device));
    if (prev_radiance_model) {
      // We want enough samples to make the regularization work, but it isn't free in runtime.
      // Only using importance samples stabilizes the foreground but not the background.
      // We could use every available basic sample, but that is more than we need. So well sample 
      // a few more just for this...
      const auto [reg_query_points, _] = computeQueryPointsFromRays(ray_origins, ray_dirs, cfg.prev_reg_num_samples);
      torch::Tensor flat_reg_query_points = reg_query_points.reshape({-1, 3});
      const auto& curr_sigma = radiance_model->pointToDensity(flat_reg_query_points);
      const auto& reg_sigma = prev_radiance_model->pointToDensity(flat_reg_query_points);
      prev_density_loss = torch::smooth_l1_loss(curr_sigma, reg_sigma);
    }

    // Density sparseness loss
    torch::Tensor density_loss = sigma.mean();

    // Weighted sum of losses.
    torch::Tensor total_loss = 
      color_loss + 
      reg_floater * cfg.floater_weight + 
      reg_gini * cfg.gini_weight + 
      distortion_loss * cfg.distortion_weight +
      not_visible_loss * cfg.visibility_weight + 
      prev_density_loss * cfg.prev_density_weight + 
      density_loss * cfg.density_weight + 
      far_away_loss * cfg.far_away_weight;

    image_code_optimizer.zero_grad();
    radiance_model_optimizer.zero_grad();
    total_loss.backward();
    image_code_optimizer.step();
    radiance_model_optimizer.step();

    if (itr % 10 == 0) {
      proposal_loss = updateProposalModel(cfg, device, dataset, radiance_model, proposal_model, proposal_model_optimizer);
    }

    XPLINFO << itr << std::setprecision(5)
      //<< "\t" << curr_lr << "\t"
      << "\ttot=" << total_loss.item<float>()
      << "\tclr=" << color_loss.item<float>()
      << "\tflt=" << reg_floater.item<float>()
      << "\tgni=" << reg_gini.item<float>()
      << "\tdst=" << distortion_loss.item<float>()
      << "\tden=" << density_loss.item<float>()
      << "\tvis=" << not_visible_loss.item<float>()
      << "\tfar=" << far_away_loss.item<float>()
      << "\tprv=" << prev_density_loss.item<float>()
      << "\tprp=" << proposal_loss;

    // Abort training if the loss is NaN.
    if (std::isnan(total_loss.item<float>())) {
      XPLERROR << "Stopping training early due to NaN. An earlier checkpoint may still be available.";
      return;
    }

    if (gui_data) {
      std::lock_guard<std::mutex> guard(gui_data->mutex);
      gui_data->plot_data_x.push_back(itr);
      gui_data->plot_data_y.push_back(total_loss.item<float>());
    }
  }

  XPLINFO << "training time (sec):\t\t" << time::timeSinceSec(train_timer);
}

cv::Mat imageTensorToCvMat(const torch::Tensor& tensor) {
  cv::Mat image(tensor.size(0), tensor.size(1), CV_32FC3);
  auto cpu_tensor = tensor.to(torch::kCPU);
  auto accessor = cpu_tensor.accessor<float, 3>();
  for (int y = 0; y < image.rows; ++y) {
    for (int x = 0; x < image.cols; ++x) {
      image.at<cv::Vec3f>(y, x) = cv::Vec3f(accessor[y][x][0], accessor[y][x][1], accessor[y][x][2]);
    }
  }
  return image;
}

// Sorts the point cloud so the farthest points from the origin are first in the array.
// This is a hack to do approximate painters algorithm (which wont update as the camera moves)
void sortPointCloud(
  std::vector<Eigen::Vector3f>& point_cloud, 
  std::vector<Eigen::Vector4f>& point_cloud_colors
) {
  std::vector<int> indices(point_cloud.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::sort(indices.begin(), indices.end(), [&](int i, int j) {
    return point_cloud[i].squaredNorm() > point_cloud[j].squaredNorm(); });

  std::vector<Eigen::Vector3f> sorted_points(point_cloud.size());
  std::vector<Eigen::Vector4f> sorted_colors(point_cloud_colors.size());

  for (int i = 0; i < indices.size(); ++i) {
    sorted_points[i] = point_cloud[indices[i]];
    sorted_colors[i] = point_cloud_colors[indices[i]];
  }

  point_cloud = std::move(sorted_points);
  point_cloud_colors = std::move(sorted_colors);
}

void samplePointCloudFromRadianceField(
    const torch::DeviceType device,
    std::shared_ptr<NeoNerfModel>& radiance_model,
    const torch::Tensor& image_code,
    const int target_num_points,
    std::vector<Eigen::Vector3f>& point_cloud,
    std::vector<Eigen::Vector4f>& point_cloud_colors,
    std::shared_ptr<std::atomic<bool>> cancel_requested
) {
  torch::NoGradGuard no_grad;
  
  const int SAMPLES_PER_BATCH = 8*64000;
  const float MAX_DIST = 10.0f;
  const float DELTA = 1e-2;
  const float ALPHA_THRESHOLD = 0.02;
  const int MAX_BATCHES = 1000; // maximum number of batches to prevent infinite loops

  int total_filtered_points = 0;
  int current_batch = 0;
  while (total_filtered_points < target_num_points && current_batch < MAX_BATCHES) {
    if (cancel_requested && *cancel_requested) break;

    XPLINFO << "Batch: " << current_batch << " / " << MAX_BATCHES << "\t# Points: " << total_filtered_points << " / " << target_num_points;

    torch::Tensor rand_dirs = torch::randn({SAMPLES_PER_BATCH, 3}, torch::kFloat32).to(device);
    rand_dirs /= torch::norm(rand_dirs, 2, 1, true) + 1e-3;

    //torch::Tensor rand_invdist = torch::rand({SAMPLES_PER_BATCH, 1}, torch::kFloat32).to(device) + 1e-3;
    //torch::Tensor rand_dist = torch::clamp(1.0 / rand_invdist, 1e-3, MAX_DIST);
    torch::Tensor rand_dist = torch::rand({SAMPLES_PER_BATCH, 1}, torch::kFloat32).to(device) * MAX_DIST;

    torch::Tensor world_points = rand_dirs * rand_dist;

    torch::Tensor ray_dirs = rand_dirs; // NOTE: this makes a big difference!

    torch::Tensor repeated_image_code = image_code.repeat({SAMPLES_PER_BATCH, 1});

    auto [rgb, sigma] = radiance_model->pointAndDirToRadiance(world_points, ray_dirs, repeated_image_code);
    torch::Tensor alpha = 1.0 - torch::exp(-sigma * DELTA).unsqueeze(-1);
    torch::Tensor mask = (alpha > ALPHA_THRESHOLD);
    
    torch::Tensor filtered_rgb = rgb.masked_select(mask.expand_as(rgb)).reshape({-1, 3});
    torch::Tensor filtered_alpha = alpha.masked_select(mask.expand_as(alpha)).reshape({-1, 1});
    torch::Tensor filtered_points = world_points.masked_select(mask.expand_as(world_points)).reshape({-1, 3});

    auto cpu_points = filtered_points.to(torch::kCPU);
    auto cpu_colors = filtered_rgb.to(torch::kCPU);
    auto cpu_alpha = filtered_alpha.to(torch::kCPU);
    auto acc_points = cpu_points.accessor<float, 2>();
    auto acc_colors = cpu_colors.accessor<float, 2>();
    auto acc_alpha = cpu_alpha.accessor<float, 2>();

    for (int i = 0; i < filtered_points.size(0); ++i) {
      point_cloud.emplace_back(
        acc_points[i][0], acc_points[i][1], acc_points[i][2]);
      point_cloud_colors.emplace_back(
        acc_colors[i][2], acc_colors[i][1], acc_colors[i][0], // Flip BGR->RGB
        acc_alpha[i][0]); 
    }

    total_filtered_points += filtered_points.size(0);
    current_batch++;
  }

  sortPointCloud(point_cloud, point_cloud_colors);
}

void epipolarVisualization(
  const std::vector<calibration::NerfKludgeCamera>& cameras,
  const std::vector<cv::Mat>& images,
  const std::string& output_dir
) {
  cv::Point2i selected_point(0, 0);
  for (int i = 0; i < cameras.size(); ++i) {
    cv::Mat viz = images[i].clone();

    if (i == 0) {
      // In the first image, get user input for which point to visualize.
      // This little gem opens a window, waits for a click, and saves the coordinates
      bool point_selected = false;
      auto param_data = std::make_tuple(&selected_point, &point_selected);
      cv::namedWindow("select_point");
      cv::setMouseCallback("select_point", [](int event, int x, int y, int, void* param) {
        auto [point, selected] = *static_cast<std::tuple<cv::Point2i*, bool*>*>(param);
        if (event == cv::EVENT_LBUTTONDOWN) {
          *point = cv::Point2i(x, y);
          *selected = true;
          cv::destroyWindow("select_point");
        }
      }, &param_data);
      cv::imshow("select_point", opencv::halfSize(viz));
      while (!point_selected) { cv::waitKey(1); } // spin until keypress
      XPLINFO << "Selected point: " << selected_point.x << " " << selected_point.y;
      selected_point *= 2; // HACK: undo the effect of scaling the window down with halfSize to make the window fit on screen

      cv::circle(viz, selected_point, 3, cv::Scalar(0, 0, 255));
      cv::circle(viz, selected_point, 50, cv::Scalar(0, 0, 255), 10);
    } else {
      // For the other images, we need to draw the epipolar line, which is constructed
      // by starting from the first camera origin in the ray direction defined by the selected pixel,
      // then going out from 0 to infinity (or 10m) and at every point along that line, projecting it back into the second image
      XPLINFO << "Constructing epipolar line from projection of point: " << selected_point.x << " " << selected_point.y;
      const Eigen::Vector3d ray_origin_in_world = cameras[0].getPositionInWorld();
      const Eigen::Vector3d ray_dir_in_cam = cameras[0].rayDirFromPixel(Eigen::Vector2d(selected_point.x, selected_point.y));
      const Eigen::Vector3d ray_dir_in_world = cameras[0].camFromWorld().linear().transpose() * ray_dir_in_cam; // TODO: precompute worldFromCam!
    
      std::vector<cv::Point2i> line_pts;
      for (float t = 0.0; t < 10.0; t += 0.05) {
        const Eigen::Vector3d point_in_world = ray_origin_in_world + ray_dir_in_world * t;
        const Eigen::Vector2d projectd_pixel = cameras[i].pixelFromCam(cameras[i].camFromWorld(point_in_world));
        line_pts.push_back(cv::Point2i(projectd_pixel.x(), projectd_pixel.y()));
      }
      cv::polylines(viz, line_pts, false, cv::Scalar(0, 0, 255), 1);
    }

    cv::imwrite(output_dir + "/epi_" + cameras[i].name() + ".jpg", viz);
  }
}

void runNerfPipeline(NeoNerfConfig& cfg, NeoNerfGuiData* gui_data)
{
  torch::manual_seed(123);  // For reproducible initialization of weights
  srand(123);               // For calls to rand()

  const torch::DeviceType device = util_torch::findBestTorchDevice();
  if (device == torch::kCPU) torch::set_num_threads(16);

  XPLINFO << "runNerfPipeline reading dataset from " << cfg.train_images_dir << ", json file " << cfg.train_json;
  calibration::MultiCameraDataset train_dataset = calibration::readDataset(cfg.train_images_dir, cfg.train_json, device, cfg.resize_max_dim);

  if (cfg.show_epipolar_viz) epipolarVisualization(train_dataset.cameras, train_dataset.images, cfg.output_dir);

  // Init or load the radiance and proposal models
  auto radiance_model = std::make_shared<NeoNerfModel>(device, kImageCodeDim);
  auto proposal_model = std::make_shared<ProposalDensityModel>(device);
  if (!cfg.load_model_dir.empty()) {
    XPLINFO << "Loading initial model from: " << cfg.load_model_dir;
    loadNerfAndProposalModels(cfg.load_model_dir, device, radiance_model, proposal_model);
  }

  // Train the models
  torch::Tensor train_image_codes = torch::zeros({int(train_dataset.images.size()), kImageCodeDim}, {torch::kFloat32}).to(device);
  std::shared_ptr<NeoNerfModel> no_prev_radiance = nullptr;
  trainRadianceModel(cfg, device, train_dataset, radiance_model, proposal_model, no_prev_radiance, train_image_codes, gui_data);

  // Save the trained model
  util_torch::saveOutputArchive(radiance_model, cfg.output_dir + "/radiance_model");
  util_torch::saveOutputArchive(proposal_model, cfg.output_dir + "/proposal_model");

  // Render novel views.
  torch::Tensor novel_image_code = train_image_codes.mean(0); // Use average training image code
  calibration::NerfKludgeCamera novel_cam(train_dataset.cameras[0]); // Copy training intrinsics.
  for (int i = 0; i < cfg.num_novel_views; ++i) {
    XPLINFO << "rendering novel view " << i;
    novel_cam.setCamFromWorld(Eigen::Isometry3d::Identity());
    Eigen::Vector3d cam_pos(-1 + 2.0 * float(i)/(cfg.num_novel_views-1), 0.0, 0.0);
    novel_cam.setPositionInWorld(cam_pos);

    torch::Tensor image_tensor = renderImageWithNerf<calibration::NerfKludgeCamera>(
      device, radiance_model, proposal_model, novel_cam, novel_image_code, cfg.num_basic_samples, cfg.num_importance_samples);
    cv::Mat image_mat = imageTensorToCvMat(image_tensor);
    cv::imwrite(cfg.output_dir + "/novel_" + string::intToZeroPad(i, 6) + ".png", image_mat * 255.0);
  }

  // Compute the loss and PSNR on the training set
  if (cfg.compute_train_psnr) {
    double sum_psnr = 0;
    for (int i = 0; i < train_dataset.images.size(); ++i) {
      const auto& cam = train_dataset.cameras[i];
      torch::Tensor predicted_image_tensor = renderImageWithNerf(
        device, radiance_model, proposal_model, cam, train_image_codes[i], cfg.num_basic_samples, cfg.num_importance_samples);

      torch::Tensor loss = torch::mse_loss(predicted_image_tensor, train_dataset.image_tensors[i]);
      const double mse = loss.item<float>();
      const double psnr = 20.0 * std::log10(1.0 / std::sqrt(mse));
      sum_psnr += psnr;
      XPLINFO << "MSE: " << mse << " PSNR: " << psnr;

      cv::Mat image_mat = imageTensorToCvMat(predicted_image_tensor);
      cv::imwrite(cfg.output_dir + "/train_" + train_dataset.image_filenames[i], image_mat * 255.0);
    }
    const double avg_psnr = sum_psnr / train_dataset.images.size();
    XPLINFO << "avg PSNR (all train images): " << avg_psnr;
  }
}

void testImportanceSampling() {
  torch::Tensor ray_origins = torch::tensor({
    {0.0, 0.0, 0.0},
    {0.0, 0.0, 0.0},
  });
  torch::Tensor ray_dirs = torch::tensor({
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
  });
  torch::Tensor depth_values = torch::tensor({
    {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
    {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
  });
  torch::Tensor sigma = torch::tensor({
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0},
  });

  const int num_importance_samples = 1000;
  const bool include_original_samples = false;
  auto [new_query_points, sorted_depth_values] = importanceSamplingHelper(
    ray_origins, ray_dirs, depth_values, sigma, num_importance_samples, include_original_samples);

  XPLINFO << "sorted_depth_values=\n" << sorted_depth_values[1];
}

void loadNerfAndProposalModels(
  const std::string& model_dir, 
  const torch::DeviceType device,
  std::shared_ptr<nerf::NeoNerfModel>& radiance_model,
  std::shared_ptr<nerf::ProposalDensityModel>& proposal_model
) {
  radiance_model = std::make_shared<nerf::NeoNerfModel>(device, nerf::kImageCodeDim);
  proposal_model = std::make_shared<nerf::ProposalDensityModel>(device);
  util_torch::loadModelArchive(radiance_model, device, model_dir + "/radiance_model");
  util_torch::loadModelArchive(proposal_model, device, model_dir + "/proposal_model");
}

void testNerfToLdi3Distillation(
  const std::string& model_dir,
  const std::string& dest_dir,
  const int ldi_resolution,
  const bool transparent_bg
) {
  torch::DeviceType device = util_torch::findBestTorchDevice();
  std::shared_ptr<nerf::NeoNerfModel> radiance_model;
  std::shared_ptr<nerf::ProposalDensityModel> proposal_model;
  loadNerfAndProposalModels(model_dir, device, radiance_model, proposal_model);

  constexpr float kFthetaScale = 1.15;
  calibration::FisheyeCamerad cam;
  cam.width = ldi_resolution;
  cam.height = ldi_resolution;
  cam.radius_at_90 = kFthetaScale * cam.width/2;
  cam.optical_center = Eigen::Vector2d(cam.width / 2.0, cam.height / 2.0);

  // Precompute a vignette
  cv::Mat vignette = projection::makeVignetteMap(
    cam, cv::Size(cam.width, cam.height),
    0.85, 0.87, 0.01, 0.02);

  torch::Tensor image_code = torch::zeros({nerf::kImageCodeDim}, {torch::kFloat32}).to(device);

  constexpr int kNumBasicSamples = 256;
  constexpr int kNumImportanceSamples = 128;
  cv::Mat fused_bgra;
  cv::Mat ldi3_image = distillNerfToLdi3(
    dest_dir,
    device,
    radiance_model,
    proposal_model,
    cam,
    vignette,
    image_code,
    kNumBasicSamples,
    kNumImportanceSamples,
    transparent_bg,
    fused_bgra);

  cv::imwrite(dest_dir + "/ldi3.png", ldi3_image);
  cv::imwrite(dest_dir + "/fused.png", fused_bgra * 255.0);
}

void testImageBasedProjectionRefinement(const std::string& data_dir) {
  cv::Mat ldi3_image = cv::imread(data_dir + "/ldi3_000000.png");
  cv::Mat primary_image_projection = cv::imread(data_dir + "/primary_projection.png", cv::IMREAD_UNCHANGED);
  XCHECK(!ldi3_image.empty());
  XCHECK(!primary_image_projection.empty());
  XCHECK_EQ(primary_image_projection.type(), CV_8UC4);

  std::vector<cv::Mat> layer_bgra, layer_invd;
  ldi::unpackLDI3(ldi3_image, layer_bgra, layer_invd);
  for (int l = 0; l < 3; ++l) {
    layer_bgra[l].convertTo(layer_bgra[l], CV_32FC4, 1.0/255.0);
  }

  ldi::refineLdiWithPrimaryImageProjection(layer_bgra, layer_invd, primary_image_projection);

  cv::Mat new_ldi3_image = ldi::make6DofGrid(layer_bgra, layer_invd, "split12", std::vector<cv::Mat>(), false);
  cv::imwrite(data_dir + "/refined_ldi.png", new_ldi3_image);
}

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
  std::vector<cv::Mat>& layer_invd
){
  auto start_timer = time::now();
  torch::NoGradGuard no_grad;
  constexpr int MAX_RAYS_PER_BATCH = 8192;
  
  cv::Size image_size(cam.width, cam.height);
  cv::Mat final_image = cv::Mat::zeros(image_size, CV_32FC3);
  cv::Mat final_depthmap = cv::Mat::zeros(image_size, CV_32F);

  // render a low resolution depth map for analysis
  constexpr int kSmall = 4;
  calibration::FisheyeCamerad small_cam(cam);
  small_cam.width /= kSmall;
  small_cam.height /= kSmall;
  small_cam.optical_center /= kSmall;
  small_cam.radius_at_90 /= kSmall;
  small_cam.is_inflated = true;
  cv::Mat small_depth = renderInverseDepthmapAsCvMat(
    device, 
    radiance_model,
    proposal_model,
    small_cam,
    num_basic_samples,
    num_importance_samples,
    world_transform,
    cancel_requested);
  if (small_depth.empty() || (cancel_requested != nullptr && (*cancel_requested))) return;

  cv::Mat fullsize_depth;
  cv::resize(small_depth, fullsize_depth, image_size);
  cv::Mat seg = heuristic3LayerSegmentation(fullsize_depth);

  // blendDepthBySegmentation is slow, so well downscale first
  cv::Size smoothing_size(cam.width/8, cam.height/8);
  cv::resize(small_depth, small_depth, smoothing_size, 0, 0, cv::INTER_AREA);
  cv::resize(seg, seg, smoothing_size, 0, 0, cv::INTER_AREA);
  cv::Mat smooth_invd0 = blendDepthBySegmentation(small_depth, seg, 0);
  cv::Mat smooth_invd1 = blendDepthBySegmentation(small_depth, seg, 1);
  cv::Mat smooth_invd2 = blendDepthBySegmentation(small_depth, seg, 2);

  cv::Mat depth_tA(small_depth.size(), CV_32F);
  cv::Mat depth_tB(small_depth.size(), CV_32F);
  for (int y = 0; y < small_depth.rows; ++y) {
    for (int x = 0; x < small_depth.cols; ++x) {
      float sinvd0 = smooth_invd0.at<float>(y, x);
      float sinvd1 = smooth_invd1.at<float>(y, x);
      float sinvd2 = smooth_invd2.at<float>(y, x);
      // Bubblesort
      if (sinvd0 > sinvd1) std::swap(sinvd0, sinvd1);
      if (sinvd1 > sinvd2) std::swap(sinvd1, sinvd2);
      if (sinvd0 > sinvd1) std::swap(sinvd0, sinvd1);

      // Put the shell bounds at the midpoint
      const float m01 = (sinvd0 + sinvd1) * 0.5;
      const float m12 = (sinvd1 + sinvd2) * 0.5;
      depth_tA.at<float>(y, x) = math::clamp<float>(kInvDepthCoef / (m12 + 1e-6), 0.0, kZFar);
      depth_tB.at<float>(y, x) = math::clamp<float>(kInvDepthCoef / (m01 + 1e-6), 0.0, kZFar);
    }
  }
  
  // These resizes are necessary
  cv::resize(depth_tA, depth_tA, image_size);
  cv::resize(depth_tB, depth_tB, image_size);
  
  constexpr int kNumLayers = 3;
  for (int l = 0; l < kNumLayers; ++l) {
    layer_bgra.emplace_back(image_size, CV_32FC4, cv::Scalar(0, 0, 0, 0));
    layer_invd.emplace_back(image_size, CV_32F, cv::Scalar(0));
  }

  Eigen::Transform<double, 3, Eigen::Affine> world_transform_inv(world_transform.inverse());
  Eigen::Matrix3d world_rotation_inv = world_transform_inv.linear();
  const Eigen::Vector3d ray_origin_in_world = world_transform_inv * cam.getPositionInWorld();

  int batch_count = 0;
  std::vector<float> ray_origin_data, ray_dir_data;
  std::vector<float> depth_tA_data, depth_tB_data;
  std::vector<std::pair<int, int>> valid_pixels;
  for (int y = 0; y < cam.height; ++y) {
    if (y % 10 ==0) XPLINFO << "y=" << y;
    for (int x = 0; x < cam.width; ++x) {
      if (cancel_requested != nullptr && (*cancel_requested)) return;
      // Skip pixels outside the image circle
      if ((cam.optical_center - Eigen::Vector2d(x, y)).norm() > cam.radius_at_90) {
        continue;
      }
      valid_pixels.emplace_back(x, y);

      const Eigen::Vector3d ray_dir_in_cam = cam.rayDirFromPixelInflated(Eigen::Vector2d(x, y));
      Eigen::Vector3d ray_dir_in_world = world_rotation_inv * cam.cam_from_world.linear().transpose() * ray_dir_in_cam;
      ray_dir_in_world.normalize();

      ray_origin_data.push_back(ray_origin_in_world.x());
      ray_origin_data.push_back(ray_origin_in_world.y());
      ray_origin_data.push_back(ray_origin_in_world.z());
      ray_dir_data.push_back(ray_dir_in_world.x());
      ray_dir_data.push_back(ray_dir_in_world.y());
      ray_dir_data.push_back(ray_dir_in_world.z());
      depth_tA_data.push_back(depth_tA.at<float>(y, x));
      depth_tB_data.push_back(depth_tB.at<float>(y, x));

      batch_count++;
      if (batch_count == MAX_RAYS_PER_BATCH || (y == cam.height - 1 && x == cam.width - 1)) {
        int num_rays_in_batch = ray_origin_data.size() / 3;

        torch::Tensor ray_origins = torch::from_blob(ray_origin_data.data(), {num_rays_in_batch, 3}, torch::kFloat32).to(device);
        torch::Tensor ray_dirs = torch::from_blob(ray_dir_data.data(), {num_rays_in_batch, 3}, torch::kFloat32).to(device);
        torch::Tensor depth_tA_tensor = torch::from_blob(depth_tA_data.data(), {num_rays_in_batch, 1}, torch::kFloat32).to(device);
        torch::Tensor depth_tB_tensor = torch::from_blob(depth_tB_data.data(), {num_rays_in_batch, 1}, torch::kFloat32).to(device);
        util_torch::cloneIfOnCPU(ray_origins);
        util_torch::cloneIfOnCPU(ray_dirs);
        util_torch::cloneIfOnCPU(depth_tA_tensor);
        util_torch::cloneIfOnCPU(depth_tB_tensor);
        torch::Tensor depth_p00_tensor = torch::zeros_like(depth_tA_tensor);	
        torch::Tensor depth_t100_tensor = torch::ones_like(depth_tA_tensor) * 10e12;

        auto [basic_query_points, basic_depth_values] = computeQueryPointsFromRays(ray_origins, ray_dirs, num_basic_samples);
        constexpr bool warmup = false;
        constexpr bool include_original_samples = true; // IMPORTANT!!! without this, occluded parts of the scene are under-sampled.
        auto [query_points, depth_values] = importanceSampling(warmup, proposal_model, ray_origins, ray_dirs, basic_query_points, basic_depth_values, num_importance_samples, include_original_samples);

        const int num_samples_per_ray = query_points.size(1);

        torch::Tensor ray_image_codes = image_code.repeat({num_rays_in_batch, 1});
        torch::Tensor flat_query_points = query_points.reshape({-1, 3});
        torch::Tensor flat_ray_dirs = ray_dirs.unsqueeze(1).repeat_interleave(num_samples_per_ray, 1).reshape({-1, 3});

        torch::Tensor flat_image_codes = ray_image_codes.unsqueeze(1).repeat_interleave(num_samples_per_ray, 1).reshape({-1, kImageCodeDim});

        const auto& [rgb, sigma] = radiance_model->pointAndDirToRadiance(flat_query_points, flat_ray_dirs, flat_image_codes);
        torch::Tensor unflat_rgb = rgb.reshape({num_rays_in_batch, num_samples_per_ray, 3});
        torch::Tensor unflat_sigma = sigma.reshape({num_rays_in_batch, num_samples_per_ray});

        // For each layer, only composite samples within a particular distance interval.
        std::vector<torch::Tensor> level_to_min_dist = {depth_tB_tensor,   depth_tA_tensor, depth_p00_tensor};
        std::vector<torch::Tensor> level_to_max_dist = {depth_t100_tensor, depth_tB_tensor, depth_tA_tensor};
        for (int l = 0; l < kNumLayers; ++l) {
          const auto& [predicted_rgb, predicted_depth, predicted_sum_weight, _1, _2] = renderVolumeDensityWithDistanceBounds(
            unflat_rgb,
            unflat_sigma,
            depth_values,
            level_to_min_dist[l],
            level_to_max_dist[l],
            transparent_bg);

          auto cpu_rgb = predicted_rgb.to(torch::kCPU);
          auto cpu_depth = predicted_depth.to(torch::kCPU);
          auto cpu_sum_weight = predicted_sum_weight.to(torch::kCPU);
          auto acc_rgb = cpu_rgb.accessor<float, 2>();
          auto acc_depth = cpu_depth.accessor<float, 1>();
          auto acc_sum_weight = cpu_sum_weight.accessor<float, 1>();

          int i = 0;
          for (const auto& [x, y] : valid_pixels) {
            float alpha = acc_sum_weight[i];
            layer_bgra[l].at<cv::Vec4f>(y, x) = cv::Vec4f(acc_rgb[i][0], acc_rgb[i][1], acc_rgb[i][2], alpha);
            layer_invd[l].at<float>(y, x) = acc_depth[i];
            ++i;
          }
        }

        // Clear data vectors and reset batch_count after processing each batch
        ray_origin_data.clear();
        ray_dir_data.clear();
        depth_tA_data.clear();
        depth_tB_data.clear();
        valid_pixels.clear();
        batch_count = 0;
      }
    }
  }

  XPLINFO << "render time(sec):\t" << time::timeSinceSec(start_timer);

  // Apply a median filter to the alpha channel of each layer, and a vignette
  for (int l = 0; l < kNumLayers; ++l) {
    cv::Mat alpha_channel;
    cv::extractChannel(layer_bgra[l], alpha_channel, 3);
    cv::medianBlur(alpha_channel, alpha_channel, 3);

    // Attenuate inverse depth by alpha
    for (int y = 0; y < alpha_channel.rows; ++y) {
      for (int x = 0; x < alpha_channel.cols; ++x) {
        const float a = alpha_channel.at<float>(y, x);
        const float s = 1.0 - std::exp(-a * 10.0);
        layer_invd[l].at<float>(y, x) *= s;
      }
    }

    // Do the alpha-vignette after attenuation by alpha so that is not affected
    alpha_channel = projection::applyVignette<float>(alpha_channel, vignette);
    cv::insertChannel(alpha_channel, layer_bgra[l], 3);

    // Filter inverse depthmaps to reduce noise. This seems kinda redundant with dilation.
    cv::medianBlur(layer_invd[l], layer_invd[l], 3);

    // Finally, dilate depth
    cv::dilate(layer_invd[l], layer_invd[l], cv::Mat(), cv::Point(-1, -1), 9);
  }
}

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
  const cv::Mat primary_image_projection,
  const Eigen::Matrix4d& world_transform,
  std::shared_ptr<std::atomic<bool>> cancel_requested
){
  std::vector<cv::Mat> layer_bgra, layer_invd;

  distillNerfToLdi3(debug_dir, device, radiance_model, proposal_model,
    cam, vignette, image_code, num_basic_samples, num_importance_samples,
    transparent_bg, out_fused_bgra, world_transform, cancel_requested,
    layer_bgra, layer_invd);
  if (cancel_requested != nullptr && (*cancel_requested)) return cv::Mat();

  //for (int l = 0; l < kNumLayers; ++l) {
  //  cv::imwrite(debug_dir + "/rgba_" + std::to_string(l) + ".png", layer_bgra[l] * 255.0);
  //  cv::imwrite(debug_dir + "/invd_" + std::to_string(l) + ".png", layer_invd[l] * 255.0);
  //}

  if (!primary_image_projection.empty()) {
    ldi::refineLdiWithPrimaryImageProjection(layer_bgra, layer_invd, primary_image_projection);
  }

  // TODO: this is for debugging. it costs runtime. disable in prod.
  cv::Mat fused_bgra, fused_invd;
  ldi::fuseLayers(layer_bgra, layer_invd, fused_bgra, fused_invd);
  out_fused_bgra = fused_bgra;

  constexpr bool kDilateInvd = false; // We dilated already elsewhere
  cv::Mat ldi3_image = ldi::make6DofGrid(layer_bgra, layer_invd, "split12", std::vector<cv::Mat>(), kDilateInvd);

  return ldi3_image;
}

cv::Mat renderLookingGlassQuilt(
  const torch::DeviceType device,
  std::shared_ptr<NeoNerfModel>& radiance_model,
  std::shared_ptr<ProposalDensityModel>& proposal_model,
  torch::Tensor image_code,
  const int num_basic_samples,
  const int num_importance_samples,
  const Eigen::Matrix4d& world_transform = Eigen::Matrix4d::Identity(),
  std::shared_ptr<std::atomic<bool>> cancel_requested = nullptr
) {
  // NOTE: these settings are for looking glass portrait, see:
  // https://docs.lookingglassfactory.com/keyconcepts/quilts
  constexpr int kQuiltWidth = 3360;
  constexpr int kQuiltHeight = 3360;
  constexpr int kDownscale = 1;
  constexpr int kNumRows = 6;
  constexpr int kNumCols = 8;
  constexpr int kTotalTiles = kNumRows * kNumCols;
  constexpr float kHorizontalFovDeg = 60;
  constexpr float kVirtualStereoBaseline = 0.5;

  cv::Mat image_mat;
  std::vector<cv::Mat> row_images;
  std::vector<cv::Mat> images;
  for (int row = 0; row < kNumRows; ++row) {
    for (int col = 0; col < kNumCols; ++col) {
      XPLINFO << "Quilt Row: " << row << " Quilt Column: " << col;
      int tile = (kNumRows - 1 - row) * kNumCols + col;

      calibration::RectilinearCamerad quilt_cam;
      quilt_cam.width = (kQuiltWidth / kDownscale) / kNumCols;
      quilt_cam.height = (kQuiltHeight / kDownscale) / kNumRows;
      quilt_cam.optical_center = Eigen::Vector2d(quilt_cam.width / 2, quilt_cam.height / 2);
      double f = quilt_cam.width / (2.0 * tan(kHorizontalFovDeg * M_PI / 360.0));
      quilt_cam.focal_length = Eigen::Vector2d(f, f);
    
      // Calculate beta, ensuring the bottom-left tile is the leftmost view
      const float beta = float(tile) / (kTotalTiles - 1) - 0.5;
      const Eigen::Vector3d cam_pos = quilt_cam.right() * kVirtualStereoBaseline * beta;
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
  image_mat.convertTo(image_mat, CV_8UC3, 255.0);
  return image_mat;
}

std::vector<cv::Mat> precomputeWarpFromPrimaryCameraToLdiProjectionForIbr(
  const cv::Size image_size, // The size of an image straight out of the video, before any resizing
  const calibration::NerfKludgeCamera& primary_camera, // The camera who's image we are projecting
  const calibration::FisheyeCamerad ldi_cam // The projection model for the LDI
) {
  XCHECK(primary_camera.is_fisheye) << "Primary camera projection / IBR is only implemented for fisheye cameras."; // #lazy
  XCHECK(primary_camera.getPositionInWorld().norm() < 0.001) << "IBR only works if the primary (first) camera is at the origin!";
  calibration::FisheyeCamerad primary_cam_full_size(primary_camera.fisheye);
  primary_cam_full_size.resizeToWidth(image_size.width);

  cv::Mat warp(cv::Size(ldi_cam.width, ldi_cam.height), CV_32FC2);
  for (int y = 0; y < ldi_cam.height; ++y) {
    for (int x = 0; x < ldi_cam.width; ++x) {
      const Eigen::Vector3d ray_dir = ldi_cam.rayDirFromPixelInflated(Eigen::Vector2d(x, y));
      const Eigen::Vector2d proj_pixel = primary_cam_full_size.pixelFromCam(ray_dir);
      warp.at<cv::Vec2f>(y, x) = cv::Vec2f(proj_pixel.x(), proj_pixel.y());
    }
  }

  std::vector<cv::Mat> warp_primary_to_ldi;
  cv::split(warp, warp_primary_to_ldi);
  return warp_primary_to_ldi;
}

void runVideoNerfPipeline(NerfVideoConfig& vid_cfg, NeoNerfConfig& cfg, NeoNerfGuiData* gui_data) {
  const torch::DeviceType device = util_torch::findBestTorchDevice();
  if (device == torch::kCPU) torch::set_num_threads(16);
  //torch::autograd::AnomalyMode::set_enabled(true); 

  std::vector<calibration::NerfKludgeCamera> cameras = calibration::readDatasetCameraJson(vid_cfg.vid_dir + "/dataset.json");
  calibration::readTimeOffsetJson(vid_cfg.vid_dir + "/time_offsets.json", cameras);

  if (cfg.resize_max_dim != 0) {
    for (auto& cam : cameras) { cam.resizeToMaxDim(cfg.resize_max_dim); }
  }
  // HACK: wont work for mix of different image sizes
  cv::Size resize_size(cameras[0].getWidth(), cameras[0].getHeight()); // The size of the resized images

  // Read manual ignore mask images. Make a binary mask from pixels that are pure red
  // in the image (0, 0, 255).
  if (file::directoryExists(vid_cfg.vid_dir + "/masks")) {
    XPLINFO << "Found /masks folder. Reading masks...";
    for (auto& cam : cameras) {
      const std::string mask_filename = vid_cfg.vid_dir + "/masks/" + cam.name() + ".png";
      if (file::fileExists(mask_filename)) {
        cv::Mat mask_rgb = cv::imread(mask_filename);
        if (cfg.resize_max_dim != 0) { cv::resize(mask_rgb, mask_rgb, resize_size); }
        cv::Mat mask_binary(mask_rgb.size(), CV_8U, cv::Scalar(0));
        for (int y = 0; y < mask_rgb.rows; ++y) {
          for (int x = 0; x < mask_rgb.cols; ++x) {
            mask_binary.at<uint8_t>(y, x) = mask_rgb.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 0, 255) ? 255 : 0;
          }
        }
        cam.ignore_mask = mask_binary;
        //cv::imshow("mask_binary", opencv::halfSize(opencv::halfSize(mask_binary))); cv::waitKey(0);
      }
    }
  }

  std::vector<cv::VideoCapture> video_captures(cameras.size());
  for (int i = 0; i < cameras.size(); ++i) {
    const std::string video_path = vid_cfg.vid_dir + "/" + cameras[i].name() + ".mp4";
    XPLINFO << "Opening video for camera: " << video_path;
    video_captures[i].open(video_path);
    XCHECK(video_captures[i].isOpened()) << video_path;

    // Pre-advance the video to the time_offset_frames for each camera, so later 
    // decoded frames are synched across all cameras
    XPLINFO << "Skipping " << cameras[i].time_offset_frames << " frames for camera: " << cameras[i].name();
    for (int j = 0; j < cameras[i].time_offset_frames; ++j) {
      cv::Mat image;
      video_captures[i] >> image;
      XCHECK(!image.empty()) << "Error skipping " << cameras[i].time_offset_frames << " frames in camera: " << cameras[i].name();
    }
  }

  // Setup for baking an LDI
  constexpr float kFthetaScale = 1.15;
  calibration::FisheyeCamerad ldi_cam;
  ldi_cam.width = vid_cfg.ldi_resolution;
  ldi_cam.height = vid_cfg.ldi_resolution;
  ldi_cam.radius_at_90 = kFthetaScale * ldi_cam.width/2;
  ldi_cam.optical_center = Eigen::Vector2d(ldi_cam.width / 2.0, ldi_cam.height / 2.0);

  // Precompute a vignette
  cv::Mat vignette = projection::makeVignetteMap(
    ldi_cam, cv::Size(ldi_cam.width, ldi_cam.height),
    0.85, 0.86, 0.01, 0.02);

  // Will be filled in on the first frame
  std::vector<cv::Mat> warp_primary_to_ldi;

  // Process video frames
  int frame_num = 0;
  bool have_more_frames = true;
  std::shared_ptr<NeoNerfModel> prev_radiance_model = nullptr; // For temporal stability regularization
  std::shared_ptr<ProposalDensityModel> prev_proposal_model = nullptr;
  //torch::Tensor image_codes = torch::zeros({cameras.size(), kImageCodeDim}, {torch::kFloat32}).to(device);
  while(have_more_frames) {
    // Re-seed RNG every frame for more temporally stable results
    torch::manual_seed(123);  // For reproducible initialization of weights
    srand(123);               // For calls to rand()

    XPLINFO << "=========== frame_num: " << frame_num;

    calibration::MultiCameraDataset frame_dataset;
    cv::Mat primary_image_in_ldi_projection;
    // Get one frame from each camera's video, maybe preprocess
    for (int cam_idx = 0; cam_idx < cameras.size(); ++cam_idx) {
      cv::Mat image;
      video_captures[cam_idx] >> image;
      if (image.empty()) { have_more_frames = false; break; }

      // Save the first frame from each video as a png for previewing the time offset
      if (frame_num == 0) {
        cv::imwrite(cfg.output_dir + "/" + cameras[cam_idx].name() + ".jpg", image);
      }

      // Precompute a warp from the first camera to the LDI projection
      // Only do this once, and for the first camera 
      if (vid_cfg.use_ibr && cam_idx == 0) {
        if (warp_primary_to_ldi.empty()) {
         warp_primary_to_ldi = precomputeWarpFromPrimaryCameraToLdiProjectionForIbr(
          image.size(), cameras[0], ldi_cam);
        }

        cv::Mat image_with_alpha;
        cv::cvtColor(image, image_with_alpha, cv::COLOR_BGR2BGRA);
        cv::remap(image_with_alpha, primary_image_in_ldi_projection, warp_primary_to_ldi[0], warp_primary_to_ldi[1], cv::INTER_CUBIC);
        if (frame_num == 0) {
          cv::imwrite(cfg.output_dir + "/primary_projection.png", primary_image_in_ldi_projection);
        }
      }

      if (cfg.resize_max_dim != 0) {
        cv::resize(image, image, resize_size, 0, 0, cv::INTER_CUBIC);
      } else {
        if (cameras[cam_idx].getWidth() != image.cols) {
          cameras[cam_idx].resizeToWidth(image.cols);
          XCHECK_EQ(cameras[cam_idx].getHeight(), image.rows) << "Resizing intrinsics failed. Inconsistent aspect ratio?";
        }
      }

      frame_dataset.images.push_back(image);
    }
    
    frame_dataset.cameras = cameras;

    if (cfg.show_epipolar_viz && frame_num == 0) epipolarVisualization(frame_dataset.cameras, frame_dataset.images, cfg.output_dir);
      
    // Train a radiance model for this frame
    auto radiance_model= std::make_shared<NeoNerfModel>(device, kImageCodeDim);
    auto proposal_model = std::make_shared<ProposalDensityModel>(device);
    torch::Tensor image_codes = torch::zeros({int(cameras.size()), kImageCodeDim}, {torch::kFloat32}).to(device);
  
    //if (prev_radiance_model && prev_proposal_model) {
    //  util_torch::deepCopyModel(prev_radiance_model, radiance_model);
    //  util_torch::deepCopyModel(prev_proposal_model, proposal_model);
    //}
    // Adjust nerf training after the first frame because we are starting with an already trained model.
    //NeoNerfConfig after_first_frame_cfg(cfg);
    //if (frame_num != 0) {
    //  after_first_frame_cfg.num_training_itrs = 2500; // HACK
    //  after_first_frame_cfg.warmup_itrs   = 0;
    //  constexpr float kLearningRateDecay = 0.333;
    //  after_first_frame_cfg.radiance_lr   *= kLearningRateDecay;
    //  after_first_frame_cfg.image_code_lr *= kLearningRateDecay;
    //  after_first_frame_cfg.prop_lr       *= kLearningRateDecay;
    //}

    trainRadianceModel(cfg, device, frame_dataset, radiance_model, proposal_model, prev_radiance_model, image_codes, gui_data);
    prev_radiance_model = radiance_model;
    prev_proposal_model = proposal_model;

    // Save the radiance model for this frame to file
    //util_torch::saveOutputArchive(radiance_model, cfg.output_dir + "/radiance_model_" + string::intToZeroPad(frame_num, 6));
    //util_torch::saveOutputArchive(proposal_model, cfg.output_dir + "/proposal_model_" + string::intToZeroPad(frame_num, 6));

    // Render one image from the training set
    //torch::Tensor predicted_image_tensor = renderImageWithNerf(
    //  device, radiance_model, proposal_model, cameras[3], image_codes[3], cfg.num_basic_samples, cfg.num_importance_samples);
    //cv::Mat image_mat = imageTensorToCvMat(predicted_image_tensor);
    //cv::imwrite(cfg.output_dir + "/predicted_" + cameras[3].name() + ".jpg", image_mat * 255.0);


    if (vid_cfg.render_looking_glass) {
      constexpr int kNumBasicSamples = 128;
      constexpr int kNumImportanceSamples = 64;
      cv::Mat quilt_image = renderLookingGlassQuilt(
        device,
        radiance_model,
        proposal_model,
        image_codes[0], // use the image code for image 0 because that is likely to be close to the origin (alternative: torch::mean(image_codes, 0)).
        kNumBasicSamples,
        kNumImportanceSamples);
      cv::imwrite(cfg.output_dir + "/quilt_" + string::intToZeroPad(frame_num, 6) + ".png", quilt_image);
    } else {
      constexpr int kNumBasicSamples = 256;
      constexpr int kNumImportanceSamples = 128;
      cv::Mat fused_bgra;
      cv::Mat ldi3_image = distillNerfToLdi3(
        cfg.output_dir,
        device,
        radiance_model,
        proposal_model,
        ldi_cam,
        vignette,
        image_codes[0], // use the image code for image 0 because that is likely to be close to the origin (alternative: torch::mean(image_codes, 0)).
        kNumBasicSamples,
        kNumImportanceSamples,
        vid_cfg.transparent_bg,
        fused_bgra,
        primary_image_in_ldi_projection);
      cv::imwrite(cfg.output_dir + "/ldi3_" + string::intToZeroPad(frame_num, 6) + ".png", ldi3_image);
      cv::imwrite(cfg.output_dir + "/fused_bgra_" + string::intToZeroPad(frame_num, 6) + ".jpg", fused_bgra * 255.0);
    }

    if( !have_more_frames) break;
    ++frame_num;
  }
}

}}  // end namespace p11::nerf
