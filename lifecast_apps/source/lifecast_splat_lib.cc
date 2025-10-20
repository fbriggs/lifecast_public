// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "lifecast_splat_lib.h"

#include <random>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "source/rectilinear_camera.h"
#include "util_math.h"
#include "util_string.h"
#include "util_opencv.h"
#include "third_party/json.h"
#include "util_math.h"
#include "util_file.h"
#include "torch_opencv.h"
#include "lifecast_splat_math.h"
#include "lifecast_splat_io.h"
#include "lifecast_splat_population.h"
#include "gsplat_lib.h"
#include "depth_anything2.h"
#include "rof.h"
#include "third_party/turbo_colormap.h"

namespace p11 { namespace splat {

namespace {
std::mt19937 rng(123); // Global RNG
};

std::tuple<
  torch::Tensor,
  torch::Tensor,
  torch::Tensor,
  torch::Tensor,
  torch::Tensor,
  torch::Tensor,
  gsplat::RasterizationMetas
> renderSplatImageGsplat(
  const torch::DeviceType device,
  const calibration::RectilinearCamerad& cam,
  std::shared_ptr<SplatModel> model,
  c10::optional<torch::Tensor> backgrounds,
  const Eigen::Matrix4d world_transform
) {
  return renderSplatImageGsplat(
    device,
    cam,
    model->splat_alive,
    model->splat_pos,
    model->splat_color,
    model->splat_alpha,
    model->splat_scale,
    model->splat_quat,
    backgrounds,
    world_transform);
}

// New: dont pass dead splats, but recombobulate metas with correct indexing afterward
std::tuple<
  torch::Tensor,
  torch::Tensor,
  torch::Tensor,
  torch::Tensor,
  torch::Tensor,
  torch::Tensor,
  gsplat::RasterizationMetas
> renderSplatImageGsplat(
  const torch::DeviceType device,
  const calibration::RectilinearCamerad& cam,
  torch::Tensor splat_alive,
  torch::Tensor splat_pos,
  torch::Tensor splat_color,
  torch::Tensor splat_alpha,
  torch::Tensor splat_scale,
  torch::Tensor splat_quat,
  c10::optional<torch::Tensor> backgrounds,
  const Eigen::Matrix4d world_transform
) {
  XCHECK(!torch::isnan(splat_pos).any().item<bool>());
  XCHECK(!torch::isinf(splat_pos).any().item<bool>());
  
  // Extract only alive splats to process
  torch::Tensor alive_mask = splat_alive.squeeze(-1);
  torch::Tensor alive_indices = alive_mask.nonzero().squeeze(1);
  int total_splats = splat_pos.size(0);
  int alive_splats = alive_indices.size(0);
  
  // Extract properties for alive splats only
  torch::Tensor alive_pos = splat_pos.index_select(0, alive_indices);
  torch::Tensor alive_quats = splat_quat.index_select(0, alive_indices);
  torch::Tensor alive_scales = splat_scale.index_select(0, alive_indices);
  torch::Tensor alive_alphas = splat_alpha.index_select(0, alive_indices);
  torch::Tensor alive_colors = splat_color.index_select(0, alive_indices);
  
  // Convert only alive splats to linear space
  torch::Tensor means = splatPosActivation(alive_pos);
  torch::Tensor quats = alive_quats;
  torch::Tensor scales = torch::exp(scaleActivation(alive_scales));
  torch::Tensor opacities = torch::sigmoid(alive_alphas).squeeze(1);
  torch::Tensor colors = torch::sigmoid(alive_colors);

  // Camera setup (unchanged)
  Eigen::Matrix4f cam_from_world_matrix = cam.cam_from_world.matrix().cast<float>();
  cam_from_world_matrix(3, 3) = 1.0;
  Eigen::Matrix4f transformed_matrix = cam_from_world_matrix * world_transform.cast<float>();
  torch::Tensor cam_from_world_transpose = torch::from_blob(transformed_matrix.data(), {4, 4}, {torch::kFloat32}).to(device);
  torch::Tensor cam_from_world = cam_from_world_transpose.transpose(0, 1);
  cam_from_world.index_put_({1, "..."}, -cam_from_world.index({1, "..."}));
  torch::Tensor viewmats = cam_from_world.unsqueeze(0);

  float K_cpu[3][3] = {
    {  cam.focal_length.x(),                    0.0,  cam.optical_center.x() },
    {                   0.0,   cam.focal_length.y(),  cam.optical_center.y() },
    {                   0.0,                    0.0,                     1.0 },
  };

  torch::Tensor Ks = torch::from_blob(K_cpu, {1, 3, 3}, {torch::kFloat32}).to(device);

  gsplat::RasterizationConfig cfg;
  cfg.render_mode = gsplat::RGBD;
  cfg.backgrounds = backgrounds;
  cfg.absgrad = true;

  // Use filtered data for rasterization
  auto [rgbd, alpha_map, filtered_metas] = gsplat::rasterization(
    means,
    quats,
    scales,
    opacities,
    colors,
    viewmats,
    Ks,
    cam.width,
    cam.height,
    cfg
  );

  // Extract image and depth components
  using namespace torch::indexing;
  XCHECK_EQ(rgbd.sizes(), (at::IntArrayRef{1, cam.height, cam.width, 4}));
  auto image = rgbd[0].index({"...", Slice(0, 3)});
  XCHECK_EQ(image.sizes(), (at::IntArrayRef{cam.height, cam.width, 3}));
  auto depth_map = rgbd[0].index({"...", Slice(3)});
  XCHECK_EQ(depth_map.sizes(), (at::IntArrayRef{cam.height, cam.width, 1}));

  // Process conics for eigenvalue calculation
  auto a = filtered_metas.conics.select(2, 0);
  auto b = filtered_metas.conics.select(2, 1);
  auto c = filtered_metas.conics.select(2, 2);
  auto cov_inv_det = a * c - b * b;
  auto trace = a + c;
  auto mid = 0.5 * trace;
  auto discriminant = torch::sqrt((mid * mid - cov_inv_det).clamp_min(1e-6));
  auto lambda1 = torch::abs(mid + discriminant);
  auto lambda2 = torch::abs(mid - discriminant);
  torch::Tensor l1 = 3.0 * torch::sqrt(1.0 / lambda1);
  torch::Tensor l2 = 3.0 * torch::sqrt(1.0 / lambda2);

  // Create full-sized tensors for return values
  torch::Tensor full_opacities = torch::zeros({total_splats}, opacities.options());
  full_opacities.index_copy_(0, alive_indices, opacities);

  // Now expand the metas structure to include all splats
  gsplat::RasterizationMetas expanded_metas;
  
  // Expand conics (shape [1, N, 3])
  torch::Tensor expanded_conics = torch::zeros({1, total_splats, 3}, filtered_metas.conics.options());
  expanded_conics.index_put_({0, alive_indices}, filtered_metas.conics.index({0}));
  expanded_metas.conics = expanded_conics;
  
  // Expand radii (shape [1, N])
  torch::Tensor expanded_radii = torch::zeros({1, total_splats}, filtered_metas.radii.options());
  expanded_radii.index_put_({0, alive_indices}, filtered_metas.radii.index({0}));
  expanded_metas.radii = expanded_radii;
  
  // Expand means2d (shape [1, N, 2])
  torch::Tensor expanded_means2d = torch::zeros({1, total_splats, 2}, filtered_metas.means2d.options());
  expanded_means2d.index_put_({0, alive_indices}, filtered_metas.means2d.index({0}));
  expanded_metas.means2d = expanded_means2d;
  
  // Expand depths (shape [1, N])
  torch::Tensor expanded_depths = torch::zeros({1, total_splats}, filtered_metas.depths.options());
  expanded_depths.index_put_({0, alive_indices}, filtered_metas.depths.index({0}));
  expanded_metas.depths = expanded_depths;
  
  // Expand opacities (shape [1, N])
  torch::Tensor expanded_metas_opacities = torch::zeros({1, total_splats}, filtered_metas.opacities.options());
  expanded_metas_opacities.index_put_({0, alive_indices}, filtered_metas.opacities.index({0}));
  expanded_metas.opacities = expanded_metas_opacities;
  
  // Handle absgrad if defined
  if (filtered_metas.means2d_absgrad.defined()) {
    // DO NOT expand this tensor or create a new one
    // Just pass it through directly, otherwise backward() doesn't do its thing. Well expand it later.
    expanded_metas.means2d_absgrad = filtered_metas.means2d_absgrad;
  }

  // Copy other metadata that doesn't relate to specific splats
  expanded_metas.tile_width = filtered_metas.tile_width;
  expanded_metas.tile_height = filtered_metas.tile_height;
  expanded_metas.width = filtered_metas.width;
  expanded_metas.height = filtered_metas.height;
  expanded_metas.tile_size = filtered_metas.tile_size;
  expanded_metas.n_cameras = filtered_metas.n_cameras;
  
  // For tile-related fields, we need to keep them as is since they're generated
  // from the filtered splats and used for rendering
  expanded_metas.tiles_per_gauss = filtered_metas.tiles_per_gauss;
  expanded_metas.isect_ids = filtered_metas.isect_ids;
  expanded_metas.flatten_ids = filtered_metas.flatten_ids;
  expanded_metas.isect_offsets = filtered_metas.isect_offsets;

  return {image, alpha_map, depth_map, full_opacities, l1, l2, expanded_metas};
}

double applyLearningRateSchedule(
  torch::optim::Optimizer& optimizer,
  double initial_lr,
  int current_iter,
  const std::vector<int>& milestones,
  double gamma // Learning rate decays by this factor each time we reach a milestone
){
  // Apply LR decay at each milestone
  double curr_lr = initial_lr;
  for (int milestone : milestones) {
    if (current_iter >= milestone) {
      curr_lr *= gamma;
    }
  }

  // Update the learning rate for each parameter group
  for (auto& group : optimizer.param_groups()) {
    auto& options = static_cast<torch::optim::AdamOptions&>(group.options());
    options.lr(curr_lr);
  }

  return curr_lr;
}

torch::Tensor smoothnessLoss(const torch::Tensor& inv_depth) {
  auto depth_2d = inv_depth.squeeze();   
  auto laplacian = torch::ones({1, 1, 5, 5}, depth_2d.options());
  laplacian[0][0][2][2] = -24;  // Center pixel
  return torch::nn::functional::conv2d(
      depth_2d.unsqueeze(0).unsqueeze(0),  // Add only needed batch and channel dims
      laplacian,
      torch::nn::functional::Conv2dFuncOptions().padding(2)
  ).abs().mean();
}

torch::Tensor calibratedDepthLoss(
  torch::Tensor rendered_depth_,  // [H, W, 1] from gsplat
  torch::Tensor mono_depth_      // [1, H, W] from MiDaS
) {
  using namespace torch::indexing;
  
  // Check input tensor shapes
  XCHECK_EQ(rendered_depth_.dim(), 3) << "rendered_depth should be 3D [H, W, 1]";
  XCHECK_EQ(rendered_depth_.size(2), 1) << "rendered_depth should have shape [H, W, 1]";
  XCHECK_EQ(mono_depth_.dim(), 3) << "mono_depth should be 3D [1, H, W]";
  XCHECK_EQ(mono_depth_.size(0), 1) << "mono_depth should have shape [1, H, W]";
  
  // Convert rendered inverse depth to linear depth
  auto rendered_depth = torch::clamp(rendered_depth_.squeeze(0-1), kZNear, kZFar);
  auto mono_depth = torch::clamp(mono_depth_.squeeze(0), kZNear, kZFar);
  
  // Create mask excluding boundary regions (10% margin)
  int height = rendered_depth.size(0);
  int width = rendered_depth.size(1);
  int margin_y = static_cast<int>(height * 0.01);
  int margin_x = static_cast<int>(width * 0.01);
  
  auto y_indices = torch::arange(height, rendered_depth.options());
  auto x_indices = torch::arange(width, rendered_depth.options());
  auto interior_mask = (y_indices.unsqueeze(1) >= margin_y) & (y_indices.unsqueeze(1) < height - margin_y) &
                       (x_indices.unsqueeze(0) >= margin_x) & (x_indices.unsqueeze(0) < width - margin_x);
  
  auto valid_mask = (rendered_depth > kZNear) & (mono_depth > 0) & interior_mask;
  
  auto valid_rendered = rendered_depth.masked_select(valid_mask);
  auto valid_mono = mono_depth.masked_select(valid_mask);
  
  // First-pass scale/bias estimation
  auto rendered_mean = valid_rendered.mean();
  auto mono_mean = valid_mono.mean();
  
  auto centered_rendered = valid_rendered - rendered_mean;
  auto centered_mono = valid_mono - mono_mean;
  
  auto numerator = (centered_rendered * centered_mono).sum();
  auto denominator = (centered_mono * centered_mono).sum();
  
  auto scale = torch::where(
    denominator > 1e-10,
    numerator / denominator,
    torch::tensor(1.0, rendered_depth.options())
  );
  
  auto bias = rendered_mean - scale * mono_mean;
  
  // Outlier rejection
  auto initial_aligned_mono = scale * valid_mono + bias;
  auto errors = torch::abs(valid_rendered - initial_aligned_mono);
  
  auto sorted_errors = std::get<0>(torch::sort(errors));
  int inlier_count = static_cast<int>(errors.numel() * 0.9);
  auto outlier_threshold = sorted_errors[inlier_count - 1];
  
  auto inlier_mask = errors <= outlier_threshold;
  
  // Second-pass scale/bias estimation with inliers only
  auto inlier_rendered = valid_rendered.masked_select(inlier_mask);
  auto inlier_mono = valid_mono.masked_select(inlier_mask);
  
  rendered_mean = inlier_rendered.mean();
  mono_mean = inlier_mono.mean();
  
  centered_rendered = inlier_rendered - rendered_mean;
  centered_mono = inlier_mono - mono_mean;
  
  numerator = (centered_rendered * centered_mono).sum();
  denominator = (centered_mono * centered_mono).sum();
  
  scale = torch::where(
    denominator > 1e-10,
    numerator / denominator,
    torch::tensor(1.0, rendered_depth.options())
  );
  
  bias = rendered_mean - scale * mono_mean;
  
  // Apply scale and bias to mono depth
  auto aligned_mono = scale * mono_depth + bias;
  
  // Simple L1 loss between aligned depths
  return torch::l1_loss(rendered_depth.masked_select(valid_mask), aligned_mono.masked_select(valid_mask));
}

void printGradientNorms(std::shared_ptr<SplatModel> model) {
  torch::NoGradGuard no_grad;
  
  auto alive_mask = model->splat_alive.squeeze(-1);
  
  auto print_norm = [&alive_mask](const torch::Tensor& param, const std::string& name) {
    if (!param.grad().defined()) {
      XPLINFO << name << " gradient not defined";
      return;
    }
    
    auto alive_grads = param.grad().index({alive_mask});
    
    auto max_norm = alive_grads.norm(2, -1).max().item<float>();
    auto min_norm = alive_grads.norm(2, -1).min().item<float>();
    auto mean_norm = alive_grads.norm(2, -1).mean().item<float>();
    
    XPLINFO << name << " gradient norms - "
              << "max: " << max_norm << ", "
              << "min: " << min_norm << ", "
              << "mean: " << mean_norm;
  };
  
  // Print norms for each parameter type
  print_norm(model->splat_pos, "Position");
  print_norm(model->splat_color, "Color");
  print_norm(model->splat_alpha, "Alpha");
  print_norm(model->splat_scale, "Scale");
  print_norm(model->splat_quat, "Quaternion");
}

torch::Tensor floaterHashLoss(
  const torch::Tensor& splat_pos,    
  const torch::Tensor& splat_alpha,  
  const torch::Tensor& splat_alive,  
  float voxel_size = 0.1                   
) {
  auto pos_alive = splat_pos.index({splat_alive.squeeze(-1)});
  if (pos_alive.numel() == 0) return torch::tensor(0.0f, splat_pos.options());

  auto alpha_alive = torch::sigmoid(splat_alpha.index({splat_alive.squeeze(-1)})); 

  auto voxel_coords = torch::floor(pos_alive / voxel_size).to(torch::kLong); // [M, 3]
  if (voxel_coords.numel() == 0) return torch::tensor(0.0f, splat_pos.options());

  auto hash = (voxel_coords.select(1, 0) * 73856093 
             ^ voxel_coords.select(1, 1) * 19349663
             ^ voxel_coords.select(1, 2) * 83492791).abs().to(torch::kLong);

  if (hash.numel() == 0) return torch::tensor(0.0f, splat_pos.options());

  auto unique_result = torch::_unique2(hash.cpu(), true, true, true);
  auto counts = std::get<2>(unique_result).to(hash.device());       // [num_voxels]
  auto inverse_idx = std::get<1>(unique_result).to(hash.device());  // [M]

  if (counts.numel() == 0 || inverse_idx.numel() == 0) {
    return torch::tensor(0.0f, splat_pos.options());
  }

  auto density_per_splat = counts.gather(0, inverse_idx).to(torch::kFloat32); // [M]
  auto isolation_mask    = (density_per_splat == 1.0f).to(torch::kFloat32);    // [M]
  auto isolation_loss    = alpha_alive.squeeze() * isolation_mask;      

  return isolation_loss.mean();
}

torch::Tensor ordinalDepthLoss(const torch::Tensor& d_pred, const torch::Tensor& d_render, int64_t num_samples = 10000000, float alpha = 1.0f) {
  // Flatten the depth maps to shape [H*W]
  auto d_pred_flat = d_pred.flatten();
  auto d_render_flat = d_render.flatten();
  auto N = d_pred_flat.size(0);

  // Generate random index pairs for sampling pixel pairs.
  // Both indices1 and indices2 are 1D tensors of shape [num_samples].
  auto options = torch::TensorOptions().dtype(torch::kLong).device(d_pred.device());
  auto indices1 = torch::randint(0, N, {num_samples}, options);
  auto indices2 = torch::randint(0, N, {num_samples}, options);

  // Select the corresponding depth values for predicted depths.
  auto d1_pred = d_pred_flat.index_select(0, indices1);
  auto d2_pred = d_pred_flat.index_select(0, indices2);

  // Compute the order indicator R: +1 if d1_pred > d2_pred, otherwise -1.
  auto order_indicator = torch::where(d1_pred > d2_pred, torch::ones_like(d1_pred), -torch::ones_like(d1_pred));

  // Select the corresponding depth values for rendered depths.
  auto d1_render = d_render_flat.index_select(0, indices1);
  auto d2_render = d_render_flat.index_select(0, indices2);
  auto diff_render = d1_render - d2_render;

  // Compute tanh(alpha * (d1_render - d2_render))
  auto tanh_val = torch::tanh(alpha * diff_render);

  // Compute the loss as the mean absolute difference between tanh_val and the order indicator.
  auto loss = torch::mean(torch::abs(tanh_val - order_indicator));

  return loss;
}

torch::Tensor createSuperellipseVignette(
  int h, int w,
  torch::Device device,
  float sigma = 0.5f,
  float r0 = 0.95f,
  float r1 = 0.99f,
  float p = 10.0f  // Higher values make it more rectangular
) {
  auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto ys = torch::linspace(-1.0f, 1.0f, h, opts).unsqueeze(1); // [h,1]
  auto xs = torch::linspace(-1.0f, 1.0f, w, opts).unsqueeze(0); // [1,w]
  
  // p-norm: (|x|^p + |y|^p)^(1/p)
  auto r = torch::pow(torch::pow(torch::abs(xs), p) + torch::pow(torch::abs(ys), p), 1.0f / p);
  
  auto r2 = r.pow(2);
  auto base = torch::exp(-sigma * r2);
  auto ramp = torch::clamp((r1 - r) / (r1 - r0), 0.0f, 1.0f);
  return base * ramp;
}

void trainSplatModel(
  SplatConfig& cfg,
  const torch::DeviceType device,
  calibration::MultiCameraDataset& dataset,
  std::shared_ptr<SplatModel> model,
  std::shared_ptr<SplatModel> prev_model,
  GaussianSplatGuiData* gui_data,
  std::shared_ptr<std::atomic<bool>> cancel_requested 
) {
  using namespace torch::optim;

  if (cancel_requested && *cancel_requested) { return;} 

  torch::Tensor vignette;
  if (cfg.use_depth_loss) {
    vignette = createSuperellipseVignette(dataset.images[0].rows, dataset.images[0].cols, device);
  }

  //torch::autograd::AnomalyMode::set_enabled(true);
  auto train_timer = time::now();

  XCHECK(!torch::isnan(model->splat_pos).any().item<bool>());
  XCHECK(!torch::isinf(model->splat_pos).any().item<bool>());

  torch::Tensor should_stabilize = (prev_model == nullptr)
    ? torch::full_like(model->splat_alive, false, torch::kBool)
    : model->splat_alive;
  
  model->splat_pos.set_requires_grad(true);
  model->splat_color.set_requires_grad(true);
  model->splat_alpha.set_requires_grad(true);
  model->splat_scale.set_requires_grad(true);
  model->splat_quat.set_requires_grad(true);

  if (prev_model) {
    prev_model->splat_pos.set_requires_grad(false);
    prev_model->splat_color.set_requires_grad(false);
    prev_model->splat_alpha.set_requires_grad(false);
    prev_model->splat_scale.set_requires_grad(false);
    prev_model->splat_quat.set_requires_grad(false);
  }

  if (!model->per_image_depth_scale.defined() || model->per_image_depth_scale.numel() == 0) {
    int num_images = dataset.images.size();
    model->per_image_depth_scale = torch::ones({num_images}, model->splat_pos.options());
    model->per_image_depth_bias = torch::zeros({num_images}, model->splat_pos.options());
    model->per_image_depth_scale.set_requires_grad(true);
    model->per_image_depth_bias.set_requires_grad(true);
  }

  constexpr float kLR_alpha = 1e-1;
  constexpr float kLR_per_image_depth_scale = 1e-2;

  float lr = (prev_model == nullptr) ? cfg.learning_rate : cfg.learning_rate * 0.333 * 0.333 * 0.333;
  float lr_alpha = (prev_model == nullptr) ? kLR_alpha : kLR_alpha * 0.333 * 0.333 * 0.333;

  auto splat_alpha_optimizer = torch::optim::Adam(
    {model->splat_alpha},
    torch::optim::AdamOptions(lr_alpha).weight_decay(1e-9).eps(1e-9));

  auto splat_optimizer = torch::optim::Adam(
    {model->splat_pos, model->splat_color, model->splat_scale, model->splat_quat},
    torch::optim::AdamOptions(lr).weight_decay(0).eps(1e-9));

  auto per_image_depth_scale_optimizer = torch::optim::Adam(
    {model->per_image_depth_scale, model->per_image_depth_bias},
    torch::optim::AdamOptions(kLR_per_image_depth_scale).weight_decay(0).eps(1e-9));

  static int log_itr = 0; // this will keep increasing after the first frame
  std::ofstream f_log(cfg.output_dir + "/color_loss.txt", log_itr == 0 ? std::ios::out : std::ios::out | std::ios::app);

  constexpr int kStartGrad2dStep = 0; // TODO flag
  // Accumulates the 2d gradient for each splat for the population update - CxN
  torch::Tensor accumulated_grad_norm = torch::zeros({cfg.max_num_splats}, {torch::kFloat32}).to(device);

  std::set<std::string> created_vis_windows; // Hack to prevent overidding user-resized windows

  int num_nans_removed = 0;
  int num_itrs = cfg.num_itrs;
  if (cfg.is_video && prev_model == nullptr) num_itrs = cfg.first_frame_warmup_itrs;
  
  constexpr float full_population_by = 0.75; // aim to get a full population this far through the run
  const int stop_population_dynamics_itr = int(num_itrs * 0.9); // stop population dynamics after this many iterations
  int num_population_updates = 0; // just keeping track of how many have occurred
  XPLINFO << "population_update_interval=" << cfg.population_update_interval;
  //int alphapocolypse_itr = num_itrs * 0.3;
  //if (alphapocolypse_itr % cfg.population_update_interval == 0) {
  //  XPLINFO << "WARNING- alphapocolypse + population update on same iteration";
  //}
  //alphapocolypse_itr = -10;
  
  constexpr double kLRMilestoneDecay = 0.333; // TODO: cosine schedule?
  std::vector<int> milestones = {
    int(num_itrs * 0.5),
    int(num_itrs * 0.7),
    int(num_itrs * 0.9)
  };
  if (prev_model != nullptr) milestones = {};


  for (int itr = 0; itr < num_itrs; ++itr, ++log_itr) {
    if (cancel_requested && *cancel_requested) return;
  
    auto timer = time::now();

    applyLearningRateSchedule(splat_optimizer, lr, itr, milestones, kLRMilestoneDecay);
    applyLearningRateSchedule(splat_alpha_optimizer, lr_alpha, itr, milestones, kLRMilestoneDecay);

    torch::Tensor batch_loss = torch::zeros({1}, torch::kFloat32).to(device);
  
    // Setup to sample camera idx's without replacement for a batch of multiple images
    std::vector<int> img_idxs_random(dataset.cameras.size());
    std::iota(img_idxs_random.begin(), img_idxs_random.end(), 0);
    std::shuffle(img_idxs_random.begin(), img_idxs_random.end(), rng);

    std::map<std::string, torch::Tensor> vis_tensors;
    if (cfg.train_vis_interval) {
      // keep the vis windows responsive
      cv::pollKey();
    }

    std::vector<torch::Tensor> batch_means2d;
    std::vector<torch::Tensor> batch_radii;
    std::vector<calibration::RectilinearCamerad> batch_cams;

    // If we only have 1 image, dont try to make a batch with more than 1 eg
    cfg.images_per_batch = std::min(cfg.images_per_batch, int(dataset.images.size()));

    // In each batch, select several images at random.
    for (int img_itr = 0; img_itr < cfg.images_per_batch; ++img_itr) {
      const int target_img_idx = img_idxs_random[img_itr];
      XCHECK(dataset.cameras[target_img_idx].is_rectilinear);
      const calibration::RectilinearCamerad& cam = dataset.cameras[target_img_idx].rectilinear;

      torch::Tensor target_image = dataset.image_tensors[target_img_idx];
      torch::Tensor target_depthmap = cfg.use_depth_loss 
        ? dataset.depthmap_tensors[target_img_idx] : torch::zeros({1, 1, 1}, {torch::kFloat32}).to(device);
      torch::Tensor ignore_mask = torch::ones_like(target_image);
      if (!dataset.ignore_masks.empty()) {
        ignore_mask = dataset.ignore_masks[target_img_idx].expand({-1, -1, 3});
      }

      // TODO: randomized backgrounds (requires kernel change)
      // TODO: change the 1 to the batch size if using parallel batches
      auto background_colors = torch::rand({1, 3}, {torch::kFloat32}).to(device);
      auto background_depths = torch::zeros({1, 1}, {torch::kFloat32}).to(device);
      auto backgrounds = torch::cat({background_colors, background_depths}, 1);

      //torch::cuda::synchronize();
      //auto timer = time::now();

      // Gsplat renderer
      auto [rendered_image, alpha_map, depth_map, sigmoid_alpha, l1, l2, metas] = renderSplatImageGsplat(
        device, cam, model, backgrounds);

      //torch::cuda::synchronize();
      //XPLINFO << "render time (sec): " << time::timeSinceSec(timer);

      if (cfg.train_vis_interval) {
        if (itr != 0 && itr % cfg.train_vis_interval == 0 && img_itr == 0) {
          auto compensated_image = rendered_image - background_colors[0] * (1 - alpha_map[0]);
          vis_tensors.insert({
            //{"vignette", vignette.squeeze().unsqueeze(-1)},
            {"target_image", target_image},
            {"rendered_image", compensated_image},
            //{"alpha_map", alpha_map},
            {"depth_map", depth_map * 0.02},
          });
          if (cfg.use_depth_loss) {
            vis_tensors.insert({"target_depth", target_depthmap.squeeze().unsqueeze(-1) * 0.02});
          }
        }
      }

      // Store state for accumulating grads for population update
      if (itr >= kStartGrad2dStep) {
        XCHECK(metas.means2d_absgrad.defined());
        batch_means2d.push_back(metas.means2d_absgrad); // values are filled in during backward pass
        batch_radii.push_back(metas.radii);
        batch_cams.push_back(cam);
      }

      torch::Tensor color_loss = 
          dataset.ignore_masks.empty() 
        ? torch::smooth_l1_loss(target_image, rendered_image)
        : torch::smooth_l1_loss(ignore_mask * target_image, ignore_mask * rendered_image);
      
      //torch::Tensor vig = vignette.unsqueeze(-1);
      //torch::Tensor color_loss = 
      //    dataset.ignore_masks.empty() 
      //  ? torch::smooth_l1_loss(vig * target_image, vig * rendered_image)
      //  : torch::smooth_l1_loss(vig * ignore_mask * target_image, vig * ignore_mask * rendered_image);

      batch_loss += color_loss;
      
      torch::Tensor depth_loss = torch::zeros({1}, torch::kFloat32).to(device);
      if (cfg.use_depth_loss) {
        //depth_loss = ordinalDepthLoss(depth_map.squeeze(), target_depthmap.squeeze());
        
        // TODO: including ignore masks made things worse on a highly masked dataset but it seems like the right thing to do
        //torch::Tensor ignore = dataset.ignore_masks.empty()
        //  ? torch::ones_like(vignette)
        //  : dataset.ignore_masks[target_img_idx].squeeze();

        auto s = model->per_image_depth_scale[target_img_idx];
        auto b = model->per_image_depth_bias[target_img_idx];
        auto calibrated_depth = vignette * (b + s * target_depthmap.squeeze());
        
        depth_loss = torch::smooth_l1_loss(vignette * depth_map.squeeze(), calibrated_depth);
        batch_loss += depth_loss * 0.001;

        //torch::Tensor depth_smoothness_loss = smoothnessLoss(depth_v);
        //batch_loss += depth_smoothness_loss * 0.000001;
      }
      //torch::Tensor splat_invdepths = torch::clamp(kInvDepthCoef / (metas.depths + 1e-6), 0.0, 1.0); // TODO: could revist this, with less bugs. see other losses
      //torch::Tensor far_away_loss = splat_invdepths.mean(); // prefer 0 inv depth all other things being equal

      //torch::Tensor alpha_map_loss = torch::mean(torch::abs(alpha_map - 1.0));
      //batch_loss += alpha_map_loss * 0.1;

      // Near floater loss
      constexpr float kZNearFloater = 0.5;
      auto r = metas.radii.squeeze();
      auto d0    = metas.depths[0].squeeze();
      auto mask  = model->splat_alive.squeeze() & (r > 0) & (d0 >  kZNear) & (d0 <  kZNearFloater);
      auto near_floater_loss = (torch::sigmoid(model->splat_alpha.squeeze()) * mask).mean();

      batch_loss += near_floater_loss * 10000.0;
      

      XPLINFO         << "train: " // NOTE: this is formatted for a progress parser in 4dgstudio
                      << itr << " / " << num_itrs
                      <<  std::setprecision(5)
                      << "\tclr=" << color_loss.item<float>()
                      //<< "\tdsm=" << depth_smoothness_loss.item<float>()
                      //<< "\talm=" << alpha_map_loss.item<float>()
                      << "\tnfl=" << near_floater_loss.item<float>()
                      << "\tdep=" << depth_loss.item<float>()
                      << "\t#nan=" << num_nans_removed
                      << "\t#alive=" << model->splat_alive.sum().item<int>();
      f_log << log_itr << "\t" << color_loss.item<float>() << std::endl;
      

    } // end loop over images within batch
  
    torch::Tensor splat_alpha_alive = torch::sigmoid(model->splat_alpha).index({ model->splat_alive.squeeze(-1) });
    auto alpha_loss = torch::mean(splat_alpha_alive);
    batch_loss += cfg.images_per_batch * alpha_loss * 0.0001;

    XPLINFO << "\t\t\t\t\t\t\t\t\t\t\t\talpha_loss=" << alpha_loss.item<float>();

    // Ratio of largest to smalles tscale
    torch::Tensor scale_exp = torch::exp(scaleActivation(model->splat_scale));
    auto alive_scales = scale_exp.index({model->splat_alive.squeeze(-1)});
    auto scale_ratios = std::get<0>(alive_scales.max(1)) / std::get<0>(alive_scales.min(1));
    const float kMaxScaleRatio = 4.0; // above this and regularization penalty kicks in
    batch_loss += cfg.images_per_batch * torch::mean(torch::relu(scale_ratios - kMaxScaleRatio)) * 1e-1;
    
    // 3D scale loss
    //auto scale_exp_alive = scale_exp.index({ model->splat_alive.squeeze(-1) });
    //auto scale_norm = torch::norm(scale_exp_alive, 2, 1, true);
    //auto size3d_loss = scale_norm.mean();
    //batch_loss += size3d_loss * 1e-2;

    // This is minimized when alpha is close to 0 or 1, and maximized at 0.5
    //auto binary_alpha_loss = torch::mean(splat_alpha_alive * (1.0 - splat_alpha_alive));
    //batch_loss += binary_alpha_loss * 1e-3;

    // Sky-ball loss (instead of projection)
    auto pos_linear_alive = splatPosActivation(model->splat_pos).index({ model->splat_alive.squeeze(-1) });
    auto pos_norm = torch::norm(pos_linear_alive, 2, 1, true);
    auto skyball_loss = torch::relu(pos_norm - kZFar).pow(2).mean();
    batch_loss += skyball_loss * 0.0001;

    if (prev_model) {
      auto s = should_stabilize.squeeze(-1);
      torch::Tensor stabilize_pos = torch::smooth_l1_loss(model->splat_pos.index({s}), prev_model->splat_pos.index({s}));
      torch::Tensor stabilize_color = torch::smooth_l1_loss(
        torch::sigmoid(model->splat_color.index({s})), torch::sigmoid(prev_model->splat_color.index({s})));
      torch::Tensor stabilize_alpha = torch::smooth_l1_loss(
        torch::sigmoid(model->splat_alpha.index({s})), torch::sigmoid(prev_model->splat_alpha.index({s})));
      torch::Tensor stabilize_scale = torch::smooth_l1_loss(
        scaleActivation(model->splat_scale.index({s})), scaleActivation(prev_model->splat_scale.index({s})));
      torch::Tensor stabilize_quat = torch::smooth_l1_loss(model->splat_quat.index({s}), prev_model->splat_quat.index({s}));
      
      //batch_loss +=
      //  stabilize_pos * 1e-1 + 
      //  stabilize_color * 1e-1 + 
      //  stabilize_alpha * 1e-1 + 
      //  stabilize_scale * 1e-1 + 
      //  stabilize_quat * 1e-1;
      batch_loss +=
        stabilize_pos * 1.0 + 
        stabilize_color * 1.0 + 
        stabilize_alpha * 1.0 + 
        stabilize_scale * 1.0 + 
        stabilize_quat * 1.0;

      std::cout << itr  << std::setprecision(5) << "\t\t\t\t\t\t\t"
                << "\tsta=" << stabilize_pos.item<float>() << " " << stabilize_color.item<float>() << " " << stabilize_alpha.item<float>() 
                << " " << stabilize_scale.item<float>() << " " << stabilize_quat.item<float>()
                << std::endl;
    }

    splat_alpha_optimizer.zero_grad();
    splat_optimizer.zero_grad();
    per_image_depth_scale_optimizer.zero_grad();

    batch_loss.backward();

    //printGradientNorms(model);

    splat_alpha_optimizer.step();
    splat_optimizer.step();
    per_image_depth_scale_optimizer.step();


    { // Accumulate grads for population dynamics. Do this before killing any splats to maintain indexing
      torch::NoGradGuard no_grad;

      for (int i = 0; i < batch_means2d.size(); ++i) {
        torch::Tensor grads = batch_means2d[i];

        // IMPORTANT: We need to handle the shape mismatch here
        // grads has shape [1, alive_splats, 2]
        // but we need a tensor of shape [1, total_splats, 2]
        
        torch::Tensor full_grads = torch::zeros({1, model->splat_pos.size(0), 2}, 
                                              grads.options());
        torch::Tensor alive_indices = model->splat_alive.squeeze(-1).nonzero().squeeze(1);
        full_grads.index_copy_(1, alive_indices, grads);

        torch::Tensor processed_grads = full_grads;
        
        // Don't accumulate dead splat gradients
        processed_grads = torch::where(
          model->splat_alive.unsqueeze(0),
          processed_grads,
          torch::zeros_like(processed_grads, {torch::kFloat32})
        );
        
        // Normalize gradients
        processed_grads.select(1,0) *= batch_cams[i].width / 2.0;
        processed_grads.select(1,1) *= batch_cams[i].height / 2.0;
        
        // Radii == 0 is how gsplat reports an invalid splat
        torch::Tensor sel = batch_radii[i] > 0.0;
        torch::Tensor gs_ids = sel.nonzero().unbind(1)[1];
        processed_grads = processed_grads.index({sel});
        
        auto norms = processed_grads.norm(std::nullopt, -1);
        accumulated_grad_norm.index_add_(0, gs_ids, norms);
      }
    }

    { // no-grad modifications happen after the gradient step to avoid double backward
      // TODO: move this out of the scope; nothing after the optimizer step should affect grads
      // Not doing this now because it would wreck this diff
      torch::NoGradGuard no_grad;

      if (cfg.train_vis_interval && itr != 0  && itr % cfg.train_vis_interval == 0) {
        for (auto [vis_name, vis_tensor]: vis_tensors) {
          //XPLINFO << vis_name << " sizes = " << vis_tensor.sizes();
          if (vis_tensor.dim() == 4) {
            XCHECK_EQ(vis_tensor.size(0), 1) << "NYI visualizing multiple images from a batch";
            vis_tensor = vis_tensor.squeeze(0);
          }

          vis_tensor = vis_tensor.to(torch::kCPU);
          cv::Mat cv_image;
          if (vis_tensor.size(2) == 3) {
            cv_image = cv::Mat(vis_tensor.size(0), vis_tensor.size(1), CV_32FC3, vis_tensor.data_ptr<float>());
          } else {
            XCHECK_EQ(vis_tensor.size(2), 1) << "Only 1 & 3 channel images can be visualized";
            cv_image = cv::Mat(vis_tensor.size(0), vis_tensor.size(1), CV_32FC1, vis_tensor.data_ptr<float>());
          }

          cv::namedWindow(vis_name, cv::WINDOW_NORMAL);

          // Resize the first time the window is created
          if (created_vis_windows.find(vis_name) == created_vis_windows.end()) {
            created_vis_windows.emplace(vis_name);
            // Default to an easily tilable size
            cv::resizeWindow(vis_name, 1920/2, 1080/2);
          }

          cv::imshow(vis_name, cv_image);
        }
      }

      // Kill nan splats that are created by the backward pass
      auto valid_mask = torch::isfinite(model->splat_pos).all(1, true);
      auto invalid_mask = ~valid_mask;
      int num_invalid = invalid_mask.sum().item<int>();
      if (num_invalid > 0) {
        XPLWARN << "****\nKilling invalid (probably NaN) splats, # invalid: " << num_invalid << "\n*****";
        
        // Debug print invalid splat properties
        auto invalid_pos = torch::masked_select(model->splat_pos, invalid_mask.expand_as(model->splat_pos)).reshape({-1, 3});
        auto invalid_color = torch::masked_select(model->splat_color, invalid_mask.expand_as(model->splat_color)).reshape({-1, 3});
        auto invalid_quat = torch::masked_select(model->splat_quat, invalid_mask.expand_as(model->splat_quat)).reshape({-1, 4});
        auto invalid_alpha = torch::masked_select(model->splat_alpha, invalid_mask.expand_as(model->splat_alpha)).reshape({-1, 1});
        auto invalid_scale = torch::masked_select(model->splat_scale, invalid_mask.expand_as(model->splat_scale)).reshape({-1, 3});
        
        XPLINFO << "invalid_pos:\n" << invalid_pos.sizes();
        XPLINFO << "invalid_color:\n" << invalid_color.sizes();
        XPLINFO << "invalid_quat:\n" << invalid_quat.sizes();
        XPLINFO << "invalid_alpha:\n" << invalid_alpha.sizes();
        XPLINFO << "invalid_scale:\n" << invalid_scale.sizes();

        num_nans_removed += num_invalid;
        model->splat_alive = torch::where(
          valid_mask,
          model->splat_alive,
          torch::full_like(model->splat_alive, false, torch::kBool));
        model->splat_pos = torch::where(
          valid_mask,
          model->splat_pos,
          splatPosInverseActivation(torch::zeros_like(model->splat_pos)));
      }
    } // no_grad

    if (itr > 0 && itr <= stop_population_dynamics_itr && itr % cfg.population_update_interval == 0) {
      XPLINFO << "Doing population dynamics step " << num_population_updates;
      XCHECK(!torch::isnan(model->splat_pos).any().item<bool>());
      XCHECK(!torch::isinf(model->splat_pos).any().item<bool>());
      int target_num_alive = std::min(int(cfg.max_num_splats * (float(itr) / float(num_itrs)) / full_population_by), cfg.max_num_splats);
      if (prev_model != nullptr) target_num_alive = cfg.max_num_splats; // Past frame 0, try for full population immediately
      XPLINFO << "target_num_alive=" << target_num_alive;

      XPLINFO << "before population dynamics, accumulated_grad_norm=" << accumulated_grad_norm.sum().item<float>();
      splatPopulationDynamics(device, cfg, dataset, model, should_stabilize, target_num_alive, accumulated_grad_norm);

      accumulated_grad_norm.zero_();

      ++num_population_updates;

      XCHECK(!torch::isnan(model->splat_pos).any().item<bool>());
      XCHECK(!torch::isinf(model->splat_pos).any().item<bool>());
    }

    // Alpha-pocolypse
    //if (itr == alphapocolypse_itr && prev_model == nullptr) { // Don't do this after the first frame of video
    //  torch::NoGradGuard no_grad;
    //  XPLINFO << "**** ALPHAPOCOLYPSE";
    //  auto new_alpha = sigmoidInverse(torch::tensor(0.001, model->splat_alpha.options()));
    //  model->splat_alpha.index_put_({ model->splat_alive.squeeze(-1) }, new_alpha);
    //}

    // NOTE: updating every iteration meaningfully slows down training!
    if (gui_data && (itr % 10 == 0 || itr == num_itrs - 1)) {
    //if (gui_data) {
      std::lock_guard<std::mutex> guard(gui_data->mutex);
      if (gui_data->current_model == nullptr) {
        gui_data->current_model = std::make_shared<SplatModel>();
      }
      gui_data->current_model->copyFrom(model);
    }

    //XPLINFO << "train itr time (sec): " << time::timeSinceSec(timer);

    if (cfg.save_steps) {
      saveEncodedSplatFileWithSizzleZ(cfg.output_dir + "/trainsplat/" + string::intToZeroPad(itr, 6) + ".png", model);
    }
  } // end loop over itrs
  f_log.close();

  // Do a little bit of final cleanup of the solution
  { // Normalize the quaternions so they do not grow unbondedly over multiple frames
    torch::NoGradGuard no_grad;
    model->splat_quat = model->splat_quat / (torch::norm(model->splat_quat, 2, 1, true) + 1e-7);
  }
}

void calculatePsnrForDataset(
  SplatConfig& cfg,
  torch::DeviceType device,
  calibration::MultiCameraDataset& dataset,
  std::shared_ptr<SplatModel> model
) {
  XPLINFO << "Calculating PSNR on training images";
  float sum_psnr = 0;
  for (int i = 0; i < dataset.cameras.size(); ++i) {
    XCHECK(dataset.cameras[i].is_rectilinear);
    calibration::RectilinearCamerad& cam = dataset.cameras[i].rectilinear;

    auto [rendered_image, _0, depth_map, _1, _2, _3, _4] = 
      renderSplatImageGsplat(device, cam, model);

    const float mse = torch::mse_loss(rendered_image, dataset.image_tensors[i]).item<float>();
    const double psnr = 20.0 * std::log10(1.0 / std::sqrt(mse));
    sum_psnr += psnr;
    XPLINFO << "MSE: " << mse << " PSNR: " << psnr;

    // Show the rendered training set images
    auto cpu_image = rendered_image.to(torch::kCPU);
    cv::Mat cv_image(cpu_image.size(0), cpu_image.size(1), CV_32FC3, cpu_image.data_ptr<float>());
    cv::imwrite(cfg.output_dir + "/train_" + std::to_string(i) + ".jpg", cv_image * 255);
    //cv::imshow("rendered", cv_image);
    //cv::waitKey(0);

    auto cpu_depth = depth_map.to(torch::kCPU);
    cv::Mat cv_depth(cpu_depth.size(0), cpu_depth.size(1), CV_32FC1, cpu_depth.data_ptr<float>());
    //constexpr float kDepthmapScale = 0.1;
    cv::normalize(cv_depth, cv_depth, 0.0, 1.0, cv::NORM_MINMAX);
    cv::Mat turbo_viz(cv_depth.size(), CV_32FC3);
    for (int y = 0; y < cv_depth.rows; ++y) {
      for (int x = 0; x < cv_depth.cols; ++x) {
        Eigen::Vector3f c = turbo_colormap::float01ToColor(0.3 * cv_depth.at<float>(y, x));
        turbo_viz.at<cv::Vec3f>(y, x) = cv::Vec3f(c.x(), c.y(), c.z());
      }
    }
    cv::imwrite(cfg.output_dir + "/depth_" + std::to_string(i) + ".jpg", turbo_viz * 255);
  }
  const float avg_psnr = sum_psnr / dataset.cameras.size();
  XPLINFO << "avg PSNR (all train images): " << avg_psnr;
}

void runSplatPipelineStatic(SplatConfig& cfg) {
  torch::manual_seed(123);  // For reproducible initialization of weights
  srand(123);               // For calls to rand()
  const torch::DeviceType device = util_torch::findBestTorchDevice();

  std::system(("rm -rf " + cfg.output_dir + "/trainsplat").c_str());
  std::system(("mkdir -p " + cfg.output_dir + "/trainsplat").c_str());

  XPLINFO << "runSplatPipelineStatic reading dataset from " << cfg.train_images_dir << ", json file " << cfg.train_json;
  calibration::MultiCameraDataset train_dataset = calibration::readDataset(cfg.train_images_dir, cfg.train_json, device, cfg.resize_max_dim);

  std::shared_ptr<SplatModel> model = initSplatPopulation(device, cfg, train_dataset);

  trainSplatModel(cfg, device, train_dataset, model);

  saveEncodedSplatFileWithSizzleZ(cfg.output_dir + "/splats.png", model);

  if (cfg.calc_psnr) {
    calculatePsnrForDataset(cfg, device, train_dataset, model);
  }

  // Render novel views from an orbit
  for (int frame_counter = 0; frame_counter < 3000000; ++frame_counter) {
    XCHECK(train_dataset.cameras[0].is_rectilinear);
    calibration::RectilinearCamerad cam(train_dataset.cameras[0].rectilinear);

    float theta = 0.5 * std::sin(0.005 * frame_counter) - (M_PI / 2.0);
    const float r = 3.0 + 2.0 * std::sin(0.01 * frame_counter);
    cam.cam_from_world.linear() = Eigen::AngleAxisd(theta + M_PI/2.0, Eigen::Vector3d::UnitY()).matrix();
    cam.setPositionInWorld(r * Eigen::Vector3d(std::cos(theta), 0.0, std::sin(theta)));

    auto [rendered_image, _0, _1, _2, _3, _4, metas] = renderSplatImageGsplat(device, cam, model);
    auto cpu_image = rendered_image.to(torch::kCPU);
    cv::Mat cv_image(cpu_image.size(0), cpu_image.size(1), CV_32FC3, cpu_image.data_ptr<float>());
    cv::namedWindow("rendered", cv::WINDOW_NORMAL);
    cv::imshow("rendered", cv_image);
    cv::waitKey(1);
  }
}

calibration::RectilinearCamerad precomputeFisheyeToRectilinearWarp(
  const calibration::FisheyeCamerad& src_cam,
  std::vector<cv::Mat>& warp_uv,
  const double focal_multiplier,
  const int resize_max_dim // Only used if not zero
) {
  calibration::RectilinearCamerad new_cam;
  new_cam.cam_from_world = src_cam.cam_from_world;
  new_cam.width = src_cam.width;
  new_cam.height = src_cam.height;
  new_cam.k1 = 0;
  new_cam.k2 = 0;
  new_cam.optical_center = Eigen::Vector2d(new_cam.width/2, new_cam.height/2);
  new_cam.focal_length = focal_multiplier * Eigen::Vector2d(new_cam.width, new_cam.width); // NOTE: this is [w, w] not [w, h] so we have a square aspect ratio
  if (resize_max_dim != 0) { new_cam.resizeToMaxDim(resize_max_dim); }

  cv::Mat warp(cv::Size(new_cam.width, new_cam.height), CV_32FC2);
  for (int y = 0; y < new_cam.height; ++y) {
    for (int x = 0; x < new_cam.width; ++x) {
      Eigen::Vector3d ray_dir(
        (x - new_cam.optical_center.x()) / new_cam.focal_length.x(),
        -(y - new_cam.optical_center.y()) / new_cam.focal_length.y(), // NOTE: we need to put a - here to flip vertically
        1);
      ray_dir.normalize();

      const Eigen::Vector2d projected_pixel = src_cam.pixelFromCam(ray_dir);
      warp.at<cv::Vec2f>(y, x) = cv::Vec2f(projected_pixel.x(), projected_pixel.y());
    }
  }
  cv::split(warp, warp_uv);
  return new_cam;
}

void runSplatPipelineVideo(SplatConfig& cfg) {
  const torch::DeviceType device = util_torch::findBestTorchDevice();

  std::system(("rm -rf " + cfg.output_dir + "/splat_frames").c_str());
  std::system(("rm -rf " + cfg.output_dir + "/trainsplat").c_str());
  std::system(("mkdir -p " + cfg.output_dir + "/splat_frames").c_str());
  std::system(("mkdir -p " + cfg.output_dir + "/trainsplat").c_str());

  std::vector<calibration::NerfKludgeCamera> cameras = calibration::readDatasetCameraJson(cfg.vid_dir + "/dataset.json");

  calibration::readTimeOffsetJson(cfg.vid_dir + "/time_offsets.json", cameras);

  // Precompute warps from fisheye to rectified and recitifed camera intrinsics
  std::vector<calibration::NerfKludgeCamera> rectified_cameras;
  std::vector<std::vector<cv::Mat>> camera_to_rectify_warp(cameras.size(), std::vector<cv::Mat>());

  // Read ignore mask images. Make a binary mask from pixels that are pure red
  // in the image (0, 0, 255).
  std::vector<torch::Tensor> ignore_mask_tensors;
  if (file::directoryExists(cfg.vid_dir + "/masks")) {
    XPLINFO << "Found /masks folder. Reading masks...";

    for (int i = 0; i < cameras.size(); ++i) {
      auto& cam = cameras[i];
      const std::string mask_filename = cfg.vid_dir + "/masks/" + cam.name() + ".png";
      if (file::fileExists(mask_filename)) {
        cv::Mat mask_rgb = cv::imread(mask_filename);
        // Warp the mask from fisheye to rectilnear
        cv::remap(mask_rgb, mask_rgb, camera_to_rectify_warp[i][0], camera_to_rectify_warp[i][1], cv::INTER_AREA, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 0));     

        cv::Mat mask_binary(mask_rgb.size(), CV_8U, cv::Scalar(0));
        for (int y = 0; y < mask_rgb.rows; ++y) {
          for (int x = 0; x < mask_rgb.cols; ++x) {
            mask_binary.at<uint8_t>(y, x) = mask_rgb.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 0, 255) ? 0 : 255;
          }
        }
        ignore_mask_tensors.push_back(torch_opencv::cvMat8UC1_to_Tensor(device, mask_binary).permute({1, 2, 0}));
        //cv::imshow("mask_binary", mask_binary); cv::waitKey(0);
      }
    }
  }

  std::vector<cv::VideoCapture> video_captures(cameras.size());
  for (int i = 0; i < cameras.size(); ++i) {
    const std::string video_path = cfg.vid_dir + "/" + cameras[i].name() + ".mp4";
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

  std::shared_ptr<SplatModel> model = nullptr;
  std::shared_ptr<SplatModel> prev_model = nullptr;

  int frame_num = 0;
  bool have_more_frames = true;
  while(have_more_frames) {
    XPLINFO << "=========== frame_num: " << frame_num;
    // Re-seed RNG every frame for more temporally stable results
    torch::manual_seed(123);  // For reproducible initialization of weights
    srand(123);               // For calls to rand()
    
    calibration::MultiCameraDataset frame_dataset;
    // Get one frame from each camera's video, maybe preprocess
    for (int cam_idx = 0; cam_idx < cameras.size(); ++cam_idx) {
      cv::Mat image;
      video_captures[cam_idx] >> image;
      if (image.empty()) { have_more_frames = false; break; }

      // Precompute warps from fisheye to rectified and recitifed camera intrinsics
      // This is deferred because we need to know the image size
      if (rectified_cameras.empty()) {
        for (int i = 0; i < cameras.size(); ++i) {
          XPLINFO << "precomputing rectification warp for camera: " << cameras[i].name();
          XCHECK(cameras[i].is_fisheye);
          cameras[i].resizeToWidth(image.cols); // Fix case where intrinsics dont match video file size
          constexpr float kGoProHero12MagicFovConstant = 0.4;
          calibration::NerfKludgeCamera rectified_cam(precomputeFisheyeToRectilinearWarp(
            cameras[i].fisheye,
            camera_to_rectify_warp[i],
            kGoProHero12MagicFovConstant,
            cfg.resize_max_dim));
          rectified_cameras.push_back(rectified_cam);
        }
      }

      cv::Mat rectified_image;
      // cv::remap with INTER_CUBIC can alias badly when mapping from a high
      // res image to a low res image. Pre-blurring reduces this issue, and can dramatically improve PSNR.
      const float downscale_ratio = rectified_cameras[cam_idx].getWidth() / float(image.cols);
      if (downscale_ratio <= 0.25) {
        const double sigma = 0.25 / downscale_ratio; // TODO: this # is hand-tuned, could be further tuned to maximize PSNR
        const int kernel_size = 2 * static_cast<int>(std::ceil(3 * sigma)) + 1;
        cv::GaussianBlur(image, image, cv::Size(kernel_size, kernel_size), sigma, sigma);
      }
      cv::remap(image, rectified_image, camera_to_rectify_warp[cam_idx][0], camera_to_rectify_warp[cam_idx][1], cv::INTER_CUBIC, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0, 0));
      
      // Save the first frame from each video as a png for previewing the time offset
      if (frame_num == 0) {
        cv::imwrite(cfg.output_dir + "/" + cameras[cam_idx].name() + ".jpg", rectified_image);
      }

      frame_dataset.images.push_back(rectified_image); // TODO: do we need this given that we have the tensor?
      
      // TODO: calibration::cvMat8UC3_to_Tensor is deprecated, should use torch_opencv::cvMat_to_Tensor
      frame_dataset.image_tensors.push_back(calibration::cvMat8UC3_to_Tensor(device, rectified_image));
    }
    frame_dataset.cameras = rectified_cameras;
    frame_dataset.ignore_masks = ignore_mask_tensors;

    // Initialize splat tensors on the first frame
    if (frame_num == 0) {
      model = initSplatPopulation(device, cfg, frame_dataset);
    }

    trainSplatModel(cfg, device, frame_dataset, model, prev_model);

    // Copy the model for use in regularizing the next frame
    prev_model = std::make_shared<SplatModel>();
    prev_model->copyFrom(model);

    cfg.save_steps = false; // HACK: turn off save steps after the first frame (it may or may not be on)

    saveEncodedSplatFileWithSizzleZ(
      cfg.output_dir + "/splat_frames/" + string::intToZeroPad(frame_num, 6) + ".png", model);

    if (cfg.calc_psnr && frame_num == 0) {
      calculatePsnrForDataset(cfg, device, frame_dataset, model);
    }

    // HACK: render a training frame
    //{ auto [train_image, _0, _1, _2, _3, _4] = renderSplatImage(device, frame_dataset.cameras[0].rectilinear, model);
    //  auto cpu_image = train_image.to(torch::kCPU);
    //  cv::Mat cv_image(cpu_image.size(0), cpu_image.size(1), CV_32FC3, cpu_image.data_ptr<float>());
    //  cv::imshow("train", cv_image);
    //  cv::waitKey(0);
    //}

    if( !have_more_frames) break;
    ++frame_num;
  }
}

}}  // end namespace p11::splat
