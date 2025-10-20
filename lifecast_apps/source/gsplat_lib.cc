// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "gsplat_lib.h"

#include "third_party/gsplat/gsplat/cuda/include/bindings.h"
#include "torch/torch.h"
#include "logger.h"

// This file is a translation of gsplat/rendering.py and gsplat/cuda/_wrapper.py to c++ libtorch

namespace p11 { namespace gsplat {

class FullyFusedProjection : public torch::autograd::Function<FullyFusedProjection> {
 public:
  static torch::autograd::variable_list forward(
    torch::autograd::AutogradContext* ctx,
    torch::Tensor means,
    //c10::optional<torch::Tensor> covars_opt, // argument order matches _wrapper.fully_fused_projection
    torch::Tensor quats,
    torch::Tensor scales,
    torch::Tensor viewmats,
    torch::Tensor Ks,
    const int width,
    const int height,
    const float eps2d = 0.3,
    const float sigma_level = 3.f,
    const float near_plane = 0.01,
    const float far_plane = 1e10,
    const float radius_clip = 0.0,
    const bool calc_compensations = false,
    ::gsplat::CameraModelType camera_model_type = ::gsplat::PINHOLE
  ) {

    // We *should* be able to pass these in as c10::optional<torch::Tensor> arguments, according
    // to torch autograd `custom_function.h`. For some reason, it's putting empty tensors and optionals
    // into the value list. For now, we only will use the quats+scales approach
    c10::optional<torch::Tensor> covars_opt, quats_opt, scales_opt;
    //if (quats.defined()) {
    //  quats_opt = quats;
    //}
    //if (scales.defined()) {
    //  scales_opt = scales;
    //}

    quats_opt = quats;
    scales_opt = scales;
    auto [radii, means2d, depths, conics, compensations] = ::gsplat::fully_fused_projection_fwd_tensor(
      means,
      covars_opt,
      quats_opt,
      scales_opt,
      viewmats,
      Ks,
      width,
      height,
      eps2d,
      sigma_level,
      near_plane,
      far_plane,
      radius_clip,
      calc_compensations,
      camera_model_type
    );

    XCHECK(!calc_compensations) << "NYI due to c10::optional bug";
    //if (!calc_compensations) {
    //  compensations = torch::Tensor();
    //}

    //torch::Tensor covars;
    //if (covars_opt.has_value()) {
    //  covars = *covars_opt;
    //}

    ctx->save_for_backward({
      means,
      //covars,
      quats,
      scales,
      viewmats,
      Ks,
      radii,
      conics,
      //compensations
    });

    ctx->saved_data["width"] = width;
    ctx->saved_data["height"] = height;
    ctx->saved_data["eps2d"] = eps2d;
    ctx->saved_data["camera_model_type"] = static_cast<int>(camera_model_type);

    //return {radii, means2d, depths, conics, compensations};
    return {radii, means2d, depths, conics};
  }

  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_output
  ) {
    torch::Tensor v_radii = grad_output[0];
    torch::Tensor v_means2d = grad_output[1];
    torch::Tensor v_depths = grad_output[2];
    torch::Tensor v_conics = grad_output[3];
    //torch::Tensor v_compensations = grad_output[4];

    std::vector<torch::Tensor> saved_vars = ctx->get_saved_variables();
    torch::Tensor means = saved_vars[0];

    //c10::optional<torch::Tensor> covars_opt, quats_opt, scales_opt;
    //torch::Tensor covars = saved_vars[1];
    //if (covars.defined()) {
    //  covars_opt = covars;
    //}
    //torch::Tensor quats = saved_vars[2];
    //if (quats.defined()) {
    //  quats_opt = quats;
    //}
    //torch::Tensor scales = saved_vars[3];
    //if (scales.defined()) {
    //  scales_opt = scales;
    //}

    //torch::Tensor viewmats = saved_vars[4];
    //torch::Tensor Ks = saved_vars[5];
    //torch::Tensor radii = saved_vars[6];
    //torch::Tensor conics = saved_vars[7];
    //torch::Tensor compensations = saved_vars[8];

    c10::optional<torch::Tensor> covars, quats, scales;
    quats = saved_vars[1];
    scales = saved_vars[2];

    torch::Tensor viewmats = saved_vars[3];
    torch::Tensor Ks = saved_vars[4];
    torch::Tensor radii = saved_vars[5];
    torch::Tensor conics = saved_vars[6];
    //torch::Tensor compensations = saved_vars[7];

    int width = ctx->saved_data["width"].toInt();
    int height = ctx->saved_data["height"].toInt();
    float eps2d = ctx->saved_data["eps2d"].toDouble();
    auto camera_model_type = static_cast<::gsplat::CameraModelType>(ctx->saved_data["camera_model_type"].toInt());

    //if (v_compensations.defined()) {
    //  v_compensations = v_compensations.contiguous();
    //}

    bool viewmats_requires_grad = ctx->needs_input_grad(4);

    c10::optional<torch::Tensor> compensations, v_compensations;

    auto [v_means, v_covars, v_quats, v_scales, v_viewmats] = ::gsplat::fully_fused_projection_bwd_tensor(
      means,
      covars,
      quats,
      scales,
      viewmats,
      Ks,
      width,
      height,
      eps2d,
      camera_model_type,
      radii,
      conics,
      compensations,
      v_means2d.contiguous(),
      v_depths.contiguous(),
      v_conics.contiguous(),
      v_compensations,
      viewmats_requires_grad
    );

    if (!ctx->needs_input_grad(0)) {
      v_means = torch::Tensor();
    }

    //if (!ctx->needs_input_grad(1)) {
    //  v_covars = torch::Tensor();
    //}

    //if (!ctx->needs_input_grad(2)) {
    //  v_quats = torch::Tensor();
    //}

    //if (!ctx->needs_input_grad(3)) {
    //  v_scales = torch::Tensor();
    //}

    //if (!ctx->needs_input_grad(4)) {
    //  v_viewmats = torch::Tensor();
    //}

    if (!ctx->needs_input_grad(1)) {
      v_quats = torch::Tensor();
    }

    if (!ctx->needs_input_grad(2)) {
      v_scales = torch::Tensor();
    }

    if (!ctx->needs_input_grad(3)) {
      v_viewmats = torch::Tensor();
    }

    return {
      v_means,
      //v_covars,
      v_quats,
      v_scales,
      v_viewmats,
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor()
    };
  }
};

class RasterizeToPixels : public torch::autograd::Function<RasterizeToPixels> {
 public:
  static torch::autograd::variable_list forward(
    torch::autograd::AutogradContext* ctx,
    torch::Tensor means2d,
    torch::Tensor conics,
    torch::Tensor colors,
    torch::Tensor opacities,
    c10::optional<torch::Tensor> backgrounds,
    //c10::optional<torch::Tensor> masks,
    const float alpha_threshold,
    const int width,
    const int height,
    const int tile_size,
    torch::Tensor isect_offsets,
    torch::Tensor flatten_ids,
    //bool absgrad
    c10::optional<torch::Tensor> absgrad_out
  ) {
    c10::optional<torch::Tensor> masks{};
    auto [render_colors, render_alphas, last_ids] = ::gsplat::rasterize_to_pixels_fwd_tensor(
      means2d,
      conics,
      colors,
      opacities,
      backgrounds,
      masks,
      alpha_threshold,
      width,
      height,
      tile_size,
      isect_offsets,
      flatten_ids
    );

    ctx->save_for_backward({
      means2d,
      conics,
      colors,
      opacities,
      //backgrounds,
      //masks,
      isect_offsets,
      flatten_ids,
      render_alphas,
      last_ids
    });
    if (absgrad_out.has_value()) {
      absgrad_out->set_requires_grad(false);
    }
    ctx->saved_data["alpha_threshold"] = alpha_threshold;
    ctx->saved_data["width"] = width;
    ctx->saved_data["height"] = height;
    ctx->saved_data["tile_size"] = tile_size;
    ctx->saved_data["absgrad_out"] = absgrad_out;
    ctx->saved_data["backgrounds"] = backgrounds;

    render_alphas = render_alphas.to(torch::kFloat32);

    return {render_colors, render_alphas};
  }

  static torch::autograd::variable_list backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::variable_list grad_output
  ) {
    torch::Tensor v_render_colors = grad_output[0];
    torch::Tensor v_render_alphas = grad_output[1];

    auto saved_vars = ctx->get_saved_variables();
    torch::Tensor means2d = saved_vars[0];
    torch::Tensor conics = saved_vars[1];
    torch::Tensor colors = saved_vars[2];
    torch::Tensor opacities = saved_vars[3];
    //c10::optional<torch::Tensor> masks = saved_vars[5];
    torch::Tensor isect_offsets = saved_vars[4];
    torch::Tensor flatten_ids = saved_vars[5];
    torch::Tensor render_alphas = saved_vars[6];
    torch::Tensor last_ids = saved_vars[7];

    float alpha_threshold = ctx->saved_data["alpha_threshold"].toDouble();
    int width = ctx->saved_data["width"].toInt();
    int height = ctx->saved_data["height"].toInt();
    int tile_size = ctx->saved_data["tile_size"].toInt();
    //bool absgrad = ctx->saved_data["absgrad"].toBool();
    auto absgrad_out = ctx->saved_data["absgrad_out"].toOptional<torch::Tensor>();

    auto backgrounds = ctx->saved_data["backgrounds"].toOptional<torch::Tensor>();

    c10::optional<torch::Tensor> masks{}; // todo
    auto [v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities] = ::gsplat::rasterize_to_pixels_bwd_tensor(
      means2d,
      conics,
      colors,
      opacities,
      backgrounds,
      masks,
      alpha_threshold,
      width,
      height,
      tile_size,
      isect_offsets,
      flatten_ids,
      render_alphas,
      last_ids,
      v_render_colors.contiguous(),
      v_render_alphas.contiguous(),
      absgrad_out.has_value()
    );

    if (absgrad_out.has_value()) {
      absgrad_out->copy_(v_means2d_abs);
    }

    torch::Tensor v_backgrounds;
    if (backgrounds.has_value() && ctx->needs_input_grad(4)) {
      v_backgrounds = (v_render_colors * (1.0 - render_alphas).to(torch::kFloat32)).sum({1, 2});
    }

    return {
      v_means2d,
      v_conics,
      v_colors,
      v_opacities,
      v_backgrounds,
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor(),
      torch::Tensor()
    };
  }
};

util_torch::TensorTuple3 isect_tiles(
  torch::Tensor means2d,
  torch::Tensor radii,
  torch::Tensor depths,
  int tile_size,
  int tile_width,
  int tile_height,
  bool packed,
  int n_cameras
) {
  torch::NoGradGuard no_grad;
  XCHECK(!packed) << "NYI";

  int C = means2d.size(0);
  int N = means2d.size(1);

  XCHECK_EQ(means2d.size(2), 2);
  XCHECK_EQ(radii.sizes(), (at::IntArrayRef{C, N}));
  XCHECK_EQ(depths.sizes(), (at::IntArrayRef{C, N}));

  return ::gsplat::isect_tiles_tensor(
    means2d.contiguous(),
    radii.contiguous(),
    depths.contiguous(),
    at::nullopt, // camera_ids TODO
    at::nullopt, // gaussian_ids
    C,
    tile_size,
    tile_width,
    tile_height,
    true, // sort
    true  // DoubleBuffer - memory efficient radixsort
  );
}

torch::Tensor isect_offset_encode(
  torch::Tensor isect_ids,
  int n_cameras,
  int tile_width,
  int tile_height
) {
  torch::NoGradGuard no_grad;
  return ::gsplat::isect_offset_encode_tensor(isect_ids.contiguous(), n_cameras, tile_width, tile_height);
}

torch::autograd::variable_list fully_fused_projection(
  torch::Tensor means,
  //c10::optional<torch::Tensor> covars, // argument order matches _wrapper.fully_fused_projection
  torch::Tensor quats,
  torch::Tensor scales,
  torch::Tensor viewmats,
  torch::Tensor Ks,
  int width,
  int height,
  float eps2d = 0.3,
  float sigma_level = 3.f,
  float near_plane = 0.01,
  float far_plane = 1e10,
  float radius_clip = 0.0,
  bool packed = false,
  bool sparse_grad = false,
  bool calc_compensations = false,
  ::gsplat::CameraModelType camera_model = ::gsplat::PINHOLE
) {
  int C = viewmats.size(0);
  int N = means.size(0);

  XCHECK_EQ(means.sizes(), (at::IntArrayRef{N, 3}));
  XCHECK_EQ(viewmats.sizes(), (at::IntArrayRef{C, 4, 4}));
  XCHECK_EQ(Ks.sizes(), (at::IntArrayRef{C, 3, 3}));

  means = means.contiguous();

  //if (covars.has_value()) {
  //  XCHECK_EQ(covars->sizes(), (at::IntArrayRef{N, 6}));
  //  covars = covars->contiguous();
  //  XCHECK(!quats.has_value()) << "covars and quats are mutually exclusive";
  //  XCHECK(!scales.has_value()) << "covars and scales are mutually exclusive";
  //} else {
  //  XCHECK(quats.has_value()) << "covars or quats is required";
  //  XCHECK(scales.has_value()) << "covars or scales is required";
  //  XCHECK_EQ(quats->sizes(), (at::IntArrayRef{N, 4}));
  //  XCHECK_EQ(scales->sizes(), (at::IntArrayRef{N, 3}));
  //}
  XCHECK_EQ(quats.sizes(), (at::IntArrayRef{N, 4}));
  XCHECK_EQ(scales.sizes(), (at::IntArrayRef{N, 3}));

  if (sparse_grad) {
    XCHECK(packed) << "sparse_grad is only supported when packed is true";
  }

  viewmats = viewmats.contiguous();
  Ks = Ks.contiguous();

  if (packed) {
    XCHECK(false) << "NYI";
  }

  //torch::Tensor covars_tensor, quats_tensor, scales_tensor;
  //if (covars.has_value()) {
  //  covars_tensor = *covars;
  //}
  //if (quats.has_value()) {
  //  quats_tensor = *quats;
  //}
  //if (scales.has_value()) {
  //  scales_tensor = *scales;
  //}

  return FullyFusedProjection::apply(
    means,
    //covars_tensor,
    quats,
    scales,
    viewmats,
    Ks,
    width,
    height,
    eps2d,
    sigma_level,
    near_plane,
    far_plane,
    radius_clip,
    calc_compensations,
    camera_model
  );
}

// From claude translation of gsplat rendering.py:244
// Only used if packed  (packed is slower but uses less ram)
//torch::Tensor reshape_view(
//  int C,
//  torch::Tensor world_view,
//  const std::vector<int>& N_world
//) {
//  // Calculate split sizes for the first dimension
//  std::vector<int64_t> split_sizes;
//  for (int N_i : N_world) {
//    split_sizes.push_back(C * N_i);
//  }
//
//  // Split world_view along the first dimension
//  auto first_split = torch::split(world_view, split_sizes, 0);
//
//  // Lambda to split each tensor
//  auto split_lambda = [C](const torch::Tensor& x) { return torch::split(x, x.size(0) / C, 0); };
//
//  // Apply split to each tensor in first_split
//  std::vector<std::vector<torch::Tensor>> view_list;
//  for (const auto& x : first_split) {
//    view_list.push_back(split_lambda(x));
//  }
//
//  // Transpose and stack
//  std::vector<torch::Tensor> stacked_tensors;
//  for (size_t i = 0; i < view_list[0].size(); ++i) {
//    std::vector<torch::Tensor> to_cat;
//    for (const auto& inner_list : view_list) {
//      to_cat.push_back(inner_list[i]);
//    }
//    stacked_tensors.push_back(torch::cat(to_cat, 0));
//  }
//
//  return torch::stack(stacked_tensors, 0);
//}

util_torch::TensorTuple2 rasterize_to_pixels(
  torch::Tensor means2d,
  torch::Tensor conics,
  torch::Tensor colors,
  torch::Tensor opacities,
  const float alpha_threshold,
  const int image_width,
  const int image_height,
  const int tile_size,
  torch::Tensor isect_offsets,
  torch::Tensor flatten_ids,
  c10::optional<torch::Tensor> backgrounds,
  const bool packed,
  c10::optional<torch::Tensor> absgrad_out
) {
  XCHECK(!packed) << "NYI";

  int C = isect_offsets.size(0);
  //auto device = means2d.device();

  // if packed {
  //  TODO
  //} else {
  
  int N = means2d.size(1);

  XCHECK_EQ(means2d.sizes(), (at::IntArrayRef{C, N, 2}));
  XCHECK_EQ(conics.sizes(), (at::IntArrayRef{C, N, 3}));

  XCHECK_EQ(colors.size(0), C);
  XCHECK_EQ(colors.size(1), N);

  XCHECK_EQ(opacities.sizes(), (at::IntArrayRef{C, N}));

  //}

  if (backgrounds.has_value()) {
    XCHECK_EQ(backgrounds->sizes(), (at::IntArrayRef{C, colors.size(-1)}));
    backgrounds = backgrounds->contiguous();
  }

  // if (masks.has_value()) NYI


  int channels = colors.size(-1);
  XCHECK(channels == 3 || channels == 4) << "Only RGB & RGBD supported";

  //int padded_channels = 0;
  // TODO: there's a bunch of stuff in gsplat's rasterization/_wrapper about different numbers of channels

  using namespace torch::indexing;
  auto tile_height = isect_offsets.size(1);
  auto tile_width = isect_offsets.size(2);
  XCHECK_GE(tile_height * tile_size, image_height);
  XCHECK_GE(tile_width * tile_size, image_width);

  auto rasterize_result = RasterizeToPixels::apply(
    means2d.contiguous(),
    conics.contiguous(),
    colors.contiguous(),
    opacities.contiguous(),
    backgrounds,
    //torch::Tensor(), // masks - TODO use from config?
    alpha_threshold,
    image_width,
    image_height,
    tile_size,
    isect_offsets.contiguous(),
    flatten_ids.contiguous(),
    absgrad_out
  );

  XCHECK_EQ(rasterize_result.size(), 2);
  torch::Tensor render_colors = rasterize_result[0];
  torch::Tensor render_alphas = rasterize_result[1];

  //if (padded_channels > 0) {
  //  render_colors = render_colors.index({"...", Slice(None, -padded_channels)});
  //}

  return {render_colors, render_alphas};
}

std::tuple<torch::Tensor, torch::Tensor, RasterizationMetas> rasterization(
  torch::Tensor means,
  torch::Tensor quats,
  torch::Tensor scales,
  //c10::optional<torch::Tensor> covars, // covars was an optional arg at the end of the list in python
  torch::Tensor opacities,
  torch::Tensor colors,
  torch::Tensor viewmats,
  torch::Tensor Ks,
  int width,
  int height,
  const RasterizationConfig& cfg
) {
  int N = means.size(0);
  int C = viewmats.size(0);

  XCHECK_EQ(means.sizes(), (at::IntArrayRef{N, 3}));

  XCHECK_EQ(quats.sizes(), (at::IntArrayRef{N, 4}));
  XCHECK_EQ(scales.sizes(), (at::IntArrayRef{N, 3}));

  //if (!covars.has_value()) {
    //XCHECK_EQ(quats->sizes(), (at::IntArrayRef{N, 4}));
    //XCHECK_EQ(scales->sizes(), (at::IntArrayRef{N, 3}));
  //} else {
    //XCHECK_EQ(covars->sizes(), (at::IntArrayRef{N, 3, 3}));
    //XCHECK(!quats.has_value());
    //XCHECK(!scales.has_value());

    //auto tri_indices_0 = torch::tensor({0, 0, 0, 1, 1, 2});
    //auto tri_indices_1 = torch::tensor({0, 1, 2, 1, 2, 2});

    //auto covars_tri = covars->index({"...", tri_indices_0, tri_indices_1});
  //}

  XCHECK_EQ(opacities.sizes(), (at::IntArrayRef{N}));
  XCHECK_EQ(viewmats.sizes(), (at::IntArrayRef{C, 4, 4}));
  XCHECK_EQ(Ks.sizes(), (at::IntArrayRef{C, 3, 3}));

  XCHECK(!cfg.sh_degree) << "NYI until optionals work";

  //if (!cfg.sh_degree) {
    XCHECK((colors.dim() == 2 && colors.size(0) == N)
        || (colors.dim() == 3 && colors.size(0) == C && colors.size(1) == N));
    if (cfg.distributed) {
      XCHECK_EQ(colors.dim(), 2) << "Distributed mode only supports per-Gaussian colors.";
    }
  //} else {
  //  XCHECK(
  //    (colors.dim() == 3 && colors.size(0) == N && colors.size(2) == 3)
  //    || (colors.dim() == 4 && colors.size(0) == C && colors.size(1) == N && colors.size(3) == 3)
  //  );
  //  auto degree_check = *cfg.sh_degree + 1;
  //  degree_check *= degree_check;
  //  XCHECK_LE(degree_check, colors.size(-2));
  //  if (cfg.distributed) {
  //    XCHECK_EQ(colors.dim(), 3) << "Distributed mode only supports per-Gaussian colors.";
  //  }
  //}

  if (cfg.absgrad) {
    XCHECK(!cfg.distributed) << "AbsGrad is not supported in distributed mode.";
  }

  // If in distributed mode, we distribute the projection computation over Gaussians
  // and the rasterize computation over cameras. So first we gather the cameras
  // from all ranks for projection.
  XCHECK(!cfg.distributed) << "NYI";
  XCHECK(!cfg.packed) << "NYI";
  //if (cfg.distributed) {
  //  auto world_rank = torch::distributed::get_rank();
  //  auto world_size = torch::distributed::get_world_size();

  //  // Gather the number of Gaussians in each rank
  //  std::vector<int32_t> N_world = c10::all_gather_int32(world_size, N, device);

  //  // Enforce that the number of cameras is the same across all ranks
  //  std::vector<int32_t> C_world(world_size, C);

  //  // Gather view matrices and intrinsics
  //  std::vector<torch::Tensor> gathered_viewmats, gathered_Ks;
  //  auto [gathered_viewmats, gathered_Ks] = all_gather_tensor_list(world_size, {viewmats, Ks});

  //  // Silently change C from local #Cameras to global #Cameras
  //  C = gathered_viewmats.size();
  //}

  bool calc_compensations = (cfg.rasterize_mode == ANTIALIASED);
  XCHECK(!calc_compensations) << "NYI torch optional bug";

  auto proj_results = fully_fused_projection(
    means,
    //covars,
    quats,
    scales,
    viewmats,
    Ks,
    width,
    height,
    cfg.eps2d,
    cfg.sigma_level,
    cfg.near_plane,
    cfg.far_plane,
    cfg.radius_clip,
    cfg.packed,
    cfg.sparse_grad,
    //calc_compensations,
    cfg.camera_model
  );


  //torch::Tensor camera_ids, gaussian_ids; // only used if packed

  // if packed {
  //   // TODO
  //} else {

  // TODO: maybe both should return tuple7 then this could be simplified?
  XCHECK_EQ(proj_results.size(), 4);
  torch::Tensor radii = proj_results[0];
  torch::Tensor means2d = proj_results[1];
  torch::Tensor depths = proj_results[2];
  torch::Tensor conics = proj_results[3];
  //torch::Tensor compensations = proj_results[0];
  opacities = opacities.repeat({C, 1});
  //}

  //if (compensations.defined()) {
  //  opacities *= compensations;
  //}

  RasterizationMetas metas;
  metas.radii = radii;
  //metas.camera_ids = camera_ids;
  //metas.gaussian_ids = gaussian_ids;
  metas.radii = radii;
  metas.means2d = means2d;
  metas.depths = depths;
  metas.conics = conics;
  metas.opacities = opacities;

  XCHECK(!cfg.sh_degree.has_value()) << "NYI";
  //if (!cfg.sh_degree.has_value()) {
    // if packed() {
    //   // TODO
    // } else {
  if (colors.dim() == 2) {
    colors = colors.expand({C, -1, -1});
  }
    // }
  //}


  // TODO if (distributed)... add checks for distributed stuff if implemented

  // TODO: maybe don't bother cat'ing ... we're just going to slice them later
  // this is keeping to the implementation in rendering.py
  if (cfg.render_mode == RGBD) { // || cfg.render_mode == RGBED
    colors = torch::cat({colors, depths.unsqueeze(-1)}, -1);
  } // else if depth only

  int tile_width = (width + cfg.tile_size - 1) / cfg.tile_size;
  int tile_height = (height + cfg.tile_size - 1) / cfg.tile_size;

  auto [tiles_per_gauss, isect_ids, flatten_ids] = isect_tiles(
    means2d,
    radii,
    depths,
    cfg.tile_size,
    tile_width,
    tile_height,
    cfg.packed,
    C//,
    //camera_ids,
    //gaussian_ids,
  );
  auto isect_offsets = isect_offset_encode(isect_ids, C, tile_width, tile_height);

  c10::optional<torch::Tensor> absgrad_out;
  if (cfg.absgrad) {
    absgrad_out = torch::zeros_like(means2d);
  }

  metas.tile_width = tile_width;
  metas.tile_height = tile_height;
  metas.tiles_per_gauss = tiles_per_gauss;
  metas.isect_ids = isect_ids;
  metas.flatten_ids = flatten_ids;
  metas.isect_offsets = isect_offsets;
  metas.width = width;
  metas.height = height;
  metas.tile_size = cfg.tile_size;
  metas.n_cameras = C;
  if (cfg.absgrad) {
    metas.means2d_absgrad = *absgrad_out;
  }

  XCHECK_LE(colors.size(-1), cfg.channel_chunk) << "NYI rasterizing in chunks";
  auto [render_colors, render_alphas] = rasterize_to_pixels(
    means2d,
    conics,
    colors,
    opacities,
    cfg.alpha_threshold,
    width,
    height,
    cfg.tile_size,
    isect_offsets,
    flatten_ids,
    cfg.backgrounds,
    cfg.packed,
    absgrad_out
  );

  return {render_colors, render_alphas, metas};
}

}}  // namespace p11::gsplat
