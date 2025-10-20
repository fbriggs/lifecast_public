// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ldi_segmentation.h"

#include <algorithm>
#include <random>

#include "logger.h"
#include "check.h"
#include "util_math.h"
#include "util_time.h"
#include "util_opencv.h"

namespace p11 { namespace ldi {

namespace {
static constexpr float kFthetaMargin = 0.9;
}

void NeuralHashmapsForegroundSegmentationModel::initHashmap()
{
  XPLINFO << "kNumLevels=" << kNumLevels;
  float res = kCourseResolution;
  int curr_offset = 0;
  for (int level = 0; level < kNumLevels; ++level) {
    level_to_resolution.push_back(res);
    res *= kLevelScale;

    level_to_hash_offset.push_back(curr_offset);
    curr_offset += level_to_resolution[level] * level_to_resolution[level];

    XPLINFO << level << "\t" << level_to_resolution[level] << "\t" << level_to_hash_offset[level];
  }
  const int hashmap_size = curr_offset;
  XPLINFO << "hashmap_size=" << hashmap_size;

  // NOTE: we add some extra singleton dimensions for convenience later
  level_resolution_tensor =
      torch::from_blob(level_to_resolution.data(), {1, 1, kNumLevels}, {torch::kInt32}).to(device);
  level_hash_offset_tensor =
      torch::from_blob(level_to_hash_offset.data(), {1, 1, kNumLevels}, {torch::kInt32}).to(device);

  // hashmap = (torch::rand({hashmap_size, kNumFeaturesPerLevel}, torch::kFloat16) - 0.5) * 2e-4;
  hashmap = torch::zeros({hashmap_size, kNumFeaturesPerLevel}, torch::kFloat16);

  register_parameter("hashmap", hashmap);
}

torch::Tensor NeuralHashmapsForegroundSegmentationModel::hashTensor(
    torch::Tensor xi, int dx, int dy)
{
  torch::Tensor i = torch::select(xi, 1, 0) + dx;
  torch::Tensor j = torch::select(xi, 1, 1) + dy;
  torch::Tensor idx = i * level_resolution_tensor + j + level_hash_offset_tensor;

  return torch::where(
      (i >= 0) & (j >= 0) & (i < level_resolution_tensor) & (j < level_resolution_tensor), idx, 0);
}

torch::Tensor NeuralHashmapsForegroundSegmentationModel::batchPointsToHashCodes(
    const torch::DeviceType device, torch::Tensor batch_points)
{
  const int num_points = batch_points.sizes()[0];

  torch::Tensor normalized_points = (batch_points + 1.0) * 0.5;  // [-1, 1] -> [0, 1]

  torch::Tensor xl = torch::unsqueeze(normalized_points, 2) * level_resolution_tensor;

  torch::Tensor xi = xl.toType(torch::kInt32);
  torch::Tensor r = xl - xi;

  torch::Tensor rx = torch::unsqueeze(torch::select(r, 1, 0), 2);
  torch::Tensor ry = torch::unsqueeze(torch::select(r, 1, 1), 2);

  // TODO: can we do this without a stack?
  constexpr int kNumCorners = 4;
  std::vector<torch::Tensor> hash_index_corner(kNumCorners);
  hash_index_corner[0] = hashTensor(xi, 0, 0);
  hash_index_corner[1] = hashTensor(xi, 1, 0);
  hash_index_corner[2] = hashTensor(xi, 0, 1);
  hash_index_corner[3] = hashTensor(xi, 1, 1);
  torch::Tensor hash_index = torch::stack(hash_index_corner, 3);

  torch::Tensor flat_hash_index = hash_index.reshape({num_points * kNumLevels * kNumCorners});
  torch::Tensor hashed_vals = torch::index_select(hashmap, 0, flat_hash_index);
  torch::Tensor codes =
      hashed_vals.reshape({num_points, kNumLevels, kNumCorners, kNumFeaturesPerLevel});

  torch::Tensor c00 = torch::select(codes, 2, 0);
  torch::Tensor c10 = torch::select(codes, 2, 1);
  torch::Tensor c01 = torch::select(codes, 2, 2);
  torch::Tensor c11 = torch::select(codes, 2, 3);

  torch::Tensor c0 = c00 * (1.0 - rx) + c10 * rx;
  torch::Tensor c1 = c01 * (1.0 - rx) + c11 * rx;
  torch::Tensor c = c0 * (1.0 - ry) + c1 * ry;

  torch::Tensor batch_features = c.reshape({num_points, kNumFeaturesPerLevel * kNumLevels});

  return batch_features;
}

torch::Tensor NeuralHashmapsForegroundSegmentationModel::pointToSeg(torch::Tensor xy)
{
  torch::Tensor h = batchPointsToHashCodes(device, xy);
  h = torch::relu(fc1->forward(h));
  h = torch::relu(fc2->forward(h));
  return torch::softmax(fc_final->forward(h), 1);
}

std::pair<cv::Mat, cv::Mat> getEdgesHiAndLow(
    const calibration::FisheyeCamerad cam_R, const cv::Mat& R_inv_depth_ftheta)
{
  cv::Mat min_depth, max_depth;
  cv::erode(R_inv_depth_ftheta, min_depth, cv::Mat(), cv::Point(-1, -1), 13);
  cv::dilate(R_inv_depth_ftheta, max_depth, cv::Mat(), cv::Point(-1, -1), 13);
  cv::Mat edge_hi(R_inv_depth_ftheta.size(), CV_32F, cv::Scalar(0.0));
  cv::Mat edge_lo(R_inv_depth_ftheta.size(), CV_32F, cv::Scalar(0.0));
  constexpr float kSteepness = 10.0;
  constexpr float kBias = 0.1;
  constexpr float kRampStart = 0.8;
  constexpr float kRampEnd = 0.9;
  for (int y = 0; y < R_inv_depth_ftheta.cols; ++y) {
    for (int x = 0; x < R_inv_depth_ftheta.rows; ++x) {
      const float dx = x - cam_R.optical_center.x();
      const float dy = y - cam_R.optical_center.y();
      if (dx * dx + dy * dy > cam_R.radius_at_90 * cam_R.radius_at_90) continue;
      const float r = std::sqrt(dx * dx + dy * dy) / cam_R.radius_at_90;
      const float ramp = 1.0 - math::clamp((r - kRampStart) / (kRampEnd - kRampStart), 0.0f, 1.0f);

      edge_hi.at<float>(y, x) =
          ramp *
          math::clamp(
              std::tanh(
                  kSteepness * (R_inv_depth_ftheta.at<float>(y, x) - min_depth.at<float>(y, x)) -
                  kBias),
              0.0f,
              1.0f);
      edge_lo.at<float>(y, x) =
          ramp *
          math::clamp(
              std::tanh(
                  kSteepness * (max_depth.at<float>(y, x) - R_inv_depth_ftheta.at<float>(y, x)) -
                  kBias),
              0.0f,
              1.0f);
    }
  }

  // HACK: erode a little bit. This actually seems to make a huge difference.
  cv::erode(edge_hi, edge_hi, cv::Mat(), cv::Point(-1, -1), 1);
  cv::erode(edge_lo, edge_lo, cv::Mat(), cv::Point(-1, -1), 1);
  cv::GaussianBlur(edge_hi, edge_hi, cv::Size(3, 3), 1.0, 1.0);
  cv::GaussianBlur(edge_lo, edge_lo, cv::Size(3, 3), 1.0, 1.0);

  // In places where there is both a high and low edge, it doesn't really make sense.
  // Zero both edges out and let other loss terms take over (like smoothness ideally).
  cv::Mat edge_hi2(edge_hi.size(), CV_32F);
  cv::Mat edge_lo2(edge_hi.size(), CV_32F);
  for (int y = 0; y < R_inv_depth_ftheta.cols; ++y) {
    for (int x = 0; x < R_inv_depth_ftheta.rows; ++x) {
      edge_hi2.at<float>(y, x) =
          math::clamp(edge_hi.at<float>(y, x) - edge_lo.at<float>(y, x), 0.0f, 1.0f);
      edge_lo2.at<float>(y, x) =
          math::clamp(edge_lo.at<float>(y, x) - edge_hi.at<float>(y, x), 0.0f, 1.0f);
    }
  }

  return {edge_hi2, edge_lo2};
}

// Returns percentiles of depth values within the valid pixels of the f-theta image
std::tuple<float, float, float, float> getInvDepthPercentiles(
    const calibration::FisheyeCamerad cam_R, const cv::Mat& R_inv_depth_ftheta)
{
  std::vector<float> invd_vals;
  for (int y = 0; y < R_inv_depth_ftheta.rows; ++y) {
    for (int x = 0; x < R_inv_depth_ftheta.cols; ++x) {
      const float dx = x - cam_R.optical_center.x();
      const float dy = y - cam_R.optical_center.y();
      if (dx * dx + dy * dy > cam_R.radius_at_90 * cam_R.radius_at_90 * kFthetaMargin) continue;
      invd_vals.push_back(R_inv_depth_ftheta.at<float>(y, x));
    }
  }
  XCHECK(!invd_vals.empty());
  std::sort(invd_vals.begin(), invd_vals.end());
  const float p0 = invd_vals[0];
  const float p5 = invd_vals[invd_vals.size() * 0.1];
  const float p95 = invd_vals[invd_vals.size() * 0.9];
  const float p100 = invd_vals[invd_vals.size() - 1];
  return {p0, p5, p95, p100};
}

// Compute depth percentiles. We'll use these to compute a map of weight for pushing some pixels
// into the top or bottom layer.
std::pair<cv::Mat, cv::Mat> computeTopBottomBiasWeightMaps(
    const calibration::FisheyeCamerad cam_R, const cv::Mat& R_inv_depth_ftheta)
{
  const auto& [invd_p0, invd_p10, invd_p90, invd_p100] =
      getInvDepthPercentiles(cam_R, R_inv_depth_ftheta);
  XPLINFO << "invd_p0: " << invd_p0;
  XPLINFO << "invd_p10: " << invd_p10;
  XPLINFO << "invd_p90: " << invd_p90;
  XPLINFO << "invd_p100: " << invd_p100;
  cv::Mat bottom_bias_weight(R_inv_depth_ftheta.size(), CV_32F, cv::Scalar(0.0));
  cv::Mat top_bias_weight(R_inv_depth_ftheta.size(), CV_32F, cv::Scalar(0.0));
  for (int y = 0; y < R_inv_depth_ftheta.rows; ++y) {
    for (int x = 0; x < R_inv_depth_ftheta.cols; ++x) {
      const float dx = x - cam_R.optical_center.x();
      const float dy = y - cam_R.optical_center.y();
      if (dx * dx + dy * dy > cam_R.radius_at_90 * cam_R.radius_at_90 * kFthetaMargin) continue;

      const float invd = R_inv_depth_ftheta.at<float>(y, x);

      // Try to make a ramp in the weight between the 0th and 5th percentile if possible, otherwise
      // step function
      if (std::abs(invd_p10 - invd_p0) < 1e-3) {
        bottom_bias_weight.at<float>(y, x) = invd <= invd_p10 ? 1.0 : 0.0;
      } else {
        bottom_bias_weight.at<float>(y, x) =
            math::clamp((invd_p10 - invd) / (invd_p10 - invd_p0), 0.0f, 1.0f);
      }

      // Same for top bias weight...
      if (std::abs(invd_p90 - invd_p100) < 1e-3) {
        top_bias_weight.at<float>(y, x) = invd >= invd_p90 ? 1.0 : 0.0;
      } else {
        top_bias_weight.at<float>(y, x) =
            math::clamp((invd - invd_p90) / (invd_p100 - invd_p90), 0.0f, 1.0f);
      }
    }
  }
  // cv::imwrite("/tmp/bottom_weight.png", bottom_bias_weight * 255.0);
  // cv::imwrite("/tmp/top_weight.png", top_bias_weight * 255.0);
  return {bottom_bias_weight, top_bias_weight};
}

cv::Mat segmentFgBgWithMultiresolutionHashmap(
    const cv::Mat& edge_hi,
    const cv::Mat& edge_lo,
    const calibration::FisheyeCamerad cam_R,
    const calibration::FisheyeCamerad cam_R_half,
    const cv::Mat& R_inv_depth_ftheta,
    const cv::Mat prior_segmentation,
    const cv::Mat prior_weight)
{
  if (prior_segmentation.empty()) XPLINFO << "prior_segmentation is empty.";

  const auto& [bottom_bias_weight, top_bias_weight] =
      computeTopBottomBiasWeightMaps(cam_R_half, R_inv_depth_ftheta);

  // Check if we can use CUDA
  torch::DeviceType device = torch::kCPU;
  if (torch::cuda::is_available()) {
    device = torch::kCUDA;
    XPLINFO << "CUDA is available";
  }

  // Create the model
  std::shared_ptr<NeuralHashmapsForegroundSegmentationModel> model =
      std::make_shared<NeuralHashmapsForegroundSegmentationModel>(device);
  model->to(device);

  // Prepare pixels for shuffling. Many pixels have no interesting loss other than smoothness,
  // if they are not an edge. So we must avoid sampling these pixels to not waste 99% of our compute
  // doing nothing.
  std::vector<std::tuple<int, int>> edge_pixels;  // {x, y}
  for (int y = 0; y < R_inv_depth_ftheta.rows; ++y) {
    for (int x = 0; x < R_inv_depth_ftheta.cols; ++x) {
      if (edge_hi.at<float>(y, x) > 0 || edge_lo.at<float>(y, x) > 0) {
        edge_pixels.push_back({x, y});
      }
    }
  }
  XPLINFO << "# edge pixels=" << edge_pixels.size();
  std::shuffle(edge_pixels.begin(), edge_pixels.end(), std::random_device());
  int curr_edge_pixel = 0;
  int num_epochs = 0;

  // Run the optimizer
  static constexpr int kPixelsPerBatch = 8192 * 16;
  static constexpr int kNumIterations = 300;

  constexpr double kStartLearningRate = 1e-2;
  auto opt = torch::optim::Adam(
      model->parameters(),
      torch::optim::AdamOptions(kStartLearningRate)
          .betas({0.9, 0.99})
          .eps(1e-5)
          .weight_decay(0.01));

  auto start_time = time::now();
  for (int itr = 0; itr < kNumIterations; ++itr) {
    opt.zero_grad();

    std::vector<float> batch_xy_data, batch_edge_hi_data, batch_edge_lo_data;
    std::vector<float> neighbor_xy_data, neighbor_hi_data, neighbor_lo_data;

    for (int i = 0; i < kPixelsPerBatch; ++i) {
      const auto& [x, y] = edge_pixels[curr_edge_pixel];
      ++curr_edge_pixel;
      if (curr_edge_pixel >= edge_pixels.size()) {
        curr_edge_pixel = 0;
        std::shuffle(edge_pixels.begin(), edge_pixels.end(), std::random_device());
        ++num_epochs;
      }
      const float u = 2.0 * (float(x) / (cam_R_half.width - 1) - 0.5);
      const float v = 2.0 * (float(y) / (cam_R_half.height - 1) - 0.5);
      batch_xy_data.push_back(u);
      batch_xy_data.push_back(v);

      batch_edge_hi_data.push_back(edge_hi.at<float>(y, x));
      batch_edge_lo_data.push_back(edge_lo.at<float>(y, x));

      constexpr float kNeighborRadius = 0.01;
      const float theta = math::randUnif() * 2.0 * M_PI;
      const float rand_r = math::randUnif() * kNeighborRadius;
      const float u1 = u + cos(theta) * rand_r;
      const float v1 = v + sin(theta) * rand_r;
      neighbor_xy_data.push_back(u1);
      neighbor_xy_data.push_back(v1);

      // Convert back to pixel coordinates
      const float nx = (cam_R_half.width - 1) * (u1 * 0.5 + 0.5);
      const float ny = (cam_R_half.height - 1) * (v1 * 0.5 + 0.5);

      neighbor_hi_data.push_back(opencv::getPixelBilinear<float>(edge_hi, nx, ny));
      neighbor_lo_data.push_back(opencv::getPixelBilinear<float>(edge_lo, nx, ny));
    }
    torch::Tensor batch_xy =
        torch::from_blob(batch_xy_data.data(), {kPixelsPerBatch, 2}, {torch::kFloat32}).to(device);
    torch::Tensor neighbor_xy =
        torch::from_blob(neighbor_xy_data.data(), {kPixelsPerBatch, 2}, {torch::kFloat32})
            .to(device);

    torch::Tensor batch_edge_hi =
        torch::from_blob(batch_edge_hi_data.data(), {kPixelsPerBatch, 1}, {torch::kFloat32})
            .to(device);
    torch::Tensor batch_edge_lo =
        torch::from_blob(batch_edge_lo_data.data(), {kPixelsPerBatch, 1}, {torch::kFloat32})
            .to(device);

    torch::Tensor neighbor_hi =
        torch::from_blob(neighbor_hi_data.data(), {kPixelsPerBatch, 1}, {torch::kFloat32})
            .to(device);
    torch::Tensor neighbor_lo =
        torch::from_blob(neighbor_lo_data.data(), {kPixelsPerBatch, 1}, {torch::kFloat32})
            .to(device);

    torch::Tensor seg = model->pointToSeg(batch_xy);
    torch::Tensor neighbor_seg = model->pointToSeg(neighbor_xy);

    torch::Tensor seg_l0 = seg.index({"...", 0}).unsqueeze(1);
    torch::Tensor seg_l1 = seg.index({"...", 1}).unsqueeze(1);
    torch::Tensor seg_l2 = seg.index({"...", 2}).unsqueeze(1);

    torch::Tensor nei_l0 = neighbor_seg.index({"...", 0}).unsqueeze(1);
    torch::Tensor nei_l1 = neighbor_seg.index({"...", 1}).unsqueeze(1);
    torch::Tensor nei_l2 = neighbor_seg.index({"...", 2}).unsqueeze(1);

    torch::Tensor loss_no_hi_in_l0 = torch::sum(seg_l0 * batch_edge_hi);
    torch::Tensor loss_no_lo_in_l2 = torch::sum(seg_l2 * batch_edge_lo);

    torch::Tensor loss_ord = torch::sum(batch_edge_lo * neighbor_hi * nei_l1 * (1.0 - seg_l0)) +
                             torch::sum(batch_edge_hi * neighbor_lo * nei_l1 * (1.0 - seg_l2)) +
                             torch::sum(batch_edge_lo * neighbor_hi * seg_l1 * (1.0 - nei_l2)) +
                             torch::sum(batch_edge_hi * neighbor_lo * seg_l1 * (1.0 - nei_l0));

    /// Do a different batch sampling for smoothing, since it applies to all pixels (not just edges)
    /// ///

    std::vector<float> smooth1_xy_data, smooth2_xy_data;
    std::vector<float> smooth1_invd_data, smooth2_invd_data;
    std::vector<float> smooth1_top_bias_weight_data, smooth1_bottom_bias_weight_data;
    std::vector<float> smooth1_prior_seg_data, smooth1_prior_weight_data;

    for (int i = 0; i < kPixelsPerBatch; ++i) {
      const float u1 = (math::randUnif() - 0.5) * 2.0;
      const float v1 = (math::randUnif() - 0.5) * 2.0;
      smooth1_xy_data.push_back(u1);
      smooth1_xy_data.push_back(v1);

      constexpr float kSmoothingRadius = 0.025;
      const float theta = math::randUnif() * 2.0 * M_PI;
      const float rand_r = math::randUnif() * kSmoothingRadius;
      const float u2 = u1 + cos(theta) * rand_r;
      const float v2 = v1 + sin(theta) * rand_r;
      smooth2_xy_data.push_back(u2);
      smooth2_xy_data.push_back(v2);

      // Convert back to pixel coordinates
      const float sx1 = (cam_R_half.width - 1) * (u1 * 0.5 + 0.5);
      const float sy1 = (cam_R_half.height - 1) * (v1 * 0.5 + 0.5);

      const float sx2 = (cam_R_half.width - 1) * (u2 * 0.5 + 0.5);
      const float sy2 = (cam_R_half.height - 1) * (v2 * 0.5 + 0.5);

      smooth1_invd_data.push_back(opencv::getPixelBilinear<float>(R_inv_depth_ftheta, sx1, sy1));
      smooth2_invd_data.push_back(opencv::getPixelBilinear<float>(R_inv_depth_ftheta, sx2, sy2));

      smooth1_top_bias_weight_data.push_back(
          opencv::getPixelBilinear<float>(top_bias_weight, sx1, sy1));
      smooth1_bottom_bias_weight_data.push_back(
          opencv::getPixelBilinear<float>(bottom_bias_weight, sx1, sy1));

      if (!prior_segmentation.empty()) {
        const cv::Vec3f ps = opencv::getPixelBilinear<cv::Vec3f>(prior_segmentation, sx1, sy1);
        const float pw = opencv::getPixelBilinear<float>(prior_weight, sx1, sy1);
        smooth1_prior_seg_data.push_back(ps[2]);  // Note the swizzle of the channel order, related
                                                  // to opencv using BGR instead of RGB
        smooth1_prior_seg_data.push_back(ps[1]);
        smooth1_prior_seg_data.push_back(ps[0]);
        smooth1_prior_weight_data.push_back(pw);
      }
    }
    torch::Tensor smooth1_xy =
        torch::from_blob(smooth1_xy_data.data(), {kPixelsPerBatch, 2}, {torch::kFloat32})
            .to(device);
    torch::Tensor smooth2_xy =
        torch::from_blob(smooth2_xy_data.data(), {kPixelsPerBatch, 2}, {torch::kFloat32})
            .to(device);

    torch::Tensor smooth1_invd =
        torch::from_blob(smooth1_invd_data.data(), {kPixelsPerBatch, 1}, {torch::kFloat32})
            .to(device);
    torch::Tensor smooth2_invd =
        torch::from_blob(smooth2_invd_data.data(), {kPixelsPerBatch, 1}, {torch::kFloat32})
            .to(device);

    torch::Tensor smooth1_top_bias_weight =
        torch::from_blob(
            smooth1_top_bias_weight_data.data(), {kPixelsPerBatch, 1}, {torch::kFloat32})
            .to(device);
    torch::Tensor smooth1_bottom_bias_weight =
        torch::from_blob(
            smooth1_bottom_bias_weight_data.data(), {kPixelsPerBatch, 1}, {torch::kFloat32})
            .to(device);

    torch::Tensor smooth1_prior_seg, smooth1_prior_weight;
    if (!prior_segmentation.empty()) {
      smooth1_prior_seg =
          torch::from_blob(
              smooth1_prior_seg_data.data(),
              {kPixelsPerBatch, NeuralHashmapsForegroundSegmentationModel::kNumLayers},
              {torch::kFloat32})
              .to(device);
      smooth1_prior_weight =
          torch::from_blob(
              smooth1_prior_weight_data.data(), {kPixelsPerBatch, 1}, {torch::kFloat32})
              .to(device);
    }

    torch::Tensor smooth1_seg = model->pointToSeg(smooth1_xy);
    torch::Tensor smooth2_seg = model->pointToSeg(smooth2_xy);

    static constexpr float kSigma = 50.0;
    torch::Tensor smoothness = torch::sum(
        torch::exp(-kSigma * torch::abs(smooth1_invd - smooth2_invd)) *
        torch::square(smooth1_seg - smooth2_seg));

    // End smoothness loss

    // For scenes where 2 layers suffice, the solution is ambiguous. This leads to flickering back
    // and forth between solutions in video. To reduce this, introduce some biases: the highest and
    // lowest percentile depth pixels are biased toward the top and and bottom layers, and
    // everything else is slightly biased toward the middle.
    torch::Tensor bottom_bias = torch::sum((1.0 - seg_l0) * smooth1_bottom_bias_weight);
    torch::Tensor top_bias = torch::sum((1.0 - seg_l2) * smooth1_top_bias_weight);
    torch::Tensor middle_bias = torch::sum(1.0 - seg_l1);

    // Prior segmentation loss (for temporal stability with previous frame, or maybe some other use)
    torch::Tensor prior_seg_loss = torch::zeros_like(smoothness);
    if (!prior_segmentation.empty()) {
      prior_seg_loss =
          torch::sum(torch::abs(smooth1_seg - smooth1_prior_seg) * smooth1_prior_weight);
    }

    torch::Tensor loss = prior_seg_loss * 0.005 + bottom_bias * 0.01 + top_bias * 0.0001 +
                         middle_bias * 0.0001 + loss_ord * 1.0 + loss_no_hi_in_l0 * 0.2 +
                         loss_no_lo_in_l2 * 0.2 + smoothness * 1.0;

    loss.backward();
    opt.step();
    if (itr % 100 == 0) {
      std::cout << itr << "\t" << loss.item<float>() / kPixelsPerBatch << "\t"
                << prior_seg_loss.item<float>() / kPixelsPerBatch << "\t"
                << loss_ord.item<float>() / kPixelsPerBatch << "\t"
                << loss_no_hi_in_l0.item<float>() / kPixelsPerBatch << "\t"
                << loss_no_lo_in_l2.item<float>() / kPixelsPerBatch << "\t"
                << bottom_bias.item<float>() / kPixelsPerBatch << "\t"
                << top_bias.item<float>() / kPixelsPerBatch << "\t"
                << middle_bias.item<float>() / kPixelsPerBatch << "\t"
                << smoothness.item<float>() / kPixelsPerBatch << std::endl;
    }
  }
  XPLINFO << "segmention optimization time (sec): " << time::timeSinceSec(start_time);

  // Render the output segmentation (use the full resolution camera model here instead of
  // half-resolution)
  torch::NoGradGuard no_grad;
  cv::Mat segmentation(cv::Size(cam_R.width, cam_R.height), CV_32FC3, cv::Scalar(0.0));
  for (int y = 0; y < cam_R.height; ++y) {
    // Do 1 line of the image as a batch
    std::vector<float> batch_xy_data;
    std::vector<int> valid_xs;
    for (int x = 0; x < cam_R.width; ++x) {
      //const float dx = x - cam_R.width / 2.0;
      //const float dy = y - cam_R.height / 2.0;
      valid_xs.push_back(x);
      batch_xy_data.push_back(2.0 * (float(x) / (cam_R.width - 1) - 0.5));
      batch_xy_data.push_back(2.0 * (float(y) / (cam_R.height - 1) - 0.5));
    }
    if (batch_xy_data.empty()) continue;

    torch::Tensor batch_xy =
        torch::from_blob(
            batch_xy_data.data(), {(int64_t)batch_xy_data.size() / 2, 2}, {torch::kFloat32})
            .to(device);
    torch::Tensor seg = model->pointToSeg(batch_xy);

    torch::Tensor seg_cpu = seg.to(torch::kCPU);
    auto seg_accessor = seg_cpu.accessor<float, 2>();
    int i = 0;
    for (const int x : valid_xs) {
      segmentation.at<cv::Vec3f>(y, x) =
          cv::Vec3f(seg_accessor[i][2], seg_accessor[i][1], seg_accessor[i][0]);
      ++i;
    }
  }

  return segmentation;
}

}}  // namespace p11::ldi
