// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ldi_pipeline_lib.h"
#include "turbojpeg_wrapper.h"
#include "projection.h"
#include "vignette.h"

namespace p11 { namespace ldi {

namespace {

static constexpr int kNumLayers = 3;

struct ImageCache {
  int capacity;
  ImageCache(int capacity) : capacity(capacity) {}
  std::map<std::string, cv::Mat> filename_to_image;
  std::list<std::string> lru_list;

  cv::Mat imread(const std::string filename, int flags)
  {
    if (filename_to_image.count(filename) > 0) {
      lru_list.remove(filename);
      lru_list.push_front(filename);
      return filename_to_image.at(filename);
    } else {
      filename_to_image[filename] = cv::imread(filename, flags);
      if (filename_to_image[filename].empty()) {
        throw std::runtime_error("Empty image after loading: " + filename);
      }

      lru_list.push_front(filename);

      // Evict the least recently used item to maintain capacity
      if (filename_to_image.size() > capacity) {
        std::string lru_filename = lru_list.back();
        lru_list.pop_back();
        filename_to_image.erase(lru_filename);
      }

      return filename_to_image[filename];
    }
  }
};

struct VR180DepthProcessor {
  // Lookup tables for warping from VR180 projection to ftheta projection
  calibration::FisheyeCamerad cam_perfect_ftheta;
  std::vector<cv::Mat> warp_vr180_to_ftheta_unscaled, warp_vr180_to_ftheta_scaled,
      warp_ftheta_to_rectified, warp_rectified_to_ftheta;

  // Model optical flow/disparity
  torch::jit::script::Module raft_module;

  LdiPipelineConfig cfg;

  VR180DepthProcessor(const LdiPipelineConfig& cfg) : cfg(cfg) {}

  // Before we can initialize (precompute warps), we need 1 frame of video decoded to know its size.
  void init(const cv::Mat& frame)
  {
    cv::Mat L_image = frame(cv::Rect(0, 0, frame.cols / 2, frame.rows));
    cv::Mat R_image = frame(cv::Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));

    cam_perfect_ftheta = projection::makePerfectFthetaCamera(cfg.ftheta_size);
    calibration::FisheyeCamerad cam_perfect_ftheta_half =
        projection::makePerfectFthetaCamera(cfg.ftheta_size / 2);
    projection::precomputeVR180toFthetaWarp(
        cam_perfect_ftheta, cfg.ftheta_size, L_image.cols, warp_vr180_to_ftheta_unscaled, 1.0);
    projection::precomputeVR180toFthetaWarp(
        cam_perfect_ftheta,
        cfg.ftheta_size,
        L_image.cols,
        warp_vr180_to_ftheta_scaled,
        cfg.ftheta_scale);
    projection::precomputeFisheyeToRectifyWarp(
        cfg.rectified_size_for_depth,
        cfg.rectified_size_for_depth,
        cam_perfect_ftheta,
        warp_ftheta_to_rectified);
    projection::precomputeRectifiedToFisheyeWarp(
        cfg.rectified_size_for_depth,
        cfg.rectified_size_for_depth,
        cv::Size(cfg.ftheta_size / 2, cfg.ftheta_size / 2),
        cam_perfect_ftheta_half,
        warp_rectified_to_ftheta,
        cfg.ftheta_scale);

    // Prevent torch from trying to optimize the disparity model (this actually wastes more time
    // than it saves here).
    torch::jit::getProfilingMode() = false;
    optical_flow::getTorchModelRAFT(raft_module, /*model_path=*/"");
  }

  void estimateDepthForFrame(const cv::Mat& frame, const int frame_index)
  {
    cv::Mat L_image = frame(cv::Rect(0, 0, frame.cols / 2, frame.rows));
    cv::Mat R_image = frame(cv::Rect(frame.cols / 2, 0, frame.cols / 2, frame.rows));

    // Do the warping
    cv::Mat L_ftheta_unscaled =
        projection::warp(L_image, warp_vr180_to_ftheta_unscaled, cv::INTER_AREA);
    cv::Mat R_ftheta_unscaled =
        projection::warp(R_image, warp_vr180_to_ftheta_unscaled, cv::INTER_AREA);
    cv::Mat L_ftheta = projection::warp(L_image, warp_vr180_to_ftheta_scaled, cv::INTER_CUBIC);
    cv::Mat R_ftheta = projection::warp(R_image, warp_vr180_to_ftheta_scaled, cv::INTER_CUBIC);
    cv::Mat L_rectified =
        projection::warp(L_ftheta_unscaled, warp_ftheta_to_rectified, cv::INTER_CUBIC);
    cv::Mat R_rectified =
        projection::warp(R_ftheta_unscaled, warp_ftheta_to_rectified, cv::INTER_CUBIC);
    cv::Mat R_ftheta_half = opencv::halfSize(R_ftheta);

    // Compute disparity in rectified projection
    auto disparity_start_timer = time::now();
    cv::Mat R_disparity = optical_flow::computeDisparityRAFT(raft_module, R_rectified, L_rectified);
    XPLINFO << "depth time(sec):\t" << time::timeSinceSec(disparity_start_timer);

    cv::Mat R_inv_depth_rectified = projection::disparityToInvDepth(R_disparity, cfg.baseline_m);

    // Warp the rectified depth to f-theta projection
    cv::Mat R_inv_depth_ftheta =
        projection::warp(R_inv_depth_rectified, warp_rectified_to_ftheta, cv::INTER_LINEAR);

    R_inv_depth_ftheta *= cfg.inv_depth_coef;
    for (int y = 0; y < R_inv_depth_ftheta.rows; ++y) {
      for (int x = 0; x < R_inv_depth_ftheta.cols; ++x) {
        R_inv_depth_ftheta.at<float>(y, x) =
            math::clamp(R_inv_depth_ftheta.at<float>(y, x), 0.0f, 1.0f);
      }
    }

    // Save the depthmap with 16 bit png.
    R_inv_depth_ftheta.convertTo(R_inv_depth_ftheta, CV_16U, 65535.0);

    // Save images to disk.
    const std::string fnum = string::intToZeroPad(frame_index, 6);
    cv::imwrite(cfg.dest_dir + "/R_ftheta_" + fnum + ".png", R_ftheta);
    cv::imwrite(cfg.dest_dir + "/R_ftheta_half_" + fnum + ".png", R_ftheta_half);
    cv::imwrite(cfg.dest_dir + "/R_depth_" + fnum + ".png", R_inv_depth_ftheta);
  }
};

void precomputeFisheyeWarpsAndVignettes(
    const LdiPipelineConfig& cfg,
    calibration::FisheyeCamerad& cam_R,
    cv::Mat& color_vignette,
    cv::Mat& depth_vignette,
    std::vector<cv::Mat>& warp_ftheta_to_inflated)
{
  cam_R = projection::makePerfectFthetaCamera(cfg.ftheta_size);
  calibration::FisheyeCamerad cam_inflated =
      projection::makePerfectFthetaCamera(cfg.inflated_ftheta_size);
  cam_inflated.radius_at_90 *= cfg.ftheta_scale;

  color_vignette = projection::makeVignetteMap(
      cam_R,
      cv::Size(cam_R.width, cam_R.height),
      0.67 * 1.25,
      0.69 * 1.25,
      0.02 * 1.25,
      0.06 * 1.25);
  depth_vignette = projection::makeVignetteMap(
      cam_R,
      cv::Size(cam_R.width, cam_R.height),
      0.68 * 1.25,
      0.70 * 1.25,
      0.01 * 1.25,
      0.02 * 1.25);

  cam_R.radius_at_90 *= cfg.ftheta_scale;

  projection::precomputeFisheyeToInflatedWarp(cam_R, cam_inflated, warp_ftheta_to_inflated);
}

}  // namespace

void runVR180toLdi3VideoPipelineAllPhases(const LdiPipelineConfig& cfg) {
    videoDepthPhase(cfg);
    
    if (cfg.cancel_requested && *cfg.cancel_requested) return;
    
    temporallyStabilizeDepth(cfg);
    
    if (cfg.cancel_requested && *cfg.cancel_requested) return;
    
    inpaintPhase(cfg);
    
    if (cfg.cancel_requested && *cfg.cancel_requested) return;
    
    if (cfg.stabilize_inpainting) stabilizeInpaintingPhase(cfg);
}

void videoDepthPhase(const LdiPipelineConfig& cfg)
{
  if (cfg.rm_dest_dir) std::system(std::string("rm " + cfg.dest_dir + "/*").c_str());

  VR180DepthProcessor proc(cfg);
  cv::VideoCapture video_capture(cfg.src_vr180);
  XCHECK(video_capture.isOpened()) << "Failed to open video: " << cfg.src_vr180;
  int num_frames_unreliable = video_capture.get(cv::CAP_PROP_FRAME_COUNT);
  // num_frames comes out as a negative number if we input a png.
  if (num_frames_unreliable < 0) num_frames_unreliable = 1;
  int last = cfg.last_frame > -1 ? cfg.last_frame : num_frames_unreliable - 1;
  int frame_index = 0;
  while (true) {
    if (cfg.cancel_requested && *cfg.cancel_requested) return;

    cv::Mat frame;
    video_capture >> frame;

    if (frame.empty()) {
      break;
    }

    if (cfg.skip_every_other_frame) {
      cv::Mat discard_frame;
      video_capture >> discard_frame;
    }

    if (frame_index == 0) {
      proc.init(frame);
    }

    if (frame_index < cfg.first_frame) {
      XPLINFO << "decoded frame " << frame_index
              << " outside [first_frame, last_frame], skipping depth";
      ++frame_index;
      continue;
    }

    XPLINFO << "phase=StereoDepth first_frame=" << cfg.first_frame << " last_frame=" << last
        << " current_frame=" << frame_index << " frames_total=" << num_frames_unreliable;

    if (cfg.last_frame > -1 && frame_index > cfg.last_frame) {
      break;
    }

    proc.estimateDepthForFrame(frame, frame_index);

    ++frame_index;
  }
}

void temporallyStabilizeDepth(const LdiPipelineConfig& cfg)
{
  // std::system(std::string("rm " + cfg.dest_dir + "/filtered_R_depth_*").c_str());
  // std::system(std::string("rm " + cfg.dest_dir + "/warpe_*").c_str());

  const int num_frames = file::countMatchingFiles(cfg.dest_dir, "^R_depth_([0-9]{6})\\.(png|jpg)");
  XPLINFO << "num_frames: " << num_frames;

  const std::vector<int> nei_offsets = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13,
                                        +1, +2, +3, +4, +5, +6, +7, +8, +9, +10, +11, +12, +13};
  const int cache_size = nei_offsets.size() + 2;
  ImageCache cache_image(cache_size), cache_depth(cache_size);

  int last = cfg.last_frame > -1 ? cfg.last_frame : num_frames - 1;
  for (int f = cfg.first_frame; f <= last; ++f) {
    if (cfg.cancel_requested && *cfg.cancel_requested) return;

    XPLINFO << "phase=StabilizeDepth first_frame=" << cfg.first_frame << " curr_frame= " << f
            << " frames_in_chunk=" << (last + 1 - cfg.first_frame)
            << " frames_total=" << num_frames;

    auto start_time = time::now();

    // Load the current frame image and depthmap, and the neighbors
    const std::string fnum = string::intToZeroPad(f, 6);
    cv::Mat curr_image =
        cache_image.imread(cfg.dest_dir + "/R_ftheta_half_" + fnum + ".png", cv::IMREAD_COLOR);
    cv::Mat curr_depth =
        cache_depth.imread(cfg.dest_dir + "/R_depth_" + fnum + ".png", cv::IMREAD_UNCHANGED);
    XCHECK_EQ(curr_depth.type(), CV_16UC1);
    curr_depth.convertTo(curr_depth, CV_32FC1, 1.0 / 65535.0f);

    // Build up a vector of the depthmaps and weight maps that will eventually be combined to form a
    // filtered estimated depth.
    std::vector<cv::Mat> filter_depths, filter_motiondiffs;
    filter_depths.push_back(curr_depth);
    cv::Mat zero_map(curr_image.size(), CV_32F, cv::Scalar(0));
    filter_motiondiffs.push_back(zero_map);

    float total_time_loading = 0;

    for (int f_offset : nei_offsets) {
      const int fn = f + f_offset;
      if (fn < 0 || fn >= num_frames) continue;
      const std::string nei_num = string::intToZeroPad(fn, 6);

      // Load the image, depthmap and error map for the neighbor image
      auto load_timer = time::now();

      cv::Mat nei_image =
          cache_image.imread(cfg.dest_dir + "/R_ftheta_half_" + nei_num + ".png", cv::IMREAD_COLOR);
      cv::Mat nei_depth =
          cache_depth.imread(cfg.dest_dir + "/R_depth_" + nei_num + ".png", cv::IMREAD_UNCHANGED);

      nei_depth.convertTo(
          nei_depth, CV_32F, 1.0 / 65535.0f);  // TODO: this is duplicated/could be cached.
      total_time_loading += time::timeSinceSec(load_timer);
      filter_depths.push_back(nei_depth);

      cv::Mat diffmap(curr_image.size(), CV_32F);
      for (int y = 0; y < curr_image.rows; ++y) {
        for (int x = 0; x < curr_image.cols; ++x) {
          static constexpr float kSensitivity = 5.0;
          static constexpr float kBias = 0.3;
          // TODO: maybe a little blur before the diff here could reduce sensititivy to noise
          cv::Vec3f color_diff = (cv::Vec3f(curr_image.at<cv::Vec3b>(y, x)) -
                                  cv::Vec3f(nei_image.at<cv::Vec3b>(y, x))) /
                                 255.0;
          const float abs_diff =
              std::abs(color_diff[0]) + std::abs(color_diff[1]) + std::abs(color_diff[2]);
          diffmap.at<float>(y, x) = std::max(0.0f, std::tanh(kSensitivity * abs_diff - kBias));
        }
      }
      constexpr int kDiffDilateRadius = 11;
      cv::dilate(diffmap, diffmap, cv::Mat(), cv::Point(-1, -1), kDiffDilateRadius);
      cv::GaussianBlur(diffmap, diffmap, cv::Size(21, 21), kDiffDilateRadius, kDiffDilateRadius);
      filter_motiondiffs.push_back(diffmap);
    }  // end loop over neighbors

    // Average the current and neighbor depthmaps, weighted by motion estimate.
    cv::Mat avg_depth(curr_image.size(), CV_32F, cv::Scalar(0));
    cv::Mat sum_weight(curr_image.size(), CV_32F, cv::Scalar(0));
    for (int i = 0; i < filter_depths.size(); ++i) {
      for (int y = 0; y < curr_image.rows; ++y) {
        for (int x = 0; x < curr_image.cols; ++x) {
          float weight = 1.0 - filter_motiondiffs[i].at<float>(y, x);
          weight = math::clamp(weight, 0.001f, 1.0f);

          sum_weight.at<float>(y, x) += weight;
          avg_depth.at<float>(y, x) += weight * filter_depths[i].at<float>(y, x);
        }
      }
    }
    for (int y = 0; y < curr_image.rows; ++y) {
      for (int x = 0; x < curr_image.cols; ++x) {
        avg_depth.at<float>(y, x) = avg_depth.at<float>(y, x) / sum_weight.at<float>(y, x);
      }
    }

    avg_depth.convertTo(avg_depth, CV_16UC1, 65535.0f);
    cv::imwrite(cfg.dest_dir + "/filtered_R_depth_" + fnum + ".png", avg_depth);
    XPLINFO << "file load time(sec):\t" << total_time_loading;
    XPLINFO << "stabilize time(sec):\t" << time::timeSinceSec(start_time);
  }  // end loop over frames

}

void inpaintPhase(const LdiPipelineConfig& cfg)
{
  const int num_frames =
      file::countMatchingFiles(cfg.dest_dir, "^filtered_R_depth_([0-9]{6})\\.(png|jpg)");
  XPLINFO << "num_frames: " << num_frames;

  calibration::FisheyeCamerad cam_R;
  cv::Mat color_vignette, depth_vignette;
  std::vector<cv::Mat> warp_ftheta_to_inflated;
  precomputeFisheyeWarpsAndVignettes(
      cfg, cam_R, color_vignette, depth_vignette, warp_ftheta_to_inflated);

  int last = cfg.last_frame > -1 ? cfg.last_frame : num_frames - 1;
  for (int f = cfg.first_frame; f <= last; ++f) {
    if (cfg.cancel_requested && *cfg.cancel_requested) return;

    XPLINFO << "phase=Inpainting first_frame=" << cfg.first_frame << " curr_frame=" << f
            << " frames_in_chunk=" << (last + 1 - cfg.first_frame)
            << " frames_total=" << num_frames;

    XPLINFO << "---- segmenting and inpainting frame: " << f << "/" << num_frames;
    // Load the current frame image and depthmap, and the neighbors
    const std::string fnum = string::intToZeroPad(f, 6);
    cv::Mat R_ftheta = cv::imread(cfg.dest_dir + "/R_ftheta_" + fnum + ".png");
    cv::Mat R_inv_depth_ftheta =
        cv::imread(cfg.dest_dir + "/filtered_R_depth_" + fnum + ".png", cv::IMREAD_UNCHANGED);
    XCHECK_EQ(R_inv_depth_ftheta.type(), CV_16UC1);
    R_inv_depth_ftheta.convertTo(R_inv_depth_ftheta, CV_32FC1, 1.0 / 65535.0f);

    std::vector<cv::Mat> layer_bgra, layer_invd;
    cv::Mat cached_seg = cv::Mat();
    if (cfg.use_cached_seg) {
      cached_seg = cv::imread(cfg.dest_dir + "/seg_" + fnum + ".png");
      cached_seg.convertTo(cached_seg, CV_32FC3, 1.0 / 255.0);
    }
    makeLdiHeuristic(
        cfg.cwd,
        cfg.dest_dir,
        cfg.inpaint_method,
        cfg.seg_method,
        cfg.sd_ver,
        kNumLayers,
        cam_R,
        R_ftheta,
        R_inv_depth_ftheta,
        color_vignette,
        depth_vignette,
        layer_bgra,
        layer_invd,
        /*write_inpainting_stabilization_files=*/cfg.stabilize_inpainting,
        /*assemble_ldi=*/!cfg.stabilize_inpainting,
        cfg.inpaint_dilate_radius,
        cfg.run_seg_only,
        cfg.write_seg,
        cached_seg,
        fnum);

    if (!cfg.stabilize_inpainting) {
     cv::Mat ldi_grid =
        make6DofGrid(layer_bgra, layer_invd, cfg.output_encoding, warp_ftheta_to_inflated);
     cv::imwrite(cfg.dest_dir + "/ldi3_" + fnum + ".png", ldi_grid);
    }
  }
}

void stabilizeInpaintingPhase(const LdiPipelineConfig& cfg)
{
  //std::system(std::string("rm " + cfg.dest_dir + "/ldi3_*").c_str());
  //std::system(std::string("rm " + cfg.dest_dir + "/l0_stabilized_inpaint_*").c_str());
  //std::system(std::string("rm " + cfg.dest_dir + "/l1_stabilized_inpaint_*").c_str());

  const int num_frames =
      file::countMatchingFiles(cfg.dest_dir, "^l0_inpainted_([0-9]{6})\\.(png|jpg)");
  XPLINFO << "num_frames: " << num_frames;

  const std::vector<int> nei_offsets = {-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13,
                                        +1, +2, +3, +4, +5, +6, +7, +8, +9, +10, +11, +12, +13};
  const int cache_size = nei_offsets.size() + 2;
  ImageCache cache_l0(cache_size), cache_l1(cache_size), cache_l0_depth(cache_size),
      cache_l1_depth(cache_size), cache_l0_mask(cache_size), cache_l1_mask(cache_size);

  calibration::FisheyeCamerad cam_R;
  cv::Mat color_vignette, depth_vignette;
  std::vector<cv::Mat> warp_ftheta_to_inflated;
  precomputeFisheyeWarpsAndVignettes(
      cfg, cam_R, color_vignette, depth_vignette, warp_ftheta_to_inflated);

  int last = cfg.last_frame > -1 ? cfg.last_frame : num_frames - 1;
  for (int f = cfg.first_frame; f <= last; ++f) {
    if (cfg.cancel_requested && *cfg.cancel_requested) return;

    XPLINFO << "phase=StabilizeInpainting first_frame=" << cfg.first_frame << " curr_frame=" << f
            << " frames_in_chunk=" << (last + 1 - cfg.first_frame)
            << " frames_total=" << num_frames;

    auto start_time = time::now();
    const std::string fnum = string::intToZeroPad(f, 6);

    cv::Mat curr_l0_inpainted =
        cache_l0.imread(cfg.dest_dir + "/l0_inpainted_" + fnum + ".png", cv::IMREAD_COLOR);
    cv::Mat curr_l1_inpainted =
        cache_l1.imread(cfg.dest_dir + "/l1_inpainted_" + fnum + ".png", cv::IMREAD_COLOR);
    cv::Mat curr_l0_inpainted_depth = cache_l0_depth.imread(
        cfg.dest_dir + "/l0_inpainted_depth_" + fnum + ".png", cv::IMREAD_UNCHANGED);
    cv::Mat curr_l1_inpainted_depth = cache_l1_depth.imread(
        cfg.dest_dir + "/l1_inpainted_depth_" + fnum + ".png", cv::IMREAD_UNCHANGED);
    cv::Mat curr_l0_inpaint_mask = cache_l0_mask.imread(
        cfg.dest_dir + "/l0_inpaint_mask_" + fnum + ".png", cv::IMREAD_UNCHANGED);
    cv::Mat curr_l1_inpaint_mask = cache_l1_mask.imread(
        cfg.dest_dir + "/l1_inpaint_mask_" + fnum + ".png", cv::IMREAD_UNCHANGED);

    cv::Mat stabilized_l0_inpainted, stabilized_l1_inpainted;
    cv::Mat stabilized_l0_inpainted_depth, stabilized_l1_inpainted_depth;
    curr_l0_inpainted.convertTo(stabilized_l0_inpainted, CV_32FC3, 1.0 / 255.0);
    curr_l1_inpainted.convertTo(stabilized_l1_inpainted, CV_32FC3, 1.0 / 255.0);
    curr_l0_inpainted_depth.convertTo(stabilized_l0_inpainted_depth, CV_32FC1, 1.0 / 65535.0f);
    curr_l1_inpainted_depth.convertTo(stabilized_l1_inpainted_depth, CV_32FC1, 1.0 / 65535.0f);

    cv::Mat l0_sum_weight(curr_l0_inpainted.size(), CV_32F, cv::Scalar(1.0));
    cv::Mat l1_sum_weight(curr_l1_inpainted.size(), CV_32F, cv::Scalar(1.0));
    cv::Mat l0_depth_sum_weight(curr_l0_inpainted_depth.size(), CV_32F, cv::Scalar(1.0));
    cv::Mat l1_depth_sum_weight(curr_l1_inpainted_depth.size(), CV_32F, cv::Scalar(1.0));

    for (int f_offset : nei_offsets) {
      const int fn = f + f_offset;
      if (fn < 0 || fn >= num_frames) continue;
      const std::string nei_num = string::intToZeroPad(fn, 6);

      cv::Mat nei_l0_inpainted =
          cache_l0.imread(cfg.dest_dir + "/l0_inpainted_" + nei_num + ".png", cv::IMREAD_COLOR);
      cv::Mat nei_l1_inpainted =
          cache_l1.imread(cfg.dest_dir + "/l1_inpainted_" + nei_num + ".png", cv::IMREAD_COLOR);
      cv::Mat nei_l0_inpainted_depth = cache_l0_depth.imread(
          cfg.dest_dir + "/l0_inpainted_depth_" + nei_num + ".png", cv::IMREAD_UNCHANGED);
      cv::Mat nei_l1_inpainted_depth = cache_l1_depth.imread(
          cfg.dest_dir + "/l1_inpainted_depth_" + nei_num + ".png", cv::IMREAD_UNCHANGED);
      cv::Mat nei_l0_inpaint_mask = cache_l0_mask.imread(
          cfg.dest_dir + "/l0_inpaint_mask_" + nei_num + ".png", cv::IMREAD_UNCHANGED);
      cv::Mat nei_l1_inpaint_mask = cache_l1_mask.imread(
          cfg.dest_dir + "/l1_inpaint_mask_" + nei_num + ".png", cv::IMREAD_UNCHANGED);
      nei_l0_inpainted_depth.convertTo(nei_l0_inpainted_depth, CV_32F, 1.0 / 65535.0f);
      nei_l1_inpainted_depth.convertTo(nei_l1_inpainted_depth, CV_32F, 1.0 / 65535.0f);

      accumulateWeightedSum<cv::Vec3b, cv::Vec3f>(
          nei_l0_inpainted,
          1.0 / 255.0,
          nei_l0_inpaint_mask,
          stabilized_l0_inpainted,
          l0_sum_weight);
      accumulateWeightedSum<cv::Vec3b, cv::Vec3f>(
          nei_l1_inpainted,
          1.0 / 255.0,
          nei_l1_inpaint_mask,
          stabilized_l1_inpainted,
          l1_sum_weight);
      accumulateWeightedSum<float, float>(
          nei_l0_inpainted_depth,
          1.0,
          nei_l0_inpaint_mask,
          stabilized_l0_inpainted_depth,
          l0_depth_sum_weight);
      accumulateWeightedSum<float, float>(
          nei_l1_inpainted_depth,
          1.0,
          nei_l1_inpaint_mask,
          stabilized_l1_inpainted_depth,
          l1_depth_sum_weight);
    }

    // Take a weighted average. Sum weight will not be 0 because we initialize  it to have a weight
    // of 1 on the curr frame.
    for (int y = 0; y < stabilized_l0_inpainted.rows; ++y) {
      for (int x = 0; x < stabilized_l0_inpainted.cols; ++x) {
        stabilized_l0_inpainted.at<cv::Vec3f>(y, x) =
            stabilized_l0_inpainted.at<cv::Vec3f>(y, x) / l0_sum_weight.at<float>(y, x);
        stabilized_l1_inpainted.at<cv::Vec3f>(y, x) =
            stabilized_l1_inpainted.at<cv::Vec3f>(y, x) / l1_sum_weight.at<float>(y, x);
      }
    }
    for (int y = 0; y < stabilized_l0_inpainted_depth.rows; ++y) {
      for (int x = 0; x < stabilized_l0_inpainted_depth.cols; ++x) {
        stabilized_l0_inpainted_depth.at<float>(y, x) =
            stabilized_l0_inpainted_depth.at<float>(y, x) / l0_depth_sum_weight.at<float>(y, x);
        stabilized_l1_inpainted_depth.at<float>(y, x) =
            stabilized_l1_inpainted_depth.at<float>(y, x) / l1_depth_sum_weight.at<float>(y, x);
      }
    }
    stabilized_l0_inpainted.convertTo(stabilized_l0_inpainted, CV_8UC3, 255.0);
    stabilized_l1_inpainted.convertTo(stabilized_l1_inpainted, CV_8UC3, 255.0);

    XPLINFO << "stabilize inpainting time(sec):\t" << time::timeSinceSec(start_time);

    auto assemble_start_time = time::now();
    cv::Mat curr_R_ftheta =
        cv::imread(cfg.dest_dir + "/R_ftheta_" + fnum + ".png", cv::IMREAD_COLOR);
    cv::Mat curr_R_inv_depth_ftheta =
        cv::imread(cfg.dest_dir + "/R_depth_" + fnum + ".png", cv::IMREAD_UNCHANGED);
    cv::Mat curr_l1_alpha =
        cv::imread(cfg.dest_dir + "/l1_alpha_" + fnum + ".png", cv::IMREAD_UNCHANGED);
    cv::Mat curr_l2_alpha =
        cv::imread(cfg.dest_dir + "/l2_alpha_" + fnum + ".png", cv::IMREAD_UNCHANGED);
    cv::Mat curr_l0_blend =
        cv::imread(cfg.dest_dir + "/l0_blend_" + fnum + ".png", cv::IMREAD_UNCHANGED);
    cv::Mat curr_l1_blend =
        cv::imread(cfg.dest_dir + "/l1_blend_" + fnum + ".png", cv::IMREAD_UNCHANGED);

    curr_R_inv_depth_ftheta.convertTo(curr_R_inv_depth_ftheta, CV_32F, 1.0 / 65535.0f);
    curr_l1_alpha.convertTo(curr_l1_alpha, CV_32F, 1.0 / 255.0f);
    curr_l2_alpha.convertTo(curr_l2_alpha, CV_32F, 1.0 / 255.0f);
    curr_l0_blend.convertTo(curr_l0_blend, CV_32F, 1.0 / 255.0f);
    curr_l1_blend.convertTo(curr_l1_blend, CV_32F, 1.0 / 255.0f);

    std::vector<cv::Mat> layer_bgra, layer_invd;
    assembleLayersAndChannels(
        color_vignette,
        depth_vignette,
        curr_R_ftheta,
        curr_R_inv_depth_ftheta,
        curr_l0_blend,
        curr_l1_blend,
        curr_l1_alpha,
        curr_l2_alpha,
        stabilized_l0_inpainted_depth,
        stabilized_l1_inpainted_depth,
        stabilized_l0_inpainted,
        stabilized_l1_inpainted,
        layer_bgra,
        layer_invd);

    cv::Mat ldi_grid =
        make6DofGrid(layer_bgra, layer_invd, cfg.output_encoding, warp_ftheta_to_inflated);
    cv::imwrite(cfg.dest_dir + "/ldi3_" + fnum + ".png", ldi_grid);

    XPLINFO << "assemble time(sec):\t" << time::timeSinceSec(assemble_start_time);

    // For debugging, we could save the stabilized inpaintings
    // stabilized_l0_inpainted_depth.convertTo(stabilized_l0_inpainted_depth, CV_16UC1, 65535.0f);
    // stabilized_l1_inpainted_depth.convertTo(stabilized_l1_inpainted_depth, CV_16UC1, 65535.0f);
    // cv::imwrite(cfg.dest_dir + "/l0_stabilized_inpaint_" + fnum + ".png",
    // stabilized_l0_inpainted); cv::imwrite(cfg.dest_dir + "/l1_stabilized_inpaint_" + fnum +
    // ".png", stabilized_l1_inpainted); cv::imwrite(cfg.dest_dir + "/l0_stabilized_inpaint_depth_"
    // + fnum + ".png", stabilized_l0_inpainted_depth); cv::imwrite(cfg.dest_dir +
    // "/l1_stabilized_inpaint_depth_" + fnum + ".png", stabilized_l1_inpainted_depth);
  }
}

void runVR180PhototoLdiPipeline(const LdiPipelineConfig& cfg)
{
  if (cfg.rm_dest_dir) std::system(std::string("rm " + cfg.dest_dir + "/*").c_str());

  XPLINFO << "phase=Initializing";
  torch::manual_seed(123);  // For reproducible initialization of weights
  srand(123);               // For calls to rand()

  cv::Mat R_ftheta, R_inv_depth_ftheta;
  calibration::FisheyeCamerad cam_unscaled, cam_R;
  std::vector<cv::Mat> warp_vr180_to_ftheta_unscaled, warp_vr180_to_ftheta_scaled,
      warp_ftheta_to_rectified, warp_rectified_to_ftheta, warp_ftheta_to_inflated;

  calibration::FisheyeCamerad cam_inflated =
      projection::makePerfectFthetaCamera(cfg.inflated_ftheta_size);
  cam_inflated.radius_at_90 *= cfg.ftheta_scale;

  if (cfg.cancel_requested && *cfg.cancel_requested) return;

  if (!cfg.src_vr180.empty()) {
    // Create a virtual camera with f-theta projection that we will warp to/from
    calibration::FisheyeCamerad cam_perfect_ftheta =
        projection::makePerfectFthetaCamera(cfg.ftheta_size);
    calibration::FisheyeCamerad cam_perfect_ftheta_half =
        projection::makePerfectFthetaCamera(cfg.ftheta_size / 2);
    cam_unscaled = projection::makePerfectFthetaCamera(cfg.ftheta_size);
    cam_R = cam_unscaled;
    cam_R.radius_at_90 *= cfg.ftheta_scale;

    // Load the input VR180 image
    cv::Mat src_vr180_image = cv::imread(cfg.src_vr180);
    XCHECK(!src_vr180_image.empty()) << cfg.src_vr180;
    cv::Mat L_image =
        src_vr180_image(cv::Rect(0, 0, src_vr180_image.cols / 2, src_vr180_image.rows));
    cv::Mat R_image = src_vr180_image(
        cv::Rect(src_vr180_image.cols / 2, 0, src_vr180_image.cols / 2, src_vr180_image.rows));

    // Precompute warp maps between VR180, f-theta, and rectified projections.
    projection::precomputeVR180toFthetaWarp(
        cam_perfect_ftheta, cfg.ftheta_size, L_image.cols, warp_vr180_to_ftheta_unscaled, 1.0);
    projection::precomputeVR180toFthetaWarp(
        cam_perfect_ftheta,
        cfg.ftheta_size,
        L_image.cols,
        warp_vr180_to_ftheta_scaled,
        cfg.ftheta_scale);
    projection::precomputeFisheyeToRectifyWarp(
        cfg.rectified_size_for_depth,
        cfg.rectified_size_for_depth,
        cam_perfect_ftheta,
        warp_ftheta_to_rectified);
    projection::precomputeRectifiedToFisheyeWarp(
        cfg.rectified_size_for_depth,
        cfg.rectified_size_for_depth,
        cv::Size(cfg.ftheta_size / 2, cfg.ftheta_size / 2),
        cam_perfect_ftheta_half,
        warp_rectified_to_ftheta,
        cfg.ftheta_scale);
    projection::precomputeFisheyeToInflatedWarp(cam_R, cam_inflated, warp_ftheta_to_inflated);

    // Do the warping
    cv::Mat L_ftheta_unscaled =
        projection::warp(L_image, warp_vr180_to_ftheta_unscaled, cv::INTER_AREA);
    cv::Mat R_ftheta_unscaled =
        projection::warp(R_image, warp_vr180_to_ftheta_unscaled, cv::INTER_AREA);
    cv::Mat L_ftheta = projection::warp(L_image, warp_vr180_to_ftheta_scaled, cv::INTER_CUBIC);
    R_ftheta = projection::warp(R_image, warp_vr180_to_ftheta_scaled, cv::INTER_CUBIC);
    cv::Mat L_rectified =
        projection::warp(L_ftheta_unscaled, warp_ftheta_to_rectified, cv::INTER_AREA);
    cv::Mat R_rectified =
        projection::warp(R_ftheta_unscaled, warp_ftheta_to_rectified, cv::INTER_AREA);

    if (cfg.cancel_requested && *cfg.cancel_requested) return;

    // Prevent torch from trying to optimize the disparity model (this actually wastes more time
    // than it saves here).
    torch::jit::getProfilingMode() = false;

    // Load the RAFT optical flow model
    cv::Mat R_disparity, L_disparity, R_error_rectified, L_error_rectified;
    {  // TODO: RAFT GPU memory is not released
      torch::jit::getProfilingMode() = false;
      XPLINFO << "phase=LoadingNet";
      torch::jit::script::Module raft_module;
      optical_flow::getTorchModelRAFT(raft_module, /*model_path=*/"");

      auto disparity_start_timer = time::now();
      R_disparity = optical_flow::computeDisparityRAFT(raft_module, R_rectified, L_rectified);
      XPLINFO << "phase=CalculatingDepth time_per_frame_s="
              << time::timeSinceSec(disparity_start_timer);
    }
    cv::Mat R_inv_depth_rectified = projection::disparityToInvDepth(R_disparity, cfg.baseline_m);

    // Warp the rectified depth to f-theta projection
    R_inv_depth_ftheta =
        projection::warp(R_inv_depth_rectified, warp_rectified_to_ftheta, cv::INTER_LINEAR);

    R_inv_depth_ftheta *= cfg.inv_depth_coef;
    for (int y = 0; y < R_inv_depth_ftheta.rows; ++y) {
      for (int x = 0; x < R_inv_depth_ftheta.cols; ++x) {
        R_inv_depth_ftheta.at<float>(y, x) =
            math::clamp(R_inv_depth_ftheta.at<float>(y, x), 0.0f, 1.0f);
      }
    }

    if (cfg.cancel_requested && *cfg.cancel_requested) return;

    // Save images to disk
    cv::imwrite(cfg.dest_dir + "/R_ftheta.png", R_ftheta);
    cv::imwrite(cfg.dest_dir + "/R_depth.png", R_inv_depth_ftheta * 255.0);
  } else {
    XCHECK(!cfg.src_ftheta_image.empty());
    XCHECK(!cfg.src_ftheta_depth.empty());

    R_ftheta = cv::imread(cfg.src_ftheta_image);
    R_inv_depth_ftheta = cv::imread(cfg.src_ftheta_depth);
    R_inv_depth_ftheta.convertTo(R_inv_depth_ftheta, CV_32FC3, 1.0f / 255.0f);
    cv::cvtColor(R_inv_depth_ftheta, R_inv_depth_ftheta, cv::COLOR_BGR2GRAY);

    cam_unscaled = projection::makePerfectFthetaCamera(R_ftheta.rows);
    cam_R = cam_unscaled;
    cam_R.radius_at_90 *= cfg.ftheta_scale;

    projection::precomputeFisheyeToInflatedWarp(cam_R, cam_inflated, warp_ftheta_to_inflated);
  }

  cv::Mat color_vignette, depth_vignette;
  color_vignette = projection::makeVignetteMap(
      cam_unscaled,
      cv::Size(cam_unscaled.width, cam_unscaled.height),
      0.67 * 1.25,
      0.69 * 1.25,
      0.02 * 1.25,
      0.06 * 1.25);
  depth_vignette = projection::makeVignetteMap(
      cam_unscaled,
      cv::Size(cam_unscaled.width, cam_unscaled.height),
      0.68 * 1.25,
      0.70 * 1.25,
      0.01 * 1.25,
      0.02 * 1.25);

  if (cfg.cancel_requested && *cfg.cancel_requested) return;

  cv::Mat cached_seg = cv::Mat();
  if (cfg.use_cached_seg) {
    cached_seg = cv::imread(cfg.dest_dir + "/seg.png");
    cached_seg.convertTo(cached_seg, CV_32FC3, 1.0 / 255.0);
  }

  std::vector<cv::Mat> layer_bgra, layer_invd;
  makeLdiHeuristic(
      cfg.cwd,
      cfg.dest_dir,
      cfg.inpaint_method,
      cfg.seg_method,
      cfg.sd_ver,
      kNumLayers,
      cam_R,
      R_ftheta,
      R_inv_depth_ftheta,
      color_vignette,
      depth_vignette,
      layer_bgra,
      layer_invd,
      /*write_inpainting_stabilization_files=*/false,
      /*assemble_ldi=*/true,
      cfg.inpaint_dilate_radius,
      cfg.run_seg_only,
      cfg.write_seg,
      cached_seg);

  // for (int l = 0; l < kNumLayers; ++l) {
  //   cv::imwrite(cfg.dest_dir + "/bgra_" + std::to_string(l) + ".png", layer_bgra[l] * 255.0);
  //   cv::imwrite(cfg.dest_dir + "/invd_" + std::to_string(l) + ".png", layer_invd[l] * 255.0);
  // }

  if (cfg.cancel_requested && *cfg.cancel_requested) return;

  cv::Mat ldi_grid =
      make6DofGrid(layer_bgra, layer_invd, cfg.output_encoding, warp_ftheta_to_inflated);

  // If it's a jpeg, write with turbojpeg. Otherwise it will fall back to cv::imwrite().
  constexpr int kJpegQuality = 90;
  turbojpeg::writeJpeg(cfg.dest_dir + "/" + cfg.output_filename, ldi_grid, kJpegQuality);

  if (cfg.make_fused_image) {
    cv::Mat fused_bgra, fused_invd;
    fuseLayers(
        {layer_bgra[0], layer_bgra[1], layer_bgra[2]},
        {layer_invd[0], layer_invd[1], layer_invd[2]},
        fused_bgra,
        fused_invd);
    cv::imwrite(cfg.dest_dir + "/fused_bgra.png", fused_bgra * 255.0);
  }
}

void cleanup(const LdiPipelineConfig& cfg) {
  XPLINFO << "Cleaning up PNG files phase=cleanup";
  std::system(std::string("rm " + cfg.dest_dir + "/R_ftheta_*").c_str());
  std::system(std::string("rm " + cfg.dest_dir + "/R_depth_*").c_str());
  std::system(std::string("rm " + cfg.dest_dir + "/filtered_R_depth*").c_str());
  std::system(std::string("rm " + cfg.dest_dir + "/l0_*").c_str());
  std::system(std::string("rm " + cfg.dest_dir + "/l1_*").c_str());
  std::system(std::string("rm " + cfg.dest_dir + "/l2_*").c_str());
  std::system(std::string("rm " + cfg.dest_dir + "/sd_*").c_str());
  std::system(std::string("rm " + cfg.dest_dir + "/seg_*").c_str());
  std::system(std::string("rm " + cfg.dest_dir + "/motion_*").c_str());
  XPLINFO << "Finished cleaning up PNG files";
}

}}  // namespace p11::ldi
