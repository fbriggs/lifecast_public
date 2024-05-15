// MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include "rectilinear_sfm_lib.h"

#include <fstream>
#include <cmath>
#include <map>
#include <algorithm>
#include <random>
#include <atomic>
#include "gflags/gflags.h"
#include "logger.h"
#include "check.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/photo.hpp"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "rectilinear_camera.h"
#include "pose_param.h"
#include "keypoint_tracking.h"
#include "point_cloud.h"
#include "util_math.h"
#include "util_time.h"
#include "util_file.h"
#include "util_string.h"
#include "util_opencv.h"
#include "third_party/json.h"
#include "Eigen/Core"
#include "Eigen/Geometry"

namespace p11 { namespace rectilinear_sfm {

namespace {
// Dont make a new track if the keypoint is within this distance of an existing track.
constexpr float kSupressionRadius = 8;
// Number of random keypoints to add each frame (in addition to those from FAST).
constexpr int kNumRandomKeypointsPerFrame = 200;
// Threshold for FAST keypoint finder. Higher -> fewer FAST keypoints.
constexpr int kFastThreshold = 1;
// Maximum number of frames to keep a track alive.
constexpr int kMaxTrackAge = 200;
// Probability of killing a track after it exceeds the max age.
constexpr float kKillOldTrackProbability = 0.01;
}

struct Track {
  // The pixel coordinates where the track is observed in the  images.
  std::map<int, cv::Point2f> frame_to_pixel;
  double avg_observation_err;
  double track_weight;

  // The estimated 3d point in the world frame. This is a decision variable in
  // the full SFM optimization.
  Eigen::Vector3d point3d_in_world;

  // Not every track will be used in bundle adjustment. This keeps track of which are,
  // so they can be visualized.
  bool used_in_bundle_adjustment;

  // Color of the corresponding pixels.
  Eigen::Vector3f color;
};

struct Frame {
  int idx;  // frame number in video sequence
  cv::Mat image;
  cv::Mat gray;

  // Indices of tracks that are active or new in this frame.
  std::vector<int> active_tracks, new_tracks;

  // Used to help make tracking more robust to varying light and exposure.
  float brightness;

  std::vector<double> cam_from_world_param;

  // This is unpacked from the SFM solution (cam_from_world_param).
  Eigen::Isometry3d world_from_cam;
};

void preprocessFrameForTracking(Frame& f, const float prev_brightness) {
  cv::cvtColor(f.image, f.gray, cv::COLOR_BGR2GRAY);
  f.brightness = cv::mean(f.gray)[0];

  if (prev_brightness > 0) {
    const float alpha = 0.95;
    const float ratio = alpha * prev_brightness / f.brightness + (1 - alpha);
    f.gray *= ratio;
  }
  f.brightness = cv::mean(f.gray)[0];
  //cv::imwrite(cfg.dest_dir + "/gray_" + string::intToZeroPad(f.idx, 6) + ".jpg", f.gray);
}

cv::Mat visualizeTracks(const Frame& f, const std::vector<Track>& tracks) {
  cv::Mat viz = f.image.clone();
  cv::cvtColor(viz, viz, cv::COLOR_BGR2BGRA);

  for (const size_t track_idx : f.active_tracks) {
    const Track& track = tracks[track_idx];

    if (!track.used_in_bundle_adjustment) continue;

    cv::Vec4f track_color = opencv::colorHash(track_idx);
    const cv::Point2f pixel = track.frame_to_pixel.at(f.idx);

    track_color = track_color * track.track_weight + cv::Vec4f(0, 0, 255, 255) * (1.0 - track.track_weight);

    // These anti-aliased crosses show the subpixel accuracy of tracking better than cv::circle.
    opencv::drawCrossAntiAliased(pixel, track_color, viz);
  }

  return viz;
}

void makeNewTracks(Frame& f, std::vector<Track>& tracks) {
  // Make candidate keypoints with a combination of FAST and random placement.
  std::vector<cv::Point2f> raw_keypoints =
      keypoint_tracking::findKeypointsFAST(f.gray, kFastThreshold);
  for (int i = 0; i < kNumRandomKeypointsPerFrame; ++i) {
    raw_keypoints.emplace_back(
        math::randUnif() * f.gray.cols, math::randUnif() * f.gray.rows);
  }
  // Mask for where it is valid to create new keypoints (so its O(1)).
  cv::Mat keypoint_coverage_mask(f.gray.size(), CV_8U, cv::Scalar(0));
  for (int track_idx : f.active_tracks) {
    const Track& track = tracks[track_idx];
    XCHECK(track.frame_to_pixel.count(f.idx));
    const cv::Point2f pixel = track.frame_to_pixel.at(f.idx);
    cv::circle(keypoint_coverage_mask, pixel, kSupressionRadius, cv::Scalar(255), cv::FILLED);
  }

  // Filter the raw keypoints down to those that aren't too close to existing keypoints.
  std::vector<cv::Point2f> new_keypoints;
  for (const cv::Point2f& kp : raw_keypoints) {
    if (opencv::getPixelExtend<uint8_t>(keypoint_coverage_mask, kp.x, kp.y) == 0) {
      new_keypoints.push_back(kp);
      cv::circle(keypoint_coverage_mask, kp, kSupressionRadius, cv::Scalar(255), cv::FILLED);
    }
  }

  // For each new candidate keypoint, run some checks to see if it should be kept.
  for (int i = 0; i < new_keypoints.size(); ++i) {
    // TODO: currently we don't have any extra checks to run right here

    Track t;
    XCHECK_EQ(f.image.type(), CV_8UC3);
    cv::Vec3f color = cv::Vec3f(opencv::getPixelBilinear<cv::Vec3b>(f.image, new_keypoints[i].x, new_keypoints[i].y));
    t.color = Eigen::Vector3f(color[2], color[1], color[0]) / 255.0f;
    
    t.frame_to_pixel[f.idx] = new_keypoints[i];
    // Add the new track.
    t.used_in_bundle_adjustment = true; // This will be updated later, but set to true for visualization purposes for now.
    t.track_weight = 1;
    tracks.push_back(t);
    const int new_track_idx = tracks.size() - 1;
    f.active_tracks.push_back(new_track_idx);
    f.new_tracks.push_back(new_track_idx);
  }
}

void updateTracking(
    Frame& prev_frame,
    Frame& curr_frame,
    std::vector<Track>& tracks
){
  // Track keypoints from the previous frame to the current frame.
  std::vector<cv::Point2f> keypoints_prev, keypoints_curr;
  for (int track_idx : prev_frame.active_tracks) {
    const Track& track = tracks[track_idx];
    auto itr = track.frame_to_pixel.find(prev_frame.idx);
    // If the track's index is in the frame's active track list, it should be in the map.
    XCHECK(itr != track.frame_to_pixel.end());
    keypoints_prev.push_back(itr->second);
  }
  
  std::vector<bool> prev_to_curr_valid;
  keypoint_tracking::trackKeyPointsFromSrcToDest(
      prev_frame.gray,
      curr_frame.gray,
      keypoints_prev,
      keypoints_curr,
      prev_to_curr_valid);

  // Update tracks that survive.
  for (int i = 0; i < prev_frame.active_tracks.size(); ++i) {
    // Skip tracks we couldn't follow from prev to curr
    if (!prev_to_curr_valid[i]) continue;

    const int track_idx = prev_frame.active_tracks[i];
    Track& track = tracks[track_idx];

    // Prevent tracks lasting forever. Add some randomness to prevent all old tracks from being
    // killed at the same time.
    if (track.frame_to_pixel.size() > kMaxTrackAge &&
        math::randUnif() < kKillOldTrackProbability)
      continue;

    // Add this track to the active list for curr_frame and update pixel observations.
    curr_frame.active_tracks.push_back(track_idx);
    track.frame_to_pixel[curr_frame.idx] = keypoints_curr[i];
  }
}

struct SfmReprojectionResidual {
  static constexpr int kResidualDim = 2;

  const calibration::RectilinearCamerad& base_cam;
  const Eigen::Vector2d pixel_observed;
  const double weight;

  SfmReprojectionResidual(
    const calibration::RectilinearCamerad& base_cam,
    const Eigen::Vector2d& pixel_observed,
    const double weight) 
  : base_cam(base_cam), pixel_observed(pixel_observed), weight(weight) {}

  template <typename T>
  bool operator()(
    const T* cam_from_world_param,
    const T* point3d_in_world_param,
    const T* cam_focal_length,
    T* residuals) const
  {
    using Vector3T = Eigen::Matrix<T, 3, 1>;
    using Vector2T = Eigen::Matrix<T, 2, 1>;
    using Pose3T = Eigen::Transform<T, 3, Eigen::Isometry>;

    const std::vector<T> T_cam_from_world_param(
        cam_from_world_param, cam_from_world_param + calibration::kPoseDim);
    const Pose3T cam_from_world = calibration::paramVecToPose(T_cam_from_world_param);

    const Eigen::Map<const Vector3T> point3d_in_world(point3d_in_world_param);

    const Vector3T point3d_in_cam = cam_from_world * point3d_in_world;

    // Clamp z to prevent divide by zero
    const Vector3T point3d_in_cam_clampz(
      point3d_in_cam.x(),
      point3d_in_cam.y(),
      point3d_in_cam.z() < T(0.1) ? T(0.1) : point3d_in_cam.z());

    // TODO: use applyIntrinsicParamVec instead. for now we only have 1 free parameter
    calibration::RectilinearCamera<T> T_cam(base_cam);
    T_cam.focal_length = Vector2T(cam_focal_length[0], cam_focal_length[0]);

    const Vector2T projected_pixel = T_cam.pixelFromCam(point3d_in_cam_clampz);

    residuals[0] = weight * (projected_pixel.x() - T(pixel_observed.x()));
    residuals[1] = weight * (projected_pixel.y() - T(pixel_observed.y()));

    return true;
  }
};

void makeSfmPointCloud(
  const std::vector<Track>& tracks,
  std::vector<Eigen::Vector3f>& sfm_point_cloud, 
  std::vector<Eigen::Vector3f>& point_cloud_colors)
{
  for (const Track& track : tracks) {
    if (track.used_in_bundle_adjustment) {
      sfm_point_cloud.push_back(track.point3d_in_world.cast<float>());
      point_cloud_colors.push_back(track.color);
    }
  }
}

void normalizeSfmSolution(
  std::vector<Frame>& frames,
  std::vector<Eigen::Vector3f>& sfm_point_cloud
) {
  // Unpack and normalize camera trajectory
  Eigen::Vector3d avg_cam_position(0, 0, 0);
  for (auto& f : frames) {
    Eigen::Isometry3d cam_from_world = calibration::paramVecToPose(f.cam_from_world_param);
    f.world_from_cam = cam_from_world.inverse();
    avg_cam_position += f.world_from_cam.translation();
  }
  avg_cam_position /= frames.size();

  // Subtract average camera position from each camera position, to center the trajectory at the origin.
  // Also do the same thing to the SFM point cloud
  float scale = 0;
  for (auto& f : frames) {
    f.world_from_cam.translation() -= avg_cam_position;
    scale += f.world_from_cam.translation().norm();
  }
  scale /= frames.size();
  scale *= 2.0;
  XPLINFO << "scale=" << scale;
  for (auto& f : frames) {
    f.world_from_cam.translation() /= scale;
  }
  for (auto& x : sfm_point_cloud) {
    x -= avg_cam_position.cast<float>();
    x /= scale;
  }
}

// A farthest-first traversal:
// Repeatedly select the next frame whose camera position maximimizes
// its distance to its nearest neighbor in the already selected set.
std::set<int> subsampleDiverseFrames(
  const std::vector<Frame>& frames,
  const int num_to_sample
) {
  std::vector<Frame> selected_frames;
  std::set<int> selected_idxs;

  // the first frame to be added will be the one where the camera is closest to the origin.
  int best_i = 0;
  double min_dist = std::numeric_limits<double>::max();
  for (int i = 0; i < frames.size(); ++i) {
    const Eigen::Vector3d cam_in_world = frames[i].world_from_cam.translation();
    const double dist_from_origin = cam_in_world.norm();
    if (dist_from_origin < min_dist) {
      min_dist = dist_from_origin;
      best_i = i;
    }
  }
  selected_frames.push_back(frames[best_i]);
  selected_idxs.insert(best_i);

  // Keep finding the next best frame until we have enough
  while(selected_idxs.size() < num_to_sample) {
    // consider frame i as a candidate for selection
    int best_i = -1;
    double best_dist = 0;
    for (int i = 0; i < frames.size(); ++i) {
      if (selected_idxs.count(i)) continue; // already selected i, skip

      // find frame i's nearest neighbor in the selected set
      const Eigen::Vector3d cam_in_world_i = frames[i].world_from_cam.translation();
      double nearest_neighbor_dist = std::numeric_limits<double>::max();
      for (int s : selected_idxs) {
        const Eigen::Vector3d cam_in_world_s = frames[s].world_from_cam.translation();
        const double dist = (cam_in_world_i - cam_in_world_s).norm(); 
        nearest_neighbor_dist = std::min(nearest_neighbor_dist, dist);
      }

      // check if this is the 'best' (largest) nearest neighbor distance
      if (nearest_neighbor_dist > best_dist) {
        best_dist = nearest_neighbor_dist;
        best_i = i;
      }
    }
    XCHECK(best_i != -1);

    // Add frame[best_i] to the selected set.
    selected_frames.push_back(frames[best_i]);
    selected_idxs.insert(best_i);
  }

  return selected_idxs;
}

void makeFrameDataJson(
  std::vector<double>& intrinsic_param,
  const std::vector<Frame>& frames,
  const std::string& dest_path
) {
  // Make a JSON file with camera intrinsics and extrinsics.
  using json = nlohmann::json;
  std::vector<json> frame_to_json;
  for (auto& f : frames) {
    json js_frame;

    js_frame["image_filename"] = "frame_" + string::intToZeroPad(f.idx, 6) + ".png";
    js_frame["timestamp"] = 0;

    js_frame["cam_model"] = "rectilinear";
    js_frame["width"] = f.image.cols;
    js_frame["height"] = f.image.rows;
    js_frame["fx"] = intrinsic_param[0];
    js_frame["fy"] = intrinsic_param[0];
    js_frame["cx"] = f.image.cols / 2.0;
    js_frame["cy"] = f.image.rows / 2.0;

    // Extract the camera pose for this frame from the ceres solution.
    std::vector<double> w_from_c_elements(f.world_from_cam.matrix().data(), f.world_from_cam.matrix().data() + f.world_from_cam.matrix().size());
    js_frame["world_from_cam"] = w_from_c_elements;
    frame_to_json.push_back(js_frame);
  }
  
  json all_frames_json;
  all_frames_json["frames_data"] = frame_to_json;
  std::ofstream json_file(dest_path);
  json_file << std::setw(4) << all_frames_json << std::endl;
  json_file.close();
}

// Given a set s containing N + M elements, randomly split it into two vectors containing N and M elements.
// This is used to split up images into training and test sets.
std::pair<std::vector<int>, std::vector<int>> splitSetRandomly(const std::set<int>& s, int N, int M) {
  XCHECK_EQ(s.size(), N + M);

  std::vector<int> s_vec(s.begin(), s.end());

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(s_vec.begin(), s_vec.end(), g);

  std::vector<int> vec_N(s_vec.begin(), s_vec.begin() + N);
  std::vector<int> vec_M(s_vec.begin() + N, s_vec.end());

  return {vec_N, vec_M};
}

// For volurama, we need to capture the ceres progress updates through XPL
class CeresProgressCallback : public ceres::IterationCallback {
public:
  int num_residuals;
  std::shared_ptr<std::atomic<bool>> cancel_requested;
  RectliinearSfmGuiData* gui_data;

  CeresProgressCallback(int num_residuals,
                        std::shared_ptr<std::atomic<bool>> cancel_requested,
                        RectliinearSfmGuiData* gui_data)
      : num_residuals(num_residuals), cancel_requested(cancel_requested), gui_data(gui_data) {}

  ceres::CallbackReturnType operator()(const ceres::IterationSummary& summary) override {
    const float cost = summary.cost / num_residuals; // TODO: this could be RMSE, or something more meaningful
    XPLINFO << summary.iteration << "\t" << cost;

    if (cancel_requested && *cancel_requested) {
      XPLWARN << "Cancel requested, stopping optimization.";
      return ceres::SOLVER_TERMINATE_SUCCESSFULLY;
    }

    if (gui_data) {
      std::lock_guard<std::mutex> guard(gui_data->mutex);
      gui_data->plot_data_x.push_back(summary.iteration);
      gui_data->plot_data_y.push_back(cost);
    }
    return ceres::SOLVER_CONTINUE;
  }
};

std::set<int> solveSfmWithCeres(
  RectilinearSfmConfig& cfg,
  std::vector<Track>& tracks,
  std::vector<Frame>& frames,
  RectliinearSfmGuiData* gui_data = nullptr
) {
  constexpr int kMinTrackAgeForSfm = 10;
  constexpr int kMaxTrackAgeForSfm = 10000;

  // Initial guess for focal length.
  std::vector<double> intrinsic_param(1);
  intrinsic_param[0] = (frames[0].image.cols + frames[0].image.rows) / 2.0;

  calibration::RectilinearCamerad base_cam;
  base_cam.width = frames[0].image.cols;
  base_cam.height = frames[0].image.rows;
  base_cam.focal_length = Eigen::Vector2d(intrinsic_param[0], intrinsic_param[0]);
  base_cam.optical_center = Eigen::Vector2d(base_cam.width/2.0, base_cam.height/2.0);

  // Initialize camera poses. NOTE this only happens once, not once per outer iteration!
  for (Frame& f : frames) {
    f.cam_from_world_param = std::vector<double>(calibration::kPoseDim, 0);
  }

  // NOTE: we must avoid initializing with z = 0, that causes NaNs.
  // TODO: better initial guess at 3D point in world. even just using ray direction + constant depth is better than all 0's
  for (Track& track : tracks) {
    track.point3d_in_world = Eigen::Vector3d(0, 0, 1);
  }

  std::vector<double> residual_to_err;
  for (int outer_itr = 0; outer_itr < cfg.outer_iterations; ++outer_itr) {
    XPLINFO << "Phase: Solve structure from motion, outer iteration: " << outer_itr;
    const int ceres_iterations = outer_itr == 0 ? cfg.ceres_iterations : cfg.ceres_iterations2;;

    // Clear the graph of cost vs inner iteration (if we have a gui)
    if (gui_data) {
      std::lock_guard<std::mutex> guard(gui_data->mutex);
      gui_data->plot_data_x.clear();
      gui_data->plot_data_y.clear();
      gui_data->ceres_iterations = ceres_iterations;
    }

    ceres::Problem problem;
    int num_sfm_tracks = 0;
    int num_sfm_residuals = 0;
    for (int track_idx = 0; track_idx < tracks.size(); ++track_idx) {
      Track& track = tracks[track_idx];

      // Skip tracks that don't exist for long enough.
      // TODO: skip tracks where too many observations have been removed as outliers?
      track.used_in_bundle_adjustment = false;
      const int track_age = track.frame_to_pixel.size();
      if (track_age < kMinTrackAgeForSfm || track_age > kMaxTrackAgeForSfm) continue;
      track.used_in_bundle_adjustment = true;

      // Make a residual for each observation in a track.
      for (const auto& kv : track.frame_to_pixel) {
        const int frame_idx = kv.first;
        const cv::Point2f& observed_pixel = kv.second;

        auto res_func = new SfmReprojectionResidual(base_cam, Eigen::Vector2d(observed_pixel.x, observed_pixel.y), track.track_weight);
        auto cost = new ceres::AutoDiffCostFunction<
                SfmReprojectionResidual,
                SfmReprojectionResidual::kResidualDim,
                calibration::kPoseDim,
                3,
                1>(res_func);
        problem.AddResidualBlock(
            cost,
            new ceres::CauchyLoss(1.0), // new ceres::HuberLoss(1.0),
            frames[frame_idx].cam_from_world_param.data(),
            track.point3d_in_world.data(),
            intrinsic_param.data());

        ++num_sfm_residuals;
        if (cfg.cancel_requested && (*cfg.cancel_requested)) {
            XPLWARN << "Cancel requested, aborting.";
            return std::set<int>();
        }
      }
      ++num_sfm_tracks;
    }

    // Lock the first camera pose, otherwise the problem is underconstrained.
    // TODO: this might work better with a motion prior, as-is it seems like it just causes the first frame to be junk
    //problem.SetParameterBlockConstant(frames[0].cam_from_world_param.data());

    XPLINFO << "#frames: "<< frames.size();
    XPLINFO << "# sfm_tracks: " << num_sfm_tracks;
    XPLINFO << "# sfm_residuals: " << num_sfm_residuals;

    CeresProgressCallback progress_callback(num_sfm_residuals, cfg.cancel_requested, gui_data);

    ceres::Solver::Options options;
    options.num_threads = 1;
    options.use_nonmonotonic_steps = false;
    options.gradient_tolerance = 1e-15;
    options.function_tolerance = 1e-15;
    options.minimizer_progress_to_stdout = false;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = ceres_iterations;
    options.callbacks.push_back(&progress_callback);
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    XPLINFO << summary.BriefReport();

    // Compute reprojection error RMSE
    double cost = 0;
    std::vector<double> residuals;
    auto eval_opts = ceres::Problem::EvaluateOptions();
    eval_opts.apply_loss_function = false;
    problem.Evaluate(eval_opts, &cost, &residuals, nullptr, nullptr);

    double sse = 0;
    residual_to_err.clear();
    for (int i = 0; i < residuals.size(); i += 2) { // Note there are 2 dimensions to each residual
      const double dx = residuals[i];
      const double dy = residuals[i + 1];
      const double square_err = dx * dx + dy * dy;
      sse += square_err;
      residual_to_err.push_back(square_err);
    }
  
    const double rmse = std::sqrt(sse / num_sfm_residuals);
    XPLINFO << "reprojection RMSE [pixels]: " << rmse;

    // Update residual and track weights
    if (outer_itr == 0) {
      int residual_idx = 0;

      std::vector<double> track_errs;
      for (Track& track : tracks) {
        if (!track.used_in_bundle_adjustment) continue;
    
        track.avg_observation_err = 0;
        for (int i = 0; i < track.frame_to_pixel.size(); ++i) {
          track.avg_observation_err += residual_to_err[residual_idx++];
        }
        track.avg_observation_err /= track.frame_to_pixel.size();
        track_errs.push_back(track.avg_observation_err);
      }

      const double soft_threshold = math::percentile(track_errs, cfg.outlier_percentile);
      XPLINFO << "soft track outlier threshold=" << soft_threshold;

      for (Track& track : tracks) {
        if (!track.used_in_bundle_adjustment) continue;
        track.track_weight = (std::tanh(cfg.outlier_weight_steepness * (soft_threshold - track.avg_observation_err)) + 1) * 0.5;
      }
    }
  }
  if (cfg.cancel_requested && (*cfg.cancel_requested)) {
      XPLWARN << "Cancel requested, aborting.";
      return std::set<int>();
  }

  XPLINFO << "optimized focal length [pixels]: " << intrinsic_param[0];

  std::vector<Eigen::Vector3f> sfm_point_cloud, point_cloud_colors;
  makeSfmPointCloud(tracks, sfm_point_cloud, point_cloud_colors);

  normalizeSfmSolution(frames, sfm_point_cloud);

  // Save the SFM point cloud.
  point_cloud::savePointCloudBinary(
    cfg.dest_dir + "/pointcloud_sfm.bin", sfm_point_cloud, point_cloud_colors);

  // Make a JSON file with camera poses and intrinsics for each frame
  makeFrameDataJson(intrinsic_param, frames, cfg.dest_dir + "/dataset_all.json");

  const std::set<int> diverse_frame_idxs = subsampleDiverseFrames(frames, cfg.num_train_frames + cfg.num_test_frames);
  
  const auto& [train_idxs, test_idxs] = splitSetRandomly(diverse_frame_idxs, cfg.num_train_frames, cfg.num_test_frames);

  std::vector<Frame> train_frames, test_frames;
  for (int i : train_idxs) { train_frames.push_back(frames[i]); }
  for (int i : test_idxs) { test_frames.push_back(frames[i]); }

  makeFrameDataJson(intrinsic_param, train_frames, cfg.dest_dir + "/dataset_train.json");
  makeFrameDataJson(intrinsic_param, test_frames, cfg.dest_dir + "/dataset_test.json");

  return diverse_frame_idxs;
}

void runRectilinearSfmPipeline(RectilinearSfmConfig& cfg, RectliinearSfmGuiData* gui_data) {
  if (cfg.video_frames_dir.empty()) cfg.video_frames_dir = cfg.dest_dir;

  // TODO: -p is not working on windows
  std::system(std::string("mkdir -p \"" + cfg.dest_dir + "\"").c_str());
  std::system(std::string("rm \"" + cfg.dest_dir + "/*.json\"").c_str());

  // The first part of the pipeline is all about building up tracks
  std::vector<Track> tracks;
  std::vector<Frame> frames;
 
  // We will use ffmpeg to unpack the video file to properly handle the HDR color space conversion from iPhone footage.
  // These parameters are not optimal for other phones or necessarily even all phones. They are chosen based on ffprobing
  // one video file from an iPhone 14 Max.
  const std::string scale_max_dim_maintain_aspect = "scale=" + std::to_string(cfg.max_image_dim) + ":" + std::to_string(cfg.max_image_dim) + ":force_original_aspect_ratio=decrease";
  const std::string ffmpeg_iphone = 
      cfg.ffmpeg + " -i " + cfg.src_vid +
      " -vf \"zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p," +
      scale_max_dim_maintain_aspect + "\" -start_number 0 \"" + cfg.video_frames_dir + "/frame_%06d.png\"";
  if (!cfg.no_ffmpeg) {
    std::system(std::string("rm \"" + cfg.video_frames_dir + "/frame_*.png\"").c_str());

    XPLINFO << ffmpeg_iphone;
    std::system(ffmpeg_iphone.c_str());
  }

  if (cfg.cancel_requested && (*cfg.cancel_requested)) return;

  const int num_frames = file::countMatchingFiles(cfg.video_frames_dir, "^frame_([0-9]{6})\\.png");
  XPLINFO << "num_frames: " << num_frames;
  XCHECK(num_frames > 0) << "No images found in " << cfg.video_frames_dir;

  XPLINFO << "Phase: Tracking keypoints";

  for (int frame_index = 0; frame_index < num_frames; ++frame_index) {
    XPLINFO << "Tracking keypoints, frame: " << frame_index << " / " << num_frames;
    cv::Mat frame = cv::imread(cfg.video_frames_dir + "/frame_" + string::intToZeroPad(frame_index, 6) + ".png");

    frames.push_back(Frame());
    Frame& f = frames.back();
    f.idx = frame_index;
    f.image =  frame;

    preprocessFrameForTracking(
      f,
      frame_index > 0
        ? frames[frame_index - 1].brightness
        : -1);

    // Do keypoint tracking.
    if (frame_index != 0) {
      updateTracking(frames[frame_index - 1], f, tracks);
    }

    makeNewTracks(f, tracks);

    if (cfg.cancel_requested && (*cfg.cancel_requested)) return;
  }

  const std::set<int> diverse_frame_idxs = solveSfmWithCeres(cfg, tracks, frames, gui_data);
  if (cfg.cancel_requested && (*cfg.cancel_requested)) return;

  XPLINFO << "Phase: Generating tracking and outlier visualization";
  for (int frame_index = 0; frame_index < num_frames; ++frame_index) {
    cv::Mat viz = visualizeTracks(frames[frame_index], tracks);
    cv::imwrite(cfg.dest_dir + "/track_" + string::intToZeroPad(frame_index, 6) + ".jpg", viz);
    if (cfg.cancel_requested && (*cfg.cancel_requested)) return;
  }
  const std::string tracking_ffmpeg_command = 
    cfg.ffmpeg + " -y -framerate 30 -i " + cfg.dest_dir + "/track_%06d.jpg -c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p -g 1 " + cfg.dest_dir + "/tracking.mp4";  
  XPLINFO << tracking_ffmpeg_command;
  std::system(tracking_ffmpeg_command.c_str());
  
  file::crossPlatformDelete(cfg.dest_dir + "/track_*.jpg");

  if (cfg.rm_unused_images) {
    XPLINFO << "Phase: Cleaning up unused images";
    for (const auto& frame : frames) {
      if (diverse_frame_idxs.count(frame.idx) == 0) {
        file::crossPlatformDelete(cfg.video_frames_dir + "/frame_" + string::intToZeroPad(frame.idx, 6) + ".png"); 
      }
    }
  }
}

}}  // end namespace p11::rectilinear_sfm
