// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once
#include "rof.h" // This seems like it should be in the .cc file but that wont compile for some reason.
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/photo.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include "torch/torch.h"
#include "torch/script.h"
#include "third_party/json.h"
#include "third_party/vfc.h"
#include "logger.h"
#include "check.h"
#include "fisheye_camera.h"
#include "rectilinear_camera.h"
#include "util_opencv.h"
#include "pose_param.h"
#include "point_cloud.h"
#include "util_math.h"
#include "util_time.h"
#include "util_file.h"
#include "depth_anything2.h"
#include "keypoint_tracking.h"
#include "multicamera_dataset.h"


namespace p11 { namespace calibration { namespace incremental_sfm {

static constexpr int kMinObservationsPerTrack = 2;
static constexpr double kOutlierThresholdPix = 5.0; // we aim to discard outilers if their reprojection error is above this
static constexpr double kOutlierWeightSteepness = 2.0;
static constexpr float kZFar = 1000.0;

struct IncrementalSfmGuiData {
  std::mutex mutex;
  std::vector<calibration::RectilinearCamerad> viz_cameras; // TODO: fisheye?
  std::vector<Eigen::Vector3f> point_cloud;
  std::vector<Eigen::Vector4f> point_cloud_colors;
  std::atomic<bool> pointcloud_needs_update = false;
};

struct IncrementalSfmObservation {
  int img_idx, kp_idx;
  cv::Point2f pixel;
  IncrementalSfmObservation(int img_idx, int kp_idx, const cv::Point2f& pixel) : img_idx(img_idx), kp_idx(kp_idx), pixel(pixel) {}
};

struct IncrementalSfmTrack {
  std::vector<IncrementalSfmObservation> observations;
  Eigen::Vector3d point3d;
  double weight;
  bool has_estimated_3d; // initially false, only becomes true if the track's 3D position is estimated in bundle adjustment, or mono depth
  bool pruned;
  bool skipped_low_obs; // for visualization, set to true if the track is skipped due to low observation count
};

template<typename TCamera>
struct IncrementalSfmReprojectionResidual {
  static constexpr int kResidualDim = 2;

  const TCamera& base_cam;
  const Eigen::Vector2d pixel_observed;
  const double weight;

  IncrementalSfmReprojectionResidual(
    const TCamera& base_cam,
    const Eigen::Vector2d& pixel_observed,
    const double weight) 
  : base_cam(base_cam), pixel_observed(pixel_observed), weight(weight) {}

  template <typename T>
  bool operator()(
    const T* cam_from_world_param,
    const T* point3d_in_world_param,
    const T* intrinsic_param,
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

    typename TCamera::template rebind<T>::type T_cam(base_cam); // TCamera<T> T_cam(base_cam);

    const std::vector<T> T_intrinsic_param(
          intrinsic_param, intrinsic_param + TCamera::kIntrinsicDim);
        T_cam.applyIntrinsicParamVec(T_intrinsic_param);

    const Vector2T projected_pixel = T_cam.pixelFromCam(point3d_in_cam_clampz);

    residuals[0] = weight * (projected_pixel.x() - T(pixel_observed.x()));
    residuals[1] = weight * (projected_pixel.y() - T(pixel_observed.y()));

    return true;
  }
};


template<typename TCamera>
struct IncrementalSfmDepthResidual {
  static constexpr int kResidualDim = 1;

  const TCamera& base_cam;
  const double weight, mono_depth;

  IncrementalSfmDepthResidual(
    const TCamera& base_cam,
    const double weight,
    const double mono_depth) 
  : base_cam(base_cam),weight(weight),  mono_depth(mono_depth) {}

  template <typename T>
  bool operator()(
    const T* cam_from_world_param,
    const T* point3d_in_world_param,
    const T* intrinsic_param,
    T* residuals) const
  {
    using Vector3T = Eigen::Matrix<T, 3, 1>;
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

    typename TCamera::template rebind<T>::type T_cam(base_cam); // TCamera<T> T_cam(base_cam);

    const std::vector<T> T_intrinsic_param(
          intrinsic_param, intrinsic_param + TCamera::kIntrinsicDim);
        T_cam.applyIntrinsicParamVec(T_intrinsic_param);

    residuals[0] = weight * (point3d_in_cam.z() - T(mono_depth));

    return true;
  }
};

template<typename TCamera>
struct IncrementalSfmFocalLengthPriorResidual {
  static constexpr int kResidualDim = 1; // Single focal length parameter

  const double prior_focal_length;
  const double stddev;
  const double weight;

  IncrementalSfmFocalLengthPriorResidual(
    const double prior_focal_length, 
    const double stddev,
    const double weight) 
  : prior_focal_length(prior_focal_length), stddev(stddev), weight(weight) {}

  template <typename T>
  bool operator()(const T* intrinsic_param, T* residuals) const {
    const T current_focal_length = intrinsic_param[0];

    residuals[0] = weight * (current_focal_length - T(prior_focal_length)) / T(stddev);

    return true;
  }
};


template<typename TCamera>
void matchKeypointsBetweenAllImagePairs(
  std::shared_ptr<std::atomic<bool>> cancel_requested,
  const std::vector<cv::Mat>& images,
  const std::vector<TCamera>& camera_intrinsics,
  const std::string& debug_dir,
  const bool show_keypoints,
  const bool show_matches,
  std::vector<std::vector<cv::KeyPoint>>& image_to_keypoints,
  std::vector<std::vector<cv::DMatch>>& image_to_matches,
  const float flow_err_threshold,
  const float match_ratio_threshold,
  const int time_window_size,
  const bool filter_with_flow = false
) {
  XCHECK_EQ(images.size(), camera_intrinsics.size());

  torch::jit::getProfilingMode() = false;
  torch::jit::script::Module raft_module;
  if (filter_with_flow) {
    optical_flow::getTorchModelRAFT(raft_module);
  }

  //constexpr int kMaxSiftKeypointsPerImage = 4096;
  constexpr int kMaxSiftKeypointsPerImage = 8192;
  cv::Ptr<cv::SIFT> sift = cv::SIFT::create(kMaxSiftKeypointsPerImage);

  constexpr int kFlowImageSize = 512;
  std::vector<cv::Mat> image_to_descriptors;
  std::vector<cv::Mat> small_images;
  for (int i = 0; i < images.size(); ++i) {
    if (cancel_requested && *cancel_requested) return;

    XPLINFO << "Extracting keypoints and descriptors for image " << i << " / " << images.size();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    XCHECK_EQ(images[i].type(), CV_32FC3);
    cv::Mat image8uc3;
    images[i].convertTo(image8uc3, CV_8UC3, 255.0);
    
    if (filter_with_flow) {
      cv::Mat small;
      cv::resize(image8uc3, small, cv::Size(kFlowImageSize, kFlowImageSize), 0, 0, cv::INTER_AREA);
      small_images.push_back(small);
    }

    sift->detectAndCompute(image8uc3, cv::noArray(), keypoints, descriptors);

    // Deduplicate keypoints before further processing
    std::vector<cv::KeyPoint> deduplicated_keypoints;
    cv::Mat deduplicated_descriptors;
    
    for (int j = 0; j < keypoints.size(); ++j) {
      const cv::KeyPoint& kp = keypoints[j];
      
      // Check if this keypoint is too close to any already accepted keypoint
      bool is_duplicate = false;
      for (const cv::KeyPoint& existing_kp : deduplicated_keypoints) {
        float dist = cv::norm(kp.pt - existing_kp.pt);
        if (dist < 0.1f) { // Within 0.1 pixel - treat as duplicate
          is_duplicate = true;
          break;
        }
      }
      
      if (!is_duplicate) {
        deduplicated_keypoints.push_back(kp);
        deduplicated_descriptors.push_back(descriptors.row(j));
      }
    }
    
    XPLINFO << "SIFT keypoints for image " << i << ": " << keypoints.size() 
            << " -> " << deduplicated_keypoints.size() << " (after deduplication)";
    
    // Replace original with deduplicated
    keypoints = std::move(deduplicated_keypoints);
    descriptors = deduplicated_descriptors.clone();

    // Filter out keypoints outside of useable radius of the image.
    std::vector<cv::KeyPoint> filtered_keypoints;
    cv::Mat filtered_descriptors;
    for (int j = 0; j < keypoints.size(); ++j) {
      cv::KeyPoint& kp = keypoints[j];
      if constexpr (is_fisheye_camera_v<TCamera>) {
        if ((Eigen::Vector2d(kp.pt.x, kp.pt.y) - camera_intrinsics[i].optical_center).norm() > camera_intrinsics[i].useable_radius) {
          continue;
        }
      }
      filtered_keypoints.push_back(kp);
      filtered_descriptors.push_back(descriptors.row(j));
    }

    image_to_keypoints.push_back(filtered_keypoints);
    image_to_descriptors.push_back(filtered_descriptors);

    if (show_keypoints) {
      cv::Mat viz;
      cv::drawKeypoints(image8uc3, filtered_keypoints, viz, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
      cv::imshow("viz", viz); cv::waitKey(0);
    }
  }

  int keyframe_interval = 5; // matching runs on all pairs of keyframes

  // Do keypoint matching between all pairs of images
  for (int i = 0; i < images.size(); ++i) {
    if (cancel_requested && *cancel_requested) return;

    bool i_is_keyframe = ((i % keyframe_interval) == 0) || (i == images.size() - 1);

    std::vector<cv::DMatch> final_matches_in_i;
    for (int j = i + 1; j < images.size(); ++j) {
      if (cancel_requested && *cancel_requested) return;

      bool j_is_keyframe = ((j % keyframe_interval) == 0) || (j == images.size() - 1);
      bool match_outside_window = i_is_keyframe && j_is_keyframe;

      // Skip some image pairs if we do time-window matching
      // TODO: add keyframes that get matched with everything?
      if (!match_outside_window && time_window_size > 0 && std::abs(i - j) > time_window_size) continue; 

      //XPLINFO << "Computing optical flow between images";
      cv::Mat flow_x, flow_y;
      if (filter_with_flow) {
        optical_flow::computeOpticalFlowRAFT(raft_module, small_images[i], small_images[j], flow_x, flow_y);
        cv::resize(flow_x, flow_x, images[i].size());
        cv::resize(flow_y, flow_y, images[i].size());
        flow_x *= float(images[i].cols) / float(small_images[i].cols); // When resizing flow, we need to scale it
        flow_y *= float(images[i].rows) / float(small_images[i].rows);
      }
  
      XPLINFO << "Matching keypoints between image pair: " << i << ", " << j;
      cv::FlannBasedMatcher matcher;
      std::vector<std::vector<cv::DMatch>> knn_matches;
      matcher.knnMatch(image_to_descriptors[i], image_to_descriptors[j], knn_matches, /*num_neighbors=*/2);

      XPLINFO << "# raw matches: " << knn_matches.size();

      // Filter out matches that fail either the ratio test or the flow consistency test
      std::vector<cv::DMatch> good_matches;
      for (const auto& knn_match : knn_matches) {
        // First, apply ratio test
        if (knn_match[0].distance >= match_ratio_threshold * knn_match[1].distance) {
          continue;  // Skip this match if it fails the ratio test
        }

        if (filter_with_flow) {
          // Apply RAFT flow filtering
          const cv::KeyPoint& kp1 = image_to_keypoints[i][knn_match[0].queryIdx];
          const cv::KeyPoint& kp2 = image_to_keypoints[j][knn_match[0].trainIdx];
          
          float flow_x_val = flow_x.at<float>(static_cast<int>(kp1.pt.y), static_cast<int>(kp1.pt.x));
          float flow_y_val = flow_y.at<float>(static_cast<int>(kp1.pt.y), static_cast<int>(kp1.pt.x));
          
          cv::Point2f predicted_pt(kp1.pt.x + flow_x_val, kp1.pt.y + flow_y_val);
          float distance = cv::norm(predicted_pt - kp2.pt);

          if (distance <= flow_err_threshold) {
            good_matches.push_back(knn_match[0]);
          }
        } else {
          good_matches.push_back(knn_match[0]);
        }
      }

      XPLINFO << "# matches after combined ratio test and flow filtering: " << good_matches.size();

      // Cross-check matching
      std::vector<std::vector<cv::DMatch>> knn_matches_rev;
      matcher.knnMatch(image_to_descriptors[j], image_to_descriptors[i], knn_matches_rev, /*num_neighbors=*/2);
      std::vector<cv::DMatch> good_matches_rev;
      for (const auto& knn_match : knn_matches_rev) {
        if (knn_match[0].distance < match_ratio_threshold * knn_match[1].distance) {
          good_matches_rev.push_back(knn_match[0]);
        }
      }
  
      std::vector<cv::DMatch> cross_matches;
      for (const auto& match : good_matches) {
        for (const auto& match_rev : good_matches_rev) {
          if (match.queryIdx == match_rev.trainIdx && match.trainIdx == match_rev.queryIdx) {
            cross_matches.push_back(match);
            break;
          }
        }
      }

      XPLINFO << "# matches after cross-checking: " << cross_matches.size();

      // Prepare points for VFC
      std::vector<cv::Point2f> points_i, points_j;
      for (const auto& match : cross_matches) {
        points_i.push_back(image_to_keypoints[i][match.queryIdx].pt);
        points_j.push_back(image_to_keypoints[j][match.trainIdx].pt);
      }

      // Perform VFC filtering
      VFC vfc;
      vfc.setData(points_i, points_j);
      vfc.optimize();
      std::vector<int> inliers = vfc.obtainCorrectMatch();
      XPLINFO << "# inliers from VFC (final # matches between image i and j): " << inliers.size();

      // Filter matches based on VFC inliers
      std::vector<cv::DMatch> final_matches_between_i_and_j; // Just for viz
      for (int idx : inliers) {
        cross_matches[idx].imgIdx = j;
        final_matches_in_i.push_back(cross_matches[idx]);
        final_matches_between_i_and_j.push_back(cross_matches[idx]);
      }
      XPLINFO << "# running total matches for image i=" << i << " is: " << final_matches_in_i.size();

      if (show_matches) {
        XCHECK_EQ(images[i].type(), CV_32FC3);
        cv::Mat image8uc3_i, image8uc3_j;
        images[i].convertTo(image8uc3_i, CV_8UC3, 255.0);
        images[j].convertTo(image8uc3_j, CV_8UC3, 255.0);

        cv::Mat viz;
        cv::drawMatches(image8uc3_i, image_to_keypoints[i], image8uc3_j, image_to_keypoints[j], final_matches_between_i_and_j, viz, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imshow("viz", viz); cv::waitKey(0);
      }
    }

    if (final_matches_in_i.size() == 0 && i != images.size() - 1) { // There will be zero in the last image (not an issue).
      XPLINFO << "WARNING: 0 keypoint matches to image i: " << i << " cam.name: " << camera_intrinsics[i].name;
      //XCHECK(false);
    }
    image_to_matches.push_back(final_matches_in_i);
  }
}

// tracks[r][s] -- r'th track, s'th observation, .first = image index, .second = pixel coordinate
// i - image index
// j - second image index
// p - indexes a keypoint in image i
// q - indexes a keypoint in image j
std::vector<IncrementalSfmTrack> buildTracks(
  const std::vector<std::vector<cv::KeyPoint>>& image_to_keypoints,
  const std::vector<std::vector<cv::DMatch>>& image_to_matches
);


// returns false if the SFM problem it builds has 0 residuals, indicating a camera with no matches
template<typename TCamera>
bool bundleAdjustment(
  const std::string& debug_dir,
  const int num_active_cameras,
  const int first_unlocked_camera,
  const int last_unlocked_camera,
  const bool lock_3d_points,
  const bool lock_all_cameras,
  const bool optimize_intrinsics,
  const bool only_optimize_tracks_with_estimated_3d,
  const bool share_intrinsics,
  const bool use_intrinsic_prior,
  const int num_outer_itrs,
  const float min_inlier_frac,
  const float depth_weight,
  const int max_solver_itrs,
  const std::vector<cv::Mat>& images,
  const std::vector<cv::Mat>& mono_depthmaps,
  const std::vector<TCamera>& base_cameras,
  std::vector<std::vector<double>>& camera_to_pose_param,
  std::vector<std::vector<double>>& camera_to_intrinsic_param,
  std::vector<TCamera>& optimized_cameras,
  std::vector<IncrementalSfmTrack>& tracks
) {
  // Start with all tracks having a weight of 1 in the first outer iteration.
  for (auto& t : tracks) { 
    t.weight = t.pruned ? 0.0 : 1.0;
    t.skipped_low_obs = false;
  }

  for (int outer_itr = 0; outer_itr < num_outer_itrs; ++outer_itr) {
    XPLINFO << "-- Outer iteration: " << outer_itr;

    ceres::Problem problem;
    int num_sfm_residuals = 0;

    std::vector<int> residual_idx_to_track_idx;
    std::set<int> valid_track_idxs;
    for (int track_idx = 0; track_idx < tracks.size(); ++track_idx) {
      IncrementalSfmTrack& track = tracks[track_idx];
      
      if (track.pruned) continue; // skip tracks that were behind cameras

      // Skip tracks without estimated 3D if required
      if (only_optimize_tracks_with_estimated_3d && !track.has_estimated_3d) continue;

      // Count observations in active cameras and unlocked cameras to determine if the track
      // should be included
      int num_obs_in_unlocked_cameras = 0;
      int num_obs_in_active_cameras = 0;
      for (const IncrementalSfmObservation& obs : track.observations) {
        if (obs.img_idx >= num_active_cameras) continue;
        num_obs_in_active_cameras++;
        
        if (obs.img_idx >= first_unlocked_camera && obs.img_idx <= last_unlocked_camera) {
          num_obs_in_unlocked_cameras++;
        }
      }
      // Need multiple observations for triangulation and at least one in unlock window
      if (num_obs_in_active_cameras < kMinObservationsPerTrack || num_obs_in_unlocked_cameras < 1) {
        tracks[track_idx].skipped_low_obs = true;
        continue;
      }

      valid_track_idxs.insert(track_idx);

      // For selected tracks, include ALL observations from active cameras
      for (IncrementalSfmObservation& obs : track.observations) {
        if (obs.img_idx >= num_active_cameras) continue; // Skip observations from inactive cameras

        auto res_func = new IncrementalSfmReprojectionResidual(
          base_cameras[obs.img_idx],
          Eigen::Vector2d(obs.pixel.x, obs.pixel.y), track.weight);
        auto cost = new ceres::AutoDiffCostFunction<
          IncrementalSfmReprojectionResidual<TCamera>,
          IncrementalSfmReprojectionResidual<TCamera>::kResidualDim,
          calibration::kPoseDim,
          3, // For points in 3D space
          TCamera::kIntrinsicDim>(res_func);

        problem.AddResidualBlock(
          cost,
          new ceres::CauchyLoss(1.0),
          camera_to_pose_param[obs.img_idx].data(),
          track.point3d.data(),
          camera_to_intrinsic_param[share_intrinsics ? 0 : obs.img_idx].data());

        if (!lock_3d_points) {
          track.has_estimated_3d = true; // point can gain a proper 3d estimate here if not locked
        }
        
        residual_idx_to_track_idx.push_back(track_idx);
        ++num_sfm_residuals;


        // Add depth residuals
        float mono_depth = opencv::getPixelBilinear<float>(
          mono_depthmaps[obs.img_idx], obs.pixel.x, obs.pixel.y);     
      
        auto depth_res_func = new IncrementalSfmDepthResidual(
          base_cameras[obs.img_idx],
          track.weight * depth_weight,
          mono_depth);
        auto depth_cost = new ceres::AutoDiffCostFunction<
          IncrementalSfmDepthResidual<TCamera>,
          IncrementalSfmDepthResidual<TCamera>::kResidualDim,
          calibration::kPoseDim,
          3, // For points in 3D space
          TCamera::kIntrinsicDim>(depth_res_func);
        problem.AddResidualBlock(
          depth_cost,
          new ceres::CauchyLoss(1.0),
          camera_to_pose_param[obs.img_idx].data(),
          track.point3d.data(),
          camera_to_intrinsic_param[share_intrinsics ? 0 : obs.img_idx].data());

      }
      
      if (lock_3d_points && problem.HasParameterBlock(track.point3d.data())) {
        problem.SetParameterBlockConstant(track.point3d.data());
      }
    }
  
    if (num_sfm_residuals == 0) return false;

    // Add focal length prior for rectilinear cameras to prevent intrinsic collapse
    if constexpr (is_rectilinear_camera_v<TCamera>) {
      if (optimize_intrinsics && use_intrinsic_prior) {
        static constexpr double kFocalLengthPriorWeight = 1.0; // Adjust weight as needed
        static constexpr double kFocalLengthStddevPercent = 0.10; // 10% uncertainty

        // Add focal length prior for each intrinsic parameter block
        const int num_intrinsic_blocks = share_intrinsics ? 1 : num_active_cameras;
        int num_priors_added = 0;
        for (int i = 0; i < num_intrinsic_blocks; ++i) {
          // CRITICAL: Only add prior if the parameter block already exists in the problem
          if (problem.HasParameterBlock(camera_to_intrinsic_param[i].data())) {

            // Use each camera's own initial focal length as prior (when sharing, use camera 0)
            const double prior_focal_length = base_cameras[share_intrinsics ? 0 : i].focal_length.x();
            const double focal_length_stddev = prior_focal_length * kFocalLengthStddevPercent;
        
            auto focal_prior_func = new IncrementalSfmFocalLengthPriorResidual<TCamera>(
              prior_focal_length, focal_length_stddev, kFocalLengthPriorWeight);
              
            auto focal_prior_cost = new ceres::AutoDiffCostFunction<
              IncrementalSfmFocalLengthPriorResidual<TCamera>,
              IncrementalSfmFocalLengthPriorResidual<TCamera>::kResidualDim,
              TCamera::kIntrinsicDim>(focal_prior_func);
    
            problem.AddResidualBlock(
              focal_prior_cost,
              nullptr, // No robust loss for priors
              camera_to_intrinsic_param[i].data());
              
            num_priors_added++;
          }
        }
        
        //XPLINFO << "Added focal length priors, weight=" << kFocalLengthPriorWeight
        //        << ", blocks=" << num_priors_added;
      }
    }

    XPLINFO << "# tracks: " << tracks.size();
    XPLINFO << "# sfm_residuals: " << num_sfm_residuals;

    // Lock/unlock camera poses based on the optimization window
    if (problem.HasParameterBlock(camera_to_pose_param[0].data())) {
      problem.SetParameterBlockConstant(camera_to_pose_param[0].data()); // always lock the first camera
    }
    
    for (int i = 1; i < num_active_cameras; ++i) {
      if (i < first_unlocked_camera || i > last_unlocked_camera) {
        if (problem.HasParameterBlock(camera_to_pose_param[i].data())) {
          problem.SetParameterBlockConstant(camera_to_pose_param[i].data());
        }
      }
    }

    if (lock_all_cameras) {
      for (int i = 0; i < camera_to_pose_param.size(); ++i) {
        if (problem.HasParameterBlock(camera_to_pose_param[i].data())) {
          problem.SetParameterBlockConstant(camera_to_pose_param[i].data());
        }
      }
    }
    
    // Lock/unlock intrinsics
    for (int i = 0; i < camera_to_intrinsic_param.size(); ++i) {
      if (!optimize_intrinsics && problem.HasParameterBlock(camera_to_intrinsic_param[i].data())) {
        problem.SetParameterBlockConstant(camera_to_intrinsic_param[i].data());
      }
    }

    ceres::Solver::Options options;
    options.num_threads = 8;
    //options.use_nonmonotonic_steps = false;
    options.use_nonmonotonic_steps = true;
    options.gradient_tolerance = 1e-15;
    options.function_tolerance = 1e-15;
    //options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = max_solver_itrs;
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
    std::vector<double> residual_to_err;

    constexpr int resdim = 2 + 1; // reprojection + depth
    XCHECK_GE(residuals.size(), num_sfm_residuals * resdim);
    float scale_err = 0;
    for (int i = 0; i < num_sfm_residuals * resdim; i += resdim) {       
      const double dx = residuals[i];
      const double dy = residuals[i + 1];
      const double square_err = dx * dx + dy * dy;
      sse += square_err;
      residual_to_err.push_back(square_err);

      scale_err += residuals[i + 2] * residuals[i + 2] 
        / (depth_weight * depth_weight); // NOTE: not quite right, we need the track weight?
    }
    scale_err /= num_sfm_residuals;
    scale_err = std::sqrt(scale_err);
    XPLINFO << "scale error RMSE [meters]:" << scale_err;

    const double rmse = std::sqrt(sse / num_sfm_residuals);
    XPLINFO << "reprojection RMSE [pixels]: " << rmse;

    if (share_intrinsics) {
      for (int i = 1; i < camera_to_intrinsic_param.size(); ++i) {
        camera_to_intrinsic_param[i] = camera_to_intrinsic_param[0];
      }
    }

    optimized_cameras = base_cameras;
    for (int i = 0; i < base_cameras.size(); ++i) {
      optimized_cameras[i].cam_from_world = calibration::paramVecToPose(camera_to_pose_param[i]);
      optimized_cameras[i].applyIntrinsicParamVec(camera_to_intrinsic_param[i]);
    }

    // Update track weights before the next outer iteration, but don't do it on
    // the last iteration so we preserve the weights for visualization.
    if (outer_itr == 0) {
      // For tracks that participated in optimization, compute their
      //  true reprojection error directly (no weights involved)
      std::vector<double> track_to_sum_err(tracks.size(), 0.0);
      std::vector<int> track_to_num_obs(tracks.size(), 0);
      for (int track_idx = 0; track_idx < tracks.size(); ++track_idx) {
        if (valid_track_idxs.count(track_idx) == 0) continue;
  
        IncrementalSfmTrack& track = tracks[track_idx];
        for (const IncrementalSfmObservation& obs : track.observations) {
          if (obs.img_idx >= num_active_cameras) continue;
  
          Eigen::Vector3d point3d_in_cam = optimized_cameras[obs.img_idx].cam_from_world * track.point3d;
          if (point3d_in_cam.z() < 0.1) { point3d_in_cam.z() = 0.1; }
          Eigen::Vector2d projected_pixel = optimized_cameras[obs.img_idx].pixelFromCam(point3d_in_cam);
  
          double dx = projected_pixel.x() - obs.pixel.x;
          double dy = projected_pixel.y() - obs.pixel.y;
          double reproj_err = dx*dx + dy*dy;
  
          track_to_sum_err[track_idx] += reproj_err;
          track_to_num_obs[track_idx] += 1;
        }
      }
      std::vector<double> track_avg_err(tracks.size(), 0.0);
      std::vector<double> valid_avg_errs;
      for (int track_idx = 0; track_idx < tracks.size(); ++track_idx) {
        if (track_to_num_obs[track_idx] == 0) continue;
        track_avg_err[track_idx] = std::sqrt(track_to_sum_err[track_idx] / track_to_num_obs[track_idx] + 1e-7);

        valid_avg_errs.push_back(track_avg_err[track_idx]);
      }

      int total_pruned_tracks = 0;
      int total_skipped_low_obs = 0;
      for (auto& t : tracks) {
        if (t.pruned) ++total_pruned_tracks;
        if (t.skipped_low_obs) ++total_skipped_low_obs;
        
      }
      XPLINFO << "# pruned tracks total: " << total_pruned_tracks << " # skipped low obs (this BA): " << total_skipped_low_obs;

      // E.g., try for an outlier threshold of 5 (kOutlierThresholdPix) pixels, but if that would throw away 
      // more than half (min_inlier_frac) of the dataset, use whatever threshold keeps half.
      // TODO: maybe we need some kind of spatial binning to make sure we dont remove too many outliers in some parts of the scene.
      double percentil_err = math::percentile(valid_avg_errs, min_inlier_frac);
      double soft_threshold = std::max(kOutlierThresholdPix, percentil_err);
      XPLINFO << "# tracks valid for threshold: " << valid_avg_errs.size();
      XPLINFO << "soft track outlier threshold (RMSE)=" << soft_threshold;
      XPLINFO << "RMSE percentile (" << min_inlier_frac << ") = " << percentil_err;
      int num_behind_camera = 0;
      int num_pruned_low_weight = 0;
      for (int track_idx = 0; track_idx < tracks.size(); ++track_idx) {
        if (track_to_num_obs[track_idx] == 0) continue;
        tracks[track_idx].weight = (std::tanh(kOutlierWeightSteepness * (soft_threshold - track_avg_err[track_idx])) + 1) * 0.5;
        
        XCHECK(!std::isnan(tracks[track_idx].weight));
        //XPLINFO << "weight = " << tracks[track_idx].weight << " track err=" << track_avg_err[track_idx];
        
        // Set weight to 0 for tracks that are behind their cameras
        for (auto& obs : tracks[track_idx].observations) {
          if (obs.img_idx >= num_active_cameras) continue;
          TCamera& cam = optimized_cameras[obs.img_idx];
          Eigen::Vector3d dir_to_point = tracks[track_idx].point3d - cam.getPositionInWorld();
          float dot = dir_to_point.dot(cam.forward());
          if (dot < 0) {
            tracks[track_idx].pruned = true;
            tracks[track_idx].weight = 0;
            ++num_behind_camera;
          }           
        }

        // If a track has lower weight than this, prune it forever
        static constexpr float kWeightPruneThreshold = 0.1;
        if (outer_itr > 0 && tracks[track_idx].weight < kWeightPruneThreshold) {
          tracks[track_idx].pruned = true;
          tracks[track_idx].weight = 0;
          ++num_pruned_low_weight;
        }

      }
      XPLINFO << "# behind camera: " << num_behind_camera << " # pruned low weight: " << num_pruned_low_weight;
    }
  }

  return true;
}

template<typename TCamera>
void normalizeSfmSolutionToRadius1(
  std::vector<TCamera>& cameras,
  std::vector<IncrementalSfmTrack>& tracks,
  int num_active_cameras
) {
  // Unpack and normalize camera trajectory
  Eigen::Vector3d avg_cam_position(0, 0, 0);
  for (int i = 0; i < num_active_cameras; ++i) {
    avg_cam_position += cameras[i].getPositionInWorld();
  }
  avg_cam_position /= num_active_cameras;

  // Subtract average camera position from each camera position, to center the trajectory at the origin.
  // Also do the same thing to the SFM point cloud
  float scale = 0;
  for (auto& cam : cameras) {
    cam.setPositionInWorld(cam.getPositionInWorld() - avg_cam_position);
    scale += cam.getPositionInWorld().norm();
  }
  scale /= num_active_cameras;
  scale *= 2.0;
  for (auto& cam : cameras) {
    cam.setPositionInWorld(cam.getPositionInWorld() / scale);
  }
  for (auto& t : tracks) {
    if (!t.has_estimated_3d) continue;
    t.point3d -= avg_cam_position;
    t.point3d /= scale;
    if (t.point3d.norm() > kZFar) { t.point3d *= kZFar / t.point3d.norm(); }
  }
}

template<typename TCamera>
void normalizeScaleWithMonoDepth(
  const std::vector<cv::Mat>& mono_depthmaps,
  std::vector<TCamera>& cameras,
  std::vector<IncrementalSfmTrack>& tracks,
  const int num_active_cameras = -1,  // -1 means use all cameras
  bool align_to_cam1 = false
) {
  const int n_cameras = (num_active_cameras <= 0) ? cameras.size() : num_active_cameras;
  if (n_cameras < 2) return;
  
  // Collect depth ratios (sfm_depth / mono_depth) for all valid observations
  std::vector<double> depth_ratios;
  for (const auto& track : tracks) {
    if (track.pruned || !track.has_estimated_3d) continue;

    for (const auto& obs : track.observations) {
      // Skip observations outside active cameras
      if (obs.img_idx >= n_cameras) continue;
      
      // Get the SfM depth (distance from camera to 3D point in camera space)
      Eigen::Vector3d point3d_in_cam = cameras[obs.img_idx].cam_from_world * track.point3d;
      double sfm_depth = point3d_in_cam.z();

      if (sfm_depth < 0.01) continue;
      
      // Get the mono depth at this pixel location
      float mono_depth = opencv::getPixelBilinear<float>(
          mono_depthmaps[obs.img_idx], obs.pixel.x, obs.pixel.y);
          
      if (mono_depth <= 0.01 || !std::isfinite(mono_depth)) continue;
      
      depth_ratios.push_back(sfm_depth / mono_depth);
    }
  }

  if (depth_ratios.size() < 20) {
    XPLINFO << "Not enough depth samples to compute reliable scale: " << depth_ratios.size();
    return;
  }
  
  std::sort(depth_ratios.begin(), depth_ratios.end());
  double median_ratio = depth_ratios[depth_ratios.size() / 2];
  
  // Apply filter to remove extreme outliers
  std::vector<double> filtered_ratios;
  const double kOutlierThreshold = 5.0;  // Filter ratios more than 2x different from median
  for (double ratio : depth_ratios) {
    if (ratio > median_ratio / kOutlierThreshold && ratio < median_ratio * kOutlierThreshold) {
      filtered_ratios.push_back(ratio);
    }
  }
  
  // Recompute scale factor with filtered ratios
  double scale_factor = median_ratio;
  if (filtered_ratios.size() >= 20) {
    std::sort(filtered_ratios.begin(), filtered_ratios.end());
    scale_factor = filtered_ratios[filtered_ratios.size() / 2];
  }
  
  XPLINFO << "Computed mono depth scale factor: " << scale_factor 
          << " from " << depth_ratios.size() << " samples (filtered: " << filtered_ratios.size() << ") median ratio: " << median_ratio;

  // Apply scale to all camera positions (inverse of the ratio because we're scaling the world)
  const double world_scale = 1.0 / scale_factor;
  for (auto& cam : cameras) {
    Eigen::Vector3d pos = cam.getPositionInWorld();
    cam.setPositionInWorld(pos * world_scale);
  }
  
  // Apply the same scale to all 3D points
  for (auto& track : tracks) {
    if (!track.pruned && track.has_estimated_3d) {
      track.point3d *= world_scale;
    }
  }

  // Subtract avg camera position
  Eigen::Vector3d avg_cam_position(0, 0, 0);
  for (int i = 0; i < n_cameras; ++i) {
    avg_cam_position += cameras[i].getPositionInWorld();
  }
  avg_cam_position /= n_cameras;
  for (auto& cam : cameras) {
    cam.setPositionInWorld(cam.getPositionInWorld() - avg_cam_position);
  }
  for (auto& t : tracks) {
    if (!t.has_estimated_3d) continue;
    t.point3d -= avg_cam_position;
  }
}


template<typename TCamera>
void initialGuessPoint3d(
  const std::vector<cv::Mat>& mono_depthmaps,
  std::vector<IncrementalSfmTrack>& tracks,
  const std::vector<TCamera>& initial_camera_intrinsics
) {
  for (IncrementalSfmTrack& track : tracks) {
    track.point3d = Eigen::Vector3d(0, 0, 1); // default initialization
    for (IncrementalSfmObservation& obs : track.observations) {
      if (obs.img_idx == 0) {
        auto& cam = initial_camera_intrinsics[0];
        auto& depthmap = mono_depthmaps[obs.img_idx];
        float d = opencv::getPixelBilinear<float>(depthmap, obs.pixel.x, obs.pixel.y);

        Eigen::Matrix3d world_R_cam = cam.cam_from_world.linear().transpose();
        Eigen::Vector3d ray = cam.rayDirFromPixel(Eigen::Vector2d(obs.pixel.x, obs.pixel.y));

        // TODO: this might not be the best thing to do for fisheye lenses!
        Eigen::Vector3d adjusted_ray = Eigen::Vector3d(
          ray.x() * d / ray.z(), 
          ray.y() * d / ray.z(), 
          d);

        track.point3d = cam.getPositionInWorld() + world_R_cam * adjusted_ray;

        // Project points to world ball
        float norm = track.point3d.norm();
        if (norm > kZFar) { track.point3d *= kZFar / norm; }

        track.has_estimated_3d = true; // pretend it came from bundle adjustment
        break;
      }
    }
  } 
}

// Construct a better initial estimate for the 3D point of any track that is entering the problem incrementally.
// We can do better than the first guess because we have a pose for the camera.
template<typename TCamera>
void guess3DPointForIcrementalTracks(
  const std::vector<cv::Mat>& mono_depthmaps,
  std::vector<IncrementalSfmTrack>& tracks,
  const std::vector<TCamera>& optimized_cameras,
  const int num_active_cameras
) {
  for (IncrementalSfmTrack& track : tracks) {
    if (track.has_estimated_3d) continue;
  
    for (IncrementalSfmObservation& obs : track.observations) {
      if (obs.img_idx == num_active_cameras - 1) {
        const auto& cam = optimized_cameras[obs.img_idx];

        Eigen::Matrix3d world_R_cam = cam.cam_from_world.linear().transpose();
        Eigen::Vector3d ray = cam.rayDirFromPixel(Eigen::Vector2d(obs.pixel.x, obs.pixel.y));
        
        float d = opencv::getPixelBilinear<float>(
          mono_depthmaps[obs.img_idx], obs.pixel.x, obs.pixel.y);

        // Wrong: d is not the distance on the ray, it is the z coordinate in camera space
        //track.point3d = cam.getPositionInWorld() + d * world_R_cam * ray;
        
        Eigen::Vector3d adjusted_ray = Eigen::Vector3d(
          ray.x() * d / ray.z(), 
          ray.y() * d / ray.z(), 
          d);

        track.point3d = cam.getPositionInWorld() + world_R_cam * adjusted_ray;

        // Keep points in a reasonably sized ball
        float norm = track.point3d.norm();
        if (norm > kZFar) { track.point3d *= kZFar / norm; }

        //track.has_estimated_3d = true;
        break;
      }
    }
  }
}

template<typename TCamera>
void vizReprojectionErrors(
  const std::vector<cv::Mat>& images,
  const std::vector<TCamera>& cameras,
  const std::vector<IncrementalSfmTrack>& tracks,
  const std::string& debug_dir
) {
  std::vector<cv::Mat> viz_images;
  for (int i = 0; i < images.size(); ++i) {
    XCHECK_EQ(images[i].type(), CV_32FC3);
    cv::Mat image8uc3;
    images[i].convertTo(image8uc3, CV_8UC3, 255);
    viz_images.push_back(image8uc3);
  }

  // Do a calculation of RMSE per image while were in here
  std::vector<double> image_to_sse(images.size(), 0); // sum squared error per image
  std::vector<int> image_to_sse_count(images.size(), 0);
  double overall_sse = 0;
  double overall_sse_count = 0;

  for (const auto& track : tracks) {
    if (track.skipped_low_obs) continue; // Dont draw points that never participated in final BA

    // Blend color from green (track weight = 1) to red (track weight = 0).
    cv::Scalar track_color = cv::Scalar(0, 255, 0) * track.weight + cv::Scalar(0, 0, 255) * (1.0 - track.weight);
    if (!track.has_estimated_3d) track_color = cv::Scalar(255, 0, 255);
    //if (track.skipped_low_obs) track_color = cv::Scalar(0, 255, 255);
    if (track.pruned) track_color = cv::Scalar(255, 255, 255);

    for (const auto& obs : track.observations) {
      if (obs.img_idx >= images.size()) continue;

      cv::circle(viz_images[obs.img_idx], obs.pixel, 5, track_color, 1);
      Eigen::Vector3d point_in_cam = cameras[obs.img_idx].camFromWorld(track.point3d);
      if (point_in_cam.z() < 0.1) point_in_cam.z() = 0.1; // Match residual. TODO: is 0.1 bad?
      const Eigen::Vector2d projected_pixel = cameras[obs.img_idx].pixelFromCam(point_in_cam);
      cv::line(viz_images[obs.img_idx], obs.pixel, cv::Point2f(projected_pixel.x(), projected_pixel.y()), track_color, 1);
    
      //double reproj_err = (Eigen::Vector2d(obs.pixel.x, obs.pixel.y) - projected_pixel).norm();
      if (!track.pruned && track.has_estimated_3d && !track.skipped_low_obs) {
        double reproj_err = (Eigen::Vector2d(obs.pixel.x, obs.pixel.y) - projected_pixel).squaredNorm();
        image_to_sse[obs.img_idx] += track.weight * reproj_err;
        image_to_sse_count[obs.img_idx] += 1;
        overall_sse += track.weight * reproj_err;
        overall_sse_count += 1;
      }
    }
  }

  double overall_rmse = std::sqrt(overall_sse / overall_sse_count);
  XPLINFO << "---- Weighted RMSE (all images): " << overall_rmse;
  XPLINFO << "---- Per-camera reprojection error (weighted):";
  for (int i = 0; i < images.size(); ++i) {
    std::string label = cameras[i].name.empty() ? std::to_string(i) : cameras[i].name;
    cv::imwrite(debug_dir + "/reproj_" + label + ".jpg", viz_images[i]);

    double cam_rmse = std::sqrt(image_to_sse[i] / image_to_sse_count[i]);
    XPLINFO << label << ": reproj RMSE: " << cam_rmse;  
  }
}

inline void updatePointCloud(
  const std::vector<cv::Mat>& images,
  const std::vector<IncrementalSfmTrack>& tracks,
  std::vector<Eigen::Vector3f>& point_cloud,
  std::vector<Eigen::Vector4f>& point_cloud_colors
) {
  point_cloud.clear();
  point_cloud_colors.clear();

  for (const auto& t : tracks) {
    // Skip bad points, they harm 3DGS reconstruction
    if (t.pruned) continue;
    //if (t.skipped_low_obs) continue;
    if (!t.has_estimated_3d) continue;
    if (t.weight < 0.01) continue;

    point_cloud.push_back(t.point3d.cast<float>());
    int img_idx = t.observations[0].img_idx;
    XCHECK_EQ(images[img_idx].type(), CV_32FC3);
    const cv::Vec3f color = opencv::getPixelBilinear<cv::Vec3f>(
      images[img_idx], t.observations[0].pixel.x, t.observations[0].pixel.y);
    point_cloud_colors.emplace_back(color[2], color[1], color[0], 0.9); // swizzle BGR->RGBA
  }
}

template<typename TCamera>
void updateGuiData(
  IncrementalSfmGuiData* gui_data,
  const std::vector<cv::Mat>& images,
  const std::vector<IncrementalSfmTrack>& tracks,
  const std::vector<TCamera>& optimized_cameras,
  const int num_active_cameras,
  const double dist_a_to_b
) {
  if (gui_data) {
    std::lock_guard<std::mutex> guard(gui_data->mutex);      

    if constexpr (is_rectilinear_camera_v<TCamera>) {
      gui_data->viz_cameras = optimized_cameras;
    }

    // Visualize fisheye cameras with proxy rectilinear cameras
    if constexpr (is_fisheye_camera_v<TCamera>) {
      gui_data->viz_cameras.clear();
      for (auto& c : optimized_cameras) {
        calibration::RectilinearCamerad proxy_cam;
        proxy_cam.cam_from_world = c.cam_from_world;
        proxy_cam.width = c.width;
        proxy_cam.height = c.height;
        proxy_cam.optical_center = c.optical_center;
        proxy_cam.focal_length = Eigen::Vector2d(c.width/2, c.height/2);
        gui_data->viz_cameras.push_back(proxy_cam);
      }
    }


    gui_data->viz_cameras.resize(num_active_cameras);
    updatePointCloud(images, tracks, gui_data->point_cloud, gui_data->point_cloud_colors);
    gui_data->pointcloud_needs_update = true;
  }
}

struct DepthCalibrationCostFunctor {
  DepthCalibrationCostFunctor(float mono_depth, float sfm_depth, float weight)
      : mono_depth_(mono_depth), sfm_depth_(sfm_depth), weight_(weight) {}

  template <typename T>
  bool operator()(const T* const scale, const T* const bias, T* residual) const {
    T calibrated_depth = scale[0] * T(mono_depth_) + bias[0];
    residual[0] = (calibrated_depth - T(sfm_depth_)) * T(weight_);
    return true;
  }

private:
  const float mono_depth_;
  const float sfm_depth_;
  const float weight_;
};

/*
template<typename TCamera>
void calibrateMonoDepthmap(
  const int cam_idx, // which depthmap to update
  const std::vector<cv::Mat>& mono_depthmaps, // original uncalibrated
  std::vector<cv::Mat>& mono_depthmaps_calibrated,
  std::vector<IncrementalSfmTrack>& tracks,
  const std::vector<TCamera>& optimized_cameras
) {
  float w = mono_depthmaps[cam_idx].cols;
  float h = mono_depthmaps[cam_idx].rows;
  float margin_x = w * 0.05;
  float margin_y = h * 0.05;

  // Extract depth percentile values from uncalibrated mono depthmaps
  std::vector<float> depth_samples;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (x < margin_x || y < margin_y || x > w - margin_x || y > h - margin_y) continue;
      float d = mono_depthmaps[cam_idx].at<float>(y, x);
      if (d < 0 || !std::isfinite(d)) continue;
      depth_samples.push_back(d);
    }
  }
  std::sort(depth_samples.begin(), depth_samples.end());
  float p10 = depth_samples[0.10 * depth_samples.size()];
  float p50 = depth_samples[0.50 * depth_samples.size()];
  float p90 = depth_samples[0.90 * depth_samples.size()];
  XPLINFO << "p10=" << p10 << " p50=" << p50 << " p90=" << p90;
  
  // Collect depths according to mono depth, SFM tracks, and corresponding weights.
  std::vector<float> sfm_depths, mono_depths, weights;
  for (const auto& track : tracks) {
    if (track.pruned || !track.has_estimated_3d) continue;
    for (const auto& obs : track.observations) {
      if (obs.img_idx != cam_idx) continue;

      // Skip observations near the edge of images where mono depth is less reliable
      if (obs.pixel.x < margin_x || obs.pixel.y < margin_y || obs.pixel.x > w - margin_x || obs.pixel.y > h - margin_y) continue;

      Eigen::Vector3d point3d_in_cam = optimized_cameras[obs.img_idx].cam_from_world * track.point3d;
      float sfm_depth = point3d_in_cam.z();
      if (sfm_depth < 0.0) continue;

      sfm_depth = std::min(sfm_depth, kZFar); // clamp to avoid large distances

      float mono_depth = opencv::getPixelBilinear<float>(
        mono_depthmaps[obs.img_idx], obs.pixel.x, obs.pixel.y);
      if (mono_depth < 0.0 || !std::isfinite(mono_depth)) continue;

      // Focus optimization on p10 to p50 range of depth values to avoid overfitting on high depth values
      if (mono_depth < p10 || mono_depth > p50) continue;

      sfm_depths.push_back(sfm_depth);
      mono_depths.push_back(mono_depth);
      weights.push_back(track.weight);
    }
  }

  if (sfm_depths.size() < 10) {
    XPLINFO << "Not enough observations to calibrate depth map for camera " << cam_idx;
    return;
  }

  double scale = 1.0;  
  double bias = 0.0;  
  ceres::Problem problem;
  for (size_t i = 0; i < sfm_depths.size(); ++i) {
    ceres::CostFunction* cost_function = 
        new ceres::AutoDiffCostFunction<DepthCalibrationCostFunctor, 1, 1, 1>(
            new DepthCalibrationCostFunctor(mono_depths[i], sfm_depths[i], weights[i]));

    problem.AddResidualBlock(cost_function, nullptr, &scale, &bias); //new ceres::HuberLoss(1.0)
  }
  
  // Lock bias to zero, only estimate scale
  problem.SetParameterBlockConstant(&bias);

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = false;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  scale = std::max(0.01, std::min(100.0, scale));
  
  double total_abs_error = 0.0;
  double total_rel_error = 0.0;
  std::vector<double> abs_errors;
  
  for (size_t i = 0; i < sfm_depths.size(); ++i) {
    double calibrated_depth = scale * mono_depths[i] + bias;
    double abs_error = std::abs(calibrated_depth - sfm_depths[i]);
    double rel_error = abs_error / sfm_depths[i];
    
    total_abs_error += abs_error;
    total_rel_error += rel_error;
    abs_errors.push_back(abs_error);
  }
  
  double mean_abs_error = total_abs_error / sfm_depths.size();
  double mean_rel_error = total_rel_error / sfm_depths.size();
  
  // Calculate median absolute error
  std::sort(abs_errors.begin(), abs_errors.end());
  double median_abs_error = abs_errors[abs_errors.size() / 2];
  
  // Log results with error metrics
  XPLINFO << "Depthmap cal camera: " << cam_idx 
          << " depth scale = " << scale << ", bias = " << bias
          << " (points: " << sfm_depths.size()
          << ", mean abs error: " << mean_abs_error
          << ", median abs error: " << median_abs_error
          << ", mean relative error: " << mean_rel_error;
          //<< ", solver status: " << summary.BriefReport() << ")";

  // Update the calibrated depth map
  cv::Mat& calibrated_depthmap = mono_depthmaps_calibrated[cam_idx];
  for (int y = 0; y < calibrated_depthmap.rows; ++y) {
    for (int x = 0; x < calibrated_depthmap.cols; ++x) {
      float d = mono_depthmaps[cam_idx].at<float>(y, x);
      if (std::isfinite(d)) {
        calibrated_depthmap.at<float>(y, x) = scale * d + bias;
      }
    }
  }
}
*/

inline void estimateMonoDepthmaps(
  std::shared_ptr<std::atomic<bool>> cancel_requested,
  const std::string& debug_dir,
  const std::vector<cv::Mat>& images,
  std::vector<cv::Mat>& mono_depthmaps//,
  //std::vector<cv::Mat>& mono_depthmaps_calibrated
) {
  XPLINFO << "Loading mono depth model";
  torch::jit::getProfilingMode() = false;
  torch::jit::script::Module mono_depth;
  depth_estimation::getTorchModelDepthAnything2(mono_depth);
  XPLINFO << "Finished loading mono depth model";

  for (int i = 0; i < images.size(); ++i) {
    if (cancel_requested && *cancel_requested) return;

    XPLINFO << "Estimating mono depthmap: " << i << " / " << images.size();
    XCHECK_EQ(images[i].type(), CV_32FC3);
    cv::Mat image0_8uc3;
    images[i].convertTo(image0_8uc3, CV_8UC3, 255);

    bool normalize_depth = false;
    bool resize_depth = true;
    cv::Mat depthmap = depth_estimation::estimateMonoDepthWithDepthAnything2(mono_depth, image0_8uc3, normalize_depth, resize_depth);

    XCHECK_EQ(depthmap.size(), images[0].size());
    cv::imshow("depthmap", depthmap * 0.1); cv::waitKey(20);
    cv::imwrite(debug_dir + "/monodepth_" + string::intToZeroPad(i, 6) + ".jpg", depthmap * 0.1 * 255.0);

    mono_depthmaps.push_back(depthmap);
    //mono_depthmaps_calibrated.push_back(depthmap);
  }
  cv::destroyAllWindows();
}

template<typename TCamera>
inline void reorderImagesByMatchCount(
  std::vector<cv::Mat>& images,
  std::vector<std::string>& image_names,
  std::vector<TCamera>& initial_camera_intrinsics,
  std::vector<std::vector<cv::KeyPoint>>& image_to_keypoints,
  std::vector<std::vector<cv::DMatch>>& image_to_matches
) {
  const int num_images = images.size();
  XCHECK_EQ(num_images, image_names.size());
  XCHECK_EQ(num_images, initial_camera_intrinsics.size());
  XCHECK_EQ(num_images, image_to_keypoints.size());
  XCHECK_EQ(num_images, image_to_matches.size());

  // Helper function to count matches between camera A and a set of selected cameras
  auto countMatchesToSelectedSet = [&](int camera_a, const std::unordered_set<int>& selected_set) {
    int count = 0;
    
    // Count matches where camera_a is source and target is in selected set
    for (const auto& match : image_to_matches[camera_a]) {
      if (selected_set.count(match.imgIdx)) {
        count++;
      }
    }
    
    // Count matches where camera_a is target and source is in selected set
    for (int selected_cam : selected_set) {
      for (const auto& match : image_to_matches[selected_cam]) {
        if (match.imgIdx == camera_a) {
          count++;
        }
      }
    }
    
    return count;
  };

  // Count total matches for each image to find the best starting camera
  std::vector<int> total_matches_per_image(num_images, 0);
  for (int i = 0; i < num_images; ++i) {
    total_matches_per_image[i] += image_to_matches[i].size();
    for (const auto& match : image_to_matches[i]) {
      total_matches_per_image[match.imgIdx]++;
    }
  }

  // Start with camera that has most matches overall
  int first_camera = std::max_element(total_matches_per_image.begin(), 
                                     total_matches_per_image.end()) 
                     - total_matches_per_image.begin();
  
  std::vector<int> new_order;
  std::unordered_set<int> selected_cameras;
  std::unordered_set<int> remaining_cameras;
  
  // Initialize sets
  for (int i = 0; i < num_images; ++i) {
    remaining_cameras.insert(i);
  }
  
  // Add first camera
  new_order.push_back(first_camera);
  selected_cameras.insert(first_camera);
  remaining_cameras.erase(first_camera);
  
  XPLINFO << "Reordering images by connectivity to existing set:";
  XPLINFO << "  Position 0: " << image_names[first_camera] 
          << " (index " << first_camera << ", " << total_matches_per_image[first_camera] << " total matches)";

  // Greedily add cameras with most connections to existing set
  while (!remaining_cameras.empty()) {
    int best_camera = -1;
    int best_match_count = -1;
    
    for (int candidate : remaining_cameras) {
      int match_count = countMatchesToSelectedSet(candidate, selected_cameras);
      if (match_count > best_match_count) {
        best_match_count = match_count;
        best_camera = candidate;
      }
    }
    
    if (best_camera != -1) {
      new_order.push_back(best_camera);
      selected_cameras.insert(best_camera);
      remaining_cameras.erase(best_camera);
      
      XPLINFO << "  Position " << new_order.size() - 1 << ": " << image_names[best_camera] 
              << " (index " << best_camera << ", " << best_match_count << " matches to existing set)";
    } else {
      // No connections found, just add the first remaining camera
      int fallback = *remaining_cameras.begin();
      new_order.push_back(fallback);
      selected_cameras.insert(fallback);
      remaining_cameras.erase(fallback);
      
      XPLINFO << "  Position " << new_order.size() - 1 << ": " << image_names[fallback] 
              << " (index " << fallback << ", 0 matches to existing set - isolated)";
    }
  }

  // Create mapping from old index to new index
  std::vector<int> old_to_new(num_images);
  for (int new_idx = 0; new_idx < num_images; ++new_idx) {
    old_to_new[new_order[new_idx]] = new_idx;
  }

  // Reorder all vectors
  auto reorder = [&](auto& vec) {
    auto original = vec;
    for (int i = 0; i < num_images; ++i) {
      vec[i] = std::move(original[new_order[i]]);
    }
  };

  reorder(images);
  reorder(image_names);
  reorder(initial_camera_intrinsics);
  reorder(image_to_keypoints);
  reorder(image_to_matches);

  // Update imgIdx in all matches to reflect new ordering
  for (auto& matches : image_to_matches) {
    for (auto& match : matches) {
      match.imgIdx = old_to_new[match.imgIdx];
    }
  }
}

// Returns the number of matches per image
inline std::vector<int> printMatchesPerImage(
  const std::vector<std::string>& image_names,
  const std::vector<std::vector<cv::DMatch>>& image_to_matches,
  const std::string& label = ""
) {
  const int num_images = image_names.size();
  XCHECK_EQ(num_images, image_to_matches.size());
  
  // Count total matches for each image (as both source and target)
  std::vector<int> total_matches_per_image(num_images, 0);
  
  for (int i = 0; i < num_images; ++i) {
    // Count matches where image i is the source
    total_matches_per_image[i] += image_to_matches[i].size();
    
    // Count matches where image i is the target (referenced in earlier images' matches)
    for (const auto& match : image_to_matches[i]) {
      total_matches_per_image[match.imgIdx]++;
    }
  }
  
  XPLINFO << "---- # matches per image" << (label.empty() ? ":" : " (" + label + "):");
  for (int i = 0; i < num_images; ++i) {
    XPLINFO << i << ", " << image_names[i] << " # matches: " << total_matches_per_image[i];
  }

  return total_matches_per_image;
}

template<typename TCamera>
inline std::vector<TCamera> estimateCameraPosesAndKeypoint3DWithIncrementalSfm(
  std::shared_ptr<std::atomic<bool>> cancel_requested,
  IncrementalSfmGuiData* gui_data,
  std::vector<cv::Mat>& images,
  std::vector<std::string>& image_names, // will be inserted into returned cameras .name field
  std::vector<TCamera>& initial_camera_intrinsics,
  const double dist_a_to_b,
  const std::string& debug_dir,
  const bool show_keypoints,
  const bool show_matches,
  const float flow_err_threshold,
  const float match_ratio_threshold,
  const float inlier_frac,
  const float depth_weight,
  const int time_window_size, // 0 means no time window
  const bool filter_with_flow,
  const bool share_intrinsics_all_cameras,
  const bool reorder_cameras,
  const bool use_intrinsic_prior,
  const int max_solver_itrs,
  std::vector<Eigen::Vector3f>& point_cloud,
  std::vector<Eigen::Vector4f>& point_cloud_colors
) {
  XCHECK_EQ(images.size(), initial_camera_intrinsics.size());
  XCHECK_EQ(images.size(), image_names.size());

  for (int i = 0; i < initial_camera_intrinsics.size(); ++i) {
    initial_camera_intrinsics[i].name = image_names[i];
  }

  std::vector<std::vector<cv::KeyPoint>> image_to_keypoints;
  std::vector<std::vector<cv::DMatch>> image_to_matches;
  std::vector<IncrementalSfmTrack> tracks;

  matchKeypointsBetweenAllImagePairs(
    cancel_requested, images, initial_camera_intrinsics, debug_dir, show_keypoints, show_matches,
    image_to_keypoints, image_to_matches, flow_err_threshold, match_ratio_threshold, time_window_size, filter_with_flow);

  if (cancel_requested && *cancel_requested) return std::vector<TCamera>();

  std::vector<int> num_matches_per_image = printMatchesPerImage(image_names, image_to_matches, "original order");
  if (reorder_cameras) {
    reorderImagesByMatchCount(
      images,
      image_names,
      initial_camera_intrinsics,
      image_to_keypoints,
      image_to_matches);  
    num_matches_per_image = printMatchesPerImage(image_names, image_to_matches, "re-ordered");
  }

  tracks = buildTracks(image_to_keypoints, image_to_matches);
  XPLINFO << "# tracks: " << tracks.size();

  //std::vector<cv::Mat> mono_depthmaps, mono_depthmaps_calibrated;
  std::vector<cv::Mat> mono_depthmaps;
  //estimateMonoDepthmaps(cancel_requested, debug_dir, images, mono_depthmaps, mono_depthmaps_calibrated);
  estimateMonoDepthmaps(cancel_requested, debug_dir, images, mono_depthmaps);

  initialGuessPoint3d(mono_depthmaps, tracks, initial_camera_intrinsics);

  // Our goal is to estimate the camera poses (first as 6d param vector, then stored in cam_from_world in a FisheyeCamera or RectilinearCamera)
  std::vector<std::vector<double>> camera_to_pose_param(images.size(), std::vector<double>(calibration::kPoseDim, 0.0));
  std::vector<TCamera> optimized_cameras = initial_camera_intrinsics;

  updateGuiData(gui_data, images, tracks, optimized_cameras, 1, dist_a_to_b);
  //std::this_thread::sleep_for(std::chrono::milliseconds(3500));

  // Camera intrinsic parameters are jointly optimized
  std::vector<std::vector<double>> camera_to_intrinsic_param;
  for (const auto& cam : initial_camera_intrinsics) {
    camera_to_intrinsic_param.push_back(cam.getIntrinsicParamVec());
  }

  // Main incremental SFM loop
  int num_active_cameras = 2;
  while(num_active_cameras <= images.size()) {
    if (cancel_requested && *cancel_requested) return std::vector<TCamera>();

    XPLINFO << "==== # active cameras: " << num_active_cameras << " / " << images.size();
    
    constexpr int kMinMatchesToIncludeCamera = 5;
    if (num_matches_per_image[num_active_cameras - 1] < kMinMatchesToIncludeCamera) {
      XPLINFO << "Camera has not enough matches. Stopping early...";
      --num_active_cameras; // dont include this last failed camera in global bundle adjustment
      break;
    }

    // Initial guess new camera is at previous camera
    camera_to_pose_param[num_active_cameras - 1] = camera_to_pose_param[num_active_cameras - 2];
    optimized_cameras[num_active_cameras - 1] = optimized_cameras[num_active_cameras - 2];

    XPLINFO << "---- estimating pose of new camera";

    bool added_ok = bundleAdjustment(
      debug_dir, 
      num_active_cameras,
      num_active_cameras - 1,         // first unlocked camera
      num_active_cameras - 1,         // last unlocked camera
      true,                           // lock 3d points
      false,                          // dont lock all cameras
      false,                          // optimize_intrinsics (keep intrinsics fixed)
      true,                           // only_optimize_tracks_with_estimated3d (all points)
      share_intrinsics_all_cameras,
      use_intrinsic_prior,
      2,                              // num_outer_itrs
      inlier_frac,
      depth_weight,
      max_solver_itrs,
      images,
      //mono_depthmaps_calibrated,
      mono_depthmaps,
      initial_camera_intrinsics, 
      camera_to_pose_param, 
      camera_to_intrinsic_param, 
      optimized_cameras, 
      tracks
    );
    
    if (!added_ok) {
      XPLINFO << "Camera has no SFM residuals. Stopping early...";
      --num_active_cameras; // dont include this last failed camera in global bundle adjustment
      break;
    }

    updateGuiData(gui_data, images, tracks, optimized_cameras, num_active_cameras, dist_a_to_b);
    //std::this_thread::sleep_for(std::chrono::milliseconds(1500));

    //XPLINFO << "---- calibrating depth map before bundle adjustment";
    //calibrateMonoDepthmap(num_active_cameras - 1, mono_depthmaps, mono_depthmaps_calibrated, tracks, optimized_cameras);

    XPLINFO << "---- windowed bundle adjustment";

    bundleAdjustment(
      debug_dir, 
      num_active_cameras,
      num_active_cameras - 1 - time_window_size, // first unlocked camera
      num_active_cameras - 1,         // last unlocked camera
      false,                          // lock 3d points
      false,                          // dont lock all cameras
      true,                           // optimize_intrinsics (keep intrinsics fixed)
      false,                          // only_optimize_tracks_with_estimated3d (all points)
      share_intrinsics_all_cameras,
      use_intrinsic_prior,
      2,                              // num_outer_itrs
      inlier_frac,
      depth_weight,
      max_solver_itrs,
      images,
      //mono_depthmaps_calibrated,
      mono_depthmaps,
      initial_camera_intrinsics, 
      camera_to_pose_param, 
      camera_to_intrinsic_param, 
      optimized_cameras, 
      tracks
    );
    updateGuiData(gui_data, images, tracks, optimized_cameras, num_active_cameras, dist_a_to_b);
    //std::this_thread::sleep_for(std::chrono::milliseconds(1500));

    XPLINFO << "---- rescaling solution to align with mono depth";

    //normalizeSfmSolutionToRadius1(optimized_cameras, tracks, num_active_cameras);
    //calibrateMonoDepthmap(num_active_cameras - 1, mono_depthmaps, mono_depthmaps_calibrated, tracks, optimized_cameras);
    normalizeScaleWithMonoDepth(mono_depthmaps, optimized_cameras, tracks, num_active_cameras, false); // use uncalibrated depthmaps for scalce normalization

    // Update pose parameters after normalizing (which only updates optimized_cameras)
    for (int i = 0; i < num_active_cameras; ++i) {
      camera_to_pose_param[i] = calibration::poseToParamVec(optimized_cameras[i].cam_from_world);
    }

    XPLINFO << "---- adding points for new camera";
    guess3DPointForIcrementalTracks(
      //mono_depthmaps_calibrated,
      mono_depthmaps,
      tracks, 
      optimized_cameras, 
      num_active_cameras);
    updateGuiData(gui_data, images, tracks, optimized_cameras, num_active_cameras, dist_a_to_b);

    ++num_active_cameras;
  }

  XPLINFO << "---- global bundle adjustment";

  if (cancel_requested && *cancel_requested) return std::vector<TCamera>();

  if (images.size() > 1) { // dont run if we only have 1 image
    bundleAdjustment(
      debug_dir, 
      num_active_cameras,
      0,                              // first unlocked camera
      num_active_cameras - 1,         // last unlocked camera
      false,                          // lock 3d points
      false,                          // dont lock all cameras
      true,                           // optimize_intrinsics (keep intrinsics fixed)
      false,                          // only_optimize_tracks_with_estimated3d (all points)
      share_intrinsics_all_cameras,
      use_intrinsic_prior,
      2,                              // num_outer_itrs.
      inlier_frac,
      depth_weight,
      max_solver_itrs,
      images,
      //mono_depthmaps_calibrated,
      mono_depthmaps,
      initial_camera_intrinsics, 
      camera_to_pose_param, 
      camera_to_intrinsic_param, 
      optimized_cameras, 
      tracks
    );
    //normalizeSfmSolutionToRadius1(optimized_cameras, tracks, num_active_cameras);
    normalizeScaleWithMonoDepth(mono_depthmaps, optimized_cameras, tracks, num_active_cameras, true); // use uncalibrated depthmaps for scalce normalization
  }
  updateGuiData(gui_data, images, tracks, optimized_cameras, num_active_cameras, dist_a_to_b);


  if (cancel_requested && *cancel_requested) return std::vector<TCamera>();

  // calibrate the final depthmaps
  for (int i = 0; i < images.size(); ++i) {
    //XPLINFO << "calibrating depthmap " << i << " / " << images.size(); // TODO: progress bar
    //calibrateMonoDepthmap(i, mono_depthmaps, mono_depthmaps_calibrated, tracks, optimized_cameras);
    //cv::imwrite(debug_dir + "/caldepth_" + string::intToZeroPad(i, 6) + ".jpg", mono_depthmaps_calibrated[i] * 0.1 * 255.0);
    //cv::imwrite(debug_dir + "/caldepth_" + file::filenamePrefix(image_names[i]) + ".exr", mono_depthmaps_calibrated[i]);
    cv::imwrite(debug_dir + "/depth_" + file::filenamePrefix(image_names[i]) + ".exr", mono_depthmaps[i]);
  }

  XPLINFO << "keeping only valid cameras: " << num_active_cameras;
  int resize_num_cams = std::min(num_active_cameras, int(images.size()));
  optimized_cameras.resize(resize_num_cams);
  images.resize(resize_num_cams);

  // Visualize reprojection error before normalizing optimized_cameras
  vizReprojectionErrors(images, optimized_cameras, tracks, debug_dir);

  return optimized_cameras;
}

template<typename TCamera>
void makeJsonSingleMovingCamera(
  std::vector<TCamera> cameras,
  const std::string& dest_path
) {
  using json = nlohmann::json;
  std::vector<json> frame_to_json;
  for (int i = 0; i < cameras.size(); ++i) {
    json js_frame;
    js_frame["image_filename"] = cameras[i].name;
    if constexpr (is_rectilinear_camera_v<TCamera>) {
      js_frame["cam_model"] = "rectilinear";
      js_frame["width"] = cameras[i].width;
      js_frame["height"] = cameras[i].height;
      js_frame["fx"] = cameras[i].focal_length.x();
      js_frame["fy"] = cameras[i].focal_length.y();
      js_frame["cx"] = cameras[i].optical_center.x();
      js_frame["cy"] = cameras[i].optical_center.y();
    } else {
      js_frame["cam_model"] = "equiangular";
      js_frame["width"] = cameras[i].width;
      js_frame["height"] = cameras[i].height;
      js_frame["cx"] = cameras[i].optical_center.x();
      js_frame["cy"] = cameras[i].optical_center.y();
      js_frame["radius_at_90"] = cameras[i].radius_at_90;
      js_frame["k1"] = cameras[i].k1;
      js_frame["tilt"] = cameras[i].tilt;
      js_frame["useable_radius"] = cameras[i].useable_radius;
    }

    // Extract the camera pose for this frame from the ceres solution.
    Eigen::Isometry3d world_from_cam = cameras[i].worldFromCam();
    Eigen::Matrix4d w_from_c = world_from_cam.matrix();
    w_from_c.row(3) << 0.0, 0.0, 0.0, 1.0; // Explicitly ensure the last row is [0,0,0,1]

    std::vector<double> w_from_c_elements(w_from_c.data(), w_from_c.data() + w_from_c.size());
    js_frame["world_from_cam"] = w_from_c_elements;
    frame_to_json.push_back(js_frame);
  }

  json all_frames_json;
  all_frames_json["frames_data"] = frame_to_json;
  std::ofstream json_file(dest_path);
  json_file << std::setw(4) << all_frames_json << std::endl;
  json_file.close();
}

}}} // end namespace p11::calibration::incremental_sfm
