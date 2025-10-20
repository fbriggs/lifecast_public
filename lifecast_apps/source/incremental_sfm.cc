// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*

bazel run -- //source:incremental_sfm \
--dest_dir ~/Desktop/incremental_sfm \
--src_vid ~/Downloads/___cave.mov \
--show_keypoints --show_matches

bazel run -- //source:point_cloud_viz \
--point_size 4 \
--cam_json ~/Desktop/incremental_sfm/dataset.json \
--point_cloud ~/Desktop/incremental_sfm/pointcloud_sfm.bin


bazel run -- //source:incremental_sfm \
--dest_dir ~/Downloads/bones720p_4dgs \
--src_vid ~/Downloads/bones720p.mov \
--show_keypoints --show_matches

*/

#include "gflags/gflags.h"
#include "third_party/json.h"
#include "logger.h"
#include "incremental_sfm_lib.h"
#include "video_transcode_lib.h"
#include "util_string.h"
#include "util_file.h"
#include "point_cloud.h"

DEFINE_string(dest_dir, "", "path write results");
DEFINE_string(src_vid, "", "path to single input video");
DEFINE_bool(show_keypoints, false, "show keypoint visualization");
DEFINE_bool(show_matches, false, "show match visualization");
DEFINE_int32(max_image_dim, 640, "resize images so the larger image dimension is this.");
DEFINE_int32(time_window_size, 10, "0 = no window (all pairs), radius of time window for matching in frames");
DEFINE_bool(cam_not_moving, false, "The input camera is stationary");
DEFINE_bool(filter_with_flow, false, "Use optical flow to filter keypoint matches");
DEFINE_bool(share_intrinsics, false, "Use the same intrinsics for all cameras");
DEFINE_bool(reorder, false, "Re-order cameras for SFM");
DEFINE_bool(intrinsic_prior, false, "Include prior residuals for intrinsics");
DEFINE_double(inlier_frac, 0.8, "expected fraction of tracks that are inliers");
DEFINE_double(depth_weight, 0.00001, "weight of mono depth residuals");
DEFINE_int32(max_solver_itrs, 100, "max number of solver iterations for bundle adjustment");


namespace p11 { namespace calibration { namespace incremental_sfm {

void runIncrementalSfmSingleInputVideo(){
  using namespace video;
  
  InputVideoStream in_stream(FLAGS_src_vid);
  XCHECK(in_stream.valid()) << "Invalid input video stream: " << FLAGS_src_vid;

  int w = in_stream.getWidth();
  int h = in_stream.getHeight();
  std::pair<int, int> frame_rate = in_stream.guessFrameRate();
  double est_duration = in_stream.guessDurationSec();
  int est_num_frames = in_stream.guessNumFrames();
  XPLINFO << "width, height: " << w << ", " << h;
  XPLINFO << "frame rate: " << frame_rate.first << "/" << frame_rate.second << " = " << (float(frame_rate.first) / frame_rate.second);
  XPLINFO << "estimated duration(sec): " << est_duration;
  XPLINFO << "estimated # frames: " << est_num_frames;

  std::vector<cv::Mat> images;
  std::vector<std::string> image_names;
  int decode_type = CV_32FC3;
  MediaFrame frame;
  int frame_count = 0;
  int subsample_frames = 10;
  VideoStreamResult result;
  while((result = in_stream.readFrame(frame, decode_type)) == VideoStreamResult::OK) {
    if (!frame.is_video()) continue;
    XPLINFO << "frame: " << frame_count;

    if (frame_count % subsample_frames == 0) {
      // Resize to max_image_dim
      if(frame.img.cols > FLAGS_max_image_dim || frame.img.rows > FLAGS_max_image_dim) {
        float scale = FLAGS_max_image_dim / static_cast<float>(std::max(frame.img.cols, frame.img.rows));
        cv::resize(frame.img, frame.img, cv::Size(frame.img.cols * scale, frame.img.rows * scale), 0, 0, cv::INTER_AREA);
        w = frame.img.cols;
        h = frame.img.rows;
      }
      images.push_back(frame.img);
      image_names.push_back("frame_" + string::intToZeroPad(frame_count, 6) + ".png");
    }

    ++frame_count;
  }

  if (result == VideoStreamResult::FINISHED) {
    XPLINFO << "Finished successfully.";
  } else {
    XCHECK_EQ(int(result), int(VideoStreamResult::ERR)) << "There was an error during transcoding.";
  }

  RectilinearCamerad guess_intrinsics = calibration::guessRectilinearIntrinsics(w, h);
  std::vector<RectilinearCamerad> initial_camera_intrinsics;
  //std::vector<FisheyeCamerad> initial_camera_intrinsics;

  for (int i = 0; i < images.size(); ++i) {
    initial_camera_intrinsics.push_back(guess_intrinsics);
  }

  double dist_a_to_b = 0; // 0 means it will be scaled automatically
  std::string debug_dir = FLAGS_dest_dir;
  float flow_err_threshold = 20.0;
  float match_ratio_threshold = 0.9;
  std::vector<Eigen::Vector3f> sfm_point_cloud;
  std::vector<Eigen::Vector4f> sfm_point_cloud_colors;

  auto cancel_requested = std::make_shared<std::atomic<bool>>(false);
  std::vector<RectilinearCamerad> sfm_cameras = incremental_sfm::estimateCameraPosesAndKeypoint3DWithIncrementalSfm(
    cancel_requested,
    nullptr, // gui data
    images,
    image_names,
    initial_camera_intrinsics,
    dist_a_to_b,
    debug_dir,
    FLAGS_show_keypoints,
    FLAGS_show_matches,
    flow_err_threshold,
    match_ratio_threshold,
    FLAGS_inlier_frac,
    FLAGS_depth_weight,
    FLAGS_time_window_size,
    FLAGS_filter_with_flow,
    FLAGS_share_intrinsics,
    FLAGS_reorder,
    FLAGS_intrinsic_prior,
    FLAGS_max_solver_itrs,
    sfm_point_cloud,
    sfm_point_cloud_colors);
  makeJsonSingleMovingCamera(sfm_cameras, FLAGS_dest_dir + "/dataset.json");
  
  // Convert the point-cloud from RGBA to RGB
  std::vector<Eigen::Vector3f> sfm_point_cloud_colors3f;
  for (auto& c : sfm_point_cloud_colors) {
    sfm_point_cloud_colors3f.emplace_back(c.x(), c.y(), c.z());
  }
  point_cloud::savePointCloudBinary(
    FLAGS_dest_dir + "/pointcloud_sfm.bin", sfm_point_cloud, sfm_point_cloud_colors3f);
}

}}} // end namespace p11::calibration::incremental_sfm


int main(int argc, char** argv)
{
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  XCHECK(!FLAGS_dest_dir.empty());
  XCHECK(!FLAGS_src_vid.empty());

  p11::calibration::incremental_sfm::runIncrementalSfmSingleInputVideo();

  return EXIT_SUCCESS;
}
