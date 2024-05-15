// MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include "gflags/gflags.h"
#include "logger.h"
#include "check.h"
#include "util_file.h"
#include "rectilinear_sfm_lib.h"

DEFINE_string(src_vid, "", "path to input video");
DEFINE_string(video_frames_dir, "", "path to where video frames are written. If empty, defaults to dest_dir");
DEFINE_string(dest_dir, "", "dir to write temporary / viz data");
DEFINE_int32(max_image_dim, 640, "resize images so the larger image dimension is this.");
DEFINE_int32(num_train_frames, 30, "how many frames to subsample when constructing a diverse subset for training.");
DEFINE_int32(num_test_frames, 10, "how many frames to subsample when constructing an adversarial test set.");
DEFINE_int32(outer_iterations, 2, "How many iterations of iteratively reweighted least squares to run.");
DEFINE_int32(ceres_iterations, 25, "How many iterations to run ceres solver in the first outer iteration.");
DEFINE_int32(ceres_iterations2, 5, "How many iterations to run ceres solver after the first outer iteration.");
DEFINE_double(outlier_percentile, 0.8, "Percentile of reprojection error to define outlier threshold at");
DEFINE_double(outlier_weight_steepness, 2.0, "How sharp the soft-threshold on outliers is.");
DEFINE_bool(rm_unused_images, true, "Delete images which are not in the training or test set.");
DEFINE_bool(no_ffmpeg, false, "Skip ffmpeg unpacking of video, just use what's in the folder already.");

int main(int argc, char** argv)
{
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  XCHECK(!FLAGS_src_vid.empty());
  XCHECK(!FLAGS_dest_dir.empty());

  p11::rectilinear_sfm::RectilinearSfmConfig cfg{
    .cancel_requested = std::make_shared<std::atomic<bool>>(false),
    .src_vid = FLAGS_src_vid,
    .dest_dir = FLAGS_dest_dir,
    .max_image_dim = FLAGS_max_image_dim,
    .num_train_frames = FLAGS_num_train_frames,
    .num_test_frames = FLAGS_num_test_frames,
    .outer_iterations = FLAGS_outer_iterations,
    .ceres_iterations = FLAGS_ceres_iterations,
    .ceres_iterations2 = FLAGS_ceres_iterations2,
    .outlier_percentile = FLAGS_outlier_percentile,
    .outlier_weight_steepness = FLAGS_outlier_weight_steepness,
    .rm_unused_images = FLAGS_rm_unused_images,
    .no_ffmpeg = FLAGS_no_ffmpeg,
    .ffmpeg = "ffmpeg"
  };

  p11::rectilinear_sfm::runRectilinearSfmPipeline(cfg);

  return EXIT_SUCCESS;
}
