// MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#pragma once

#include <string>
#include <memory>
#include <atomic>
#include "logger.h"

namespace p11 { namespace rectilinear_sfm {

struct RectilinearSfmConfig {
  std::shared_ptr<std::atomic<bool>> cancel_requested;  
  std::string src_vid;
  std::string video_frames_dir;
  std::string dest_dir;
  int max_image_dim;
  int num_train_frames;
  int num_test_frames;
  int outer_iterations;
  int ceres_iterations;
  int ceres_iterations2; // number of iterations after first outer iteration
  double outlier_percentile;
  double outlier_weight_steepness;
  bool rm_unused_images;
  bool no_ffmpeg;
  std::string ffmpeg; // path to ffmpeg command
};

struct RectliinearSfmGuiData {
  std::mutex mutex;
  std::vector<float> plot_data_x, plot_data_y;
  int ceres_iterations; // It changes by outer iteration, and we need it for progress bars
};

static void printConfig(const RectilinearSfmConfig& cfg) {
  XPLINFO << "cancel_requested=" << cfg.cancel_requested;
  XPLINFO << "src_vid=" << cfg.src_vid;
  XPLINFO << "video_frames_dir=" << cfg.video_frames_dir;
  XPLINFO << "dest_dir=" << cfg.dest_dir;
  XPLINFO << "max_image_dim=" << cfg.max_image_dim;
  XPLINFO << "num_train_frames=" << cfg.num_train_frames;
  XPLINFO << "num_test_frames=" << cfg.num_test_frames;
  XPLINFO << "outer_iterations=" << cfg.outer_iterations;
  XPLINFO << "ceres_iterations=" << cfg.ceres_iterations;
  XPLINFO << "ceres_iterations2=" << cfg.ceres_iterations2;
  XPLINFO << "outlier_percentile=" << cfg.outlier_percentile;
  XPLINFO << "outlier_weight_steepness=" << cfg.outlier_weight_steepness;
  XPLINFO << "rm_unused_images=" << cfg.rm_unused_images;
  XPLINFO << "no_ffmpeg=" << cfg.no_ffmpeg;
  XPLINFO << "ffmpeg=" << cfg.ffmpeg;
}

void runRectilinearSfmPipeline(RectilinearSfmConfig& cfg, RectliinearSfmGuiData* gui_data = nullptr);

}}  // end namespace p11::rectilinear_sfm
