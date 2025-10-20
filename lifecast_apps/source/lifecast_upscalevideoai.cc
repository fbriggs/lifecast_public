// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
== example ==

bazel run -c opt -- //source:lifecast_upscalevideoai \
--model_name edsr2 \
--model_path ~/dev/p11/ml_models/edsr2_mystery.pt \
--dest_dir ~/Desktop/enhance \
--src_vid ~/Downloads/short.mp4 \
--inflate --inflated_size 8192 --final_resize 2880 --make_ab --crf 24 --video_encoder libx265

== Output h264 ==

(upscale only):

bazel run -c opt -- //source:lifecast_upscalevideoai \
--model_name edsr2 \
--model_path ~/dev/p11/ml_models/edsr2_noaug.pt \
--dest_dir ~/Desktop/enhance \
--src_vid ~/Downloads/tiger_tiny.mp4 \
--video_encoder libx264 \
--crf 19

(upscale with denoise + deblur):

bazel run -c opt -- //source:lifecast_upscalevideoai \
--model_name edsr2 \
--model_path ~/dev/p11/ml_models/edsr2_mystery.pt \
--dest_dir ~/Desktop/enhance \
--src_vid ~/Downloads/tiger_tiny.mp4 \
--video_encoder libx264 \
--crf 19

== Output h265, 10 Bit ==

bazel run -c opt -- //source:lifecast_upscalevideoai \
--model_path ~/dev/p11/ml_models/edsr2_mystery.pt \
--dest_dir ~/Desktop/enhance \
--src_vid ~/Downloads/tiger_tiny.mp4 \
--video_encoder libx265 \
--crf 19

== VR180 to inflated equiangular ===

bazel run -c opt -- //source:lifecast_upscalevideoai \
--model_name edsr2 \
--model_path ~/dev/p11/ml_models/edsr2_mystery.pt \
--dest_dir ~/Desktop/enhance \
--src_vid ~/Downloads/elements8k.mov \
--inflate

== A/B test: VR180 / inflated equiangular ===

bazel run -c opt -- //source:lifecast_upscalevideoai \
--model_name edsr2 \
--model_path ~/dev/p11/ml_models/edsr2_mystery.pt \
--dest_dir ~/Desktop/enhance \
--src_vid ~/Downloads/h15.jpg \
--inflate --inflated_size 4096 --make_ab 

bazel run -c opt -- //source:lifecast_upscalevideoai \
--model_name edsr2 \
--model_path ~/dev/p11/ml_models/edsr2_mystery.pt \
--dest_dir ~/Desktop/enhance \
--src_vid ~/Downloads/elements8k.mov \
--inflate --inflated_size 4096 --make_ab

5.7K might be the highes res we can do an AB test video:

bazel run -c opt -- //source:lifecast_upscalevideoai \
--model_name edsr2 \
--model_path ~/dev/p11/ml_models/edsr2_mystery.pt \
--dest_dir ~/Desktop/enhance \
--src_vid ~/Downloads/elements8k.mov \
--inflate --inflated_size 2880 --crf 22 --make_ab

== Make a histogram to show the difference between 8 and 10 bit ==

ffmpeg -y -i ~/Desktop/enhance/enhanced_960x540_h265_crf19_10bit.mp4 \
-vf "split=2[a][b];[b]histogram=display_mode=overlay[v];[a][v]overlay" \
-c:v libx265 -crf 19 -pix_fmt yuv420p10le -tag:v hvc1 \
~/Desktop/enhance/histo10.mp4

== Just encode plain h265 for comparison ==

ffmpeg -y \
-i ~/Downloads/elements8k.mov \
-c:v libx265 -preset medium -crf 25 -pix_fmt yuv420p -movflags faststart \
-profile:v main -tag:v hvc1 \
~/Desktop/enhance/elements8k_vr180_h265_crf25.mp4
*/
#include <csignal>
#include <atomic>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "torch/torch.h"
#include "gflags/gflags.h"
#include "third_party/json.h"
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "logger.h"
#include "util_string.h"
#include "util_file.h"
#include "fisheye_camera.h"
#include "projection.h"
#include "rectilinear_camera.h"
#include "tinysr_lib.h"
#include "util_torch.h"
#include "util_time.h"
#include "video_transcode_lib.h"
#include "util_opencv.h"

DEFINE_string(src_vid, "", "path to source VR180 video file");
DEFINE_string(dest_dir, "", "path to write outputs");
DEFINE_int32(inflated_size, 4096, "size of inflated projection");
DEFINE_double(inflate_exponent, 1.3, "controls how much infilation there is. classic LDI3 uses 3.0");
DEFINE_string(model_name, "edsr2", "name of super resolution model architecture");
DEFINE_string(model_path, "", "path to super resolution model");
DEFINE_bool(inflate, false, "If true, do VR180 to inflated equiangular projection, otherwise just process mono video.");
DEFINE_double(resize_after_sr, 2.0, "super res 2x up -> down to 1.5x for faster processing. Try 1.40625 to get 5760 from 8k");
DEFINE_double(resize_before_sr, 1.0, "scale the image before super resolution");
DEFINE_int32(final_resize, 0, "if not 0, resize the inflated projection (down) to this");
DEFINE_bool(make_ab, false, "If true, generate an output where we stack the original VR180 input on top and the inflated version on bottom");
DEFINE_string(video_encoder, "libx264", "libx264,libx265,prores");
DEFINE_int32(crf, 25, "crf to use when encoding h264 or h265");
DEFINE_bool(skip_sr, false, "If true, dont run super resolution at all");
// TODO: ffmpeg "preset" open (encoder speed / quality tradeoff)

namespace p11 { namespace vr180 {

namespace {
std::atomic<bool> g_interrupt_received(false);
}

// Handle ctrl-C better so we close the output video file
void signalHandler(int signum) {
  g_interrupt_received = true;
  XPLINFO << "Interrupted. Finishing video file as cleanly as possible...";
}

std::string suggestFilenameSuffix(std::string video_encoder, int width, int height, int crf) {
  std::string s = "_" + std::to_string(width) + "x" + std::to_string(height);
  if (video_encoder == "libx264") {
    s += "_h264_crf_"+std::to_string(crf)+".mp4";
  }
  if (video_encoder == "libx265") {
    s += "_h265_crf_"+std::to_string(crf)+".mp4";
  }
  if (video_encoder == "prores") {
    s += "_prores.mov";
  }
  return s;
}


void runEnhanceVR180Pipeline() {
  // Setup interupt handler for ctrl-C

  // Setup the super resolution model
  constexpr float kSuperResScale = 2.0;
  torch::NoGradGuard no_grad;
  const torch::DeviceType device = util_torch::findBestTorchDevice();
  std::shared_ptr<enhance::Base_SuperResModel> sr_model = enhance::makeSuperResModelByName(FLAGS_model_name);
  torch::load(sr_model, FLAGS_model_path);
  sr_model->to(device);

  // We'll need to fill in projection warps once we load a frame
  calibration::FisheyeCamerad cam_ftheta, cam_inflated;
  std::vector<cv::Mat> warp_vr180_to_ftheta, warp_ftheta_to_inflated, warp_vr180_to_inflated;

  // Open the input video for reading
  video::InputVideoStream in_stream(FLAGS_src_vid);
  XCHECK(in_stream.valid()) << FLAGS_src_vid;
  std::unique_ptr<video::OutputVideoStream> out_stream = nullptr; // Allocated later when we know the output size

  std::string file_ext = file::filenameExtension(FLAGS_src_vid);
  bool is_photo = file_ext == "png" || file_ext == "jpg" || file_ext == "jpeg";
  XPLINFO << "is_photo=" << is_photo;

  // Prepare to write output video using ffmpeg stream
  int output_vid_width = 0;
  int output_vid_height = 0;  
  int frame_num = 0;
  video::MediaFrame frame;
  while(!g_interrupt_received) {
    XPLINFO << "=========== frame_num: " << frame_num;
    auto frame_timer = time::now();

    auto res = in_stream.readFrame(frame, CV_32FC3);
    if (res == video::VideoStreamResult::FINISHED) {
      XPLINFO << "finished video";
      break;
    }
    if (res == video::VideoStreamResult::ERR) {
      XPLINFO << "erro while reading frame";
      break;
    }
    if (!frame.is_video()) continue;
    cv::Mat src_image = frame.img;

    //cv::Mat src_image;
    //if (!in_stream.readFrame(src_image, CV_32FC3)) { break; }

    if (FLAGS_resize_before_sr != 1.0) {
      cv::resize(src_image, src_image, cv::Size(src_image.cols * FLAGS_resize_before_sr, src_image.rows * FLAGS_resize_before_sr), 0, 0, cv::INTER_CUBIC);
    }

    auto sr_timer = time::now();
    cv::Mat image_super, image_bicubic;
    if (FLAGS_skip_sr) {
      image_super = src_image;
    } else {
      superResolutionEnhance(device, kSuperResScale, sr_model, src_image, image_super, image_bicubic, CV_32FC3);
      if (FLAGS_resize_after_sr != kSuperResScale) {
        cv::resize(image_super, image_super, cv::Size(src_image.cols * FLAGS_resize_after_sr, src_image.rows * FLAGS_resize_after_sr), 0, 0, cv::INTER_CUBIC);
      }
    }
    XPLINFO << "super res time (sec):\t\t" << time::timeSinceSec(sr_timer);
    XPLINFO << "size after super res" << image_super.size(); 

    cv::Mat output_frame;
    if (!FLAGS_inflate) {
      output_frame = image_super;
      //cv::imwrite(FLAGS_dest_dir + "/src_" + string::intToZeroPad(frame_num, 6) + ".png", src_image);
      //cv::imwrite(FLAGS_dest_dir + "/super_" + string::intToZeroPad(frame_num, 6) + ".png", image_super);
    } else {
      cv::Mat L_super, R_super;
      if (image_super.rows == image_super.cols) { // If it is square, assume mono
        L_super = image_super.clone();
        R_super = image_super.clone();
      } else {
        L_super = image_super(cv::Rect(0, 0, image_super.cols / 2, image_super.rows));
        R_super = image_super(cv::Rect(image_super.cols / 2, 0, image_super.cols / 2, image_super.rows));
      }

      //cv::imwrite(FLAGS_dest_dir + "/L_super_" + string::intToZeroPad(frame_num, 6) + ".png", L_super);

      // We need to know the image size before we can precompute projection warps
      if (frame_num == 0) {
        const int ftheta_size = L_super.cols; // ftheta size doesn't necessarily have to be equal to this, especially with warp composition, but this is fine.
        cam_ftheta = projection::makePerfectFthetaCamera(ftheta_size);
        cam_inflated = projection::makePerfectFthetaCamera(FLAGS_inflated_size);
        projection::precomputeVR180toFthetaWarp(
          cam_ftheta, 
          ftheta_size, 
          L_super.cols,
          warp_vr180_to_ftheta,
          1.0);
        projection::precomputeFisheyeToInflatedWarp(
          cam_ftheta,
          cam_inflated,
          warp_ftheta_to_inflated,
          FLAGS_inflate_exponent);
        // Make the warp from VR180 directly to inflated using warp composition
        warp_vr180_to_inflated = projection::composeWarps(warp_ftheta_to_inflated, warp_vr180_to_ftheta);
      }

      cv::Mat L_inflated = projection::warp(L_super, warp_vr180_to_inflated, cv::INTER_CUBIC);
      cv::Mat R_inflated = projection::warp(R_super, warp_vr180_to_inflated, cv::INTER_CUBIC);
      if (FLAGS_final_resize != 0) {
        const double sigma = 0.5;
        const int kernel_size = 2 * static_cast<int>(std::ceil(3 * sigma)) + 1;
        cv::GaussianBlur(L_inflated, L_inflated, cv::Size(kernel_size, kernel_size), sigma, sigma);
        cv::GaussianBlur(R_inflated, R_inflated, cv::Size(kernel_size, kernel_size), sigma, sigma);
        cv::resize(L_inflated, L_inflated, cv::Size(FLAGS_final_resize, FLAGS_final_resize), 0, 0, cv::INTER_CUBIC);
        cv::resize(R_inflated, R_inflated, cv::Size(FLAGS_final_resize, FLAGS_final_resize), 0, 0, cv::INTER_CUBIC);
      }
      cv::hconcat(L_inflated, R_inflated, output_frame);
      XPLINFO << "L_inflated size=" << L_inflated.size();
      cv::imwrite(FLAGS_dest_dir + "/L_inflated.jpg", L_inflated * 255.0);
  
      // L_ftheta and L_image are unused, just for visualization
      //cv::Mat L_ftheta = projection::warp(L_super, warp_vr180_to_ftheta, cv::INTER_CUBIC);
      //cv::imwrite(FLAGS_dest_dir + "/L_ftheta_" + string::intToZeroPad(frame_num, 6) + ".png", L_ftheta);
      //cv::Mat L_image = src_image(cv::Rect(0, 0, src_image.cols / 2, src_image.rows));
      //cv::imwrite(FLAGS_dest_dir + "/L_image_" + string::intToZeroPad(frame_num, 6) + ".png", L_image);
      //cv::imwrite(FLAGS_dest_dir + "/L_super_" + string::intToZeroPad(frame_num, 6) + ".png", L_super);
      //cv::imwrite(FLAGS_dest_dir + "/L_inflated" + string::intToZeroPad(frame_num, 6) + ".png", L_inflated);

      if (FLAGS_make_ab) {
        // Resize the input frame (VR180) to the same size as the output (inflated)
        cv::Mat vr180_resized;
        if (src_image.rows == src_image.cols) { // assume square -> mono
          cv::Mat lr;
          cv::hconcat(src_image, src_image, lr);
          cv::resize(lr, vr180_resized, output_frame.size(), 0, 0, cv::INTER_CUBIC);
        } else {
          cv::resize(src_image, vr180_resized, output_frame.size(), 0, 0, cv::INTER_CUBIC);
        }

        cv::putText(vr180_resized, "VR180",
          cv::Point(vr180_resized.cols * 0.25, vr180_resized.rows * 0.3),
          cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(0, 255, 0), 1, cv::LINE_AA);
        cv::putText(vr180_resized, "VR180",
          cv::Point(vr180_resized.cols * 0.75, vr180_resized.rows * 0.3),
          cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(0, 0, 0), 1, cv::LINE_AA);

        cv::vconcat(vr180_resized, output_frame, output_frame);
      }
    }

    // Write the final frame to ffmpeg stream
    if (frame_num == 0) {
      //if (is_photo || FLAGS_make_ab) {
      if (is_photo) { // allow AB video
        const std::string dest_photo_path = FLAGS_dest_dir + "/enhanced"+(FLAGS_inflate ? "_inflated":"")+".jpg";
        cv::imwrite(
            dest_photo_path,
            output_frame * (output_frame.type()==CV_32FC3 ? 255.0f : 1),
            {cv::IMWRITE_JPEG_QUALITY, 95});
        return;
      }

      output_vid_width = output_frame.cols;
      output_vid_height = output_frame.rows;

      const std::string dest_vid_path = FLAGS_dest_dir +
        "/enhanced" + (FLAGS_inflate ? "_inflated":"") + (FLAGS_make_ab ? "_ab":"") 
        + suggestFilenameSuffix(FLAGS_video_encoder, output_vid_width, output_vid_height, FLAGS_crf);

      XCHECK(out_stream == nullptr);
      video::EncoderConfig cfg;
      cfg.crf = FLAGS_crf;
      out_stream = std::make_unique<video::OutputVideoStream>(
        dest_vid_path, output_vid_width, output_vid_height,
        in_stream.guessFrameRate(), FLAGS_video_encoder, cfg);
    }

    if (!out_stream->valid()) {
      XPLERROR << "Invalid output stream. Cannot write frame " << frame_num;
      break;
    }

    video::MediaFrame mf;
    mf.img = output_frame;
    if (!out_stream->writeFrame(mf)) {
      XPLERROR << "Error writing frame";
      break;
    }

    XPLINFO << "total frame time (sec):\t" << time::timeSinceSec(frame_timer);
    ++frame_num;
  }
}

}} // namespace p11::vr180

int main(int argc, char** argv)
{
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::signal(SIGINT, p11::vr180::signalHandler);

  XCHECK(!FLAGS_src_vid.empty());
  XCHECK(!FLAGS_dest_dir.empty());
  XCHECK(!FLAGS_model_path.empty());

  p11::vr180::runEnhanceVR180Pipeline();

  return EXIT_SUCCESS;
}
