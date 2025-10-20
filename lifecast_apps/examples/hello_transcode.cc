// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
bazel run -c opt -- //examples:hello_transcode --src ~/Downloads/tiger_tiny.mp4 --dest ~/Downloads/test.mp4 --encoder libx264 --crf 18
*/
#include "source/logger.h"
#include "gflags/gflags.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "source/video_transcode_lib.h"

DEFINE_string(src, "", "path to read input video");
DEFINE_string(dest, "", "path to write output video");
DEFINE_string(encoder, "libx264", "libx264,libx265,prores");
DEFINE_int32(crf, -1, "if specified, override default crf for h264 or h265 encoder");

namespace p11 { namespace video {

void testGetInfo() {
  int width, height, guess_num_frames;
  double guess_fps;
  p11::video::getVideoInfo(FLAGS_src, width, height, guess_fps, guess_num_frames);
  XPLINFO << "width, height: " << width << ", " << height;
  XPLINFO << "guess # frames: " << guess_num_frames;
  XPLINFO << "guess fps: " << guess_fps;
}

void testTranscode() {
  InputVideoStream in_stream(FLAGS_src);
  XCHECK(in_stream.valid()) << "Invalid input video stream: " << FLAGS_src;

  int w = in_stream.getWidth();
  int h = in_stream.getHeight();
  std::pair<int, int> frame_rate = in_stream.guessFrameRate();
  double est_duration = in_stream.guessDurationSec();
  int est_num_frames = in_stream.guessNumFrames();
  XPLINFO << "width, height: " << w << ", " << h;
  XPLINFO << "frame rate: " << frame_rate.first << "/" << frame_rate.second << " = " << (float(frame_rate.first) / frame_rate.second);
  XPLINFO << "estimated duration(sec): " << est_duration;
  XPLINFO << "estimated # frames: " << est_num_frames;

  EncoderConfig cfg;
  cfg.crf = FLAGS_crf;
  OutputVideoStream out_stream(FLAGS_dest, w, h, frame_rate, FLAGS_encoder, cfg);
  XCHECK(out_stream.valid()) << "Invalid output video stream: " << FLAGS_dest;

  int decode_type = CV_32FC3; // CV_8UC3 or CV_32FC3
  MediaFrame frame;
  int frame_count = 0;

  VideoStreamResult result;
  while((result = in_stream.readFrame(frame, decode_type)) == VideoStreamResult::OK) {
    if (!frame.is_video()) continue;

    XPLINFO << "frame: " << frame_count;
    XCHECK_EQ(frame.img.type(), decode_type);

    cv::blur(frame.img, frame.img, cv::Size(5, 5));
    cv::imshow("frame", frame.img); cv::waitKey(100);

    if (!out_stream.writeFrame(frame)) {
      XPLERROR << "Error writing frame";
      break;
    }

    ++frame_count;
  }

  if (result == VideoStreamResult::FINISHED) {
    XPLINFO << "Finished successfully.";
  } else {
    XCHECK_EQ(int(result), int(VideoStreamResult::ERR)) << "There was an error during transcoding.";
  }
}

}} // namespace p11::video


int main(int argc, char** argv) {

  google::ParseCommandLineFlags(&argc, &argv, true);

  p11::video::testTranscode();

  return EXIT_SUCCESS;
}
