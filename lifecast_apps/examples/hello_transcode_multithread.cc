// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
bazel run -c opt -- //examples:hello_transcode_multithread --src ~/Downloads/tiger_tiny.mp4 --dest ~/Downloads/test.mp4 --encoder libx264 --crf 18
*/

#include <memory>
#include <thread>

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

constexpr int kMaxParallelTasks = 3;
bool testTranscode() {
  video::InputVideoStream in_stream(FLAGS_src);

  std::atomic<bool> failed{false};

  auto decode_func = [&](std::shared_ptr<std::atomic<bool>> cancel_requested, int frame_num) -> video::MediaFrame {
    MediaFrame input_frame;
    // Skip over non-video frames
    VideoStreamResult result;
    while ((result = in_stream.readFrame(input_frame, CV_32FC3)) == VideoStreamResult::OK) {
      if (input_frame.is_video()) {
        XPLINFO << "Decoded frame " << frame_num;
        return input_frame;
      }
    }
    if (result == VideoStreamResult::ERR) {
      XPLERROR << "There was an error decoding the video";
      *cancel_requested = true; // Stop the presses
    }
    return MediaFrame();
  };

  auto process_func = [](std::shared_ptr<std::atomic<bool>> cancel_requested, video::MediaFrame frame, int frame_num) -> video::MediaFrame {
    XPLINFO << "Processing frame " << frame_num;

    if (*cancel_requested) return video::MediaFrame();

    XCHECK(frame.is_video());

    using namespace std::chrono_literals;
    std::this_thread::sleep_for(1000ms);
    cv::blur(frame.img, frame.img, cv::Size(5, 5));
    return frame;
  };

  std::unique_ptr<video::OutputVideoStream> out_stream;
  auto encode_func = [&](std::shared_ptr<std::atomic<bool>> cancel_requested, video::MediaFrame frame, int frame_num) {
    XCHECK(frame.is_video());
    XPLINFO << "Encode thread received frame " << frame_num;
    if (frame_num == 0) {
      XPLINFO << "Initializing output stream";
      XCHECK(!out_stream);
      video::EncoderConfig cfg;
      cfg.crf = FLAGS_crf;
      out_stream = std::make_unique<OutputVideoStream>(
        FLAGS_dest,
        frame.img.cols,
        frame.img.rows,
        in_stream.guessFrameRate(),
        FLAGS_encoder,
        cfg);
      if (!out_stream->valid()) {
        XPLERROR << "Failed to construct output stream";
        failed = true;
        *cancel_requested = true;
        return;
      }
    }
    XCHECK(out_stream);

    if (!out_stream->writeFrame(frame)) {
      XPLERROR << "Error writing frame.";
      failed = true;
      *cancel_requested = true;
    } else {
      XPLINFO << "Frame " << frame_num << " encoded";
    }
  };

  video::transcodeWithThreading(
    decode_func,
    process_func,
    encode_func,
    kMaxParallelTasks);
  
  return true;
}

}} // namespace p11::video


int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);

  p11::video::testTranscode();

  return EXIT_SUCCESS;
}
