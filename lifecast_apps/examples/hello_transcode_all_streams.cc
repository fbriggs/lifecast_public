// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
bazel run -c opt -- //examples:hello_transcode_all_streams --src ~/Downloads/tiger_tiny.mp4 --dest ~/Downloads/test.mp4 --encoder libx264 --crf 18
*/

#include <csignal>
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

std::shared_ptr<std::atomic<bool>> cancel_requested;

constexpr int kMaxParallelTasks = 1;
bool testTranscode() {
  video::InputVideoStream in_stream(FLAGS_src);

  std::atomic<bool> failed{false};

  auto decode_func = [&](std::shared_ptr<std::atomic<bool>> cancel_requested, int frame_num) -> MediaFrame {
    MediaFrame input_frame;
    VideoStreamResult result = in_stream.readFrame(input_frame, CV_32FC3);
    if (result == VideoStreamResult::FINISHED) {
        return MediaFrame();
    } else if (result == VideoStreamResult::ERR) {
        failed = true;
        *cancel_requested = true;
        return MediaFrame();
    }

    if (input_frame.is_video()) {
      XPLINFO << "Decoded frame " << frame_num;
    } else {
      XPLINFO << "Got non-video frame " << frame_num;
    }
    return input_frame;
  };

  auto process_func = [](std::shared_ptr<std::atomic<bool>> cancel_requested, MediaFrame frame, int frame_num) -> MediaFrame {
    if (*cancel_requested) return MediaFrame();

    if (frame.is_video()) {
      XPLINFO << "Processing frame " << frame_num;

      using namespace std::chrono_literals;
      std::this_thread::sleep_for(1000ms);
      cv::blur(frame.img, frame.img, cv::Size(5, 5));
    }
    return frame;
  };

  std::unique_ptr<video::OutputVideoStream> out_stream;
  std::vector<MediaFrame> frames_to_write;
  auto encode_func = [&](std::shared_ptr<std::atomic<bool>> cancel_requested, MediaFrame frame, int frame_num) {
    frames_to_write.push_back(frame);

    if (!frame.is_video()) {
      if (!out_stream) {
        XPLINFO << "Buffering non-video frame";
        return;
      }
    } else {
      XPLINFO << "Encoding frame " << frame_num;

      if (frame_num == 0) {
        XPLINFO << "Initializing output stream";
        XCHECK(!out_stream);
        video::EncoderConfig cfg;
        cfg.crf = FLAGS_crf;
        out_stream = std::make_unique<OutputVideoStream>(
          in_stream,
          FLAGS_dest,
          frame.img.cols,
          frame.img.rows,
          in_stream.guessFrameRate(),
          FLAGS_encoder,
          cfg);
      }

      if (!out_stream->valid()) {
        XPLERROR << "Failed to create output stream";
        *cancel_requested = true;
        failed = true;
        return;
      }
    }

    XCHECK(out_stream);

    for (auto& f: frames_to_write) {
      if (*cancel_requested) return;
      if (!out_stream->writeFrame(f)) {
        XPLERROR << "Error writing frame.";
        *cancel_requested = true;
        failed = true;
      }
    }

    frames_to_write.clear();
  };

  video::transcodeWithThreading(
    decode_func,
    process_func,
    encode_func,
    kMaxParallelTasks,
    cancel_requested);

  return !failed;
}

void signalHandler(int signum) {
  *cancel_requested = true;
  XPLINFO << "interrupted. canceling...";
}

}} // namespace p11::video


int main(int argc, char** argv) {
  p11::video::cancel_requested = std::make_shared<std::atomic<bool>>(false);
  std::signal(SIGINT, p11::video::signalHandler);

  google::ParseCommandLineFlags(&argc, &argv, true);

  return p11::video::testTranscode() ? EXIT_SUCCESS : EXIT_FAILURE;
}
