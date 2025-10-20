// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

// These forward-decls are not part of public API; TODO move to OutputVideoStreamImpl
struct AVCodec;
struct AVFormatContext;
struct AVStream;

namespace p11 { namespace video {

extern const std::unordered_set<std::string> image_extensions;
bool hasImageExt(std::string filename);

extern const std::unordered_set<std::string> video_extensions;
bool hasVideoExt(std::string filename);

void getVideoInfo( // Alternative to util_ffmpeg::getVideoInfo
  const std::string& input_video_path,
  int& width,
  int& height,
  double& guess_fps,
  int& guess_num_frames);

struct InputVideoStreamImpl;
struct OutputVideoStreamImpl;

struct NonVideoFrame; // Opaque handle

// Either video is a valid mat or non_video is non-null
struct MediaFrame {
  cv::Mat img;
  std::shared_ptr<NonVideoFrame> non_video;

  // video mat can be empty even for video frames if the conversion to cv::Mat failed
  // so is_video uses the status of non_video;
  bool is_video() const { return !non_video; }
};

enum class VideoStreamResult {
  OK,
  FINISHED,
  ERR,
};

struct InputVideoStream {
  InputVideoStream(const std::string& filename);

  InputVideoStream(const InputVideoStream& other) = delete;
  InputVideoStream(InputVideoStream&& other) = default;

  ~InputVideoStream();
  bool valid() const;
  VideoStreamResult readFrame(MediaFrame& frame, int cv_type = CV_32FC3);

  int getWidth() const;
  int getHeight() const;

  std::pair<int, int> guessFrameRate() const { return guessed_frame_rate; }; // returns a rational number e.g. 30/1 for 30 fps
  double guessDurationSec() const { return guessed_duration_sec; } // best guess, may not be accurate
  int guessNumFrames() const { return guessed_num_frames; }

private:
  friend struct OutputVideoStream; // for access to libav internals when duping the non-video streams
  InputVideoStreamImpl* impl;

  std::pair<int, int> guessed_frame_rate = {30, 1};
  double guessed_duration_sec = 0.0;
  int guessed_num_frames = 0;
};

enum ProResProfile {
  PRORES_422LT,
  PRORES_422HQ,
  PRORES_4444,
};

struct EncoderConfig {
  // h264/h265
  int crf = -1;
  std::string preset = "medium";

  // ProRes
  ProResProfile prores_profile = PRORES_422LT;
};

struct OutputVideoStream {
  // Only creates a video stream for the output file. Other frame types are ignored.
  OutputVideoStream(
    const std::string& filename,
    int width,
    int height,
    std::pair<int, int> framerate,
    const std::string encoder_name = "libx264",
    const EncoderConfig& encoder_config = EncoderConfig{});

  // Recreates all the non-video streams in the output file
  OutputVideoStream(
    const InputVideoStream& input_stream,
    const std::string& filename,
    int width,
    int height,
    std::pair<int, int> framerate,
    const std::string encoder_name = "libx264",
    const EncoderConfig& encoder_config = EncoderConfig{});

  OutputVideoStream(const OutputVideoStream& other) = delete;
  OutputVideoStream& operator=(const OutputVideoStream& other) = delete;
  OutputVideoStream(OutputVideoStream&& other) = default;

  ~OutputVideoStream();
  bool valid() const;
  bool writeFrame(const MediaFrame& frame); // returns true if writing succeeds

private:
  OutputVideoStreamImpl* impl = nullptr;
  std::string filename;
  std::string encoder_name;
  int frame_num;

  OutputVideoStream(
    std::function<AVStream*(AVFormatContext*, const AVCodec*)> initialize_streams,
    const std::string& filename,
    int width,
    int height,
    std::pair<int, int> framerate,
    const std::string encoder_name,
    const EncoderConfig& encoder_config
  );

  bool writeImage(cv::Mat& image); // returns true if writing succeeds
};

using Canceler = std::shared_ptr<std::atomic<bool>>;

void transcodeWithThreading(
    std::function<MediaFrame(Canceler, int)> decode_func,
    std::function<MediaFrame(Canceler, MediaFrame, int)> process_func,
    std::function<void(Canceler, MediaFrame, int)> encode_func,
    int max_parallel_tasks,
    Canceler cancel_requested = std::make_shared<std::atomic<bool>>(false));

}}  // namespace p11::video
