// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "video_transcode_lib.h"
#include "logger.h"
#include "scope_exit.h"
#include "util_string.h"
#include "util_file.h"
#include "util_opencv.h"
#include "concurrency_lib.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
#include <libavutil/display.h> 
}

namespace p11 { namespace video {

namespace {
struct StaticInitializer {
  StaticInitializer() {
    avformat_network_init();
  }

  ~StaticInitializer() {
    avformat_network_deinit();
  }

} staticInitializer;
}

const std::unordered_set<std::string> image_extensions = {
    "png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp", "heic", "heif"};

bool hasImageExt(std::string filename) {
  return image_extensions.count(file::filenameExtension(filename)) == 1;
}

const std::unordered_set<std::string> video_extensions = {"mp4", "mov", "mkv", "avi", "webm"};

bool hasVideoExt(std::string filename) {
  return video_extensions.count(file::filenameExtension(filename)) == 1;
}

void getVideoInfo(
  const std::string& input_video_path,
  int& width,
  int& height,
  double& guess_fps,
  int& guess_num_frames
) {
  width = 0;
  height = 0;
  guess_fps = 0;
  guess_num_frames = 0;

  if (hasVideoExt(input_video_path)) {
    InputVideoStream in_stream(input_video_path);
    if (!in_stream.valid()) return;
    width = in_stream.getWidth();
    height = in_stream.getHeight();
    std::pair<int, int> frame_rate = in_stream.guessFrameRate();
    guess_fps = float(frame_rate.first) / frame_rate.second;
    guess_num_frames = in_stream.guessNumFrames();
  } else if (hasImageExt(input_video_path)) {
    const cv::Mat image = cv::imread(input_video_path);
    if (image.empty()) return;
    width = image.cols;
    height = image.rows;
    guess_fps = 0;
    guess_num_frames = 1;
  } else {
    XPLWARN << "Can't get info for non-image/non-video file " << input_video_path;
  }
}

namespace {

cv::Mat avframeToCvMat(const AVFrame* frame, int cv_type) {
  const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(static_cast<AVPixelFormat>(frame->format));
  int bits_per_channel = desc->comp[0].depth;

  AVPixelFormat dst_format =
      (bits_per_channel > 8 && cv_type == CV_32FC3) ? AV_PIX_FMT_BGR48LE : AV_PIX_FMT_BGR24;
  int buffer_cv_type = (dst_format == AV_PIX_FMT_BGR48LE) ? CV_16UC3 : CV_8UC3;

  SwsContext* conversion = sws_getContext(
    frame->width, frame->height, static_cast<AVPixelFormat>(frame->format),
    frame->width, frame->height, dst_format,
    SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);

  if (!conversion) return cv::Mat();

  // Create an intermediate image buffer with the initial type
  cv::Mat image(frame->height, frame->linesize[0], buffer_cv_type);
  uint8_t* dest[AV_NUM_DATA_POINTERS] = {image.data, nullptr /*...*/};
  int dest_linesize[AV_NUM_DATA_POINTERS] = {static_cast<int>(image.step[0]), 0 /*...*/};

  // Perform the conversion into the intermediate buffer
  sws_scale(conversion, frame->data, frame->linesize, 0, frame->height, dest, dest_linesize);
  sws_freeContext(conversion);

  cv::Rect roi(0, 0, frame->width, frame->height);
  image = image(roi);

  if (cv_type == CV_32FC3) {
    image.convertTo(image, CV_32FC3, (bits_per_channel > 8) ? (1.0 / 65535.0) : (1.0 / 255.0));
  }
  return image;
}

AVFrame* cvMatToAvframe(const cv::Mat& image, AVPixelFormat pix_fmt) {
  AVFrame* frame = av_frame_alloc();
  frame->format = pix_fmt;
  frame->width = image.cols;
  frame->height = image.rows;

  if (av_frame_get_buffer(frame, 32) < 0) {
    av_frame_free(&frame);
    return nullptr;
  }

  AVPixelFormat src_fmt;
  const uint8_t* src_data;
  int src_stride;
  cv::Mat temp;

  switch (image.type()) {
    case CV_8UC3:
      src_fmt = AV_PIX_FMT_BGR24;
      src_data = image.data;
      src_stride = static_cast<int>(image.step[0]);
      break;

    case CV_32FC3:
      if (pix_fmt == AV_PIX_FMT_YUV420P10LE || pix_fmt == AV_PIX_FMT_YUV422P10LE || pix_fmt == AV_PIX_FMT_YUV444P10LE) {
        temp = cv::Mat(image.size(), CV_16UC3);
        image.convertTo(temp, CV_16UC3, 65535.0);
        src_fmt = AV_PIX_FMT_BGR48LE;
      } else {
        temp = cv::Mat(image.size(), CV_8UC3);
        image.convertTo(temp, CV_8UC3, 255.0);
        src_fmt = AV_PIX_FMT_BGR24;
      }
      src_data = temp.data;
      src_stride = static_cast<int>(temp.step[0]);
      break;

    default:
      av_frame_free(&frame);
      return nullptr;
  }

  SwsContext* conversion = sws_getContext(
    image.cols, image.rows, src_fmt,
    frame->width, frame->height, pix_fmt,
    SWS_BICUBIC, nullptr, nullptr, nullptr);

  const uint8_t* src_slices[1] = {src_data};
  int src_stride_arr[1] = {src_stride};

  sws_scale(conversion, src_slices, src_stride_arr, 0, image.rows, frame->data, frame->linesize);
  sws_freeContext(conversion);
  return frame;
}

}  // anonymous namespace

struct InputVideoStreamImpl {
  AVFormatContext* fmt_ctx = nullptr;
  AVCodecContext* codec_ctx = nullptr;
  int video_stream_index = -1;
  AVFrame* frame = nullptr;
  int width = 0;
  int height = 0;
  int rotated_width = 0;
  int rotated_height = 0;
  AVPixelFormat pix_fmt;
  cv::Mat image_frame;
  bool finished_single_image = false; // when decoding png, jpg, etc, we need to return OK once, then FINISHED after that
  int rotation_theta = 0; // degrees
};

InputVideoStream::InputVideoStream(const std::string& filename)
  : impl(new InputVideoStreamImpl())
{
  if (hasImageExt(filename)) {
    impl->image_frame = cv::imread(filename);
    if (impl->image_frame.empty()) {
      XPLERROR << "Could not open input file.";
    } else {
      impl->width = impl->image_frame.cols;
      impl->height = impl->image_frame.rows;
      impl->rotated_width = impl->width;
      impl->rotated_height = impl->height;
    }
    return;
  }

  if (avformat_open_input(&impl->fmt_ctx, filename.c_str(), nullptr, nullptr) != 0) {
    XPLERROR << "Could not open input file.";
    return;
  }

  if (avformat_find_stream_info(impl->fmt_ctx, nullptr) < 0) {
    XPLERROR << "Could not find stream information.";
    return;
  }

  // Find a video stream
  for (int i = 0; i < impl->fmt_ctx->nb_streams; ++i) {
    if (impl->fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      impl->video_stream_index = i;
      break;
    }
  }
  if (impl->video_stream_index == -1) {
    XPLERROR << "Could not find a video stream.";
    return;
  }

  // Set up the decoder
  AVCodecParameters* codecpar = impl->fmt_ctx->streams[impl->video_stream_index]->codecpar;
  const AVCodec* decoder = avcodec_find_decoder(codecpar->codec_id);
  if (!decoder) {
    XPLERROR << "Unsupported codec";
    return;
  }

  impl->codec_ctx = avcodec_alloc_context3(decoder);
  if (!impl->codec_ctx) {
    XPLERROR << "Could not allocate codec context.";
    return;
  }

  if (avcodec_parameters_to_context(impl->codec_ctx, codecpar) < 0) {
    XPLERROR << "Could not copy codec parameters.";
    return;
  }

  if (avcodec_open2(impl->codec_ctx, decoder, nullptr) < 0) {
    XPLERROR << "Could not open codec.";
    return;
  }

  impl->frame = av_frame_alloc();
  impl->width = impl->codec_ctx->width;
  impl->height = impl->codec_ctx->height;
  impl->rotated_width = impl->width;
  impl->rotated_height = impl->height;
  impl->pix_fmt = impl->codec_ctx->pix_fmt;

  if (impl->rotation_theta == 90 || impl->rotation_theta == 270) {
    std::swap(impl->rotated_width, impl->rotated_height);
  }

  // Guess frame rate
  AVStream* stream = impl->fmt_ctx->streams[impl->video_stream_index];
  AVRational frame_rate = av_guess_frame_rate(impl->fmt_ctx, stream, nullptr);
  if (frame_rate.num == 0 || frame_rate.den == 0) {
    XPLWARN << "Unable to guess frame rate. Using default.";
    frame_rate = {30, 1};
  }
  guessed_frame_rate = {frame_rate.num, frame_rate.den};

  // First try to get duration from the stream
  if (stream->duration != AV_NOPTS_VALUE) {
    guessed_duration_sec = stream->duration * av_q2d(stream->time_base);
  } else if (impl->fmt_ctx->duration != AV_NOPTS_VALUE) {
    // Fall back to format context duration
    guessed_duration_sec = impl->fmt_ctx->duration / static_cast<double>(AV_TIME_BASE);
  } else {
    XPLWARN << "Could not guess video duration";
  }

  // First try to use nb_frames if it's available
  if (stream->nb_frames > 0) {
    guessed_num_frames = stream->nb_frames;
  } else {
    // Calculate based on duration and frame rate
    double duration = guessDurationSec();
    if (duration > 0) {
      std::pair<int, int> fps = guessFrameRate();
      guessed_num_frames = static_cast<int64_t>(duration * fps.first / fps.second + 0.5);
    } else {
      XPLWARN << "Could not guess frame count";
    }
  }

  // Check if the video is rotated
  AVDictionaryEntry* rotate_tag = av_dict_get(stream->metadata, "rotate", nullptr, 0);
  if (rotate_tag) {
    impl->rotation_theta = std::atoi(rotate_tag->value);
  }
  uint8_t* display_matrix = av_stream_get_side_data(stream, AV_PKT_DATA_DISPLAYMATRIX, nullptr);
  if (display_matrix) {
    impl->rotation_theta = -av_display_rotation_get((int32_t*)display_matrix);
  }
  XPLINFO << "metadata: video rotation theta=" << impl->rotation_theta; 
}

InputVideoStream::~InputVideoStream() {
  if (impl->image_frame.empty()) {
    // clean up video
    av_frame_free(&impl->frame);
    avcodec_free_context(&impl->codec_ctx);
    avformat_close_input(&impl->fmt_ctx);
  }
  delete impl;
}

bool InputVideoStream::valid() const {
  const bool has_image = !impl->image_frame.empty();
  const bool has_video =
      (impl->fmt_ctx != nullptr) &&
      (impl->codec_ctx != nullptr) &&
      (impl->frame != nullptr) &&
      (impl->video_stream_index >= 0) &&
      (impl->width > 0) &&
      (impl->height > 0);
  return has_image || has_video;
}

void rotateImageByAngle(cv::Mat& image, int theta) {
  switch(theta) {
    case 90: cv::rotate(image, image, cv::ROTATE_90_CLOCKWISE); break;
    case 180: cv::rotate(image, image, cv::ROTATE_180); break;
    case 270: cv::rotate(image, image, cv::ROTATE_90_COUNTERCLOCKWISE); break;
    default: XPLWARN << "Unusual rotation angle: " << theta; break;
  }
}

VideoStreamResult InputVideoStream::readFrame(MediaFrame& frame, int cv_type) {
  frame.img = cv::Mat();
  frame.non_video.reset();

  if (impl->finished_single_image) {
    return VideoStreamResult::FINISHED;
  }

  if (!impl->image_frame.empty()) {
    // Swap the image from the impl with the empty image created above
    // Subsequent readFrame will return false due to the `valid()` check below
    // This makes an image behave like a single frame video
    std::swap(frame.img, impl->image_frame);
    opencv::convertToWithAutoScale(frame.img, frame.img, cv_type);
    impl->finished_single_image = true;
    return VideoStreamResult::OK;
  }

  if (!valid()) return VideoStreamResult::ERR;

  AVPacket* packet = av_packet_alloc();
  ScopeExit free_packet([&]{ av_packet_free(&packet); });

  int result;

  while ((result = av_read_frame(impl->fmt_ctx, packet)) >= 0) {
    ScopeExit unref_packet([=]{ av_packet_unref(packet); });

    if (packet->stream_index == impl->video_stream_index) {
      if (avcodec_send_packet(impl->codec_ctx, packet) < 0) {
        continue;
      }
      if (avcodec_receive_frame(impl->codec_ctx, impl->frame) == 0) {
        frame.img = avframeToCvMat(impl->frame, cv_type);
        if (frame.img.empty()) {
          return VideoStreamResult::ERR;
        } else {
          if (impl->rotation_theta != 0) {
            rotateImageByAngle(frame.img, impl->rotation_theta);
          }
          return VideoStreamResult::OK;
        }
      }
    } else {
      AVPacket* packet_ref = av_packet_alloc();
      XCHECK(packet_ref) << "failed to allocate AVPacket. out of memory?";
      av_packet_ref(packet_ref, packet);
      frame.non_video = std::shared_ptr<NonVideoFrame>(
        reinterpret_cast<NonVideoFrame*>(packet_ref),
        [](NonVideoFrame* handle) {
          auto packet = reinterpret_cast<AVPacket*>(handle);
          av_packet_unref(packet);
          av_packet_free(&packet);
        }
      );
      return VideoStreamResult::OK;
    }
  }

  if (result == AVERROR_EOF) {
    return VideoStreamResult::FINISHED;
  } else {
    return VideoStreamResult::ERR;
  }
}

int InputVideoStream::getWidth() const { return impl->rotated_width; }

int InputVideoStream::getHeight() const { return impl->rotated_height; }


struct OutputVideoStreamImpl {
  AVFormatContext* fmt_ctx = nullptr;
  AVCodecContext* codec_ctx = nullptr;
  AVStream* stream = nullptr;
  int frame_count = 0;
  int width = 0;
  int height = 0;
  std::vector<int> stream_index_map;
  AVPixelFormat pix_fmt;

  bool encodeFrame(AVFrame* frame) {
    int ret = avcodec_send_frame(codec_ctx, frame);
    if (ret < 0) {
      XPLERROR << "Error sending frame to encoder.";
      return false;
    }

    AVPacket pkt;
    av_init_packet(&pkt);
    pkt.data = nullptr;
    pkt.size = 0;

    while (ret >= 0) {
      ret = avcodec_receive_packet(codec_ctx, &pkt);
      if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        av_packet_unref(&pkt);
        break;
      } else if (ret < 0) {
        av_packet_unref(&pkt);
        XPLERROR << "Error encoding frame.";
        return false;
      }

      av_packet_rescale_ts(&pkt, codec_ctx->time_base, stream->time_base);
      pkt.stream_index = stream->index;

      if (av_interleaved_write_frame(fmt_ctx, &pkt) < 0) {
        av_packet_unref(&pkt);
        XPLERROR << "Error writing video frame.";
        return false;
      }
      av_packet_unref(&pkt);
    }
    return true;
  }
};

OutputVideoStream::OutputVideoStream(
  std::function<AVStream*(AVFormatContext*, const AVCodec*)> initialize_streams,
  const std::string& filename,
  int width,
  int height,
  std::pair<int, int> framerate,
  const std::string encoder_name,
  const EncoderConfig& encoder_config
) : filename(filename), encoder_name(encoder_name), frame_num(0)
{
  XCHECK(encoder_name == "libx264" || encoder_name == "libx265" || encoder_name == "prores" || encoder_name == "png" || encoder_name == "jpg");
  if (encoder_name == "png" || encoder_name == "jpg") return;

  impl = new OutputVideoStreamImpl();

  if (avformat_alloc_output_context2(&impl->fmt_ctx, nullptr, nullptr, filename.c_str()) < 0) {
    XPLERROR << "Could not create output context.";
    return;
  }

  std::string libav_encoder_name = encoder_name;
  if (encoder_name == "prores") {
    libav_encoder_name = "prores_ks";
#ifdef __APPLE__
    if (encoder_config.prores_profile == PRORES_422LT) {
      libav_encoder_name = "prores_aw";
    }
#endif
  }

  const AVCodec* encoder = avcodec_find_encoder_by_name(libav_encoder_name.c_str());
  if (!encoder) {
    XPLERROR << "Could not find encoder: " << encoder;
  }

  impl->stream = initialize_streams(impl->fmt_ctx, encoder);

  impl->codec_ctx = avcodec_alloc_context3(encoder);
  if (!impl->codec_ctx) {
    XPLERROR << "Could not allocate codec context.";
    return;
  }

  impl->codec_ctx->codec_type = AVMEDIA_TYPE_VIDEO;
  impl->codec_ctx->width = width;
  impl->codec_ctx->height = height;

  AVRational framerate_q;
  framerate_q.num = framerate.first;
  framerate_q.den = framerate.second;
  impl->codec_ctx->framerate = framerate_q;
  impl->codec_ctx->time_base = av_inv_q(framerate_q);

  AVDictionary* codec_options = nullptr;

  if (encoder_name == "libx264") {
    impl->codec_ctx->codec_id = AV_CODEC_ID_H264;
    impl->codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
  } else if (encoder_name == "libx265") {
    impl->codec_ctx->codec_id = AV_CODEC_ID_H265;

#ifdef _WIN32
    impl->codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P; // TODO: for now fall back to 8 bit on windows because 10 bit doesn't work.
#else
    impl->codec_ctx->pix_fmt =  AV_PIX_FMT_YUV420P10LE;
#endif

    // Set HVC1 tag for H.265/HEVC to make it open in QuickTime player on Mac
    impl->stream->codecpar->codec_tag = MKTAG('h', 'v', 'c', '1');
    impl->codec_ctx->codec_tag = MKTAG('h', 'v', 'c', '1');
    av_dict_set(&codec_options, "tag", "hvc1", 0);
  } else if (encoder_name == "prores") {
    impl->codec_ctx->codec_id = AV_CODEC_ID_PRORES;
    impl->codec_ctx->colorspace = AVCOL_SPC_BT709;
    impl->codec_ctx->color_range = AVCOL_RANGE_MPEG;
    switch (encoder_config.prores_profile) {
      case ProResProfile::PRORES_422LT:
        impl->codec_ctx->pix_fmt = AV_PIX_FMT_YUV422P10LE;
#ifdef __linux__
        impl->codec_ctx->profile = FF_PROFILE_PRORES_LT;
#else
        impl->codec_ctx->profile = AV_PROFILE_PRORES_LT;
#endif
#ifdef __APPLE__
        av_dict_set(&codec_options, "profile", "1", 0); // for encoder prores_aw
#else
        av_dict_set(&codec_options, "profile", "lt", 0);
#endif
        break;
      case ProResProfile::PRORES_422HQ:
        impl->codec_ctx->pix_fmt = AV_PIX_FMT_YUV422P10LE;
#ifdef __linux__
        impl->codec_ctx->profile = FF_PROFILE_PRORES_HQ;
#else
        impl->codec_ctx->profile = AV_PROFILE_PRORES_HQ;
#endif
#ifdef __APPLE__
        av_dict_set(&codec_options, "profile", "3", 0);
#else
        av_dict_set(&codec_options, "profile", "hq", 0);
#endif
        break;
      case ProResProfile::PRORES_4444:
        impl->codec_ctx->pix_fmt = AV_PIX_FMT_YUV444P10LE;
#ifdef __linux__
        impl->codec_ctx->profile = FF_PROFILE_PRORES_4444;
#else
        impl->codec_ctx->profile = AV_PROFILE_PRORES_4444;
#endif
#ifdef __APPLE__
        av_dict_set(&codec_options, "profile", "4", 0);
#else
        av_dict_set(&codec_options, "profile", "4444", 0); // TODO: double check on windows?
#endif
        break;
    }
  }

  // Set CRF and preset for h264 and h265.
  // According to Claude, codec_options is allocated the first time we call av_dict_set
  if (encoder_name == "libx264" || encoder_name == "libx265") {
    av_dict_set(&codec_options, "preset", encoder_config.preset.c_str(), 0);
    if (encoder_config.crf >= 0) {
      av_dict_set(&codec_options, "crf", std::to_string(encoder_config.crf).c_str(), 0);
    }
  }

  // Configure codec to include a header if necessary
  if (impl->fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER) {
    impl->codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }

  if (avcodec_open2(impl->codec_ctx, encoder, &codec_options) < 0) {
    XPLERROR << "Could not open codec.";
    return;
  }
  if (codec_options) {
    av_dict_free(&codec_options);
  }

  if (avcodec_parameters_from_context(impl->stream->codecpar, impl->codec_ctx) < 0) {
    XPLERROR << "Could not copy codec parameters.";
    return;
  }

  impl->stream->time_base = impl->codec_ctx->time_base;

  if (!(impl->fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
    if (avio_open(&impl->fmt_ctx->pb, filename.c_str(), AVIO_FLAG_WRITE) < 0) {
      XPLERROR << "Could not open output file:" << filename;
      return;
    }
  }

  if (avformat_write_header(impl->fmt_ctx, nullptr) < 0) {
    XPLERROR << "Error occurred when writing header.";
    return;
  }

  impl->width = width;
  impl->height = height;
  impl->pix_fmt = impl->codec_ctx->pix_fmt;
}

OutputVideoStream::OutputVideoStream(
  const InputVideoStream& input_stream,
  const std::string& filename,
  int width,
  int height,
  std::pair<int, int> framerate,
  const std::string encoder_name,
  const EncoderConfig& encoder_config
) : OutputVideoStream(
      [&input_stream,this](AVFormatContext* fmt_ctx, const AVCodec* encoder) mutable -> AVStream* {
        int video_stream_index = input_stream.impl->video_stream_index;
        AVFormatContext* input_fmt_ctx = input_stream.impl->fmt_ctx;
        AVStream* output_video_stream = nullptr;

        for (int i = 0; i < input_stream.impl->fmt_ctx->nb_streams; ++i) {
          //XPLINFO << "creating output stream " << i;
          impl->stream_index_map.push_back(i);
          if (i == video_stream_index) {
            // Create our custom video stream
            output_video_stream = avformat_new_stream(fmt_ctx, encoder);
            if (!output_video_stream) {
              XPLERROR << "Could not create output video stream.";
              return nullptr;
            }
          } else {
            AVStream* input_avstream = input_fmt_ctx->streams[i];

            if (!avformat_query_codec(
                    fmt_ctx->oformat, input_avstream->codecpar->codec_id, FF_COMPLIANCE_NORMAL)) {
              XPLWARN << "Skipping incompatible input stream " << i;
              impl->stream_index_map.back() = -1;
              continue;
            }

            // Duplicate the non-video stream
            AVStream* dupe_stream = avformat_new_stream(fmt_ctx, nullptr);
            if (!dupe_stream) {
              XPLERROR << "Failed to create non-video output stream " << i;
              return nullptr;
            }

            if (avcodec_parameters_copy(dupe_stream->codecpar, input_avstream->codecpar) < 0) {
              XPLERROR << "Failed to copy parameters for stream " << i;
              return nullptr;
            }

            dupe_stream->time_base = input_avstream->time_base;
            dupe_stream->avg_frame_rate = input_avstream->avg_frame_rate;
            dupe_stream->r_frame_rate = input_avstream->r_frame_rate;
            dupe_stream->sample_aspect_ratio = input_avstream->sample_aspect_ratio;
            dupe_stream->disposition = input_avstream->disposition;

            if (input_avstream->metadata) {
              av_dict_copy(&dupe_stream->metadata, input_avstream->metadata, 0);
              if (!dupe_stream->metadata) {
                XPLERROR << "failed to copy metadata dict (OOM?)";
                return nullptr;
              }
            }

            if (avformat_transfer_internal_stream_timing_info(
                    fmt_ctx->oformat, dupe_stream, input_avstream, AVFMT_TBCF_AUTO) < 0) {
              XPLERROR << "failed to transfer stream timing";
              return nullptr;
            }
          }
        }
        return output_video_stream;
      },
      filename,
      width,
      height,
      framerate,
      encoder_name,
      encoder_config)
{}

OutputVideoStream::OutputVideoStream(
  const std::string& filename,
  int width,
  int height,
  std::pair<int, int> framerate,
  const std::string encoder_name,
  const EncoderConfig& encoder_config
) : OutputVideoStream(
      [](AVFormatContext* fmt_ctx, const AVCodec* encoder)->AVStream* {
        AVStream* output_video_stream = avformat_new_stream(fmt_ctx, encoder);
        if (!output_video_stream) {
          XPLERROR << "Could not create output video stream.";
          return nullptr;
        }
        return output_video_stream;
      },
      filename,
      width,
      height,
      framerate,
      encoder_name,
      encoder_config)
{}

OutputVideoStream::~OutputVideoStream() {
  if (encoder_name == "png" || encoder_name == "jpg") return;

  if (valid()) {
    impl->encodeFrame(nullptr); // flush the encoder
    av_write_trailer(impl->fmt_ctx);
  }

  if (impl->codec_ctx) {
    avcodec_free_context(&impl->codec_ctx);
  }

  if (impl->fmt_ctx) {
    if (!(impl->fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
      avio_closep(&impl->fmt_ctx->pb);
    }
    avformat_free_context(impl->fmt_ctx);
  }

  delete impl;
}

bool OutputVideoStream::valid() const {
  if (encoder_name == "png" || encoder_name == "jpg") return true; // We are just calling imwrite, there is no way to determine an invalid state here

  return  impl->fmt_ctx != nullptr &&
          impl->stream != nullptr &&
          impl->codec_ctx != nullptr &&
          impl->width > 0 &&
          impl->height > 0;
}

bool OutputVideoStream::writeImage(cv::Mat& image) {
  XCHECK(image.type() == CV_8UC3 || image.type() == CV_32FC3);

  if (encoder_name == "png" || encoder_name == "jpg") {
    std::string param = "{FRAME_NUMBER}";
    std::string frame_str = string::intToZeroPad(frame_num, 6);
    std::string filename_with_param(filename);
    size_t pos = filename.find(param);
    if (pos != std::string::npos) {
      filename_with_param.replace(pos, param.length(), frame_str);
    }

    cv::Mat image_quantized;
    opencv::convertToWithAutoScale(image, image_quantized, encoder_name == "png" ? CV_16UC3 : CV_8UC3);
    cv::imwrite(filename_with_param, image_quantized, {cv::IMWRITE_JPEG_QUALITY, 95});

    ++frame_num;
    return true;
  }
  
  AVFrame* frame = cvMatToAvframe(image, impl->pix_fmt);
  if (!frame) {
    XPLERROR << "cvMatToAvframe failed to encode frame";
    return false;
  }

  frame->pts = impl->frame_count++;
  bool result = impl->encodeFrame(frame);
  av_frame_free(&frame);
  ++frame_num;
  return result;
}

bool OutputVideoStream::writeFrame(const MediaFrame& frame) {
  if (frame.is_video()) {
    // TODO: just do the encoding to a packet and fall through to sending a packet?
    return writeImage(const_cast<MediaFrame&>(frame).img);
  }

  AVPacket* packet = reinterpret_cast<AVPacket*>(frame.non_video.get());

  // Remap input stream index to output stream index
  XCHECK_LT(packet->stream_index, impl->stream_index_map.size()) << "Did you forget to pass in_stream to the OutputVideoStream constructor?";
  packet->stream_index = impl->stream_index_map[packet->stream_index];

  if (packet->stream_index < 0) {
    // ignore packets that don't have a mapping (incompatible)
    return true;
  }

  if (av_interleaved_write_frame(impl->fmt_ctx, packet) < 0) {
    XPLERROR << "Error writing non-video frame.";
    return false;
  }
  return true;
}

void transcodeWithThreading(
    std::function<MediaFrame(concurrency::Canceler, int)> decode_func,
    std::function<MediaFrame(concurrency::Canceler, MediaFrame, int)> process_func,
    std::function<void(concurrency::Canceler, MediaFrame, int)> encode_func,
    int max_parallel_tasks,
    std::shared_ptr<std::atomic<bool>> cancel_requested)
{
  // TODO: don't allocate new buffers every time everywhere
  concurrency::CancelableTaskQueue<MediaFrame> output_queue(max_parallel_tasks, cancel_requested);

  std::atomic<bool> done = false;

  std::thread decode_thread([&done, &output_queue, cancel_requested, decode_func, process_func]{
    int frame_num = 0;
    while(!*cancel_requested) {
      MediaFrame frame = decode_func(cancel_requested, frame_num);

      if (*cancel_requested) {
        break;
      }

      if (frame.is_video() && frame.img.empty()) {
        // No more frames
        break;
      }

      concurrency::CancelableTask<MediaFrame> task(
        [frame, frame_num, process_func = std::move(process_func)](concurrency::Canceler inner_canceler) -> MediaFrame {
          return process_func(inner_canceler, frame, frame_num);
      });

      // blocks until queue has room
      output_queue.push(std::move(task));

      if (frame.is_video()) {
        ++frame_num;
      }
    }

    done = true;

    if (*cancel_requested) {
      output_queue.cancel();
    }
  });

  std::thread encode_thread([&done, &output_queue, cancel_requested, &encode_func]{
    int frame_num = 0;
    while (!*cancel_requested) {
      // pop first future from queue (may block if queue empty)
      if (output_queue.empty() && done) break;

      // wait for next item to be available in queue
      std::optional<MediaFrame> maybe_frame = output_queue.waitPop();

      if (!maybe_frame) break;
      if (*cancel_requested) break;

      encode_func(cancel_requested, *maybe_frame, frame_num);
      if (*cancel_requested) {
        output_queue.cancel();
      }

      if (maybe_frame->is_video()) {
        ++frame_num;
      }
    }
  });

  decode_thread.join();
  encode_thread.join();
}

}}  // namespace p11::video
