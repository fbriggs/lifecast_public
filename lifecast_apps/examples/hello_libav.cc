// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
bazel run -c opt -- //examples:hello_libav --src ~/Downloads/tiger_tiny.mp4 --dest ~/Downloads/test.mp4
*/
#include "source/logger.h"
#include "gflags/gflags.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

// Shenanagins to avoid silent linker explosion
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/avutil.h>
#include <libavutil/imgutils.h>
}
#ifdef __WIN32
#pragma comment(lib, "avcodec.lib")
#pragma comment(lib, "avformat.lib")
#endif

DEFINE_string(src, "", "path to read input video");
DEFINE_string(dest, "", "path to write output video");

cv::Mat avframeToCvMat(const AVFrame* frame) {
  SwsContext* conversion = sws_getContext(
    frame->width, frame->height, static_cast<AVPixelFormat>(frame->format),
    frame->width, frame->height, AV_PIX_FMT_BGR24,
    SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);

  cv::Mat image(frame->height, frame->width, CV_8UC3);
  uint8_t* dest[4] = { image.data, nullptr, nullptr, nullptr };
  int dest_linesize[4] = { static_cast<int>(image.step[0]), 0, 0, 0 };

  sws_scale(conversion, frame->data, frame->linesize, 0, frame->height, dest, dest_linesize);

  sws_freeContext(conversion);
  return image;
}

AVFrame* cvMatToAvframe(const cv::Mat& image, AVPixelFormat pix_fmt) {
  AVFrame* frame = av_frame_alloc();
  frame->format = pix_fmt;
  frame->width = image.cols;
  frame->height = image.rows;

  if (av_frame_get_buffer(frame, 32) < 0)
    throw std::runtime_error("Could not allocate frame data.");

  SwsContext* conversion = sws_getContext(
    image.cols, image.rows, AV_PIX_FMT_BGR24,
    frame->width, frame->height, pix_fmt,
    SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);

  const uint8_t* src_slices[1] = { image.data };
  int src_stride[1] = { static_cast<int>(image.step[0]) };

  sws_scale(conversion, src_slices, src_stride, 0, image.rows, frame->data, frame->linesize);

  sws_freeContext(conversion);
  return frame;
}

struct VideoReader {
  AVFormatContext* fmt_ctx = nullptr;
  AVCodecContext* codec_ctx = nullptr;
  int video_stream_index = -1;
  AVFrame* frame = nullptr;
  AVPacket* packet = nullptr;
  int width = 0;
  int height = 0;
  AVPixelFormat pix_fmt;

  VideoReader(const std::string filename) {

    if (avformat_open_input(&fmt_ctx, filename.c_str(), nullptr, nullptr) != 0)
      throw std::runtime_error("Could not open input file.");

    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0)
      throw std::runtime_error("Could not find stream information.");

    // Find the video stream
    for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {
      if (fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
        video_stream_index = i;
        break;
      }
    }
    if (video_stream_index == -1)
      throw std::runtime_error("Could not find a video stream.");

    // Set up the decoder
    AVCodecParameters* codecpar = fmt_ctx->streams[video_stream_index]->codecpar;
    const AVCodec* decoder = avcodec_find_decoder(codecpar->codec_id);
    if (!decoder)
      throw std::runtime_error("Unsupported codec.");

    codec_ctx = avcodec_alloc_context3(decoder);
    if (!codec_ctx)
      throw std::runtime_error("Could not allocate codec context.");

    if (avcodec_parameters_to_context(codec_ctx, codecpar) < 0)
      throw std::runtime_error("Could not copy codec parameters.");

    if (avcodec_open2(codec_ctx, decoder, nullptr) < 0)
      throw std::runtime_error("Could not open codec.");

    // Allocate frame and packet
    frame = av_frame_alloc();
    packet = av_packet_alloc();

    width = codec_ctx->width;
    height = codec_ctx->height;
    pix_fmt = codec_ctx->pix_fmt;
  }

  ~VideoReader() {
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&codec_ctx);
    avformat_close_input(&fmt_ctx);
  }

  bool readFrame(cv::Mat& image) {
    while (av_read_frame(fmt_ctx, packet) >= 0) {
      if (packet->stream_index == video_stream_index) {
        if (avcodec_send_packet(codec_ctx, packet) < 0) {
          av_packet_unref(packet);
          continue;
        }
        av_packet_unref(packet);

        if (avcodec_receive_frame(codec_ctx, frame) == 0) {
          image = avframeToCvMat(frame);
          return true;
        }
      } else {
        av_packet_unref(packet);
      }
    }
    return false;
  }
};

struct VideoWriter {
  AVFormatContext* fmt_ctx = nullptr;
  AVCodecContext* codec_ctx = nullptr;
  AVStream* stream = nullptr;
  int frame_count = 0;
  int width = 0;
  int height = 0;
  AVPixelFormat pix_fmt;

  VideoWriter(const std::string filename, int width, int height, AVRational time_base) {
    if (avformat_alloc_output_context2(&fmt_ctx, nullptr, nullptr, filename.c_str()) < 0)
      throw std::runtime_error("Could not create output context.");

    // Find the encoder
    const AVCodec* encoder = avcodec_find_encoder_by_name("libx264");
    //const AVCodec* encoder = avcodec_find_encoder_by_name("libx265");
    //const AVCodec* encoder = avcodec_find_encoder_by_name("prores_ks");

    if (!encoder)
      throw std::runtime_error("Encoder not found.");

    // Create the output stream
    stream = avformat_new_stream(fmt_ctx, encoder);
    if (!stream)
      throw std::runtime_error("Could not create stream.");

    codec_ctx = avcodec_alloc_context3(encoder);
    if (!codec_ctx)
      throw std::runtime_error("Could not allocate codec context.");

    //avcodec_get_context_defaults3 (codec_ctx, encoder);

    // Example parameters for h264 or h265 (not necessarily correct)
    codec_ctx->codec_id = AV_CODEC_ID_H264;
    //codec_ctx->codec_id = AV_CODEC_ID_H265;
    codec_ctx->codec_type = AVMEDIA_TYPE_VIDEO;
    codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    codec_ctx->width = width;
    codec_ctx->height = height;
    codec_ctx->time_base = time_base;
    codec_ctx->framerate = av_inv_q(time_base);
    codec_ctx->gop_size = 12;
    codec_ctx->max_b_frames = 2;

    XPLINFO << "codec_ctx->time_base=" << codec_ctx->time_base.num << "/" << codec_ctx->time_base.den;
    XPLINFO << "codec_ctx->framerate=" << codec_ctx->framerate.num << "/" << codec_ctx->framerate.den;

    //codec_ctx->width = width;
    //codec_ctx->height = height;
    //codec_ctx->time_base = time_base;
    //codec_ctx->framerate = av_inv_q(time_base);
    //codec_ctx->codec_id = AV_CODEC_ID_PRORES;
    //codec_ctx->colorspace = AVCOL_SPC_BT709;
    //codec_ctx->color_range = AVCOL_RANGE_MPEG;
    //codec_ctx->pix_fmt = AV_PIX_FMT_YUV422P10LE;
    ////codec_ctx->profile = AV_PROFILE_PRORES_STANDARD;

    // TODO: not sure what this does
    if (fmt_ctx->oformat->flags & AVFMT_GLOBALHEADER)
      codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    if (avcodec_open2(codec_ctx, encoder, nullptr) < 0)
      throw std::runtime_error("Could not open codec.");

    if (avcodec_parameters_from_context(stream->codecpar, codec_ctx) < 0)
      throw std::runtime_error("Could not copy codec parameters.");

    stream->time_base = codec_ctx->time_base;
    XPLINFO << "Stream time base: " << av_q2d(stream->time_base);
    
    if (!(fmt_ctx->oformat->flags & AVFMT_NOFILE)) {
      if (avio_open(&fmt_ctx->pb, filename.c_str(), AVIO_FLAG_WRITE) < 0)
        throw std::runtime_error("Could not open output file.");
    }

    if (avformat_write_header(fmt_ctx, nullptr) < 0)
      throw std::runtime_error("Error occurred when writing header.");

    this->width = width;
    this->height = height;
    pix_fmt = codec_ctx->pix_fmt;
  
    XPLINFO << "-----";
    XPLINFO << "Codec time base: " << av_q2d(codec_ctx->time_base);
    XPLINFO << "Stream time base: " << av_q2d(stream->time_base);
  }

  ~VideoWriter() {
    // Flush encoder
    encodeFrame(nullptr);

    // Write trailer
    av_write_trailer(fmt_ctx);

    // Close codec and file
    avcodec_free_context(&codec_ctx);
    if (!(fmt_ctx->oformat->flags & AVFMT_NOFILE))
      avio_closep(&fmt_ctx->pb);
    avformat_free_context(fmt_ctx);

    avformat_network_deinit();
  }

  void writeFrame(const cv::Mat& image) {
    AVFrame* frame = cvMatToAvframe(image, pix_fmt);
    frame->pts = frame_count++;

    encodeFrame(frame);

    av_frame_free(&frame);
  }

  void encodeFrame(AVFrame* frame) {

    int ret = avcodec_send_frame(codec_ctx, frame);
    if (ret < 0)
      throw std::runtime_error("Error sending frame to encoder.");

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
        throw std::runtime_error("Error encoding frame.");
      }

      av_packet_rescale_ts(&pkt, codec_ctx->time_base, stream->time_base);
      pkt.stream_index = stream->index;

      if (av_interleaved_write_frame(fmt_ctx, &pkt) < 0) {
        av_packet_unref(&pkt);
        throw std::runtime_error("Error writing frame.");
      }
      av_packet_unref(&pkt);
    }
  }
};

int main(int argc, char** argv) {

  google::ParseCommandLineFlags(&argc, &argv, true);

  avformat_network_init(); // init ffmpeg. TODO: only call this once?

  try {
    XPLINFO << "Attempting to open video: " << FLAGS_src;
    VideoReader reader(FLAGS_src);
    //AVRational time_base = reader.codec_ctx->time_base;
    int width = reader.width;
    int height = reader.height;
    XPLINFO << "width=" << width << " height=" << height;

    // Get input frame rate
    AVStream* in_stream = reader.fmt_ctx->streams[reader.video_stream_index];
    AVRational frame_rate = av_guess_frame_rate(reader.fmt_ctx, in_stream, nullptr);
    AVRational time_base = av_inv_q(frame_rate);
    if (time_base.num == 0 || time_base.den == 0) {
      XPLINFO << "Using default time base / frame rate";
      time_base = {1, 1}; // Default to 1 fps
    }
    XPLINFO << "input stream guess time_base: " << time_base.num << "/" << time_base.den;

    VideoWriter writer(FLAGS_dest, width, height, time_base);

    cv::Mat frame;
    while (reader.readFrame(frame)) {
      cv::blur(frame, frame, cv::Size(5, 5));
      cv::imshow("frame", frame); cv::waitKey(1);
      writer.writeFrame(frame);
    }
  } catch (const std::exception& ex) {
    XPLERROR << ex.what();
  }

  avformat_network_deinit();

  return EXIT_SUCCESS;
}
