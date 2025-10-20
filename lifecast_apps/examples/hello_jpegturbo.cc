// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
bazel run -c opt -- //examples:hello_jpegturbo \
--src_image ~/Desktop/spongebob.jpg \
--dest_image ~/Desktop/turbo.jpg
*/
#include <chrono>
#include <string>

#include "gflags/gflags.h"
#include "source/logger.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/optflow.hpp"
#include "turbojpeg.h"

DEFINE_string(src_image, "", "Path to image to load");
DEFINE_string(dest_image, "", "Path to write image");

typedef std::chrono::high_resolution_clock Clock;

void writeJpegTurbo(const std::string& dest_filename, const cv::Mat& image)
{
  constexpr int kJpegQuality = 75;
  long unsigned int jpeg_size = 0;  // Memory is allocated by tjCompress2 if == 0
  tjhandle jpeg_compressor = tjInitCompress();

  auto start_time = Clock::now();
  unsigned char* compressed_image = nullptr;
  tjCompress2(
      jpeg_compressor,
      image.ptr(),
      image.cols,
      0,
      image.rows,
      TJPF_BGR,
      &compressed_image,
      &jpeg_size,
      TJSAMP_444,
      kJpegQuality,
      TJFLAG_FASTDCT);

  FILE* file = fopen(dest_filename.c_str(), "wb");
  XCHECK(file) << "Failed to open file: " << dest_filename;

  fwrite(compressed_image, jpeg_size, 1, file);
  fclose(file);

  XPLINFO << "Turbo compress + write time:"
          << std::chrono::duration<double>(Clock::now() - start_time).count() << std::endl;

  tjDestroy(jpeg_compressor);
  tjFree(compressed_image);
}

int main(int argc, char** argv)
{
  google::ParseCommandLineFlags(&argc, &argv, true);

  const cv::Mat image = cv::imread(FLAGS_src_image);
  cv::imshow("image", image);
  cv::waitKey(0);

  writeJpegTurbo(FLAGS_dest_image, image);

  auto start_time = Clock::now();
  cv::imwrite(FLAGS_dest_image, image);
  XPLINFO << "cv::imwrite time:" << std::chrono::duration<double>(Clock::now() - start_time).count()
          << std::endl;
}
