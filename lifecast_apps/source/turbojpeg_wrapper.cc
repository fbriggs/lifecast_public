// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "turbojpeg_wrapper.h"
#include <fstream>
#include <iostream>
#include "logger.h"
#include "turbojpeg.h"
#include "util_file.h"

namespace p11 { namespace turbojpeg {

void writeJpeg(const std::string& dest_filename, const cv::Mat& image, const int jpeg_quality)
{
  if (file::filenameExtension(dest_filename) != "jpg") {
    cv::imwrite(dest_filename, image);
    return;
  }

  long unsigned int jpeg_size = 0;              // Memory is allocated by tjCompress2 if == 0
  tjhandle jpeg_compressor = tjInitCompress();  // TODO: pre-allocate jpeg_compressor and buffer

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
      jpeg_quality,
      TJFLAG_FASTDCT);

  FILE* file = fopen(dest_filename.c_str(), "wb");
  XCHECK(file) << "Failed to open file: " << dest_filename;

  fwrite(compressed_image, jpeg_size, 1, file);
  fclose(file);

  tjDestroy(jpeg_compressor);
  tjFree(compressed_image);
}

cv::Mat readJpeg(const std::string& filename)
{
  if (file::filenameExtension(filename) != "jpg") {
    // LOG(WARNING) << "image is not a jpeg, falling back to cv::imread(), filename: " << filename;
    return cv::imread(filename);
  }

  std::ifstream file(filename, std::ifstream::binary);
  XCHECK(file.is_open()) << "filename: " << filename;
  file.seekg(0, file.end);
  long unsigned int file_length = file.tellg();
  file.seekg(0, file.beg);
  char* compressed_data = new char[file_length];
  file.read(compressed_data, file_length);
  file.close();

  int width, height;
  tjhandle jpeg_decompressor = tjInitDecompress();
  tjDecompressHeader(
      jpeg_decompressor, (unsigned char*)compressed_data, file_length, &width, &height);

  cv::Mat image(cv::Size(width, height), CV_8UC3);
  tjDecompress2(
      jpeg_decompressor,
      (unsigned char*)compressed_data,
      file_length,
      image.ptr(),
      width,
      /*pitch=*/0,
      height,
      TJPF_BGR,
      TJFLAG_FASTDCT);

  tjDestroy(jpeg_decompressor);
  delete[] compressed_data;

  return image;
}

void compressJpegInMemory(
    const cv::Mat& image, const int jpeg_quality, std::vector<unsigned char>& buffer)
{
  long unsigned int jpeg_size = 0;              // Memory is allocated by tjCompress2 if == 0
  tjhandle jpeg_compressor = tjInitCompress();  // TODO: pre-allocate jpeg_compressor and buffer

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
      jpeg_quality,
      TJFLAG_FASTDCT);

  // TODO: this is making a copy. there must be a more efficient way.
  // we want the data in an std::vector primarily for MJPEGStreamer.
  buffer = std::vector<unsigned char>(compressed_image, compressed_image + jpeg_size);

  tjDestroy(jpeg_compressor);
  tjFree(compressed_image);
}

}}  // namespace p11::turbojpeg
