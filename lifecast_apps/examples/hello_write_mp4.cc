// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
bazel run -- //examples:hello_write_mp4 --dest_vid ~/Desktop/test.mp4
*/
#include "gflags/gflags.h"
#include "source/logger.h"

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/optflow.hpp"

DEFINE_string(dest_vid, "", "Path to write video");

int main(int argc, char** argv)
{
  google::ParseCommandLineFlags(&argc, &argv, true);

  XCHECK(!FLAGS_dest_vid.empty());

  cv::Mat image(cv::Size(3840, 3840), CV_8UC3, cv::Scalar(0, 0, 0));

  cv::VideoWriter video_writer(
      FLAGS_dest_vid,
      cv::VideoWriter::fourcc('H', '2', '6', '4'),
      // cv::VideoWriter::fourcc('p','n','g',' '),
      30,
      image.size());

  XPLINFO << "Encoder: " << video_writer.getBackendName();

  for (int frame = 0; frame < 30; ++frame) {
    XPLINFO << frame;
    for (int y = 0; y < image.rows; ++y) {
      for (int x = 0; x < image.cols; ++x) {
        image.at<cv::Vec3b>(y, x) = cv::Vec3b(rand() % 255, rand() % 255, rand() % 255);
      }
    }
    video_writer.write(image);
  }
}
