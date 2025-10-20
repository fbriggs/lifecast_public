// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
bazel run -- //examples:hello_depth_anything2 --src ~/Downloads/frame1.png
*/

#include "gflags/gflags.h"
#include "source/logger.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "source/depth_anything2.h"
#include "torch/torch.h"
#include "torch/script.h"

#ifdef _WIN32
#include <Windows.h>
#endif

DEFINE_string(src, "", "Path to image");

int main(int argc, char** argv)
{
// Workaround a bug in libtorch for windows where it links against the wrong dll and doesn't support
// CUDA. See https://github.com/pytorch/pytorch/issues/72396
#ifdef _WIN32
  LoadLibraryA("torch_cuda.dll");
#endif

  google::ParseCommandLineFlags(&argc, &argv, true);

  XCHECK(!FLAGS_src.empty());

  cv::Mat image = cv::imread(FLAGS_src);

  torch::jit::getProfilingMode() = false;
  torch::jit::script::Module model;
  p11::depth_estimation::getTorchModelDepthAnything2(model);

  XPLINFO << "Finished loading model";

  bool normalize_output = true;
  bool resize_output = true;
  cv::Mat depth = p11::depth_estimation::estimateMonoDepthWithDepthAnything2(model, image, normalize_output, resize_output);
  cv::imshow("depth", depth * 0.1);
  cv::waitKey(0);


  //cv::Mat depth2 = p11::depth_estimation::estimateMonoDepthWithPatchesHighRes(model, image);
  //cv::imshow("depth2", depth2);
  //cv::waitKey(0);
}
