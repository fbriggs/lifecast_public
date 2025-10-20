// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

/*
Example usage (Mac or Linux):

bazel run -- //examples:hello_rof \
--src_image1 000000.jpg \
--src_image2 000001.jpg
*/

#include "gflags/gflags.h"
#include "source/logger.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "source/rof.h"
#include "torch/torch.h"
#include "torch/script.h"

#ifdef _WIN32
#include <Windows.h>
#endif

DEFINE_string(src_image1, "", "Path to read image 1");
DEFINE_string(src_image2, "", "Path to read image 2");
DEFINE_string(model_path, "", "Path to RAFT model (optional, if empty, a default will be used)");

int main(int argc, char** argv)
{
// Workaround a bug in libtorch for windows where it links against the wrong dll and doesn't support
// CUDA. See https://github.com/pytorch/pytorch/issues/72396
#ifdef _WIN32
  LoadLibraryA("torch_cuda.dll");
#endif

  google::ParseCommandLineFlags(&argc, &argv, true);

  XCHECK(!FLAGS_src_image1.empty());
  XCHECK(!FLAGS_src_image2.empty());

  cv::Mat image1 = cv::imread(FLAGS_src_image1);
  cv::Mat image2 = cv::imread(FLAGS_src_image2);

  torch::jit::script::Module raft_module;
  p11::optical_flow::getTorchModelRAFT(raft_module, FLAGS_model_path);

  XPLINFO << "Finished loading model";

  cv::Mat flow_x, flow_y;
  p11::optical_flow::computeOpticalFlowRAFT(raft_module, image1, image2, flow_x, flow_y);

  XPLINFO << "Finshed computing optical flow";

  cv::imshow("flow_x", flow_x * 0.05 * 0.1);
  cv::waitKey(0);
}
