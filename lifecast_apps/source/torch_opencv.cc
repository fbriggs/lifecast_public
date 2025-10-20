// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "torch_opencv.h"
#include "util_torch.h"

namespace p11 { namespace torch_opencv {

torch::Tensor cvMat8UC3_to_Tensor(const torch::DeviceType device, cv::Mat image) {
  XCHECK_EQ(image.type(), CV_8UC3); 
  std::vector<uint8_t> pixel_data(image.rows * image.cols * 3);
  int pix = 0;
  for (int y = 0; y < image.rows; ++y) {
    for (int x = 0; x < image.cols; ++x) {
      const cv::Vec3b bgr = image.at<cv::Vec3b>(y, x);
      pixel_data[pix + 0] = bgr[0];
      pixel_data[pix + 1] = bgr[1];
      pixel_data[pix + 2] = bgr[2];
      pix += 3;
    }
  }
  torch::Tensor image_tensor = torch::from_blob(pixel_data.data(), {image.rows, image.cols, 3}, {torch::kUInt8}).to(device);
  image_tensor = image_tensor.to(torch::kFloat32).div(255.0);
  image_tensor = image_tensor.permute({2, 0, 1});
  util_torch::cloneIfOnCPU(image_tensor);
  return image_tensor;
}

torch::Tensor cvMat8UC1_to_Tensor(const torch::DeviceType device, cv::Mat image) {
  XCHECK_EQ(image.type(), CV_8UC1); 
  std::vector<uint8_t> pixel_data(image.rows * image.cols);
  int pix = 0;
  for (int y = 0; y < image.rows; ++y) {
    for (int x = 0; x < image.cols; ++x) {
      const uint8_t val = image.at<uint8_t>(y, x);
      pixel_data[pix] = val;
      ++pix;
    }
  }
  torch::Tensor image_tensor = torch::from_blob(pixel_data.data(), {image.rows, image.cols, 1}, {torch::kUInt8}).to(device);
  image_tensor = image_tensor.to(torch::kFloat32).div(255.0);
  image_tensor = image_tensor.permute({2, 0, 1});
  util_torch::cloneIfOnCPU(image_tensor);
  return image_tensor;
}

torch::Tensor cvMat_to_Tensor(const torch::DeviceType device, const cv::Mat& image) {
  if (image.type() == CV_8UC1) { // Fall back to original optimized code for 8 bit input
    return cvMat8UC1_to_Tensor(device, image);
  }
  if (image.type() == CV_8UC3) { // Fall back to original optimized code for 8 bit input
    return cvMat8UC3_to_Tensor(device, image);
  }
  XCHECK(image.type() == CV_8UC3 || image.type() == CV_16UC3 || image.type() == CV_32FC3 || image.type() == CV_32FC1 || image.type() == CV_32FC2);

  cv::Mat image_32f;
  if (image.type() == CV_16UC3) {
    image.convertTo(image_32f, CV_32FC3, 1.0 / 65535.0f);
  } else {
    image_32f = image;
  }
  int num_channels = image.channels();
  std::vector<float> pixel_data(image.rows * image.cols * num_channels);
  int pix = 0;
  if (num_channels == 3) {
    for (int y = 0; y < image_32f.rows; ++y) {
      for (int x = 0; x < image_32f.cols; ++x) {
        const cv::Vec3f bgr = image_32f.at<cv::Vec3f>(y, x);
        pixel_data[pix + 0] = bgr[0];
        pixel_data[pix + 1] = bgr[1];
        pixel_data[pix + 2] = bgr[2];
        pix += 3;
      }
    }
  } else if (num_channels == 1) {
    for (int y = 0; y < image_32f.rows; ++y) {
      for (int x = 0; x < image_32f.cols; ++x) {
        pixel_data[pix] = image_32f.at<float>(y, x);
        ++pix;
      }
    }
  } else if (num_channels == 2) { // e.g. for optical flow
    for (int y = 0; y < image_32f.rows; ++y) {
      for (int x = 0; x < image_32f.cols; ++x) {
        const cv::Vec2f fxy = image_32f.at<cv::Vec2f>(y, x);
        pixel_data[pix + 0] = fxy[0];
        pixel_data[pix + 1] = fxy[1];
        pix += 2;
      }
    }
  } else { XCHECK(false) << "Unsupported number of channels"; }

  torch::Tensor image_tensor = torch::from_blob(pixel_data.data(), {image.rows, image.cols, num_channels}, torch::kFloat32).to(device);
  image_tensor = image_tensor.permute({2, 0, 1});
  util_torch::cloneIfOnCPU(image_tensor);
  return image_tensor;
}

void fastTensor_To_CvMat(torch::Tensor image_tensor, cv::Mat& dest) {
  // bring to CPU and ensure contiguous layout
  auto img = image_tensor.to(torch::kCPU).contiguous();
  int d = img.dim();

  if (d == 2) {
    // single-channel [h,w]
    int h = img.size(0), w = img.size(1);
    dest.create(h, w, CV_32FC1);
    auto view = torch::from_blob(dest.data, {h, w}, torch::kFloat32);
    view.copy_(img);

  } else if (d == 3) {
    // multi-channel [h,w,c]
    int h = img.size(0), w = img.size(1), c = img.size(2);
    int type = (c == 1 ? CV_32FC1 : CV_32FC3);
    dest.create(h, w, type);
    auto view = torch::from_blob(dest.data, {h, w, c}, torch::kFloat32);
    view.copy_(img);

  } else {
    XCHECK(false) << "fastTensor_To_CvMat: unsupported tensor dims: " + d;
  }
}

}}  // namespace p11::torch_opencv
