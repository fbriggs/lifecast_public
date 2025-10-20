// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "logger.h"
#include "util_math.h"

namespace p11 { namespace opencv {

// Like cv::Mat.converTo but automatically determines the appropriate scaling
inline void convertToWithAutoScale(const cv::Mat& src, cv::Mat& dst, int rtype) {
  if (rtype == src.type()) { dst = src.clone(); return ; }
  int src_depth = src.depth();
  int dst_depth = CV_MAT_DEPTH(rtype);
  double scale = 1.0;
  if (src_depth == CV_8U && dst_depth == CV_16U) {
    scale = 256.0; // 8U -> 16U
  } else if (src_depth == CV_8U && dst_depth == CV_32F) {
    scale = 1.0 / 255.0; // 8U -> 32F
  } else if (src_depth == CV_16U && dst_depth == CV_8U) {
    scale = 1.0 / 256.0; // 16U -> 8U. GPT-4o claims this is correct.
  } else if (src_depth == CV_16U && dst_depth == CV_32F) {
    scale = 1.0 / 65535.0; // 16U -> 32F
  } else if (src_depth == CV_32F && dst_depth == CV_8U) {
    scale = 255.0; // 32F -> 8U
  } else if (src_depth == CV_32F && dst_depth == CV_16U) {
    scale = 65535.0; // 32F -> 16U
  } else {
    XCHECK(false) << "Unsupported combination of types for conversion: " << rtype << " " << src.type();
  }
  src.convertTo(dst, rtype, scale);
}

inline std::string typeToString(int type) {
  std::string r, channel;
  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);
  switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }
  channel = (chans > 1) ? ("C" + std::to_string(chans)) : "";
  return r + channel;
}

template <typename T>
T getPixelExtendIntCoord(const cv::Mat& img, int ix, int iy)
{
  ix = p11::math::clamp(ix, 0, img.cols - 1);
  iy = p11::math::clamp(iy, 0, img.rows - 1);
  return img.at<T>(iy, ix);
}

template <typename T>
T getPixelExtend(const cv::Mat& img, float x, float y)
{
  int ix = std::floor(x);
  int iy = std::floor(y);
  ix = p11::math::clamp(ix, 0, img.cols - 1);
  iy = p11::math::clamp(iy, 0, img.rows - 1);
  return img.at<T>(iy, ix);
}

template <typename T>
T getPixelBilinear(const cv::Mat& img, float x, float y)
{
  // get the surrounding pixels and find the relative coordinates
  const float x0 = std::floor(x);
  const float y0 = std::floor(y);
  const float x1 = std::ceil(x);
  const float y1 = std::ceil(y);
  const float xR = x - x0;
  const float yR = y - y0;

  // get the image value at each neighbor pixel
  const T f00 = getPixelExtend<T>(img, x0, y0);
  const T f01 = getPixelExtend<T>(img, x0, y1);
  const T f10 = getPixelExtend<T>(img, x1, y0);
  const T f11 = getPixelExtend<T>(img, x1, y1);

  // compute the linear regression parameters (closed form solution)
  const T a1 = f00;
  const T a2 = f10 - f00;
  const T a3 = f01 - f00;
  const T a4 = f00 + f11 - f10 - f01;

  // use the model parameters to predict the value at (x,y)
  return a1 + a2 * xR + a3 * yR + a4 * xR * yR;
}

// BGRA->RGB
inline Eigen::Vector3f cvColor4fToEigenColor3f(const cv::Vec4f& v)
{
  return Eigen::Vector3f(v[2], v[1], v[0]);
}

inline float distSq(const cv::Point2f& a, const cv::Point2f& b)
{
  return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}

// TODO: rename or better abstraction vs 3f version
inline void drawDotAntiAliased(
    const cv::Point2f& point, const cv::Vec4f& color, cv::Mat& dest_image)
{
  XCHECK_EQ(dest_image.type(), CV_8UC4);

  const int ix = int(point.x);
  const int iy = int(point.y);
  const float rx = point.x - ix;
  const float ry = point.y - iy;

  if (ix < 0 || iy < 0 || ix + 1 >= dest_image.cols || iy + 1 >= dest_image.rows) {
    return;
  }

  const cv::Vec4f color00(dest_image.at<cv::Vec4b>(iy, ix));
  const cv::Vec4f color10(dest_image.at<cv::Vec4b>(iy + 1, ix));
  const cv::Vec4f color01(dest_image.at<cv::Vec4b>(iy, ix + 1));
  const cv::Vec4f color11(dest_image.at<cv::Vec4b>(iy + 1, ix + 1));

  const float alpha00 = (1.0 - rx) * (1.0 - ry);
  const float alpha10 = (1.0 - rx) * ry;
  const float alpha01 = rx * (1.0 - ry);
  const float alpha11 = rx * ry;

  dest_image.at<cv::Vec4b>(iy, ix) = cv::Vec4b(color * alpha00 + color00 * (1.0 - alpha00));
  dest_image.at<cv::Vec4b>(iy + 1, ix) = cv::Vec4b(color * alpha10 + color10 * (1.0 - alpha10));
  dest_image.at<cv::Vec4b>(iy, ix + 1) = cv::Vec4b(color * alpha01 + color01 * (1.0 - alpha01));
  dest_image.at<cv::Vec4b>(iy + 1, ix + 1) = cv::Vec4b(color * alpha11 + color11 * (1.0 - alpha11));
}

inline void drawDotAntiAliased4f(
    const cv::Point2f& point, const cv::Vec4f& color, cv::Mat& dest_image)
{
  XCHECK_EQ(dest_image.type(), CV_32FC4);

  const int ix = int(point.x);
  const int iy = int(point.y);
  const float rx = point.x - ix;
  const float ry = point.y - iy;

  if (ix < 0 || iy < 0 || ix + 1 >= dest_image.cols || iy + 1 >= dest_image.rows) {
    return;
  }

  const cv::Vec4f color00(dest_image.at<cv::Vec4f>(iy, ix));
  const cv::Vec4f color10(dest_image.at<cv::Vec4f>(iy + 1, ix));
  const cv::Vec4f color01(dest_image.at<cv::Vec4f>(iy, ix + 1));
  const cv::Vec4f color11(dest_image.at<cv::Vec4f>(iy + 1, ix + 1));

  const float alpha00 = (1.0 - rx) * (1.0 - ry);
  const float alpha10 = (1.0 - rx) * ry;
  const float alpha01 = rx * (1.0 - ry);
  const float alpha11 = rx * ry;

  dest_image.at<cv::Vec4f>(iy, ix) = color * alpha00 + color00 * (1.0 - alpha00);
  dest_image.at<cv::Vec4f>(iy + 1, ix) = color * alpha10 + color10 * (1.0 - alpha10);
  dest_image.at<cv::Vec4f>(iy, ix + 1) = color * alpha01 + color01 * (1.0 - alpha01);
  dest_image.at<cv::Vec4f>(iy + 1, ix + 1) = color * alpha11 + color11 * (1.0 - alpha11);
}

inline void drawCrossAntiAliased(
    const cv::Point2f& point, const cv::Vec4f& color, cv::Mat& dest_image)
{
  drawDotAntiAliased(point, color, dest_image);
  drawDotAntiAliased(point + cv::Point2f(+1, 0), color, dest_image);
  drawDotAntiAliased(point + cv::Point2f(+2, 0), color, dest_image);
  drawDotAntiAliased(point + cv::Point2f(-1, 0), color, dest_image);
  drawDotAntiAliased(point + cv::Point2f(-2, 0), color, dest_image);
  drawDotAntiAliased(point + cv::Point2f(0, +1), color, dest_image);
  drawDotAntiAliased(point + cv::Point2f(0, +2), color, dest_image);
  drawDotAntiAliased(point + cv::Point2f(0, -1), color, dest_image);
  drawDotAntiAliased(point + cv::Point2f(0, -2), color, dest_image);
}

inline cv::Vec4f colorHash(const size_t x)
{
  return cv::Vec4f(
      255.0 * (0.5 + 0.5 * cosf(x * 5123 + 34)),
      255.0 * (0.5 + 0.5 * cosf(x * 1234 + 12)),
      255.0 * (0.5 + 0.5 * cosf(x * 6734 + 66)),
      255.0);
}

// shift an image horizontally and wrap past edges
inline cv::Mat shiftAndWrap(const cv::Mat& image, const int shift_x, const int shift_y)
{
  cv::Mat affine = (cv::Mat_<double>(2, 3) << 1, 0, shift_x, 0, 1, shift_y);
  cv::Mat dest;
  warpAffine(image, dest, affine, image.size(), cv::INTER_NEAREST, cv::BORDER_WRAP);
  return dest;
}

template <typename T>
cv::Point2f toCvPoint2f(const Eigen::Matrix<T, 2, 1>& p)
{
  return cv::Point2f(p.x(), p.y());
}

template <typename T>
void copyCircle(
    const cv::Mat& circle_image, cv::Mat& dest_image, const int offset_x, const int offset_y)
{
  for (int y = 0; y < circle_image.rows; ++y) {
    for (int x = 0; x < circle_image.cols; ++x) {
      int dx = x - circle_image.cols / 2;
      int dy = y - circle_image.rows / 2;
      if (dx * dx + dy * dy > circle_image.rows * circle_image.rows / 4) continue;

      int dest_x = x + offset_x;
      int dest_y = y + offset_y;
      if (dest_x < 0 || dest_y < 0 || dest_x >= dest_image.cols || dest_y >= dest_image.rows)
        continue;
      dest_image.at<T>(dest_y, dest_x) = circle_image.at<T>(y, x);
    }
  }
}

inline cv::Mat halfSize(const cv::Mat& image)
{
  cv::Mat _small;
  cv::resize(image, _small, cv::Size(image.cols / 2, image.rows / 2), 0, 0, cv::INTER_AREA);
  return _small;
}

// O3-mini optimized version
inline cv::Mat bilateralDenoise(cv::Mat& src, int radius, float sigma, float sigma_color) {
  XCHECK_EQ(src.type(), CV_32FC3);
  const int h = src.rows;
  const int w = src.cols;
  const int kernel_size = 2 * radius + 1;

  cv::Mat dest(h, w, CV_32FC3);

  // Precompute spatial weights (depend only on offset u,v)
  std::vector<float> spatial_weights(kernel_size * kernel_size);
  const float sigma2 = sigma * sigma;
  for (int v = -radius; v <= radius; ++v) {
    for (int u = -radius; u <= radius; ++u) {
      float weight = std::exp(-(u * u + v * v) / (2.0f * sigma2));
      spatial_weights[(v + radius) * kernel_size + (u + radius)] = weight;
    }
  }

  #pragma omp parallel for
  for (int y = 0; y < h; ++y) {
    cv::Vec3f* dest_row = dest.ptr<cv::Vec3f>(y);
    for (int x = 0; x < w; ++x) {
      cv::Vec3f center_color = src.at<cv::Vec3f>(y, x);
      cv::Vec3f sum_color(0, 0, 0);
      float sum_weight = 0.0f;
      for (int v = -radius; v <= radius; ++v) {
        int y2 = y + v;
        if (y2 < 0 || y2 >= h) continue;
        const cv::Vec3f* src_row = src.ptr<cv::Vec3f>(y2);
        for (int u = -radius; u <= radius; ++u) {
          int x2 = x + u;
          if (x2 < 0 || x2 >= w) continue;
          cv::Vec3f sample_color = src_row[x2];

          // Use precomputed spatial weight
          float spatial_weight = spatial_weights[(v + radius) * kernel_size + (u + radius)];

          // Compute range weight based on color difference
          float diff0 = sample_color[0] - center_color[0];
          float diff1 = sample_color[1] - center_color[1];
          float diff2 = sample_color[2] - center_color[2];
          float color_weight = std::exp(-(diff0 * diff0 + diff1 * diff1 + diff2 * diff2) /
                                         (2.0f * sigma_color * sigma_color));
          float weight = spatial_weight * color_weight;
          sum_color += sample_color * weight;
          sum_weight += weight;
        }
      }
      dest_row[x] = sum_color / sum_weight;
    }
  }

  return dest;
}

}}  // namespace p11::opencv
