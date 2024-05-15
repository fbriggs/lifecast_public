// MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#pragma once

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "Eigen/Core"
#include "Eigen/Geometry"
#include "logger.h"
#include "util_math.h"

namespace p11 { namespace opencv {
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
static Eigen::Vector3f cvColor4fToEigenColor3f(const cv::Vec4f& v)
{
  return Eigen::Vector3f(v[2], v[1], v[0]);
}

static inline float distSq(const cv::Point2f& a, const cv::Point2f& b)
{
  return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}

// TODO: rename or better abstraction vs 3f version
static void drawDotAntiAliased(
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

static void drawDotAntiAliased4f(
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

static void drawCrossAntiAliased(
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

static cv::Vec4f colorHash(const size_t x)
{
  return cv::Vec4f(
      255.0 * (0.5 + 0.5 * cosf(x * 5123 + 34)),
      255.0 * (0.5 + 0.5 * cosf(x * 1234 + 12)),
      255.0 * (0.5 + 0.5 * cosf(x * 6734 + 66)),
      255.0);
}

// shift an image horizontally and wrap past edges
static cv::Mat shiftAndWrap(const cv::Mat& image, const int shift_x, const int shift_y)
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

static cv::Mat halfSize(const cv::Mat& image)
{
  cv::Mat _small;
  cv::resize(image, _small, cv::Size(image.cols / 2, image.rows / 2), 0, 0, cv::INTER_AREA);
  return _small;
}

static cv::Mat warp(
    const cv::Mat& src,
    const std::vector<cv::Mat>& warp_uv,
    cv::InterpolationFlags interp_mode = cv::INTER_CUBIC)
{
  cv::Mat output_image;
  cv::remap(
      src,
      output_image,
      warp_uv[0],
      warp_uv[1],
      interp_mode,
      cv::BORDER_CONSTANT,
      cv::Scalar(0, 0, 0, 0));
  return output_image;
}

}}  // namespace p11::opencv
