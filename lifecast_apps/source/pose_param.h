// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// ceres functions for going back and forth between a 6d parameter vector and a pose
// (Eigen::Isometry)
#pragma once

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace p11 { namespace calibration {

constexpr int kPoseDim = 6;

template <typename T>
std::vector<T> poseToParamVec(const Eigen::Transform<T, 3, Eigen::Isometry>& pose)
{
  T angle_axis[3];
  Eigen::Matrix<T, 3, 3> rotation = pose.linear();
  // WARNING: do not try to pass pose.linear().data() directly to RotationMatrixToAngleAxis. It
  // doesn't work correctly.
  ceres::RotationMatrixToAngleAxis(rotation.data(), angle_axis);

  std::vector<T> param(kPoseDim);
  param[0] = angle_axis[0];
  param[1] = angle_axis[1];
  param[2] = angle_axis[2];
  param[3] = pose.translation().x();
  param[4] = pose.translation().y();
  param[5] = pose.translation().z();
  return param;
}

template <typename T>
Eigen::Transform<T, 3, Eigen::Isometry> paramVecToPose(const std::vector<T>& param)
{
  Eigen::Transform<T, 3, Eigen::Isometry> pose;
  pose.translation().x() = param[3];
  pose.translation().y() = param[4];
  pose.translation().z() = param[5];

  Eigen::Matrix<T, 3, 3> R;
  ceres::AngleAxisToRotationMatrix(param.data(), R.data());
  pose.linear() = R;
  return pose;
}

}}  // namespace p11::calibration
