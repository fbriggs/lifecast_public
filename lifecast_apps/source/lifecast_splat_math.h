// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "torch/torch.h"
#include "lifecast_splat_config.h"
#include "util_math.h"

namespace p11 { namespace splat {

constexpr float kLinearEncodeRadius = 5.0f;

inline torch::Tensor quaternionTo3x3(torch::Tensor quat) {
  auto x = quat.select(-1, 0);
  auto y = quat.select(-1, 1);
  auto z = quat.select(-1, 2);
  auto w = quat.select(-1, 3);
  auto x2 = x * x;
  auto y2 = y * y;
  auto z2 = z * z;
  auto w2 = w * w;
  auto xy = x * y;
  auto xz = x * z;
  auto xw = x * w;
  auto yz = y * z;
  auto yw = y * w;
  auto zw = z * w;
  auto m00 = x2 - y2 - z2 + w2;
  auto m10 = 2.0 * (xy + zw);
  auto m20 = 2.0 * (xz - yw);
  auto m01 = 2.0 * (xy - zw);
  auto m11 = -x2 + y2 - z2 + w2;
  auto m21 = 2.0 * (yz + xw);
  auto m02 = 2.0 * (xz + yw);
  auto m12 = 2.0 * (yz - xw);
  auto m22 = -x2 - y2 + z2 + w2;
  auto col0 = torch::stack({m00, m10, m20}, -1);
  auto col1 = torch::stack({m01, m11, m21}, -1);
  auto col2 = torch::stack({m02, m12, m22}, -1);
  auto matrix = torch::stack({col0, col1, col2}, -1);
  return matrix;
}

// Assumes R is a tensor of shape (N, 3, 3) representing rotation matrices.
// Returns a tensor of shape (N, 4) containing unit quaternions in XYZW order.
inline torch::Tensor rotationMatrixToQuaternion(const torch::Tensor& R) {
  auto eps = 1e-8;
  auto m00 = R.select(-2, 0).select(-1, 0);
  auto m11 = R.select(-2, 1).select(-1, 1);
  auto m22 = R.select(-2, 2).select(-1, 2);
  auto trace = m00 + m11 + m22;
  
  auto qw = 0.5 * torch::sqrt(torch::clamp(1.0 + trace, eps));
  auto qx = 0.5 * torch::sqrt(torch::clamp(1.0 + m00 - m11 - m22, eps));
  auto qy = 0.5 * torch::sqrt(torch::clamp(1.0 - m00 + m11 - m22, eps));
  auto qz = 0.5 * torch::sqrt(torch::clamp(1.0 - m00 - m11 + m22, eps));
  qx = torch::copysign(qx, R.select(-2, 2).select(-1, 1) - R.select(-2, 1).select(-1, 2));
  qy = torch::copysign(qy, R.select(-2, 0).select(-1, 2) - R.select(-2, 2).select(-1, 0));
  qz = torch::copysign(qz, R.select(-2, 1).select(-1, 0) - R.select(-2, 0).select(-1, 1));
  
  auto quat = torch::stack({qx, qy, qz, qw}, -1);
  quat = quat / torch::norm(quat, 2, -1, true);
  return quat;
}

// Compute the hat operator for a batch of 3-vectors.
// Input: tensor of shape [N, 3]
// Output: tensor of shape [N, 3, 3] where each 3x3 is the skew-symmetric matrix.
inline torch::Tensor hat(const torch::Tensor& omega) {
  auto wx = omega.select(-1, 0);
  auto wy = omega.select(-1, 1);
  auto wz = omega.select(-1, 2);
  auto zeros = torch::zeros_like(wx);
  auto row1 = torch::stack({zeros, -wz, wy}, -1);
  auto row2 = torch::stack({wz, zeros, -wx}, -1);
  auto row3 = torch::stack({-wy, wx, zeros}, -1);
  return torch::stack({row1, row2, row3}, -2);
}

// Exponential map from se(3) to SE(3).
// Input: twist of shape [N, 6] (first three: rotation, last three: translation)
// Output: tuple: (R, t) where R is [N, 3, 3] and t is [N, 3]
inline std::tuple<torch::Tensor, torch::Tensor> se3Exp(const torch::Tensor& xi) {
  auto omega = xi.index({"...", torch::indexing::Slice(0, 3)}); // [N,3]
  auto v     = xi.index({"...", torch::indexing::Slice(3, 6)}); // [N,3]
  auto theta = torch::norm(omega, 2, -1, true);  // [N,1]
  auto eps = 1e-7;
  auto N = xi.size(0);
  auto I = torch::eye(3, torch::TensorOptions().dtype(xi.dtype()).device(xi.device()))
             .unsqueeze(0).expand({N, 3, 3});
  auto omega_hat = hat(omega); // [N,3,3]
  auto A = torch::sin(theta) / (theta + eps);         // [N,1]
  auto B = (1 - torch::cos(theta)) / (theta * theta + eps); // [N,1]
  auto C = (theta - torch::sin(theta)) / (theta * theta * theta + eps); // [N,1]
  auto R = I + A.unsqueeze(-1) * omega_hat + B.unsqueeze(-1) * torch::bmm(omega_hat, omega_hat);
  auto J = I + B.unsqueeze(-1) * omega_hat + C.unsqueeze(-1) * torch::bmm(omega_hat, omega_hat);
  auto t = torch::bmm(J, v.unsqueeze(-1)).squeeze(-1); // [N,3]
  return std::make_tuple(R, t);
}

inline float sigmoidInverse(float x) { 
  x = math::clamp(x, 0.001f, 0.999f);
  float y = std::log(x) - std::log(1.0 - x);
  XCHECK(!std::isnan(y) && !std::isinf(y)) << "x=" << x;
  return y;
}

inline torch::Tensor sigmoidInverse(torch::Tensor x) {
  torch::Tensor x_ = torch::clamp(x, 1e-7, 1.0 - 1e-7);
  return torch::log(x_ / (1.0 - x_));
}

// [-infinity, infinity] -> [-kMaxScaleExponent, 0]
inline torch::Tensor scaleActivation(torch::Tensor s) {
  return -kMaxScaleExponent * torch::sigmoid(s) + kScaleBias;
}

// [-kMaxScaleExponent, 0] -> [-infinity, infinity] 
inline float inverseScaleActivation(float s) {
  const float v = -(s - kScaleBias)/kMaxScaleExponent;
   // avoid taking sigmoidInverse of values outside [0, 1]. These
   // could arise if we load a PLY file with splats that have scales outside [-kMaxScaleExponent, 0]
  return sigmoidInverse(math::clamp(v, 1e-6f, 1.0f - 1e-6f));
}

inline torch::Tensor inverseScaleActivation(torch::Tensor s) {
  return sigmoidInverse(-(s - kScaleBias) / kMaxScaleExponent);
}

// similar to expandUnbounded / contractUnbounded
inline torch::Tensor splatPosInverseActivation(torch::Tensor x) {
  return x;

  torch::Tensor scaled = x / kLinearEncodeRadius;
  torch::Tensor mag = scaled.norm(2, -1, /*keepdim=*/true);
  torch::Tensor safe_mag = torch::clamp(mag, 1e-6, 1e10);
  torch::Tensor direction = scaled / safe_mag;
  return torch::where(
    mag < 1.0,
    scaled,
    direction * (2.0 - (1.0 / safe_mag)));
}

inline torch::Tensor splatPosActivation(torch::Tensor y) {
  return y;

  torch::Tensor mag = y.norm(2, -1, /*keepdim=*/true);
  torch::Tensor safe_factor = mag * (2.0 - mag);
  safe_factor = torch::clamp(safe_factor, 1e-6, 1e10);
  return kLinearEncodeRadius * torch::where(
    mag < 1.0,
    y,
    y / safe_factor);
}

// expand from [-2, +2] to [-infinity, +infinity]
inline Eigen::Vector3f expandUnbounded(const Eigen::Vector3f& y) {
  const float mag = y.norm();
  return kLinearEncodeRadius * (mag < 1.0f ? y : y / (mag * (2.0f - mag)));
}

// contract from [-infinity, +infinity] to [-2, +2]
inline Eigen::Vector3f contractUnbounded(Eigen::Vector3f x) {
  x /= kLinearEncodeRadius;
  const float mag = x.norm();
  return mag < 1.0f ? x : (2.0f - (1.0f / mag)) * (x / mag); 
}

inline torch::Tensor isometryToTensor(const torch::DeviceType device, const Eigen::Isometry3d& isometry) {
  // HACK this does an implicit transpose because Eigen and Torch use different major order
  Eigen::Matrix4f matrix = isometry.matrix().cast<float>();
  auto tensor = torch::from_blob(matrix.data(), {4, 4}, {torch::kFloat32}).to(device);
  return tensor;
}


}}  // end namespace p11::splat
