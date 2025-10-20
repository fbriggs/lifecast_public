// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "lifecast_splat_io.h"

#include <map>
#include "lifecast_splat_math.h"
#include "util_math.h"
#include "util_torch.h"
#include "util_file.h"

namespace p11 { namespace splat {

namespace {
// How many pixels vertically are occupied by data from each splat
constexpr int kRowsPerSplat = 20;
}

std::tuple<uint8_t, uint8_t, uint8_t> encode12bitIn3Pixels(float v) {
  static constexpr int kNumBits = 12;
  v = std::clamp(v, 0.0f, 1.0f);
  const int iv = static_cast<int>(v * ((1 << kNumBits) - 1));

  // Extract 4 bits for each pixel
  int low4 = iv & 0xF;
  int mid4 = (iv >> 4) & 0xF;
  int high4 = (iv >> 8) & 0xF;

  // Scale to use the full 0-255 range, centering the values
  uint8_t pixel1 = static_cast<uint8_t>(low4 * 16 + 8);
  uint8_t pixel2 = static_cast<uint8_t>(mid4 * 16 + 8);
  uint8_t pixel3 = static_cast<uint8_t>(high4 * 16 + 8);

  return {pixel1, pixel2, pixel3};
}

float decode12BitFrom3Pixels(uint8_t p1, uint8_t p2, uint8_t p3) {
  int low4 = (p1 / 16) & 0xF;
  int mid4 = (p2 / 16) & 0xF;
  int high4 = (p3 / 16) & 0xF;
  int i12 = (high4 << 8) | (mid4 << 4) | low4;
  float f12 = float(i12) / float((1 << 12) - 1);
  return f12;
}

cv::Mat encodeSplatsInImage(
  const std::vector<SerializableSplat>& splats,
  const int w,
  const int h
) {
  XCHECK_LE(splats.size(), (w * (h / kRowsPerSplat))) << "Too many splats for image size";

  cv::Mat image(cv::Size(w, h), CV_8UC3, cv::Scalar(0, 0, 0));
  int num_encoded = 0; // valid splat index, may be <= j due to some splats outside bounding box
  for (int j = 0; j < splats.size(); ++j) {
    // [-infinity, +infinity] --> [-2, +2] --> [0, 1]
    const Eigen::Vector3f npos = 0.25 * (contractUnbounded(splats[j].pos) + Eigen::Vector3f(2, 2, 2));
    if (!(npos.x() >= 0 && npos.y() >= 0 && npos.z() >= 0 && npos.x() <= 1.0 && npos.y() <= 1.0 && npos.z() <= 1.0)
      || (splats[j].quat.norm() == 0.0f)) {
      //XPLINFO << "WARNING: skipping splat unfit for encoding";
      //XPLINFO << "Bad splat pos: " << splats[j].pos.x() << " " << splats[j].pos.y() << " " << splats[j].pos.z();
      ++num_encoded; // Without this, null splats accumulate at the end, and disrupt temporal stability. With it, null splats become zero in the encoding.
      continue;
    }

    const int row = (num_encoded / w) * kRowsPerSplat;
    const int col = num_encoded % w;

    // Position - 12 bit encoding (9 pixels per splat)
    const auto [x_p1, x_p2, x_p3] = encode12bitIn3Pixels(npos.x());
    const auto [y_p1, y_p2, y_p3] = encode12bitIn3Pixels(npos.y());
    const auto [z_p1, z_p2, z_p3] = encode12bitIn3Pixels(npos.z());
    image.at<cv::Vec3b>(row + 0, col) = cv::Vec3b(x_p1, x_p1, x_p1);
    image.at<cv::Vec3b>(row + 1, col) = cv::Vec3b(x_p2, x_p2, x_p2);
    image.at<cv::Vec3b>(row + 2, col) = cv::Vec3b(x_p3, x_p3, x_p3);
    image.at<cv::Vec3b>(row + 3, col) = cv::Vec3b(y_p1, y_p1, y_p1);
    image.at<cv::Vec3b>(row + 4, col) = cv::Vec3b(y_p2, y_p2, y_p2);
    image.at<cv::Vec3b>(row + 5, col) = cv::Vec3b(y_p3, y_p3, y_p3);
    image.at<cv::Vec3b>(row + 6, col) = cv::Vec3b(z_p1, z_p1, z_p1);
    image.at<cv::Vec3b>(row + 7, col) = cv::Vec3b(z_p2, z_p2, z_p2);
    image.at<cv::Vec3b>(row + 8, col) = cv::Vec3b(z_p3, z_p3, z_p3);

    // Color - well use 3 pixels to store RGB so it survives yuv420 pixel formats
    Eigen::Vector4f rgba = splats[j].color * 255;
    image.at<cv::Vec3b>(row + 9, col) = cv::Vec3b(rgba.x(), rgba.x(), rgba.x()); 
    image.at<cv::Vec3b>(row + 10, col) = cv::Vec3b(rgba.y(), rgba.y(), rgba.y()); 
    image.at<cv::Vec3b>(row + 11, col) = cv::Vec3b(rgba.z(), rgba.z(), rgba.z()); 
    image.at<cv::Vec3b>(row + 12, col) = cv::Vec3b(rgba.w(), rgba.w(), rgba.w()); 
    
    // Scale
    //XCHECK(splats[j].scale.x() <= 0); // This shouldn't happen. We have an activation function.
    //XCHECK(splats[j].scale.y() <= 0);
    //XCHECK(splats[j].scale.z() <= 0);
    //Eigen::Vector3f encoded_scale = -splats[j].scale; // Flip negative to positive
    Eigen::Vector3f encoded_scale = Eigen::Vector3f(kScaleBias, kScaleBias, kScaleBias) - splats[j].scale; // Flip negative to positive, include bias term
    encoded_scale.x() = p11::math::clamp(encoded_scale.x() / kMaxScaleExponent, 0.0f, 1.0f);
    encoded_scale.y() = p11::math::clamp(encoded_scale.y() / kMaxScaleExponent, 0.0f, 1.0f);
    encoded_scale.z() = p11::math::clamp(encoded_scale.z() / kMaxScaleExponent, 0.0f, 1.0f);
    const Eigen::Vector3f s = encoded_scale * 255;
    image.at<cv::Vec3b>(row + 13, col) = cv::Vec3b(s.x(), s.x(), s.x());
    image.at<cv::Vec3b>(row + 14, col) = cv::Vec3b(s.y(), s.y(), s.y());
    image.at<cv::Vec3b>(row + 15, col) = cv::Vec3b(s.z(), s.z(), s.z());

    // Rotation (4 element quaternion). Normalized so that max element is either +1 or -1.
    Eigen::Vector4f q = splats[j].quat / splats[j].quat.cwiseAbs().maxCoeff();
    q = 255.0f * (q + Eigen::Vector4f(1.0f, 1.0f, 1.0f, 1.0f)) / 2.0f;
    image.at<cv::Vec3b>(row + 16, col) = cv::Vec3b(q.x(), q.x(), q.x());
    image.at<cv::Vec3b>(row + 17, col) = cv::Vec3b(q.y(), q.y(), q.y());
    image.at<cv::Vec3b>(row + 18, col) = cv::Vec3b(q.z(), q.z(), q.z());
    image.at<cv::Vec3b>(row + 19, col) = cv::Vec3b(q.w(), q.w(), q.w());

    ++num_encoded;
  }
  return image;
}

cv::Mat encodeSerializableSplatsWithSizzleZ(std::vector<SerializableSplat>& splats, int w, int h) {
  // Swizzle from lifecast (+z = forward) to opengl (-z = forward)
  for (auto& s : splats) {
    s.pos = Eigen::Vector3f(s.pos.x(), s.pos.y(), -s.pos.z());
    s.quat = Eigen::Vector4f(-s.quat.x(), s.quat.y(), s.quat.z(), -s.quat.w()); // NOTE: the quaternion swizzle corresponding to a -z flip was found by brute force enumeration.
  }
  
  // Pre-sort splats by distance from the origin, assuming the player won't do this. This destroys temporal coherence when compressing video.
  //std::sort(splats.begin(), splats.end(), [](const SerializableSplat& a, const SerializableSplat& b) { return a.pos.norm() > b.pos.norm(); });
  
  return encodeSplatsInImage(splats, w, h);
}

cv::Mat encodeSplatModelWithSizzleZ(std::shared_ptr<SplatModel> model, int w, int h) {
  std::vector<SerializableSplat> splats = splatModelToSerializable(model);
  return encodeSerializableSplatsWithSizzleZ(splats, w, h);
}

void saveEncodedSplatFileWithSizzleZ(const std::string& png_filename, std::shared_ptr<SplatModel> model, int w, int h) {
  const cv::Mat encoded_image = encodeSplatModelWithSizzleZ(model, w, h);
  file::createDirectoryIfNotExists(file::getDirectoryName(png_filename));
  XCHECK(cv::imwrite(png_filename, encoded_image));
}

std::vector<SerializableSplat> loadSplatPLYFile(const std::string& filename) {
  std::ifstream file(filename, std::ios::binary);
  XCHECK(file) << "Error opening ply: " << filename;

  std::string line;
  size_t vertex_count = 0;
  while (std::getline(file, line)) {
    if (line.find("element vertex") != std::string::npos) {
      std::sscanf(line.c_str(), "element vertex %zu", &vertex_count);
    }
    if (line == "end_header") {
      break;
    }
  }

  std::vector<SerializableSplat> splats;
  for (size_t i = 0; i < vertex_count; ++i) {
    Eigen::Vector3f pos, normal, dc, scale;
    Eigen::Vector4f rot;
    float opacity;
    Eigen::Matrix<float, 1, 45> rest;

    file.read(reinterpret_cast<char*>(&pos[0]), sizeof(float) * 3);
    file.read(reinterpret_cast<char*>(&normal[0]), sizeof(float) * 3);
    file.read(reinterpret_cast<char*>(&dc[0]), sizeof(float) * 3);
    file.read(reinterpret_cast<char*>(rest.data()), sizeof(float) * 45);
    file.read(reinterpret_cast<char*>(&opacity), sizeof(float));
    file.read(reinterpret_cast<char*>(&scale[0]), sizeof(float) * 3);
    file.read(reinterpret_cast<char*>(&rot[0]), sizeof(float) * 4);
    
    // Scales are (usually negative) exponents
    scale.x() = std::min(scale.x(), 0.0f);
    scale.y() = std::min(scale.y(), 0.0f);
    scale.z() = std::min(scale.z(), 0.0f);

    SerializableSplat splat;
    splat.pos = pos;
    splat.scale = scale;
    splat.quat = rot; // Not necessarily a normalized quaternion.

    // convert to "DC" terms to RGB in [0, 1]
    // Opacity has to go through a sigmoid to get [0, 1] range.
    constexpr float SH_C0 = 0.28209479177387814;
    splat.color = Eigen::Vector4f(
      p11::math::clamp(0.5f + dc.x() * SH_C0, 0.0f, 1.0f),
      p11::math::clamp(0.5f + dc.y() * SH_C0, 0.0f, 1.0f),
      p11::math::clamp(0.5f + dc.z() * SH_C0, 0.0f, 1.0f),
      1.0 / (1.0 + std::exp(-opacity))
    );

    splats.push_back(splat);
  }
  file.close();

  return splats;
}

std::vector<SerializableSplat> decodeSplatImage(const cv::Mat& image) {
  const int num_splats = image.cols * (image.rows / kRowsPerSplat); // NOTE: parens here are important for order of rounding

  std::vector<SerializableSplat> splats;
  for (int j = 0; j < num_splats; ++j) {
    SerializableSplat splat;
    const int row = (j / image.cols) * kRowsPerSplat;
    const int col = j % image.cols;
    
    uint8_t x_p1 = image.at<cv::Vec3b>(row + 0, col)[0];
    uint8_t x_p2 = image.at<cv::Vec3b>(row + 1, col)[0];
    uint8_t x_p3 = image.at<cv::Vec3b>(row + 2, col)[0];
    uint8_t y_p1 = image.at<cv::Vec3b>(row + 3, col)[0];
    uint8_t y_p2 = image.at<cv::Vec3b>(row + 4, col)[0];
    uint8_t y_p3 = image.at<cv::Vec3b>(row + 5, col)[0];
    uint8_t z_p1 = image.at<cv::Vec3b>(row + 6, col)[0];
    uint8_t z_p2 = image.at<cv::Vec3b>(row + 7, col)[0];
    uint8_t z_p3 = image.at<cv::Vec3b>(row + 8, col)[0];
    const Eigen::Vector3f npos( // [0, 1]
      decode12BitFrom3Pixels(x_p1, x_p2, x_p3),
      decode12BitFrom3Pixels(y_p1, y_p2, y_p3),
      decode12BitFrom3Pixels(z_p1, z_p2, z_p3));

    splat.pos = expandUnbounded(4.0 * (npos - Eigen::Vector3f(0.5, 0.5, 0.5)));
  
    splat.color = Eigen::Vector4f(
      image.at<cv::Vec3b>(row + 9, col)[0],
      image.at<cv::Vec3b>(row + 10, col)[0],
      image.at<cv::Vec3b>(row + 11, col)[0],
      image.at<cv::Vec3b>(row + 12, col)[0]) / 255.0f;
    if (splat.color.w() == 0) continue; // skip zero alpha splats

    splat.scale = Eigen::Vector3f(
      image.at<cv::Vec3b>(row + 13, col)[0],
      image.at<cv::Vec3b>(row + 14, col)[0],
      image.at<cv::Vec3b>(row + 15, col)[0]) * (-kMaxScaleExponent / 255.0f) +
      Eigen::Vector3f(kScaleBias, kScaleBias, kScaleBias);

    splat.quat = Eigen::Vector4f(
      image.at<cv::Vec3b>(row + 16, col)[0],
      image.at<cv::Vec3b>(row + 17, col)[0],
      image.at<cv::Vec3b>(row + 18, col)[0],
      image.at<cv::Vec3b>(row + 19, col)[0]) * (2.0f / 255.0f)
      - Eigen::Vector4f(1, 1, 1, 1);

    // Undo swizzle we did when encoding
    splat.pos = Eigen::Vector3f(splat.pos.x(), splat.pos.y(), -splat.pos.z());
    splat.quat = Eigen::Vector4f(-splat.quat.x(), splat.quat.y(), splat.quat.z(), -splat.quat.w()); 

    splats.push_back(splat);
  }
  return splats;
}

std::vector<SerializableSplat> loadSplatImageFile(const std::string& filename) {
  cv::Mat image = cv::imread(filename);
  XCHECK(!image.empty());
  XCHECK_EQ(image.type(), CV_8UC3);
  return decodeSplatImage(image);
}

std::shared_ptr<SplatModel> serializableSplatsToModel(
  torch::DeviceType device,
  const std::vector<SerializableSplat>& splats
) {
  std::vector<float> splat_pos_data, splat_color_data, splat_alpha_data, splat_scale_data, splat_quat_data;
  std::vector<uint8_t> splat_alive_data;
  for (int i = 0; i < splats.size(); ++i) {
    const SerializableSplat& s = splats[i];
    splat_pos_data.push_back(s.pos.x());
    splat_pos_data.push_back(s.pos.y());
    splat_pos_data.push_back(s.pos.z());
    splat_color_data.push_back(sigmoidInverse(s.color.z())); // swizzle
    splat_color_data.push_back(sigmoidInverse(s.color.y()));
    splat_color_data.push_back(sigmoidInverse(s.color.x()));
    splat_alpha_data.push_back(sigmoidInverse(s.color.w()));
    splat_scale_data.push_back(inverseScaleActivation(s.scale.x()));
    splat_scale_data.push_back(inverseScaleActivation(s.scale.y()));
    splat_scale_data.push_back(inverseScaleActivation(s.scale.z()));
    Eigen::Vector4f q = s.quat.normalized();
    splat_quat_data.push_back(q.x());
    splat_quat_data.push_back(q.y());
    splat_quat_data.push_back(q.z());
    splat_quat_data.push_back(q.w());
    splat_alive_data.push_back(uint8_t(s.color.w() != 0.0)); // TODO: or use kDeadSplatAlphaThreshold?
  }

  const int num_splats = splats.size();
  auto model = std::make_shared<SplatModel>();
  model->splat_pos   = torch::from_blob(splat_pos_data.data(),   {num_splats, 3}, {torch::kFloat32}).to(device);
  model->splat_color = torch::from_blob(splat_color_data.data(), {num_splats, 3}, {torch::kFloat32}).to(device);
  model->splat_alpha = torch::from_blob(splat_alpha_data.data(), {num_splats, 1}, {torch::kFloat32}).to(device);
  model->splat_scale = torch::from_blob(splat_scale_data.data(), {num_splats, 3}, {torch::kFloat32}).to(device);
  model->splat_quat  = torch::from_blob(splat_quat_data.data(),  {num_splats, 4}, {torch::kFloat32}).to(device);
  model->splat_alive  = torch::from_blob(splat_alive_data.data(),  {num_splats, 1}, {torch::kUInt8}).to(device).to(torch::kBool);
  model->splat_pos = splatPosInverseActivation(model->splat_pos);
  util_torch::cloneIfOnCPU(model->splat_pos);
  util_torch::cloneIfOnCPU(model->splat_color);
  util_torch::cloneIfOnCPU(model->splat_alpha);
  util_torch::cloneIfOnCPU(model->splat_scale);
  util_torch::cloneIfOnCPU(model->splat_quat);
  util_torch::cloneIfOnCPU(model->splat_alive);
  return model;
}

std::vector<SerializableSplat> splatModelToSerializable(std::shared_ptr<SplatModel> model) {
  const int num_splats = model->splat_pos.size(0);
  torch::Tensor cpu_pos = splatPosActivation(model->splat_pos).to(torch::kCPU);
  torch::Tensor cpu_color = torch::sigmoid(model->splat_color).to(torch::kCPU);
  torch::Tensor cpu_alpha = torch::sigmoid(model->splat_alpha).to(torch::kCPU);
  torch::Tensor cpu_scale = scaleActivation(model->splat_scale).to(torch::kCPU);
  torch::Tensor cpu_quat = model->splat_quat.to(torch::kCPU);
  torch::Tensor cpu_alive = model->splat_alive.to(torch::kCPU);
  auto acc_pos = cpu_pos.accessor<float, 2>();
  auto acc_color = cpu_color.accessor<float, 2>();
  auto acc_alpha = cpu_alpha.accessor<float, 2>();
  auto acc_scale = cpu_scale.accessor<float, 2>();
  auto acc_quat = cpu_quat.accessor<float, 2>();
  auto acc_alive = cpu_alive.accessor<bool, 2>();

  std::vector<SerializableSplat> splats;
  for (int i = 0; i < num_splats; ++i) {
    SerializableSplat s;
    if (acc_alive[i][0]) {
      s.pos = Eigen::Vector3f(acc_pos[i][0], acc_pos[i][1], acc_pos[i][2]);
      s.scale = Eigen::Vector3f(acc_scale[i][0], acc_scale[i][1], acc_scale[i][2]);
      s.color = Eigen::Vector4f(acc_color[i][2], acc_color[i][1], acc_color[i][0], acc_alpha[i][0]);
      s.quat = Eigen::Vector4f(acc_quat[i][0], acc_quat[i][1], acc_quat[i][2], acc_quat[i][3]).normalized();
    } else { // If the splat is not alive, fill with dummy data to avoid disrupting ordering
      s.pos = Eigen::Vector3f(0, 0, 0);
      s.scale = Eigen::Vector3f(0, 0, 0);
      s.color = Eigen::Vector4f(0, 0, 0, 0);
      s.quat = Eigen::Vector4f(1, 0, 0, 0);
    }
    splats.push_back(s);
  }
  return splats;
}

}}  // end namespace p11::splat
