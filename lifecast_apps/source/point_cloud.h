// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include <fstream>
#include <iostream>
#include <string>

#include "Eigen/Core"
#include "logger.h"

#include "util_math.h"
#include "util_string.h"

namespace p11 { namespace point_cloud {

// Saves a point cloud in the "PCD" format from the PCL library
template<typename TColor> // This will ignore the A channel for RGBA
inline void savePointCloudPCL(
    const std::string& filename,
    const std::vector<Eigen::Vector3f>& point_cloud,
    const std::vector<TColor>& point_cloud_colors)
{
  XCHECK_EQ(point_cloud.size(), point_cloud_colors.size());
  std::ofstream f;
  f.open(filename);
  f << "# .PCD v0.7 - Point Cloud Data file format" << std::endl;
  f << "VERSION 0.7" << std::endl;
  f << "FIELDS x y z rgb" << std::endl;
  f << "SIZE 4 4 4 4" << std::endl;
  f << "TYPE F F F U" << std::endl;
  f << "COUNT 1 1 1 1" << std::endl;
  f << "WIDTH " << point_cloud.size() << std::endl;
  f << "HEIGHT 1" << std::endl;
  f << "VIEWPOINT 0 0 0 1 0 0 0" << std::endl;
  f << "POINTS " << point_cloud.size() << std::endl;
  f << "DATA ascii" << std::endl;

  for (size_t i = 0; i < point_cloud.size(); ++i) {
    const int r = point_cloud_colors[i].x() * 255;
    const int g = point_cloud_colors[i].y() * 255;
    const int b = point_cloud_colors[i].z() * 255;
    const unsigned int rgb = (r << 16) + (g << 8) + b;
    f << point_cloud[i].x() << " "  //
      << point_cloud[i].y() << " "  //
      << point_cloud[i].z() << " "  //
      << rgb << std::endl;          //
  }
  f.close();
}

inline void savePointCloudPly(
    const std::string& filename,
    const std::vector<Eigen::Vector3f>& point_cloud,
    const std::vector<Eigen::Vector3f>& point_cloud_colors)
{
  XCHECK_EQ(point_cloud.size(), point_cloud_colors.size());
  std::ofstream f;
  f.open(filename);
  f << "ply" << std::endl;
  f << "format ascii 1.0" << std::endl;
  f << "element vertex " << point_cloud.size() << std::endl;
  f << "property float x" << std::endl;
  f << "property float y" << std::endl;
  f << "property float z" << std::endl;
  f << "property uint8 red" << std::endl;
  f << "property uint8 green" << std::endl;
  f << "property uint8 blue" << std::endl;
  f << "end_header" << std::endl;

  for (size_t i = 0; i < point_cloud.size(); ++i) {
    f << point_cloud[i].x() << " "                           //
      << point_cloud[i].y() << " "                           //
      << point_cloud[i].z() << " "                           //
      << int(point_cloud_colors[i].x() * 255) << " "         //
      << int(point_cloud_colors[i].y() * 255) << " "         //
      << int(point_cloud_colors[i].z() * 255) << std::endl;  //
  }
  f.close();
}


inline void savePointCloudCsv(
    const std::string& filename,
    const std::vector<Eigen::Vector3f>& point_cloud,
    const std::vector<Eigen::Vector3f>& point_cloud_colors)
{
  XCHECK_EQ(point_cloud.size(), point_cloud_colors.size());
  std::ofstream f;
  f.open(filename);
  for (size_t i = 0; i < point_cloud.size(); ++i) {
    f << point_cloud[i].x() << ","                           //
      << point_cloud[i].y() << ","                           //
      << point_cloud[i].z() << ","                           //
      << int(point_cloud_colors[i].x() * 255) << ","         //
      << int(point_cloud_colors[i].y() * 255) << ","         //
      << int(point_cloud_colors[i].z() * 255) << std::endl;  //
  }
  f.close();
}

inline void savePointCloudCsv(
    const std::string& filename,
    const std::vector<Eigen::Vector3f>& point_cloud,
    const std::vector<Eigen::Vector4f>& point_cloud_colors)
{
  XCHECK_EQ(point_cloud.size(), point_cloud_colors.size());
  std::ofstream f;
  f.open(filename);
  for (size_t i = 0; i < point_cloud.size(); ++i) {
    f << point_cloud[i].x() << ","                           //
      << point_cloud[i].y() << ","                           //
      << point_cloud[i].z() << ","                           //
      << int(point_cloud_colors[i].x() * 255) << ","         //
      << int(point_cloud_colors[i].y() * 255) << ","         //
      << int(point_cloud_colors[i].z() * 255) << ","         //
      << int(point_cloud_colors[i].w() * 255) << std::endl;  //
  }
  f.close();
}

inline void loadPointCloudCsv(
    const std::string& filename,
    std::vector<Eigen::Vector3f>& point_cloud,
    std::vector<Eigen::Vector3f>& point_cloud_colors,
    float subsample = 1.0)
{
  std::ifstream file(filename);
  int line_count = 0;
  for (std::string line; std::getline(file, line);) {
    // skip some points at random if we are sub-sampling
    if (subsample != 1.0 && math::randUnif() > subsample) {
      continue;
    }

    std::vector<std::string> tokens = p11::string::split(line, ',');
    XCHECK_EQ(tokens.size(), 6) << "error parsing line: " + line;
    point_cloud.emplace_back(
        std::atof(tokens[0].c_str()), std::atof(tokens[1].c_str()), std::atof(tokens[2].c_str()));
    point_cloud_colors.emplace_back(
        std::atof(tokens[3].c_str()) / 255.0f,
        std::atof(tokens[4].c_str()) / 255.0f,
        std::atof(tokens[5].c_str()) / 255.0f);
    ++line_count;
    if (line_count % 100000 == 0) {
      XPLINFO << "loading line " << line_count;
    }
  }
  file.close();
}

inline void savePointCloudBinary(
    const std::string& filename,
    const std::vector<Eigen::Vector3f>& point_cloud,
    const std::vector<Eigen::Vector3f>& point_cloud_colors)
{
  XCHECK_EQ(point_cloud.size(), point_cloud_colors.size());
  std::ofstream f;
  f.open(filename, std::ios::out | std::ios::binary);
  int num_points = point_cloud.size();
  f.write((char*)&num_points, sizeof(int));
  for (size_t i = 0; i < point_cloud.size(); ++i) {
    const Eigen::Vector3f& p = point_cloud[i];

    unsigned char r = point_cloud_colors[i].x() * 255;
    unsigned char g = point_cloud_colors[i].y() * 255;
    unsigned char b = point_cloud_colors[i].z() * 255;

    f.write((char*)&p.x(), sizeof(float));
    f.write((char*)&p.y(), sizeof(float));
    f.write((char*)&p.z(), sizeof(float));
    f.write((char*)&r, sizeof(unsigned char));
    f.write((char*)&g, sizeof(unsigned char));
    f.write((char*)&b, sizeof(unsigned char));
  }
  f.close();
}

inline void loadPointCloudBinary(
    const std::string& filename,
    std::vector<Eigen::Vector3f>& point_cloud,
    std::vector<Eigen::Vector3f>& point_cloud_colors,
    float subsample = 1.0)
{
  std::ifstream f(filename, std::ios::in | std::ios::binary);
  int num_points;
  f.read((char*)&num_points, sizeof(int));

  for (int i = 0; i < num_points; ++i) {
    float x, y, z;
    unsigned char r, g, b;
    f.read((char*)&x, sizeof(float));
    f.read((char*)&y, sizeof(float));
    f.read((char*)&z, sizeof(float));
    f.read((char*)&r, sizeof(unsigned char));
    f.read((char*)&g, sizeof(unsigned char));
    f.read((char*)&b, sizeof(unsigned char));

    // skip some points at random if we are sub-sampling
    if (subsample != 1.0 && math::randUnif() > subsample) {
      continue;
    }

    point_cloud.emplace_back(x, y, z);
    point_cloud_colors.emplace_back(r / 255.0f, g / 255.0f, b / 255.0f);
  }

  f.close();
}

}}  // namespace p11::point_cloud
