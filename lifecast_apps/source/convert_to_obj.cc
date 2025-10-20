// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "convert_to_obj.h"

#include "gflags/gflags.h"
#include "logger.h"
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include "fisheye_camera.h"
#include "projection.h"
#include "depth_estimation.h"
#include "turbojpeg_wrapper.h"
#include "ldi_common.h"
#include "util_file.h"
#include "util_string.h"
#include "util_time.h"
#include "util_opencv.h"

namespace p11 { namespace vr180 {

void generateMtlFile(const std::string& obj_filename, const std::string& texture_filename)
{
  // base_filename: myfile.obj
  std::string base_filename = obj_filename.substr(0, obj_filename.find_last_of('.'));
  base_filename = base_filename.substr(base_filename.find_last_of('/') + 1);
  // mtl_filename: path/to/myfile.mtl
  std::string mtl_filename = obj_filename.substr(0, obj_filename.find_last_of('.')) + ".mtl";
  // base_texture_filename: myfile.png
  std::string base_texture_filename =
      texture_filename.substr(texture_filename.find_last_of('/') + 1);

  std::ofstream mtl_file(mtl_filename);
  // NOTE: the newmtl name must match the usemtl name in the obj file
  mtl_file << "newmtl " << base_filename << std::endl;
  mtl_file << "Ka 1.0 1.0 1.0" << std::endl;
  mtl_file << "Kd 1.0 1.0 1.0" << std::endl;
  mtl_file << "Ks 0.0 0.0 0.0" << std::endl;
  mtl_file << "map_Kd " << base_texture_filename << std::endl;
  mtl_file.close();

  std::cout << "Generated " << mtl_filename << " successfully." << std::endl;
}

void writeTexturedMeshObj(
    const ConvertToOBJConfig& cfg,
    const std::vector<cv::Mat>& layer_bgra,
    const std::vector<cv::Mat>& layer_invd)
{
  XPLINFO << "writing OBJ mesh: " << cfg.output_obj;
  XCHECK(!cfg.output_obj.empty());
  XCHECK(layer_bgra.size() == layer_invd.size());

  const int num_layers = layer_bgra.size();

  // HACK: connecting the triangle count with the pixel resolution makes huge meshes.
  // Ideally these two parameters are specific separately. As a hack to work around this,
  // here I arbitrarily select a factor to scale.
  constexpr int kDepthDownscale = 2;
  cv::Size half_size(layer_bgra[0].cols / kDepthDownscale, layer_bgra[0].rows / kDepthDownscale);

  // Write obj mesh
  std::vector<Eigen::Vector3d> verts;
  std::vector<Eigen::Vector2f> uvs;
  std::vector<std::array<int, 3>> triangles;

  for (int layer = 0; layer < num_layers; ++layer) {
    cv::Mat invd_small;
    cv::resize(layer_invd[layer], invd_small, half_size);
    long vertex_offset = verts.size();

    for (int j = 0; j < invd_small.rows; ++j) {
      for (int i = 0; i < invd_small.cols; ++i) {
        const float u = i / float(invd_small.cols - 1);
        const float v = j / float(invd_small.rows - 1);
        const float a = 2.0 * (u - 0.5);
        const float b = 2.0 * (v - 0.5);

        const float theta = atan2(b, a);
        float r = sqrt(a * a + b * b) / cfg.ftheta_scale;
        // Apply inflated-equiangular projection
        r = 0.5 * r + 0.5 * std::pow(r, cfg.ftheta_inflation);
        const float phi = r * M_PI / 2.0;

        static constexpr float kEpsilon = 1e-6;
        static constexpr float kMaxDist = 50;

        float d = float(cfg.inv_depth_encoding_coef) / (kEpsilon + invd_small.at<float>(j, i));
        d = math::clamp(d, 0.0f, kMaxDist);

        // This is swizzled to come out straight in Blender coordinates
        const float x = cos(theta) * sin(phi) * d;
        const float y = -sin(theta) * sin(phi) * d;
        const float z = -cos(phi) * d;

        verts.emplace_back(x, y, z);

        // Adjust UVs based on the layer
        float offset = layer / float(num_layers);
        uvs.emplace_back((u / float(num_layers)) + offset, 1.0 - v);
      }
    }
    assert(verts.size() == vertex_offset + invd_small.rows * invd_small.cols);

    static constexpr int kMargin = 3;
    const int r_clip = invd_small.cols / 2;
    for (int j = 0; j < invd_small.rows; ++j) {
      for (int i = 0; i < invd_small.cols; ++i) {
        const int di = i - r_clip;
        const int dj = j - r_clip;
        if (di * di + dj * dj > (r_clip - kMargin) * (r_clip - kMargin)) continue;

        const int a = i + (invd_small.cols * j) + vertex_offset;
        const int b = a + 1;
        const int c = a + invd_small.cols;
        const int d = c + 1;
        const std::array<int, 3> t1 = {a, c, b};
        const std::array<int, 3> t2 = {c, d, b};
        XCHECK(a < verts.size() && b < verts.size() && c < verts.size() && d < verts.size());
        triangles.emplace_back(t1);
        triangles.emplace_back(t2);
      }
    }
  }

  std::ofstream f;
  f.open(cfg.output_obj);
  XCHECK(f.is_open()) << "Failed to open file to write obj: " << cfg.output_obj;

  // For a volumetric file "foobar.png", use "foobar" as the material name
  std::string base_filename = cfg.output_obj.substr(0, cfg.output_obj.find_last_of('.'));
  base_filename = base_filename.substr(base_filename.find_last_of('/') + 1);

  // NOTE: The "mtllib" in the OBJ file contains a reference to the .mtl filename
  // Most software will look for the .mtl file in the same directory as the .obj file
  f << "mtllib " << base_filename << ".mtl" << std::endl;
  // NOTE: the usemtl name must match the newmtl name in the mtl file
  f << "usemtl " << base_filename << std::endl;

  for (const auto& v : verts) {
    f << "v " << v.x() << " " << v.y() << " " << v.z() << std::endl;
  }
  for (const auto& uv : uvs) {
    f << "vt " << uv.x() << " " << uv.y() << std::endl;
  }
  for (const auto& t : triangles) {
    std::string a = std::to_string(t[0]);
    std::string b = std::to_string(t[1]);
    std::string c = std::to_string(t[2]);
    f << "f " << a << "/" << a << " " << b << "/" << b << " " << c << "/" << c << std::endl;
  }
  f.close();

  // Concatenate all 3 layers into a single output .png texture
  cv::Mat bgra_atlas;
  cv::hconcat(layer_bgra, bgra_atlas);
  std::string texture_filename = cfg.output_obj;
  texture_filename.replace(texture_filename.end() - 3, texture_filename.end(), "png");
  if (bgra_atlas.depth() != CV_8U) bgra_atlas.convertTo(bgra_atlas, CV_8U, 255.0);
  cv::imwrite(texture_filename, bgra_atlas);

  // Output a .mtl texture file
  generateMtlFile(cfg.output_obj, texture_filename);
}

void convertToOBJ(const ConvertToOBJConfig& cfg)
{
  try {
    XPLINFO << "OBJ converter converting " << cfg.input_vid << " to " << cfg.output_obj;

    cv::Mat image6dof = cv::imread(cfg.input_vid);
    if (image6dof.empty()) {
      XPLERROR << "Failed to load image: " << cfg.input_vid;
      return;
    }
    std::vector<cv::Mat> layer_bgra, layer_invd;
    ldi::unpackLDI3(image6dof, layer_bgra, layer_invd);
    writeTexturedMeshObj(cfg, layer_bgra, layer_invd);
    XPLINFO << "OBJ converter finished";

  } catch (std::runtime_error& e) {
    XPLINFO << "Exception:\n---\n" << e.what() << "\n---";
  }
}

}}  // namespace p11::vr180
