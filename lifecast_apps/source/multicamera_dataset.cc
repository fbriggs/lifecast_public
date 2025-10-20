// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "multicamera_dataset.h"

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "util_file.h"
#include "util_torch.h"
#include "torch_opencv.h"
#include "third_party/json.h"

namespace p11 { namespace calibration {

std::vector<calibration::NerfKludgeCamera> readDatasetCameraJson(const std::string& json_path) {
  using json = nlohmann::json;
  XCHECK(file::fileExists(json_path)) << json_path;

  std::vector<calibration::NerfKludgeCamera> cameras;

  std::ifstream train_json_file(json_path);
  json train_json_data = json::parse(train_json_file);
  auto& frames_data = train_json_data["frames_data"];

  for (auto& frame_data : frames_data) {
    XPLINFO << frame_data;

    std::vector<double> world_from_cam_data = frame_data["world_from_cam"];
    Eigen::Matrix4d world_from_cam = Eigen::Map<Eigen::Matrix<double, 4, 4>>(world_from_cam_data.data());
    // Fix the bottom row to ensure it's a proper transformation matrix, or it might not be invertible
    world_from_cam.row(3) << 0, 0, 0, 1;
    Eigen::Matrix4d cam_from_world = world_from_cam.inverse();

    calibration::NerfKludgeCamera cam;
    if (frame_data["cam_model"] == "rectilinear") {
      cam.is_rectilinear = true;
      cam.is_fisheye = false;
      cam.rectilinear.name = frame_data["image_filename"];
      cam.rectilinear.width = frame_data["width"];
      cam.rectilinear.height = frame_data["height"];
      cam.rectilinear.focal_length = Eigen::Vector2d(frame_data["fx"], frame_data["fy"]);
      cam.rectilinear.optical_center = Eigen::Vector2d(frame_data["cx"], frame_data["cy"]);
      if (frame_data.contains("radial_distortion")) {
        cam.rectilinear.k1 = frame_data["radial_distortion"][0];
        cam.rectilinear.k2 = frame_data["radial_distortion"][1];
      }
      cam.rectilinear.cam_from_world = cam_from_world;
    } else if (frame_data["cam_model"] == "equiangular") {
      cam.is_rectilinear = false;
      cam.is_fisheye = true;
      cam.fisheye.name = frame_data["image_filename"];
      cam.fisheye.width = frame_data["width"];
      cam.fisheye.height = frame_data["height"];
      cam.fisheye.radius_at_90 = frame_data["radius_at_90"];
      cam.fisheye.useable_radius = frame_data["useable_radius"];
      cam.fisheye.optical_center = Eigen::Vector2d(frame_data["cx"], frame_data["cy"]);
      cam.fisheye.k1 = frame_data["k1"];
      cam.fisheye.tilt = frame_data["tilt"];
      cam.fisheye.cam_from_world = cam_from_world;
    } else {
      XCHECK(false) << "Unrecognized camera model: " << frame_data["cam_model"];
    }

    cameras.push_back(cam);
  }
  return cameras;
}

void createEmptyTimeOffsetJson(const std::string& path, const std::vector<std::string> camera_names) {
  using json = nlohmann::json;

  json time_offset_json;
  for (const std::string& name : camera_names) {
    time_offset_json[name] = 0;
  }

  std::ofstream json_file(path);
  json_file << std::setw(4) << time_offset_json << std::endl;
  json_file.close();
}

// Updates cameras by filling in their time offsets
void readTimeOffsetJson(
  const std::string& time_offset_json_path,
  std::vector<calibration::NerfKludgeCamera>& cameras
) {
  using json = nlohmann::json;
  if (!file::fileExists(time_offset_json_path)) {
    XPLINFO << "WARNING: time offset json file not found: " << time_offset_json_path;
    XPLINFO << "Continuing with 0 time offsets.";
    return;
  }

  std::ifstream json_file(time_offset_json_path);
  json time_offset_json = json::parse(json_file);

  for (auto& cam : cameras) {
    XCHECK(time_offset_json.contains(cam.name())) << "Missing time offset in json file: " << time_offset_json_path << " for camera: " << cam.name();
    cam.time_offset_frames = time_offset_json[cam.name()]; 
  }
}

std::map<std::string, int> readTimeOffsetJsonAsMap(const std::string& time_offset_json_path) {
  using json = nlohmann::json;
  std::map<std::string, int> offsets;

  if (!file::fileExists(time_offset_json_path)) {
    XPLINFO << "WARNING: time offset json file not found: " << time_offset_json_path;
    XPLINFO << "Continuing with empty time offsets.";
    return offsets;
  }

  std::ifstream json_file(time_offset_json_path);
  json j = json::parse(json_file);

  for (auto& [name, value] : j.items()) {
    offsets[name] = value.get<int>();
  }

  return offsets;
}

torch::Tensor cvMat8UC3_to_Tensor(const torch::DeviceType device, cv::Mat image) {
  XCHECK_EQ(image.type(), CV_8UC3); 
  std::vector<float> pixel_data;
  for (int y = 0; y < image.rows; ++y) {
    for (int x = 0; x < image.cols; ++x) {
      const cv::Vec3f bgr = cv::Vec3f(image.at<cv::Vec3b>(y, x)) / 255.0f;
      pixel_data.push_back(bgr[0]);
      pixel_data.push_back(bgr[1]);
      pixel_data.push_back(bgr[2]);
    }
  }
  torch::Tensor image_tensor = torch::from_blob(pixel_data.data(), {image.rows, image.cols, 3}, {torch::kFloat32}).to(device);
  util_torch::cloneIfOnCPU(image_tensor);
  return image_tensor;
}

calibration::MultiCameraDataset readDataset(
  const std::string& images_dir,
  const std::string& json_path,
  const torch::DeviceType device,
  const int resize_max_dim,
  bool load_depthmaps
) {
  using json = nlohmann::json;

  calibration::MultiCameraDataset dataset;
  dataset.cameras = readDatasetCameraJson(json_path);

  if (resize_max_dim != 0) {
    for (auto& cam : dataset.cameras) { cam.resizeToMaxDim(resize_max_dim); }
  }

  std::ifstream train_json_file(json_path);
  json train_json_data = json::parse(train_json_file);
  auto& frames_data = train_json_data["frames_data"];

  cv::Size raw_image_size;
  for (auto& frame_data : frames_data) {
    const std::string image_filename = frame_data["image_filename"];
    dataset.image_filenames.push_back(image_filename);
    cv::Mat image = cv::imread(images_dir + "/" + image_filename);
    // Check if the image file exists
    if (image.empty()) {
      XPLINFO << "Error: failed to load image file " << images_dir + "/" + image_filename;
      XPLINFO << "image_filename: " << image_filename;
      XPLINFO << "frame_data=" << frame_data;
      exit(1);
    }

    // HACK: this won't work if there are images of different sizes in the dataset
    raw_image_size = image.size();
    if (resize_max_dim !=0 ) {
      cv::resize(image, image, cv::Size(dataset.cameras[0].getWidth(), dataset.cameras[0].getHeight()), 0, 0, cv::INTER_CUBIC);
    }

    dataset.images.push_back(image);
    dataset.image_tensors.push_back(cvMat8UC3_to_Tensor(device, image)); // TECH DEBT: not using torch_opencv, different tensor shape?

    // Load EXR depthmaps if they exist
    if (load_depthmaps) {
      std::string exr_filename = "depth_" + file::filenamePrefix(image_filename) + ".exr";
      std::string exr_path = images_dir + "/" + exr_filename;
      if (file::fileExists(exr_path)) {
        XPLINFO << "Loading EXR depthmap: " << exr_filename;
        cv::Mat depthmap = cv::imread(exr_path, cv::IMREAD_UNCHANGED);
        if (depthmap.size() != image.size()) {
          cv::resize(depthmap, depthmap, image.size(), 0, 0, cv::INTER_AREA);
        }
        XCHECK_EQ(depthmap.type(), CV_32FC1);
        dataset.depthmaps.push_back(depthmap);
        dataset.depthmap_tensors.push_back(torch_opencv::cvMat_to_Tensor(device, depthmap));
      }
    }
  }

  if (resize_max_dim == 0 ) {
    for (auto& cam : dataset.cameras) {
      int max = std::max(raw_image_size.width, raw_image_size.height);
      cam.resizeToMaxDim(max);

      if (raw_image_size.width > raw_image_size.height) {
        XCHECK_EQ(cam.getHeight(), raw_image_size.height) << "Resizing camera intrinsics failed. Mismatched aspect ratio?";
      } else {
        XCHECK_EQ(cam.getWidth(), raw_image_size.width) << "Resizing camera intrinsics failed. Mismatched aspect ratio?";
      }
    }
  }

  return dataset;
}

}}  // end namespace p11::calibration
