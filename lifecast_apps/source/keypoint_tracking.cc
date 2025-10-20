// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "keypoint_tracking.h"

#include "logger.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/optflow.hpp"
#include "opencv2/video/tracking.hpp"

namespace p11 { namespace keypoint_tracking {
std::vector<cv::Point2f> findKeypointsFAST(const cv::Mat& image, const int threshold)
{
  constexpr bool kNonMaxSupress = true;

  std::vector<cv::KeyPoint> keypoints;
  std::vector<cv::Point2f> detection_coords;
  cv::FAST(image, keypoints, threshold, kNonMaxSupress);
  cv::KeyPoint::convert(keypoints, detection_coords, std::vector<int>());
  return detection_coords;
}

void trackKeyPointsFromSrcToDest(
    const cv::Mat& src_gray,
    const cv::Mat& dest_gray,
    const std::vector<cv::Point2f>& points_in_src,
    std::vector<cv::Point2f>& points_in_dest,
    std::vector<bool>& valid_mask)
{
  if (points_in_src.empty()) {
    return;
  }  // flow will crash with 0 input

  std::vector<uchar> status;
  std::vector<float> err;
  calcOpticalFlowPyrLK(
      src_gray,
      dest_gray,
      points_in_src,
      points_in_dest,
      status,
      err,
      cv::Size(21, 21),
      3,
      cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30.0, 0.01),
      0,
      0.001);

  XCHECK_EQ(points_in_src.size(), status.size());
  valid_mask.resize(status.size());
  for (int i = 0; i < status.size(); ++i) {
    valid_mask[i] = bool(status[i]);
    // TODO: check err as well to get better filtering?
  }
}


nlohmann::json matchesToJson(
  const std::vector<std::string>& image_filenames,
  const std::vector<cv::Mat>& images,  // Added images parameter
  const std::vector<std::vector<cv::KeyPoint>>& image_to_keypoints,
  const std::vector<std::vector<cv::DMatch>>& image_to_matches
) {
  using json = nlohmann::json;
  json j;

  // Metadata
  j["metadata"]["num_images"] = image_filenames.size();
  j["metadata"]["image_filenames"] = image_filenames;
  
  // Store image dimensions
  json image_sizes = json::array();
  for (const auto& img : images) {
    image_sizes.push_back({
      {"width", img.cols},
      {"height", img.rows}
    });
  }
  j["metadata"]["image_sizes"] = image_sizes;

  // Keypoints (unchanged)
  for (int i = 0; i < image_to_keypoints.size(); ++i) {
    json kp_array = json::array();
    for (const auto& kp : image_to_keypoints[i]) {
      kp_array.push_back({ {"x", kp.pt.x}, {"y", kp.pt.y}, {"size", kp.size} });
    }
    j["keypoints"].push_back(kp_array);
  }

  // Matches as flat array of 4-tuples
  json matches_array = json::array();
  for (int i = 0; i < image_to_matches.size(); ++i) {
    for (const auto& m : image_to_matches[i]) {
      matches_array.push_back({
        {"image_idx1", i},                    
        {"keypoint_idx1", m.queryIdx},       
        {"image_idx2", m.imgIdx},            
        {"keypoint_idx2", m.trainIdx},       
        {"distance", m.distance}
      });
    }
  }
  j["matches"] = matches_array;

  return j;
}

void matchesFromJson(
  const nlohmann::json& j,
  std::vector<std::string>& image_filenames,
  std::vector<std::vector<cv::KeyPoint>>& image_to_keypoints,
  std::vector<std::vector<cv::DMatch>>& image_to_matches
) {
  image_filenames.clear();
  image_to_keypoints.clear();
  image_to_matches.clear();

  // Load metadata
  image_filenames = j["metadata"]["image_filenames"];
  
  // Load keypoints
  for (const auto& kp_array : j.at("keypoints")) {
    std::vector<cv::KeyPoint> v;
    for (const auto& kp : kp_array) {
      v.emplace_back(cv::Point2f(kp["x"], kp["y"]), kp["size"]);
    }
    image_to_keypoints.push_back(std::move(v));
  }

  // Reconstruct the vector-of-vectors structure from flat matches
  image_to_matches.resize(image_filenames.size());
  
  for (const auto& match : j["matches"]) {
    cv::DMatch dm;
    dm.queryIdx = match["keypoint_idx1"];
    dm.trainIdx = match["keypoint_idx2"];
    dm.imgIdx = match["image_idx2"];
    dm.distance = match["distance"];
    
    int image1_idx = match["image_idx1"];
    image_to_matches[image1_idx].push_back(dm);
  }
}


}}  // namespace p11::keypoint_tracking
