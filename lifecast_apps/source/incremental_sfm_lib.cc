// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "incremental_sfm_lib.h"

namespace p11 { namespace calibration { namespace incremental_sfm {

// tracks[r][s] -- r'th track, s'th observation, .first = image index, .second = pixel coordinate
// i - image index
// j - second image index
// p - indexes a keypoint in image i
// q - indexes a keypoint in image j
std::vector<IncrementalSfmTrack> buildTracks(
  const std::vector<std::vector<cv::KeyPoint>>& image_to_keypoints,
  const std::vector<std::vector<cv::DMatch>>& image_to_matches
) {
  XCHECK_EQ(image_to_keypoints.size(), image_to_matches.size());
  const int num_images = image_to_keypoints.size();

  std::map<std::pair<int, int>, int> keypoint_to_track; // Map from (image_idx, keypoint_idx) to track_idx
  std::vector<IncrementalSfmTrack> tracks;

  // If we only have 1 image, still make tracks so we can output a rough point cloud from mono depth
  if (num_images == 1) {
    for (int p = 0; p < image_to_keypoints[0].size(); ++p) {
      IncrementalSfmTrack new_track;
      new_track.pruned = false;
      new_track.has_estimated_3d = false;
      new_track.observations.emplace_back(0, p, image_to_keypoints[0][p].pt);
      tracks.push_back(new_track);
    }
    return tracks;
  }

  for (int i = 0; i < num_images; ++i) {
    for (const cv::DMatch& match : image_to_matches[i]) {
      const int j = match.imgIdx;
      const int p = match.queryIdx; // keypoint index in image i
      const int q = match.trainIdx; // keypoint index in image j
      XCHECK(p < image_to_keypoints[i].size());
      XCHECK(q < image_to_keypoints[j].size());

      auto keypoint_i = std::make_pair(i, p);
      auto keypoint_j = std::make_pair(j, q);

      // Check if keypoints are already part of a track
      auto it_i = keypoint_to_track.find(keypoint_i);
      auto it_j = keypoint_to_track.find(keypoint_j);

      if (it_i == keypoint_to_track.end() && it_j == keypoint_to_track.end()) {
        // Neither keypoint is part of a track, create a new track
        IncrementalSfmTrack new_track;
        new_track.pruned = false;
        new_track.has_estimated_3d = false;
        new_track.observations.emplace_back(i, p, image_to_keypoints[i][p].pt);
        new_track.observations.emplace_back(j, q, image_to_keypoints[j][q].pt);
        tracks.push_back(new_track);
        int new_track_idx = tracks.size() - 1;
        keypoint_to_track[keypoint_i] = new_track_idx;
        keypoint_to_track[keypoint_j] = new_track_idx;
      } else if (it_i != keypoint_to_track.end() && it_j == keypoint_to_track.end()) {
        // Keypoint i is part of a track, add keypoint j to the same track
        int track_idx = it_i->second;
        tracks[track_idx].observations.emplace_back(j, q, image_to_keypoints[j][q].pt);
        keypoint_to_track[keypoint_j] = track_idx;
      } else if (it_i == keypoint_to_track.end() && it_j != keypoint_to_track.end()) {
        // Keypoint j is part of a track, add keypoint i to the same track
        int track_idx = it_j->second;
        tracks[track_idx].observations.emplace_back(i, p, image_to_keypoints[i][p].pt);
        keypoint_to_track[keypoint_i] = track_idx;
      } else {
        // Both keypoints are part of different tracks, merge the tracks
        int track_idx_i = it_i->second;
        int track_idx_j = it_j->second;
        if (track_idx_i != track_idx_j) {
          // Merge track j into track i
          for (const auto& obs : tracks[track_idx_j].observations) {
            tracks[track_idx_i].observations.push_back(obs);
            keypoint_to_track[std::make_pair(obs.img_idx, obs.kp_idx)] = track_idx_i;
          }
          tracks[track_idx_j].observations.clear(); // Clear the merged track
        }
      }
    }
  }

  // Remove empty tracks after merging
  tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [num_images](const IncrementalSfmTrack& track) {
    return track.observations.size() < std::min(num_images, kMinObservationsPerTrack);
  }), tracks.end());

  return tracks;
}

}}} // end namespace p11::calibration::incremental_sfm
