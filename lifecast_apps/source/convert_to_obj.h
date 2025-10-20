// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <string>
#include <mutex>
#include <atomic>
#include "opencv2/core.hpp"

namespace p11 { namespace vr180 {

struct ConvertToOBJConfig {
  //std::string format;
  std::string input_vid;
  std::string output_obj;
  int ftheta_size;
  double ftheta_scale;
  double ftheta_inflation;
  double inv_depth_encoding_coef;
};

void convertToOBJ(const ConvertToOBJConfig& cfg);

void writeTexturedMeshObj(
    const ConvertToOBJConfig& cfg,
    const std::vector<cv::Mat>& layer_bgra,
    const std::vector<cv::Mat>& layer_invd);

}}  // namespace p11::vr180
