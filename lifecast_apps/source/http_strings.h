// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

inline const char* kLifecastPlayerHTML = R"(
<!DOCTYPE html>
<html>
  <head>
  <script src="https://cdn.jsdelivr.net/gh/fbriggs/lifecast_public/lifecast.min.js"></script>
  </head>
  <body>
  <script>
  LifecastVideoPlayer.init({
    _media_urls: ["ldi3.png"],
    _enable_intro_animation: false,
  });
  </script>
  </body>
</html>
)";

inline const char* kLifecastPlayerVideoHTML = R"(
<!DOCTYPE html>
<html>
  <head>
  <script src="https://cdn.jsdelivr.net/gh/fbriggs/lifecast_public/lifecast.min.js"></script>
  </head>
  <body>
  <script>
  LifecastVideoPlayer.init({
    _media_urls: [
      "ldi3_h265_hvc1_5760x5760.mp4",
      "ldi3_h264_5760x5760.mp4",
      "ldi3_h264_3840x3840.mp4",
      "ldi3_h264_1920x1920.mp4",
    ],
    _autoplay_muted: true,
    _loop: true,
    _enable_intro_animation: false,
  });
  </script>
  </body>
</html>
)";
