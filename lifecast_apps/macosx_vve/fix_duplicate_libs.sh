#!/usr/bin/env bash

# MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

LIBS_DIR="bazel-bin/macosx_vve/volumetric_video_editor.app/Contents/libs"

for dylib in "$LIBS_DIR"/*.dylib; do
  chmod u+w "$dylib"

  # Fix install_name of each dylib
  base_dylib=$(basename "$dylib")
  install_name_tool -id "@executable_path/../libs/$base_dylib" "$dylib"

  # Update references to other bundled dylibs explicitly
  otool -L "$dylib" | grep "$LIBS_DIR" | awk '{print $1}' | while read dep; do
    dep_base=$(basename "$dep")
    install_name_tool -change "$dep" "@executable_path/../libs/$dep_base" "$dylib"
  done

  # Remove duplicate RPATH entries
  existing_rpaths=$(otool -l "$dylib" | grep LC_RPATH -A2 | grep path | awk '{print $2}')
  for rpath in $existing_rpaths; do
    install_name_tool -delete_rpath "$rpath" "$dylib"
  done
done

# Finally, ensure executable itself has correct RPATH
EXECUTABLE="bazel-bin/macosx_vve/volumetric_video_editor.app/Contents/MacOS/volumetric_video_editor"
chmod u+w "$EXECUTABLE"
install_name_tool -add_rpath "@executable_path/../libs/" "$EXECUTABLE"



APP="bazel-bin/macosx_vve/volumetric_video_editor.app"
SIGN_ID="${SIGN_ID:--}"  # default to ad-hoc; override by exporting SIGN_ID="Developer ID Application: Your Name (...)"

/usr/bin/codesign --force --sign "$SIGN_ID" --timestamp=none "$LIBS_DIR"/*.dylib
/usr/bin/codesign --force --sign "$SIGN_ID" --timestamp=none "$EXECUTABLE"
/usr/bin/codesign --force --sign "$SIGN_ID" --timestamp=none "$APP"

/usr/bin/codesign --verify --deep --strict --verbose=2 "$APP"
