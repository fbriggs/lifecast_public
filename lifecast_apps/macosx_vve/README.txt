Instructions for preparing a Mac OS X self-contained distributable application

== Install tools ==

If this isn't compiling, you might need to install X Code (not just command line tools)!

brew install create-dmg
brew install dylibbundler

TODO: More about building universal apps here:

https://github.com/auriamg/macdylibbundler

== Build the application and prepare it for distribution ==

Run these commands 1 at a time, in order:

1)
bazel build --define look_for_runfiles_in_mac_app=true //macosx_vve:volumetric_video_editor

2)
unzip bazel-bin/macosx_vve/volumetric_video_editor.zip -d bazel-bin/macosx_vve

3) new working version:
sudo dylibbundler -od -b -x bazel-bin/macosx_vve/volumetric_video_editor.app/Contents/MacOS/volumetric_video_editor -d bazel-bin/macosx_vve/volumetric_video_editor.app/Contents/libs/

4) Fix issue with duplicate library paths that worked until Mac OS X 15.4 broke it:

sudo ./macosx_vve/fix_duplicate_libs.sh

== Make the .dmg file ==

rm -rf ~/Desktop/Lifecast_Volumetric_Video_Editor && \
rm -f ~/Desktop/Lifecast_Volumetric_Video_Editor.dmg && \
mkdir -p ~/Desktop/Lifecast_Volumetric_Video_Editor && \
cp -rf bazel-bin/macosx_vve/volumetric_video_editor.app ~/Desktop/Lifecast_Volumetric_Video_Editor && \
dot_clean ~/Desktop/Lifecast_Volumetric_Video_Editor && \
chmod -R 755 ~/Desktop/Lifecast_Volumetric_Video_Editor \
&& \
create-dmg \
--volname "Install Lifecast Volumetric Video Editor" \
--window-pos 200 200 \
--window-size 460 350 \
--icon-size 64 \
--icon "volumetric_video_editor.app" 130 60 \
--app-drop-link 325 60 \
--no-internet-enable \
--eula ./macosx_vve/EULA.txt \
~/Desktop/Lifecast_Volumetric_Video_Editor.dmg \
~/Desktop/Lifecast_Volumetric_Video_Editor

Note the above command assumes you are running from ~/dev/p11 for the path to EULA.txt
