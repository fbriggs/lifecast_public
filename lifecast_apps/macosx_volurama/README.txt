Instructions for preparing a Mac OS X self-contained distributable application

== Install tools ==

If this isn't compiling, you might need to install X Code (not just command line tools)!

brew install create-dmg
brew install dylibbundler

NOTE: as of writing, the homebrew version of dylibbundler is 1.0.5, and it works OK. Version 2 exists, but doesn't work.
In case we want to use version 2 for some reason, here's how to install it:

    cd ~/dev
    git clone git@github.com:SCG82/macdylibbundler.git
    cd macdylibbundler
    sudo make install

Related:

https://github.com/auriamg/macdylibbundler

== Build the application and prepare it for distribution ==

Run these commands 1 at a time, in order:

1)
bazel build --define look_for_runfiles_in_mac_app=true //macosx_volurama:volurama

2)
unzip bazel-bin/macosx_volurama/volurama.zip -d bazel-bin/macosx_volurama

3) new working version:
sudo dylibbundler -od -b -x bazel-bin/macosx_volurama/volurama.app/Contents/MacOS/volurama -d bazel-bin/macosx_volurama/volurama.app/Contents/libs/

4) fix dylibs

sudo ./macosx_volurama/fix_duplicate_libs.sh

== Make the .dmg file ==

rm -rf ~/Desktop/Lifecast_Volurama && \
rm -f ~/Desktop/Lifecast_Volurama.dmg && \
mkdir -p ~/Desktop/Lifecast_Volurama && \
cp -rf bazel-bin/macosx_volurama/volurama.app ~/Desktop/Lifecast_Volurama && \
dot_clean ~/Desktop/Lifecast_Volurama && \
chmod -R 755 ~/Desktop/Lifecast_Volurama \
&& \
create-dmg \
--volname "Install Lifecast Volurama" \
--window-pos 200 200 \
--window-size 460 350 \
--icon-size 64 \
--icon "volurama.app" 130 60 \
--app-drop-link 325 60 \
--no-internet-enable \
--eula ./macosx_volurama/EULA.txt \
~/Desktop/Lifecast_Volurama.dmg \
~/Desktop/Lifecast_Volurama

Note the above command assumes you are running from ~/dev/p11 for the path to EULA.txt
