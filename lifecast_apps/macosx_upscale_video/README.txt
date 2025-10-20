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
bazel build --define look_for_runfiles_in_mac_app=true //macosx_upscale_video:upscale_video

2)
unzip bazel-bin/macosx_upscale_video/upscale_video.zip -d bazel-bin/macosx_upscale_video

3) new working version:

sudo dylibbundler -ns -od -b -x bazel-bin/macosx_upscale_video/upscale_video.app/Contents/MacOS/upscale_video -d bazel-bin/macosx_upscale_video/upscale_video.app/Contents/libs/

4) Fix issue with duplicate library paths that worked until Mac OS X 15.4 broke it:

sudo ./macosx_upscale_video/fix_duplicate_libs.sh

4) Copy the .app to another location that is not behind a symbolic link to avoid triggering issues with code-sign

rm -rf ~/dev/dylib_sign_workaround
mkdir -p ~/dev/dylib_sign_workaround
cp -r bazel-bin/macosx_upscale_video/upscale_video.app ~/dev/dylib_sign_workaround


TODO: maybe use ditto instead of cp (part of hardening, related to symlinks)


5) Sign the code

INTEL ONLY: Fix busted libraries which are missing sdk versions, which breaks signing:

for dylib in ~/dev/dylib_sign_workaround/upscale_video.app/Contents/libs/*.dylib; do
  if [[ -f "$dylib" ]]; then
    sdk=$(vtool -show-build "$dylib" | grep -i "sdk" | awk '{print $2}')
    if [[ "$sdk" == "n/a" ]]; then
      echo "SDK is NA for: $dylib"
      codesign --remove-signature $dylib
      vtool -set-build-version macos 14.0 14.0 -replace -output $dylib $dylib
    fi
  fi
done

BOTH INTEL AND APPLE (ALWAYS):

<APPLE_TEAM_ID> will be similar to M62W1T7349

codesign --verbose --force --sign <APPLE_TEAM_ID> ~/dev/dylib_sign_workaround/upscale_video.app/Contents/libs/*
codesign --verbose --force --options=runtime --sign <APPLE_TEAM_ID> ~/dev/dylib_sign_workaround/upscale_video.app

NOTE: options=runtime specifies that we are using a "hardened" runtime, which is required to notarize.

6) Make the .dmg file

rm -rf ~/Desktop/Lifecast_UpscaleVideo && \
rm -f ~/Desktop/Lifecast_UpscaleVideo.dmg && \
mkdir -p ~/Desktop/Lifecast_UpscaleVideo && \
cp -rf ~/dev/dylib_sign_workaround/upscale_video.app ~/Desktop/Lifecast_UpscaleVideo/UpscaleVideo.app && \
dot_clean ~/Desktop/Lifecast_UpscaleVideo && \
chmod -R 755 ~/Desktop/Lifecast_UpscaleVideo \
&& \
create-dmg \
--volname "Install Lifecast UpscaleVideo" \
--window-pos 200 200 \
--window-size 460 350 \
--icon-size 64 \
--icon "UpscaleVideo.app" 130 60 \
--app-drop-link 325 60 \
--no-internet-enable \
--eula ./macosx_upscale_video/EULA.txt \
~/Desktop/Lifecast_UpscaleVideo.dmg \
~/Desktop/Lifecast_UpscaleVideo

7) Sign the disk image

codesign --verbose --force --sign <APPLE_TEAM_ID> ~/Desktop/Lifecast_UpscaleVideo.dmg

Note the above command assumes you are running from ~/dev/p11 for the path to EULA.txt

8) ONE TIME SETUP for notorization:

https://developer.apple.com/documentation/security/customizing-the-notarization-workflow#Upload-your-app-to-the-notarization-service

https://support.apple.com/en-us/102654

After created an app-specific password on the apple site, you get a long password string similar to: gpqd-fgef-ghwe-fiel

xcrun notarytool store-credentials "notarytool-password" \
               --apple-id "yourappleemailid@example.com" \
               --team-id <APPLE_TEAM_ID> \
               --password gpqd-fgef-ghwe-fiel


9) notarize the installer image (dmg):

xcrun notarytool submit \
~/Desktop/Lifecast_UpscaleVideo.dmg \
--keychain-profile "notarytool-password" \
--wait

If it fails, here's how to get log (replace the id after log with whatever came out of the command above):

xcrun notarytool log d5d5636d-8d75-4e62-9fc0-70cd9efbc030 --keychain-profile "notarytool-password"

10) Stable the notory result to the dmg:

xcrun stapler staple ~/Desktop/Lifecast_UpscaleVideo.dmg
