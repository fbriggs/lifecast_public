# Lifecast Inc. Open Source

Lifecast makes software for immersive volumetric VR videos and photos. Lifecast's 6DOF format for 3D photos and videos can be generated using tools available on https://lifecastvr.com. This repo hosts the player for this format, which is open source under the MIT license. Several implementations of the player are available, in javascript, Unity and Unreal Engine.

Do you just want to make VR content without writing code? We do the same processing in the cloud and automatically host the results on https://holovolo.tv

# Unity

To use the Lifecast player in Unity, we recommend going through the Unity Asset store to directly import the package into your project:

https://assetstore.unity.com/packages/tools/video/lifecast-volumetric-video-player-221239

Tutorials on using Lifecast's player with Unity are available here:

https://lifecastvr.com/tutorial_unity.html

https://youtu.be/9Ja20bitA-w

# Unreal Engine

For Unreal 5, we have simplified the setup; there is no video tutorial currently. The setup is as follows:
* Download either Lifecast_2Layer_PlayerUE5.1.zip or Lifecast_LDI3_Player_UE5.1.zip from the /UnrealEngine5 directory of this repo (depending on which format you want to play, Lifecast 2-layer, or Lifecast LDI3).
* Create a new project -> Film / Video & Live Events -> Blank.
* Delete everything except Player Start from the level.
* Open the Content folder for the project in Explorer.
* Quit/Exit Unreal Engine completely.
* Copy all of the files from the zip into the Content folder, and replace Main.umap.
* Re-open the project, and you are good to go (with a static texture)!
* To play a video, the steps are similar in UE4 or UE5. Refer to the video tutorial below.

For Unreal 4 and the original Lifecast 2-layer format, follow this tutorial:

https://www.youtube.com/watch?v=HpGLGWP0tFo&ab_channel=Lifecast

Sample 2-layer 6DOF video file for testing: 

https://lifecastvideocdn.s3.us-west-1.amazonaws.com/public/marsh_r5a_h264.mp4

How to render a video to a texture (relevant for UE4 or UE5):

https://docs.unrealengine.com/4.26/en-US/WorkingWithMedia/MediaFramework/HowTo/FileMediaSource/

# WebGL / WebVR / Javascript

An example page is available in the /web directory of this repo.

Refer to this tutorial for more information:

https://lifecastvr.com/tutorial_webvr.html
