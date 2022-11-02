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

Video tutorial on using Lifecast player with Unreal Engine:

https://www.youtube.com/watch?v=HpGLGWP0tFo&ab_channel=Lifecast

Sample 6DOF video file for testing: 

https://lifecastvideocdn.s3.us-west-1.amazonaws.com/public/marsh_r5a_h264.mp4

How to render a video to a texture:

https://docs.unrealengine.com/4.26/en-US/WorkingWithMedia/MediaFramework/HowTo/FileMediaSource/

# WebGL / WebVR / Javascript

An example page is available in the /web directory of this repo.

Refer to this tutorial for more information:

https://lifecastvr.com/tutorial_webvr.html
