# Lifecast.ai shut down... but we made our tools free and open source

Unfortunately, Lifecast.ai shut down. The good news is, we made all of our tools free and open source. The code for UpscaleVideo.ai, Volumetric Video Editor, and 4D Gaussian Studio are in the /lifecast_apps directory. Precompiled apps for Windows and Mac are available under [Releases](https://github.com/fbriggs/lifecast_public/releases). Many thanks to the Lifecast team, and all of our supporters!

# 4D Gaussian Studio

A complete studio for creating 3D and 4D Gaussian video.

* Custom GUI and engine for 3D and 4D Gaussian reconstruction
* Input video from one or more iPhones or GoPros (rectilinear or fisheye lenses)
* In-house structure from motion solver estimates camera poses
* For each frame of video, it builds a 3D Gaussian video scene
* Render flythroughs as 2D video
* Export into our 4DGS compression format (stored in mp4), playback in THREE.js

https://github.com/user-attachments/assets/d171b665-6039-4cca-aeff-499757e27c13

# UpscaleVideo.ai

Resize, sharpen, and de-noise videos and photos.

* Batch process video and image files with drag and drop
* Increase video resolution 2x, e.g., 1080p to 4K, 4K to 8K, or 8K to 16K
* Handles very high resolution videos, e.g., 16K output in ProRes
* Input .mp4, .mov, .mkv, .png, .jpg, .gif
* Input any size of video
* Encode output video as h264, h265, ProRes, or 16-bit PNG sequence
* Output 10-bit ProRes (Windows/Mac) or 10-bit h265 (Mac only)
* Enhance videos or photos
* Content-adaptive de-noising and de-blurring / sharpening
* Reduce aliasing and compression artifacts
* Scale AI generated videos to 4K
* De-noise with bilateral filter for photos with high grain

Demo video: https://youtu.be/Htnr8ipVnN0?si=8pBwtJGe5B9M-ohB

# Volumetric Video Editor

Unlike traditional volumetric capture systems that focus inward on a single subject, Lifecast's software enables you to capture an entire immersive scene with both foreground and background. Whether you're building an immersive mixed reality experience or creating 3D background environments for virtual production in Unreal, Lifecast offers a state-of-the-art solution for photorealistic volumetric capture, editing, and deployment.

Unlike other methods, Lifecast doesn't rely on limited-resolution depth sensors. Instead, it seamlessly integrates with VR180 cameras, such as the Canon EOS R5, that offer a wider field of view and greater resolution. These off-the-shelf cameras can be used in any environment, giving you the freedom to unleash your creativity.

Powered by cutting-edge machine learning and 3D geometric computer vision, Lifecast's Volumetric Video Editor reconstructs a 3D model for every frame of your video. The software then efficiently compresses it into a standard video file format, ready to be deployed on the web using JavaScript, WebXR, or in popular game engines like Unreal or Unity.

* Use any VR180 camera, such as the Canon EOS R5 with RF 5.2mm f/2.8 Dual Fisheye 3D lens
* VR180 cameras have higher resolution and field of view than traditional depth sensors
* Lifecast enables you to capture a full scene including the background, not just a single person or object
* Beyond light stages, VR180 cameras are highly portable and can be used in remote locations
* Beyond photogrammetry and NeRF, Lifecast's LDI3 format and rendering pipeline handles video and motion
* Render VR180 to volumetric (LDI3) format using state-of-the-art machine learning for stereo depth estimation
* Edit volumetric video channels and preview 3D results in real-time
* Automatically encode video in standard formats for deployment in Unreal, Unity, or javascript (web, mobile, VR)

Demo video: https://youtu.be/nZ0x_RItfD0?si=fbpux0YtFRUVx4EN

# Volurama

A GUI tool for creating NeRF (neural radiance field) 3D models of a scene with an iPhone, and rendering them as videos (including render to VR180).

* Capture a short video (~10 seconds) of a scene using an iPhone. Move the camera in a square or circle to capture from different points of view.
* Process the video with Volurama on Windows or Mac. Volurama creates a 3D model of the scene from your video.
* Photorealistically simulate camera motions that would typically require expensive hardware, such as dolly, boom, and orbit.
* Render eye-catching 2D videos to stand out and/or save production costs.
* Render VR180 (stereoscopic 3D) video. Create 3D VR photos and videos using only a phone.
* Render holograms for the Looking Glass Portrait.

https://github.com/user-attachments/assets/e01cd9a0-d62b-41fc-a2f9-1b8478167654

# Lifecast NeRF video engine

Create immersive volumetric video for XR with arrays of cameras. Compress for distribution on the web using Lifecast's LDI3 format. Command line tools for Linux.

(Lifecast NeRF video engine)[https://github.com/fbriggs/lifecast_public/tree/main/nerf]

# Volumetric Video Player for  WebGL / WebXR / Javascript

Here is a minimal example of embedding Lifecast ldi3 player in a div. For more examples, check out the /web directory.

```html
<html>
  <head>
  <script src="https://cdn.jsdelivr.net/gh/fbriggs/lifecast_public/lifecast.min.js"></script>
  </head>
  <body>
  <div id="player_div" style="width: 600px; height: 500px;"></div>
  <script>
  LifecastVideoPlayer.init({
    _media_urls: ["https://lifecast.ai/media/orrery_transp_ldi3.jpg"],
    _embed_in_div: "player_div",
  });
  </script>
  </body>
</html>
```

For more examples, see the web/ directory of this repo.

Notable forks:
* [akbartus's A-Frame component and simplified THREE.js player](https://github.com/akbartus/A-Frame-LifeCast-Volumetric-Player-Component)

# Volumetric Video Player for Unreal Engine

Download project files:
https://github.com/fbriggs/lifecast_public/raw/main/UnrealEngine5/Lifecast_LDI3_Player_UE5.1.zip

Tutorial:
https://www.youtube.com/watch?v=ekEXxo1neVo

# Volumetric Video Player for Unity

Download sample project:
https://drive.google.com/file/d/197Ea3MHUKMsS4BUy86iVwGukmClV0V_9/view?usp=share_link


