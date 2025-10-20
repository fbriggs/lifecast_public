# Lifecast.ai shut down... but we made our tools free and open source

Unfortunately, Lifecast.ai shut down. The good news is, we made all of our tools free and open source. The code for UpscaleVideo.ai, Volumetric Video Editor, and 4D Gaussian Studio are in the /lifecast_apps directory. Precompiled apps for Windows and Mac are available under [Releases](https://github.com/fbriggs/lifecast_public/releases). Many thanks to the Lifecast team, and all of our supporters!

# 4D Gaussian Studio

* Custom GUI and engine for 3D and 4D Gaussian reconstruction
* Input video from one or more iPhones or GoPros (rectilinear or fisheye lenses)
* In-house structure from motion solver estimates camera poses
* For each frame of video, it builds a 3D Gaussian video scene
* Render flythroughs as 2D video
* Export into our 4DGS compression format (stored in mp4), playback in THREE.js



https://github.com/user-attachments/assets/d171b665-6039-4cca-aeff-499757e27c13



# UpscaleVideo.ai

# Volumetric Video Editor

# Volurama

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


