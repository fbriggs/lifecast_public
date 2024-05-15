# Open soure immersive volumetric media by lifecast.ai

Lifecast is dedicated to pushing the limits of immersive volumetric media.
We created a new format for volumetric photos and videos called 'ldi3' which enables real-time 6DOF photorealistic rendering on a wide variety of platforms.

Visit https://lifecast.ai for some demos of what can be done with ldi3 and NeRF, which work on Vision Pro and Quest!

There are a lot of ways to create and edit ldi3. In the /nerf directory, we provide an open source nerf video engine that compresses into ldi3 for web streaming of volumetric/holographic video. We also offer some commercial software for working with ldi3:

* [Volurama](https://volurama.com/) - a Windows/Mac GUI for the NeRF engine here, for reconstructing static scenes from iPhone video input.
* [Volumetric Video Editor](https://lifecastvr.com/volumetric_video_editor.html) - Windows/Mac tool for converting VR180 to ldi3 (not NeRF-based, uses fisheye stereo depth estimation).
* [holovolo.tv](https://holovolo.tv) - text-to-ldi3 with Stable Diffusion

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

# Volumetric Video Player for Unreal Engine

Download project files:
https://github.com/fbriggs/lifecast_public/raw/main/UnrealEngine5/Lifecast_LDI3_Player_UE5.1.zip

Tutorial:
https://www.youtube.com/watch?v=ekEXxo1neVo

# Volumetric Video Player for Unity

Download sample project:
https://drive.google.com/file/d/197Ea3MHUKMsS4BUy86iVwGukmClV0V_9/view?usp=share_link


