# Lifecast Inc. Open Source

Lifecast makes software for immersive volumetric VR videos and photos. Lifecast's 6DOF format for 3D photos and videos can be generated using tools available on https://lifecastvr.com. This repo hosts the player for this format, which is open source under the MIT license. Several implementations of the player are available, in javascript, Unity and Unreal Engine.

Do you just want to make VR content without writing code? We do the same processing in the cloud and automatically host the results on https://holovolo.tv


# Unreal Engine

Download project files:
https://github.com/fbriggs/lifecast_public/raw/main/UnrealEngine5/Lifecast_LDI3_Player_UE5.1.zip

Tutorial:
https://www.youtube.com/watch?v=ekEXxo1neVo

# Unity

Download sample project:
https://drive.google.com/file/d/197Ea3MHUKMsS4BUy86iVwGukmClV0V_9/view?usp=share_link


# WebGL / WebVR / Javascript

Use the following HTML to display an LDI file on your own site:

```html
<!DOCTYPE html>
<html>
<head>
<script src="https://cdn.jsdelivr.net/gh/fbriggs/lifecast_public/lifecast.min.js"></script>
</head>
<body>
<script>
LifecastVideoPlayer.init({
  _format: "ldi3",
  _media_urls: ["ldi3.png"],
});
</script>
</body>
</html>
```

For more examples, see the web/ directory of this repo.

