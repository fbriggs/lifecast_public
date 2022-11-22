/*
The MIT License

Copyright © 2021 Lifecast Incorporated

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

export const debugVertexShader = `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

export const debugFragmentShader = `
precision highp float;

#include <common>
uniform sampler2D uTexture;
varying vec2 vUv;

void main() {
  vec2 texture_uv = vec2(vUv.s, vUv.t * 0.5 + 0.5);
  gl_FragColor = texture2D(uTexture, texture_uv);
}
`;

const decodeStego = `
// If we are in Chrome, the current frame's rotation is available in a uniform "simply".
// Otherwise, we need to decode the frame index from stenography. This costs 6 extra
// texture reads + a buttload of shenanigans to stream in the uniforms in blocks.
#if (defined(CHROME) || defined(PHOTO) || defined(NO_METADATA))
  mat3 ftheta_rotation = uFthetaRotation;
#elif !defined(CHROME)
  float block_ds = (16.0 / 2048.0) / 2.0; // This constant comes from the C++ code for stenography. It should scale regardless of the actual texture size. There is an extra factor of 1/2 here because the texture is not square. TODO: what if it is square!?!!!!!
  vec3 blocks[6];
  int bits[18];

  blocks[0] = texture2D(uTexture, vec2(block_ds * 0.5, 1.0)).rgb;
  blocks[1] = texture2D(uTexture, vec2(block_ds * 1.5, 1.0)).rgb;
  blocks[2] = texture2D(uTexture, vec2(block_ds * 2.5, 1.0)).rgb;
  blocks[3] = texture2D(uTexture, vec2(block_ds * 3.5, 1.0)).rgb;
  blocks[4] = texture2D(uTexture, vec2(block_ds * 4.5, 1.0)).rgb;
  blocks[5] = texture2D(uTexture, vec2(block_ds * 5.5, 1.0)).rgb;

  bits[0] =  blocks[0].r > 0.5 ? 1  : 0; // Note: I would write these powers of 2 with bit shifts, but Safari doesn't like bit shift operator in GLSL (regardless of version).
  bits[1] =  blocks[0].g > 0.5 ? 2  : 0;
  bits[2] =  blocks[0].b > 0.5 ? 4  : 0;
  bits[3] =  blocks[1].r > 0.5 ? 8  : 0;
  bits[4] =  blocks[1].g > 0.5 ? 16  : 0;
  bits[5] =  blocks[1].b > 0.5 ? 32  : 0;
  bits[6] =  blocks[2].r > 0.5 ? 64  : 0;
  bits[7] =  blocks[2].g > 0.5 ? 128  : 0;
  bits[8] =  blocks[2].b > 0.5 ? 256  : 0;
  bits[9] =  blocks[3].r > 0.5 ? 512  : 0;
  bits[10] = blocks[3].g > 0.5 ? 1024 : 0;
  bits[11] = blocks[3].b > 0.5 ? 2048 : 0;
  bits[12] = blocks[4].r > 0.5 ? 4096 : 0;
  bits[13] = blocks[4].g > 0.5 ? 8192 : 0;
  bits[14] = blocks[4].b > 0.5 ? 16384 : 0;
  bits[15] = blocks[5].r > 0.5 ? 32768 : 0;
  bits[16] = blocks[5].g > 0.5 ? 65536 : 0;
  bits[17] = blocks[5].b > 0.5 ? 131072 : 0;

  int frame_num = bits[0] + bits[1] + bits[2] + bits[3] + bits[4] + bits[5] + bits[6] + bits[7] + bits[8] + bits[9] + bits[10] + bits[11] + bits[12] + bits[13] + bits[14]+ bits[15]+ bits[16]+ bits[17];

  mat3 ftheta_rotation = uFrameIndexToFthetaRotation[frame_num - uFirstFrameInFthetaTable];
#endif
`;

export const fthetaFgVertexShader = `
precision highp float;

uniform sampler2D uTexture;

#if (defined(CHROME) || defined(PHOTO) || defined(NO_METADATA))
  uniform mat3 uFthetaRotation;
#elif !defined(CHROME)
  uniform mat3 uFrameIndexToFthetaRotation[60]; // This MUST match FTHETA_UNIFORM_ROTATION_BUFFER_SIZE in the js code.
  uniform int uFirstFrameInFthetaTable;
#endif

varying vec2 vUv;
varying float vMaxD;
//varying float vMaxD2;

void main() {
  vUv = uv;
  vec2 depth_uv = vec2(vUv.s * 0.5 + 0.5, vUv.t * 0.5 + 0.5);
`
+ decodeStego +
`

  float border_thickness = 1.0;
  float eps = border_thickness / (32.0 * 18.0);
  float eps2 = eps * 2.0;

  float ds1 = texture2D(uTexture, depth_uv + vec2(-eps, 0)).r;
  float ds2 = texture2D(uTexture, depth_uv + vec2(eps, 0)).r;
  float ds3 = texture2D(uTexture, depth_uv + vec2(0, -eps)).r;
  float ds4 = texture2D(uTexture, depth_uv + vec2(0, eps)).r;
  //float max_d = ds; // NOTE: this was a bug before. ds was never a valid sample..., it was epsilon.!
  float max_d =  ds1;
  max_d = max(max_d, ds2);
  max_d = max(max_d, ds3);
  max_d = max(max_d, ds4);
  //vMaxD = max_d;

  float ds5 = texture2D(uTexture, depth_uv + vec2(-eps2, 0)).r;
  float ds6 = texture2D(uTexture, depth_uv + vec2(eps2, 0)).r;
  float ds7 = texture2D(uTexture, depth_uv + vec2(0, -eps2)).r;
  float ds8 = texture2D(uTexture, depth_uv + vec2(0, eps2)).r;
  float max_d2 =  ds5;
  max_d2 = max(max_d2, ds6);
  max_d2 = max(max_d2, ds7);
  max_d2 = max(max_d2, ds8);
  vMaxD = max_d2;

  float s = clamp(0.3 / max_d, 0.01, 50.0); // TODO: maybe we can go higher than 50 here.

  vec4 position_shifted = vec4(ftheta_rotation * normalize(position.xyz) * s, 1.0);
  gl_Position = projectionMatrix * modelViewMatrix * position_shifted;
}
`;

export const fthetaFgFragmentShader = `
precision highp float;

#include <common>
uniform sampler2D uTexture;
uniform float uDistCamFromOrigin;

varying vec2 vUv;
varying float vMaxD;

void main() {
  vec2 depth_uv   = vec2(vUv.s * 0.5 + 0.5, vUv.t * 0.5 + 0.5);
  vec2 texture_uv = vec2(vUv.s * 0.5,       vUv.t * 0.5 + 0.5);

  float depth_sample = clamp(texture2D(uTexture, depth_uv).r, 0.0001, 1.0);

  //float lcp = clamp(uDistCamFromOrigin - 0.01, 0.01, 0.1) + 0.01;
  float lcp = 0.1;
  float a = 1.0 - clamp(lcp * 17.0 * (vMaxD - depth_sample), 0.0, 1.0);

  float vignette = 4.0 * pow(2.0 * length(vUv - vec2(0.5, 0.5)), 15.0);
  vec3 rgb = texture2D(uTexture, texture_uv).rgb;

  // Make it so we can't discard texels with zero gradient. This helps avoid making holes
  // in the vignette.
  //vec3 coefs = vec3(0.33, 0.33, 0.33);
  //float delta = 4.0/4096.0;
  //vec3 diff1 = rgb - texture2D(uTexture, texture_uv + vec2(-delta, -delta)).rgb;
  //vec3 diff2 = rgb - texture2D(uTexture, texture_uv + vec2(-delta, delta)).rgb;
  //vec3 diff3 = rgb - texture2D(uTexture, texture_uv + vec2(delta, -delta)).rgb;
  //vec3 diff4 = rgb - texture2D(uTexture, texture_uv + vec2(delta, -delta)).rgb;
  //float g = length(diff1) + length(diff2) + length(diff3) + length(diff4);
  //g = clamp(1.0 - g * 10.0, 0.0, 1.0);

  //a = a + g * 0.1;
  //a = a + vignette + g * 0.1;
  a = a + vignette;

  if (a < 0.84) discard;

  gl_FragColor = vec4(rgb, 1.0);
  //gl_FragColor = vec4(a, a, a, 1.0);
  //gl_FragColor = vec4(g, g, g, 1.0);
  //gl_FragColor = vec4(vignette, vignette, vignette, 1.0);
}
`;

export const fthetaBgVertexShader = `
precision highp float;

uniform sampler2D uTexture;

#if defined(CHROME) || defined(PHOTO) || defined(NO_METADATA)
  uniform mat3 uFthetaRotation;
#elif !defined(CHROME)
  uniform mat3 uFrameIndexToFthetaRotation[60]; // This MUST match FTHETA_UNIFORM_ROTATION_BUFFER_SIZE in the js code.
  uniform int uFirstFrameInFthetaTable;
#endif

varying vec2 vUv;

void main() {
`
+ decodeStego +
`
  vUv = uv;
  vec2 depth_uv    = vec2(vUv.s * 0.5 + 0.5, vUv.t * 0.5);

  float depth_sample = clamp(texture2D(uTexture, depth_uv).r, 0.0001, 1.0);
  float hack = 1.03; // without this, the background sometimes goes in front with depth testing when it shouldn't
  float s = hack * clamp(0.3 / depth_sample, 0.01, 50.0);

  vec4 position_shifted = vec4(ftheta_rotation * normalize(position.xyz) * s, 1.0);

  gl_Position = projectionMatrix * modelViewMatrix * position_shifted;
}
`;

export const fthetaBgFragmentShader = `
precision highp float;

#include <common>
uniform sampler2D uTexture;
varying vec2 vUv;

void main() {
  vec2 texture_uv = vec2(vUv.s * 0.5, vUv.t * 0.5);
  gl_FragColor = texture2D(uTexture, texture_uv);
  //gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
`;

