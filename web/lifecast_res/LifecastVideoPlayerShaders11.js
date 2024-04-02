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

const decode12bit = `
float decodeInverseDepth(vec2 depth_uv_unscaled, vec2 cell_offset) {
#if defined(DECODE_12BIT)

  vec2 depth_uv_lo = cell_offset + (depth_uv_unscaled + vec2(0.0, 1.0)) * 0.33333333 * 0.5;
  vec2 depth_uv_hi = cell_offset + (depth_uv_unscaled + vec2(1.0, 1.0)) * 0.33333333 * 0.5;

  // Sampling the texture with interpolation causes errors when reconstructing bits,
  // so we'll use texelFetch instead
  //float ds_lo = texture2D(uTexture, depth_uv_lo).r;
  //float ds_hi = texture2D(uTexture, depth_uv_hi).r;

  ivec2 texture_size = textureSize(uTexture, 0);
  ivec2 texel_coord_lo = ivec2(vec2(texture_size) * depth_uv_lo);
  ivec2 texel_coord_hi = ivec2(vec2(texture_size) * depth_uv_hi);
  float ds_lo = texelFetch(uTexture, texel_coord_lo, 0).r;
  float ds_hi = texelFetch(uTexture, texel_coord_hi, 0).r;

  int lo = int(ds_lo * 255.0) & 255;
  int hi = int(ds_hi * 255.0) & 255;
  hi = hi / 16; // decode error correcting code
  lo = (hi & 1) == 0 ? lo : 255 - lo; // unfold

  int i12 = (lo & 255) | ((hi & 15) << 8);
  float f12 = float(i12) / float((1 << 12) - 1);

  return clamp(f12, 0.0001, 1.0);

#else

  // Classic: interpolated texture2D
  //vec2 depth_uv_8bit = cell_offset + depth_uv_unscaled * 0.33333;
  //float depth_sample_8bit = clamp(texture2D(uTexture, depth_uv_8bit).r, 0.0001, 1.0);
  //return depth_sample_8bit;

  // New (maybe faster): texelFetch
  vec2 depth_uv_8bit = cell_offset + depth_uv_unscaled * 0.33333;
  ivec2 texture_size = textureSize(uTexture, 0);
  ivec2 texel_coord = ivec2(vec2(texture_size) * depth_uv_8bit);
  float v = texelFetch(uTexture, texel_coord, 0).r;
  return clamp(v, 0.0001, 1.0);

#endif
}
`;

//////////////////////////// LDI3 shaders ////////////////////////////////////////////////


export const LDI3_fthetaFgVertexShader = `
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

`
+ decode12bit +
`

void main() {
`
+ decodeStego +
`
  vUv = uv;
#if defined(LAYER2)
  float depth_sample = decodeInverseDepth(vUv, vec2(0.33333333, 0.66666666));
#else
  float depth_sample = decodeInverseDepth(vUv, vec2(0.33333333, 0.33333333));
#endif

  float s = clamp(0.3 / depth_sample, 0.01, 50.0);

  vec4 position_shifted = vec4(ftheta_rotation * normalize(position.xyz) * s, 1.0);
  gl_Position = projectionMatrix * modelViewMatrix * position_shifted;
}
`;

export const LDI3_fthetaFgFragmentShader = `
precision highp float;

#include <common>
uniform sampler2D uTexture;
uniform float uDistCamFromOrigin;

varying vec2 vUv;
varying float vMaxD;

void main() {
#if defined(LAYER2)
  vec2 alpha_uv   = vec2(vUv.s * 0.33333 + 0.66666, vUv.t * 0.33333 + 0.66666);
  vec2 depth_uv   = vec2(vUv.s * 0.33333 + 0.33333, vUv.t * 0.33333 + 0.66666);
  vec2 texture_uv = vec2(vUv.s * 0.33333,           vUv.t * 0.33333 + 0.66666);
#else
  vec2 alpha_uv   = vec2(vUv.s * 0.33333 + 0.66666, vUv.t * 0.33333 + 0.33333);
  vec2 depth_uv   = vec2(vUv.s * 0.33333 + 0.33333, vUv.t * 0.33333 + 0.33333);
  vec2 texture_uv = vec2(vUv.s * 0.33333,           vUv.t * 0.33333 + 0.33333);
#endif

  vec3 rgb = texture2D(uTexture, texture_uv).rgb;
  float a = texture2D(uTexture, alpha_uv).r;

  // Prevent very low alpha regions (which may have weird depth) from screwing up
  // the depth buffer for the next layer.
  if (a < 0.05) discard;

  gl_FragColor = vec4(rgb, a);
}
`;

export const LDI3_fthetaBgVertexShader = `
precision highp float;

uniform sampler2D uTexture;

#if defined(CHROME) || defined(PHOTO) || defined(NO_METADATA)
  uniform mat3 uFthetaRotation;
#elif !defined(CHROME)
  uniform mat3 uFrameIndexToFthetaRotation[60]; // This MUST match FTHETA_UNIFORM_ROTATION_BUFFER_SIZE in the js code.
  uniform int uFirstFrameInFthetaTable;
#endif

`
+ decode12bit +
`

varying vec2 vUv;

void main() {
`
+ decodeStego +
`
  vUv = uv;

  float depth_sample = decodeInverseDepth(vUv, vec2(0.33333333, 0.0));
  float s = clamp(0.3 / depth_sample, 0.01, 50.0);

  vec4 position_shifted = vec4(ftheta_rotation * normalize(position.xyz) * s, 1.0);

  gl_Position = projectionMatrix * modelViewMatrix * position_shifted;
}
`;

export const LDI3_fthetaBgFragmentShader = `
precision highp float;

#include <common>
uniform sampler2D uTexture;
varying vec2 vUv;

void main() {
  vec2 texture_uv = vec2(vUv.s * 0.33333, vUv.t * 0.33333);
  gl_FragColor = texture2D(uTexture, texture_uv);
}
`;
