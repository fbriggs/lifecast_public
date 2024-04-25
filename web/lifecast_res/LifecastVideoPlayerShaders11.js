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
varying vec2 vUv;
`
+ decode12bit +
`
void main() {
  vUv = uv;
#if defined(LAYER2)
  float depth_sample = decodeInverseDepth(vUv, vec2(0.33333333, 0.66666666));
#else
  float depth_sample = decodeInverseDepth(vUv, vec2(0.33333333, 0.33333333));
#endif

  float s = clamp(0.3 / depth_sample, 0.01, 50.0);

  vec4 position_shifted = vec4(position.xyz * s, 1.0);
  gl_Position = projectionMatrix * modelViewMatrix * position_shifted;
}
`;

export const LDI3_fthetaFgFragmentShader = `
precision highp float;

#include <common>
uniform sampler2D uTexture;

varying vec2 vUv;

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
  if (a < 0.1) discard;

  gl_FragColor = vec4(rgb, a);
}
`;

export const LDI3_fthetaBgVertexShader = `
precision highp float;
uniform sampler2D uTexture;
`
+ decode12bit +
`
varying vec2 vUv;

void main() {
  vUv = uv;

  float depth_sample = decodeInverseDepth(vUv, vec2(0.33333333, 0.0));
  float s = clamp(0.3 / depth_sample, 0.01, 50.0);

  vec4 position_shifted = vec4(position.xyz * s, 1.0);

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

  vec2 alpha_uv   = vec2(vUv.s * 0.33333 + 0.66666, vUv.t * 0.33333);
  float a = texture2D(uTexture, alpha_uv).r;
  if (a < 0.05) discard;
  gl_FragColor = vec4(texture2D(uTexture, texture_uv).rgb, a);
}
`;
