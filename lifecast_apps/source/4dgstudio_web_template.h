// MIT License. Copyright (c) 2025 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

namespace p11 { namespace studio4dgs {

inline const char* kSimplifiedWebTemplateHTML = R"END(
<!DOCTYPE html>
<body style="margin:0">
  <script type="importmap">
  {
    "imports": {
      "three": "https://unpkg.com/three/build/three.module.js",
      "three/addons/": "https://unpkg.com/three/examples/jsm/"
    }
  }
  </script>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { LifecastSplatMesh } from './LifecastSplatMesh.js';

    const media_url = "{INPUT_FILE}";
    let video, texture;

    if (media_url.match(/\.(mp4|webm)$/i)) {
      video = document.createElement('video');
      video.src = media_url;
      video.loop = true;
      video.playsInline = true;
      video.muted = false;
      texture = new THREE.VideoTexture(video);
      document.addEventListener("click", function() { video.play(); });
    } else {
      texture = new THREE.TextureLoader().load(media_url);
    }

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(devicePixelRatio);
    renderer.setSize(innerWidth, innerHeight);
    document.body.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, innerWidth/innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 5);
    const controls = new OrbitControls(camera, renderer.domElement);

    const num_splats = {NUM_SPLATS};
    const splat_mesh = new LifecastSplatMesh(texture, renderer, num_splats);
    scene.add(splat_mesh);

    const transform = new THREE.Matrix4().fromArray({MESH_TRANSFORM});
    splat_mesh.matrixAutoUpdate = false;
    splat_mesh.applyMatrix4(transform);
    splat_mesh.setSplatScale({SPLAT_SCALE});

    document.addEventListener('click', () => video?.play());

    renderer.setAnimationLoop(() => {
      controls.update();
      if (splat_mesh.sortByDistanceToCamera)
        splat_mesh.sortByDistanceToCamera(camera.position);
      renderer.render(scene, camera);
    });
  </script>
</body>
</html>
)END";

inline const char* kWebXRTemplateHTML = R"END(
<!DOCTYPE html>
<body style="margin: 0;">
  <script type="importmap">
    {
      "imports": {
        "three": "https://unpkg.com/three/build/three.module.js",
        "three/addons/": "https://unpkg.com/three/examples/jsm/"
      }
    }
  </script>
  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { VRButton } from 'three/addons/webxr/VRButton.js';
    import { LifecastSplatMesh } from './LifecastSplatMesh.js';
    
    let media_url = "{INPUT_FILE}";

    let renderer, scene, camera;
    let splat_texture;
    let prev_vr_camera_position;
    let world_group, interface_group;
    let video; // if not null, we have a video (otherwise its a photo)
    let vr_controller0, vr_controller1, hand0, hand1; // used for getting controller state
    let progress_div;

    function vrSessionIsActive() { return renderer?.xr?.isPresenting; }

    // If we aren't in VR, just return the camera position. In VR, try to get the midpoint between the two eyes.
    function getCameraPositionForVR() {
      if (!vrSessionIsActive() || renderer.xr.getCamera().cameras.length != 2) {
        return camera.position;
      }

      var p0 = new THREE.Vector3(0, 0, 0);
      var p1 = new THREE.Vector3(0, 0, 0);
      p0.applyMatrix4(renderer.xr.getCamera().cameras[0].matrix);
      p1.applyMatrix4(renderer.xr.getCamera().cameras[1].matrix);
      return new THREE.Vector3((p0.x + p1.x) / 2.0, (p0.y + p1.y) / 2.0, (p0.z + p1.z) / 2.0);
    }

    function resetVRToCenter() {
      if (!vrSessionIsActive() ) { return; }

      const pos = getCameraPositionForVR();
      world_group.position.set(pos.x, pos.y, pos.z);
    }

    // During the first few frames of VR, the camera position from head tracking is often
    // unreliable. For example, on the Vision Pro, it usually teleports ~1 meter after 1, 2
    // or sometimes 3 frames (its random). Instead of handling this with a timer, we'll just
    // detect any time the tracking jumps by an unreasonable amount (0.25m in 1 frame).
    function fixAppleVisionProHeadTracking() {
      if (!vrSessionIsActive() ) { return; }

      var vr_camera_position = renderer.xr.getCamera().position.clone();

      if (prev_vr_camera_position) {
        const TRACKING_JUMP_THRESHOLD_SQ = 0.25 * 0.25;
        if (vr_camera_position.distanceToSquared(prev_vr_camera_position) > TRACKING_JUMP_THRESHOLD_SQ) {
          resetVRToCenter();
        }
      }
      prev_vr_camera_position = vr_camera_position.clone();
    }

    function loadTextureFromFile(filename) {
      let use_video = filename.endsWith('.mp4') || filename.endsWith('.webm');
      if (use_video) {
        video = document.createElement('video');
        video.loop = true;
        video.setAttribute('playsinline', '');
        video.setAttribute('webkit-playsinline', '');
        video.playsInline = true;
        splat_texture = new THREE.VideoTexture(video);

        var xhr = new XMLHttpRequest();
        xhr.open('GET', filename, true);
        xhr.responseType = 'blob';
        xhr.onload = function(e) {
          console.log("Finished loading video")
          video.src = window.URL.createObjectURL(this.response);
          video.autoplay = true;
          progress_div.style.display = 'none';
        };
        xhr.onprogress = function(e) {
          if (e.lengthComputable) {
            var percentComplete = (e.loaded / e.total) * 100;
            console.log('Loading: ' + percentComplete.toFixed(2) + '%');
            progress_div.innerHTML = "Loading " + percentComplete.toFixed(2) + "% (" + (e.loaded/(1024*1024)).toFixed(2) + " / " +  (e.total/(1024*1024)).toFixed(2) + " MB)";
          } else {
            console.log('Loading: ' + e.loaded + ' bytes loaded');
          }
        };
        xhr.send();
     } else { // photo
        splat_texture = new THREE.TextureLoader().load(filename, function(t) {
          progress_div.style.display = 'none';
        }, undefined, function(error) {
          progress_div.innerHTML = 'Error loading texture: ' + error.message;
        });
        splat_texture.minFilter = THREE.LinearFilter;
        splat_texture.magFilter = THREE.LinearFilter;
        splat_texture.generateMipmaps = false;
        splat_texture.format = THREE.RGBAFormat;
      } // end photo clause
    } // end of loadTextureFromFile

    function playVideo() {
      if (!video) return;
      video.play();
    }

    function pauseVideo() {
      if (!video) return;
      video.pause();
    }

    function initVrController(vr_controller) {
      vr_controller.addEventListener('select', playVideo);

      vr_controller.addEventListener('connected', function(e) {
        vr_controller.gamepad = e.data.gamepad;
      });
    }

    function initHandControllers(left_hand, right_hand) {
      if (!left_hand || !right_hand) { return; }

      right_hand.addEventListener('pinchstart', pauseVideo);
      right_hand.addEventListener('pinchend', playVideo);
      left_hand.addEventListener('pinchstart', pauseVideo);
      left_hand.addEventListener('pinchend', playVideo);
    }

    function setupHandsOrControllers() {
      vr_controller0 = renderer.xr.getController(0);
      vr_controller1 = renderer.xr.getController(1);
      hand0 = renderer.xr.getHand(0);
      hand1 = renderer.xr.getHand(1);
      initVrController(vr_controller0);
      initVrController(vr_controller1);
      initHandControllers(hand0, hand1);
    }

    renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: "high-performance"
    });
    renderer.xr.enabled = true;
    renderer.setPixelRatio(window.devicePixelRatio); // TODO: might be much faster on some systems
    //renderer.xr.setFramebufferScaleFactor(0.95); // TODO: might help on some systems
    //renderer.xr.setFoveation(0.9);
    renderer.xr.setReferenceSpaceType('local'); // TODO: do we need this?
    renderer.setSize(window.innerWidth, window.innerHeight);

    document.body.appendChild(renderer.domElement);
    document.body.appendChild(VRButton.createButton(renderer));

    scene = new THREE.Scene();
    world_group = new THREE.Group();
    interface_group = new THREE.Group();
    scene.add(world_group);
    scene.add(interface_group);

    camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 0, 5);
    const controls = new OrbitControls(camera, renderer.domElement);

    progress_div = document.createElement('div');
    document.body.appendChild(progress_div);
    progress_div.style.position = 'fixed';
    progress_div.style.top = '0';
    progress_div.style.left = '0';
    progress_div.style.backgroundColor = '#ddd';
    progress_div.style.zIndex = '1000'; // Ensure it floats above everything
    progress_div.style.display = 'inline';
    progress_div.innerHTML = "Loading: " + media_url;

    let texture = loadTextureFromFile(media_url);
 
    const num_splats = {NUM_SPLATS};
    const splat_mesh = new LifecastSplatMesh(splat_texture, renderer, num_splats);
    world_group.add(splat_mesh);

    const transform = new THREE.Matrix4().fromArray({MESH_TRANSFORM});
    splat_mesh.matrixAutoUpdate = false;
    splat_mesh.applyMatrix4(transform);
    splat_mesh.setSplatScale({SPLAT_SCALE});

    document.addEventListener('click', function(e) {
      if (video) video.play();
    });

    setupHandsOrControllers();

    function render() {
      fixAppleVisionProHeadTracking();
      controls.update();
      if (splat_mesh.sortByDistanceToCamera) {
        splat_mesh.sortByDistanceToCamera(getCameraPositionForVR());
      }
      renderer.render(scene, camera);
    }
    renderer.setAnimationLoop(render);
  </script>
</body>
</html>
)END";

inline const char* kLifecastSplatMeshJs = R"END(
/*
Copyright 2024 Lifecast Incorporated. All rights reserved.
Unauthorized copying, modification, or redistribution of this file, via any medium is strictly prohibited.
Proprietary and confidential.
*/

import * as THREE from 'three';
import { GPUComputationRenderer } from 'three/addons/misc/GPUComputationRenderer.js';

const decode_position_shader = `
float decode12bitFrom3Pixels(float pixel1, float pixel2, float pixel3) {
  // Convert from 0-1 range to 0-255 range
  int p1 = int(pixel1 * 255.0);
  int p2 = int(pixel2 * 255.0);
  int p3 = int(pixel3 * 255.0);

  // Extract 4 bits from each pixel
  int low4 = (p1 / 16) & 0xF;
  int mid4 = (p2 / 16) & 0xF;
  int high4 = (p3 / 16) & 0xF;

  // Combine the 12 bits
  int i12 = (high4 << 8) | (mid4 << 4) | low4;

  // Convert to float in 0-1 range
  float f12 = float(i12) / float((1 << 12) - 1);

  return f12;
}

const float kLinearEncodeRadius = 5.0; // must match encoder code
// Expand from [-2, +2] to [-infinity, +infinity]
vec3 expandUnbounded(vec3 y) {
  float mag = length(y);
  return kLinearEncodeRadius * (mag < 1.0 ? y : y / (mag * (2.0 - mag)));
}

vec3 decodeSplatPosition(int col, int row) {
  vec3 pos = vec3(
    decode12bitFrom3Pixels(
      texelFetch(uSplatDataTexture, ivec2(col, row - 0), 0).r,
      texelFetch(uSplatDataTexture, ivec2(col, row - 1), 0).r,
      texelFetch(uSplatDataTexture, ivec2(col, row - 2), 0).r),
    decode12bitFrom3Pixels(
      texelFetch(uSplatDataTexture, ivec2(col, row - 3), 0).r,
      texelFetch(uSplatDataTexture, ivec2(col, row - 4), 0).r,
      texelFetch(uSplatDataTexture, ivec2(col, row - 5), 0).r),
    decode12bitFrom3Pixels(
      texelFetch(uSplatDataTexture, ivec2(col, row - 6), 0).r,
      texelFetch(uSplatDataTexture, ivec2(col, row - 7), 0).r,
      texelFetch(uSplatDataTexture, ivec2(col, row - 8), 0).r));

  vec3 splat_pos = pos * 4.0 - 2.0; // scale from [0, 1] to [-2, +2]
  return expandUnbounded(splat_pos);
}
`;

const createSplatVertexShader = () => `
const int kRowsPerSplat = 20;
uniform sampler2D uSplatDataTexture;
uniform sampler2D uSortedIndices;
uniform float uSortTextureSize;
uniform float uSplatScale;
attribute float splatIndex;
varying float vSplatIndex;
varying vec2 vUV;
varying vec4 vSplatColor;

const float kScaleBias = 1.0;
const float kMaxScaleExponent = 7.0;
float decodeScale(float encoded) { return exp(kScaleBias - encoded * kMaxScaleExponent); }

${decode_position_shader}

mat3 covariance3DFromScaleAndQuat(vec3 scale, vec4 q) {
  mat3 S = mat3(
    scale.x, 0.0, 0.0,
    0.0, scale.y, 0.0,
    0.0, 0.0, scale.z
  );
  mat3 R = mat3(
    1.0 - 2.0 * (q.z * q.z + q.w * q.w),
    2.0 * (q.y * q.z - q.x * q.w),
    2.0 * (q.y * q.w + q.x * q.z),
    2.0 * (q.y * q.z + q.x * q.w),
    1.0 - 2.0 * (q.y * q.y + q.w * q.w),
    2.0 * (q.z * q.w - q.x * q.y),
    2.0 * (q.y * q.w - q.x * q.z),
    2.0 * (q.z * q.w + q.x * q.y),
    1.0 - 2.0 * (q.y * q.y + q.z * q.z)
  );
  mat3 M = S * R;
  return transpose(M) * M;
}

mat3 jacobianOfProjection(vec3 t) {
  float fx = projectionMatrix[0][0];
  float fy = projectionMatrix[1][1];

  return mat3(
    fx / t.z,       0.0, 0.0,
    0.0,       fy / t.z, 0.0,
    -(fx * t.x) / (t.z * t.z), -(fy * t.y) / (t.z * t.z), 0.0
  );
}

void main() {
  vec2 sortedUV = vec2(mod(splatIndex, uSortTextureSize) / uSortTextureSize, floor(splatIndex / uSortTextureSize) / uSortTextureSize);
  float sortedIndex = texture2D(uSortedIndices, sortedUV).r;
  vSplatIndex = sortedIndex;
  int index = int(sortedIndex);

  ivec2 textureSize = textureSize(uSplatDataTexture, 0);
  int row = textureSize.y - 1 - ((index / textureSize.x) * kRowsPerSplat);
  int col = index % textureSize.x;

  vSplatColor = vec4(
    texelFetch(uSplatDataTexture, ivec2(col, row - 9), 0).r,
    texelFetch(uSplatDataTexture, ivec2(col, row - 10), 0).r,
    texelFetch(uSplatDataTexture, ivec2(col, row - 11), 0).r,
    texelFetch(uSplatDataTexture, ivec2(col, row - 12), 0).r);
  
  // Early skip for zero alpha (dead) splats
  if (vSplatColor.w < 0.001) return;

  vec3 splat_pos = decodeSplatPosition(col, row);

  vec3 splat_scale = uSplatScale * vec3(
    decodeScale(texelFetch(uSplatDataTexture, ivec2(col, row - 13), 0).r),
    decodeScale(texelFetch(uSplatDataTexture, ivec2(col, row - 14), 0).r),
    decodeScale(texelFetch(uSplatDataTexture, ivec2(col, row - 15), 0).r));

  vec4 splat_quat = vec4(
    texelFetch(uSplatDataTexture, ivec2(col, row - 16), 0).r,
    texelFetch(uSplatDataTexture, ivec2(col, row - 17), 0).r,
    texelFetch(uSplatDataTexture, ivec2(col, row - 18), 0).r,
    texelFetch(uSplatDataTexture, ivec2(col, row - 19), 0).r);
  // Convert from [0, 1] to [-1, 1], then normalize
  splat_quat = normalize(splat_quat * 2.0 - 1.0);

  // Compute the projection of the Gaussian center according to model view projection.
  vec4 center_in_cam_space = modelViewMatrix * vec4(splat_pos, 1.0);

  float near_clip_dist = 0.1;
  if (-center_in_cam_space.z < near_clip_dist) return;

  vec4 projected_center = projectionMatrix * center_in_cam_space;
  vec2 projected_center_perspective = projected_center.xy / projected_center.w;

  // 3D -> 2D Projection: S' = J W S W^T J^T
  mat3 J = jacobianOfProjection(center_in_cam_space.xyz);
  mat3 S = covariance3DFromScaleAndQuat(splat_scale, splat_quat);
  mat3 W = normalMatrix; // from THREE.WebGLProgram, "inverse transpose of modelViewMatrix"
  mat3 S_proj3x3 = J * W * S * transpose(W) * transpose(J);

  // Extract the 2x2 part (proven in EWA splat paper)
  mat2 cov2d = mat2(S_proj3x3[0].xy, S_proj3x3[1].xy);

  // Calculate eigenvectors of the 2D covariance matrix for major and minor axis of elipse.
  float trace = cov2d[0][0] + cov2d[1][1];
  float det = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[1][0];
  float mid = trace / 2.0;
  float radius = sqrt(mid * mid - det);
  float lambda1 = mid + radius;
  float lambda2 = mid - radius;
  if (lambda2 < 0.0) return;
  vec2 eigenvector1;
  if (cov2d[0][1] != 0.0) {
    eigenvector1 = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
  } else {
    eigenvector1 = vec2(1.0, 0.0); // Handle special case where the off-diagonal element is zero
  }
  vec2 eigenvector2 = vec2(-eigenvector1.y, eigenvector1.x); // Orthogonal to eigenvector1
  // Calculate the major and minor axes (4 standard deviations)
  float max_axis_len = 0.5; // NOTE: this limits huge splats, which can be slow (but we need them for correct rendering, so pick your poison)
  vec2 v1 = min(3.0 * sqrt(lambda1), max_axis_len) * eigenvector1;
  vec2 v2 = min(3.0 * sqrt(lambda2), max_axis_len) * eigenvector2;

  vUV = uv;
  gl_Position = vec4(projected_center_perspective + position.x * v1 + position.y * v2, 0.0, 1.0);
}
`;

const splat_fragment_shader = `
precision highp float;

varying vec2 vUV;
varying vec4 vSplatColor;

float gaussian(mat2 V, vec2 x) {
  return exp(-0.5 * dot(x, inverse(V) * x)) / sqrt(pow(2.0 * 3.1415926535, 2.0) * (determinant(V)));
}

void main() {
  vec2 uv = (vUV - vec2(0.5)) * 2.0; // uv in [-1, 1], same as clip-space coords
  float g = gaussian(mat2(1.0/9.0, 0.0, 0.0, 1.0/9.0), uv); // 1/9 --> 3 std devs

  float a = g * vSplatColor.a;
  gl_FragColor = vec4(vSplatColor.rgb, a);
}
`;

const createDistanceComputeShader = (computeTextureSize) => `
#define COMPUTE_TEXTURE_SIZE ${computeTextureSize}

uniform sampler2D uSplatDataTexture;
uniform vec3 uCameraPosition;
uniform mat4 uModelMatrix;
uniform int uNumSplats;

const int kRowsPerSplat = 20;

${decode_position_shader}

void main() {
  ivec2 pixelCoords = ivec2(gl_FragCoord.xy);
  int index = pixelCoords.y * COMPUTE_TEXTURE_SIZE + pixelCoords.x;

  // If this pixel represents a splat beyond our actual count, set a very large distance
  // so it gets sorted to the end and effectively ignored
  if (index >= uNumSplats) {
    gl_FragColor = vec4(1e10, 0.0, 0.0, 1.0);
    return;
  }

  ivec2 textureSize = textureSize(uSplatDataTexture, 0);
  int row = textureSize.y - 1 - ((index / textureSize.x) * kRowsPerSplat);
  int col = index % textureSize.x;

  // Get splat position and transform it according to the model matrix (into world coords)
  vec3 splat_pos = (uModelMatrix * vec4(decodeSplatPosition(col, row), 1.0)).xyz;
  float distance = length(splat_pos - uCameraPosition);

  gl_FragColor = vec4(distance, 0.0, 0.0, 1.0);
}
`;

const createSortFragmentShader = () => `
  uniform vec3 Param1;
  uniform vec3 Param2;
  uniform sampler2D uSortKeyTexture;
  uniform sampler2D uIndexTexture;

  #define Stage          Param1.x
  #define Pass           Param1.y
  #define Unused         Param1.z  
  #define Width          Param2.x
  #define Height         Param2.y
  #define PassDistance   Param2.z

  void main() {
    vec2 uv = gl_FragCoord.xy / vec2(Width, Height);
    float self_index = texture2D(uIndexTexture, uv).r;
    int self_index_int = int(self_index);
    float self_value = texture2D(uSortKeyTexture, vec2(mod(float(self_index_int), Width) / Width, floor(float(self_index_int) / Width) / Height)).r;

    float i = floor(gl_FragCoord.x) + floor(gl_FragCoord.y) * Width;
    int iPos = int(i);
    int iPass = int(PassDistance);
    
    float compare;
    if ((iPos & iPass) == 0) {
      compare = 1.0;  // compare with right partner at +Pass distance
    } else {
      compare = -1.0; // compare with left partner at -Pass distance  
    }
    
    float adr = i + compare * PassDistance;
    vec2 partner_uv = vec2(mod(adr, Width) / Width, floor(adr / Width) / Height);
    float partner_index = texture2D(uIndexTexture, partner_uv).r;
    int partner_index_int = int(partner_index);
    float partner_value = texture2D(uSortKeyTexture, vec2(mod(float(partner_index_int), Width) / Width, floor(float(partner_index_int) / Width) / Height)).r;

    // Bitonic sequence direction - alternating ascending/descending
    int iStage = int(Stage);
    float direction = ((iPos & iStage) == 0) ? 1.0 : -1.0;

    gl_FragColor = vec4((self_value * compare * direction < partner_value * compare * direction) ? self_index : partner_index, 0.0, 0.0, 1.0);
  }
`;

// Helper function to calculate next power of 2
function nextPowerOfTwo(n) {
  return Math.pow(2, Math.ceil(Math.log2(n)));
}

export class LifecastSplatMesh extends THREE.Mesh {
  constructor(texture, renderer, numSplats) {
    // Calculate the minimum compute texture size that can accommodate all splats
    const minTextureSize = Math.ceil(Math.sqrt(numSplats));
    const computeTextureSize = nextPowerOfTwo(minTextureSize);

    const geometry = new THREE.BufferGeometry();
    const splat_index = new Float32Array(numSplats * 4);
    const vertex_index_dynamic = [];
    let positions_dynamic = [];
    let uvs_dynamic = [];
    let vertexCount = 0;
    
    for (let i = 0; i < numSplats; i++) {
        let first_idx = vertexCount;
        positions_dynamic.push(
            -1.0, -1.0, 0.0,
            1.0, -1.0, 0.0,
            1.0, 1.0, 0.0,
            -1.0, 1.0, 0.0
        );
        uvs_dynamic.push(
            0.0, 0.0,
            1.0, 0.0,
            1.0, 1.0,
            0.0, 1.0
        );
        for (let j = 0; j < 4; ++j) {
            // We sorted in ascending order, so we must draw in reverse order (back-to-front)
            splat_index[vertexCount + j] = numSplats - i - 1;
        }
        vertex_index_dynamic.push(
            first_idx + 0, first_idx + 1, first_idx + 2,
            first_idx + 0, first_idx + 2, first_idx + 3
        );
        vertexCount += 4;
    }

    const positions = new Float32Array(positions_dynamic);
    const uvs = new Float32Array(uvs_dynamic);
    const vertex_index = new Uint32Array(vertex_index_dynamic);

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    geometry.setAttribute('splatIndex', new THREE.BufferAttribute(splat_index, 1));
    geometry.setIndex(new THREE.BufferAttribute(vertex_index, 1));

    // Create material with dynamic shader
    const material = new THREE.ShaderMaterial({
      uniforms: {
        uSplatDataTexture: { value: texture },
        uSortedIndices: { value: null },
        uSortTextureSize: { value: 0 },
        uSplatScale: { value: 1.0 }
      },
      vertexShader: createSplatVertexShader(),
      fragmentShader: splat_fragment_shader,
      transparent: true,
      blending: THREE.NormalBlending,
      depthWrite: false,
      depthTest: false
    });

    super(geometry, material); // super must be called before we can use .this
    this.frustumCulled = false;
    this.numSplats = numSplats;
    this.computeTextureSize = computeTextureSize;
    this.setupGPUComputation(renderer, texture);
  }

  setSplatScale(s) { this.material.uniforms.uSplatScale.value = s * s; }

  setupGPUComputation(renderer, texture) {
    this.gpu_compute = new GPUComputationRenderer(this.computeTextureSize, this.computeTextureSize, renderer);

    this.distance_render_target = this.gpu_compute.createRenderTarget();

    this.distance_shader_material = this.gpu_compute.createShaderMaterial(createDistanceComputeShader(this.computeTextureSize), {
      uSplatDataTexture: { value: texture },
      uCameraPosition: { value: new THREE.Vector3(0, 0, 0) },
      uModelMatrix: { value: new THREE.Matrix4() },
      uNumSplats: { value: this.numSplats }
    });

    this.sort_material = this.gpu_compute.createShaderMaterial(
      createSortFragmentShader(),
      {
        Param1: { value: new THREE.Vector3(2, 1, 0) },
        Param2: { value: new THREE.Vector3(this.computeTextureSize, this.computeTextureSize, 1) },
        uSortKeyTexture: { value: this.distance_render_target.texture },
        uIndexTexture: { value: null }
      }
    );

    this.sort_render_target1 = this.gpu_compute.createRenderTarget();
    this.sort_render_target2 = this.gpu_compute.createRenderTarget();

    // Create initial index texture - pad with indices beyond numSplats for sorting
    const totalTexturePixels = this.computeTextureSize * this.computeTextureSize;
    const index_texture_data = new Float32Array(totalTexturePixels);
    for (let i = 0; i < totalTexturePixels; i++) {
      index_texture_data[i] = i;
    }
    this.index_texture = new THREE.DataTexture(index_texture_data, this.computeTextureSize, this.computeTextureSize, THREE.RedFormat, THREE.FloatType);
    this.index_texture.needsUpdate = true;
  }

  sortByDistanceToCamera(camera_pos) {
    this.distance_shader_material.uniforms['uCameraPosition'].value.copy(camera_pos);
    this.distance_shader_material.uniforms['uModelMatrix'].value.copy(this.matrixWorld);
    this.gpu_compute.doRenderTarget(this.distance_shader_material, this.distance_render_target);
  
    // Bitonic sort (works on the full texture size, padding handled by distance shader)
    let curr_render_target = this.sort_render_target1;
    let next_render_target = this.sort_render_target2;
    this.sort_material.uniforms.uIndexTexture.value = this.index_texture;
    const totalTexturePixels = this.computeTextureSize * this.computeTextureSize;
    for (let stage = 2; stage <= totalTexturePixels; stage *= 2) {
      for (let pass = stage / 2; pass >= 1; pass = Math.floor(pass / 2)) {
        this.sort_material.uniforms.Param1.value.set(stage, pass, 0);
        this.sort_material.uniforms.Param2.value.set(this.computeTextureSize, this.computeTextureSize, pass);
        
        this.sort_material.uniforms.uIndexTexture.value = 
          (stage === 2 && pass === 1) ? this.index_texture : curr_render_target.texture;
  
        this.gpu_compute.doRenderTarget(this.sort_material, next_render_target);
  
        [curr_render_target, next_render_target] = [next_render_target, curr_render_target];
      }
    }
  
    this.material.uniforms.uSortedIndices.value = curr_render_target.texture;
    this.material.uniforms.uSortTextureSize.value = this.computeTextureSize;
  }
}
)END";

}} // namespace p11::studio4dgs
