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

import {FTHETA_UNIFORM_ROTATION_BUFFER_SIZE, LdiFthetaMesh} from "./LdiFthetaMesh11.js";
import * as THREE from './three152.module.min.js';
import {OrbitControls} from "./OrbitControls.js";
import {TimedVideoTexture} from "./TimedVideoTexture.js";
import {HTMLMesh} from './HTMLMesh.js';
import {HelpGetVR} from './HelpGetVR11.js';
import {GestureControlModule} from './GestureControlModule.js';

const CubeFace = {
  FRONT_LEFT:   0,
  BACK_LEFT:    1,
  BACK_RIGHT:   2,
  FRONT_RIGHT:  3,
  BOTTOM_LEFT:  4,
  BOTTOM_RIGHT: 5,
  TOP_LEFT:     6,
  TOP_RIGHT:    7
};


const gesture_control = new GestureControlModule();

let enable_debug_text = false; // Turn this on if you want to use debugLog() or setDebugText().
let debug_text_mesh, debug_text_div;
let debug_log = "";
let debug_msg_count = 0;

let container, camera, scene, renderer, vr_controller0, vr_controller1;
let hand0, hand1; // for XR hand-tracking
let ldi_ftheta_mesh;
let world_group; // A THREE.Group that stores all of the meshes (foreground and background), so they can be transformed together by modifying the group.
let prev_vr_camera_position;
let left_finger_indicator, right_finger_indicator;
let axis_indicator;

let video;
let texture;
let vid_framerate = 30;
let nonvr_menu_fade_counter = 0;
let mouse_is_down = false;

let prev_mouse_u = 0.5;
let prev_mouse_v = 0.5;
let cam_drag_u = 0.0;
let cam_drag_v = 0.0;
let right_mouse_is_down = false;

let nonvr_controls;
let video_is_buffering = false;
let vr_session_active = false; // true if we are in VR
let vrbutton3d, vrbutton_material, vrbutton_texture_rewind, vrbutton_texture_buffering; // THREE.js object to render VR-only buttons
// Used to keep track of whether a click or a drag occured. When a mousedown event occurs,
// this becomes true, and a timer starts. When the timer expires, it becomes false.
// If the mouseup even happens before the timer, it will be counted as a click.
let maybe_click = false;
let delay1frame_reset = false; // The sessionstart event happens one frame too early. We need to wait 1 frame to reset the view after entering VR.
let photo_mode = false;
let embed_mode = false;
let cam_mode = "default";
let vscroll_bias = 0; // Offsets the scroll effect in vscroll camera mode.
let metadata;
let metadata_url;
let next_video_url;
let next_video_thumbnail;
let slideshow;
let slideshow_index = 0;

let lock_position = false;
let orbit_controls;
let create_button_url;
let next_video_button, create_button;
let mouse_last_moved_time = 0;

let has_played_video = false;

let get_vr_button;

let looking_glass_config;

// Used for IMU based control on mobile
let got_orientation_data = false;
let init_orientation_a = 0;
let init_orientation_b = 0;
let init_orientation_c = 0;

let mobile_drag_u = 0.0;
let mobile_drag_v = 0.0;

// Used for programmatic camera animation
let anim_fov_offset = 80;
let anim_x_offset = 0;
let anim_y_offset = 0;
let anim_z_offset = 0;
let anim_u_offset = 0;
let anim_v_offset = 0;
let anim_fov = 0;
let anim_x = 0.15;
let anim_y = 0.10;
let anim_z = 0.05;
let anim_u = 0.15;
let anim_v = 0.10;
let anim_fov_speed = 3000;
let anim_x_speed = 7500;
let anim_y_speed = 5100;
let anim_z_speed = 6100;
let anim_u_speed = 4500;
let anim_v_speed = 5100;

let AUTO_CAM_MOVE_TIME = 5000;

var is_firefox = navigator.userAgent.indexOf("Firefox") != -1;
var is_oculus = (navigator.userAgent.indexOf("Oculus") != -1);
var is_chrome =  (navigator.userAgent.indexOf("Chrome")  != -1) || is_oculus;
var is_safari =  (navigator.userAgent.indexOf("Safari")  != -1) && !is_chrome;
var is_ios = navigator.userAgent.match(/iPhone|iPad|iPod/i);
// TODO: android?

function byId(id) { return document.getElementById( id ); };

function filenameExtension(filename) { return filename.split('.').pop(); }

function hasAudio (video) {
    return video.mozHasAudio ||
    Boolean(video.webkitAudioDecodedByteCount) ||
    Boolean(video.audioTracks && video.audioTracks.length);
}

function loadJSON(json_path, callback) {
  var xobj = new XMLHttpRequest();
  xobj.overrideMimeType("application/json");
  xobj.open('GET', json_path, true);
  xobj.onreadystatechange = function() {
    if (xobj.readyState == 4 && xobj.status == "200") { callback(JSON.parse(xobj.responseText)); }
  };
  xobj.send(null);
}

function setBodyStyle() {
  document.body.style.margin              = "0px";
  document.body.style.padding             = "0px";
}

function makeUnselectable(element) {
  element.style["-webkit-touch-callout"]  = "none";
  element.style["-webkit-user-select"]    = "none";
  element.style["-khtml-user-select"]     = "none";
  element.style["-moz-user-select"]       = "none";
  element.style["-ms-user-select"]        = "none";
  element.style["user-select"]            = "none";
}

function trackMouseStatus(element) {
  element.addEventListener('mouseover', function() { element.mouse_is_over = true; });
  element.addEventListener('mouseout', function() { element.mouse_is_over = false; });
}

function makeNonVrControls() {
  var right_buttons_width = 0;
  if (next_video_url && !embed_mode) {
    next_video_button = document.createElement("a");
    next_video_button.id                      = "next_video_button";
    next_video_button.href                    = next_video_url;
    next_video_button.innerHTML               = `<img src="${next_video_thumbnail}" style="width:96px;height:60px;border-radius:16px;object-fit:cover">`;
    next_video_button.style.cursor            = "pointer";
    next_video_button.draggable               = false;
    next_video_button.style.position          = "absolute";
    next_video_button.style.width             = '100px';
    next_video_button.style.right             = "16px";
    next_video_button.style.bottom            = "16px";
    right_buttons_width += 116;
    document.body.appendChild(next_video_button);
  }

  if (create_button_url && !embed_mode) {
    create_button = document.createElement("a");
    create_button.id                      = "create_button";
    create_button.href                    = create_button_url;
    create_button.target                  = "_blank";
    create_button.innerHTML               = `<img src="lifecast_res/plus_button.png" style="width: 48px; height: 48px;">`;
    create_button.style.cursor            = "pointer";
    create_button.draggable               = false;
    create_button.style.position          = "absolute";
    create_button.style.right             = (right_buttons_width + 16).toString() + "px";
    create_button.style.bottom            = "22px";
    document.body.appendChild(create_button);
  }

  if (photo_mode || embed_mode) return;

  nonvr_controls = document.createElement("div");
  nonvr_controls.id = "nonvr_controls";
  nonvr_controls.style["margin"]            = "auto";
  nonvr_controls.style["position"]          = "absolute";
  nonvr_controls.style["top"]               = "1";
  nonvr_controls.style["left"]              = "0";
  nonvr_controls.style["bottom"]            = "0";
  nonvr_controls.style["margin-left"]       = "16px";
  nonvr_controls.style["margin-bottom"]     = "22px";

  let sz = "64px";
  nonvr_controls.style["width"]             = sz;
  nonvr_controls.style["height"]            = sz;
  nonvr_controls.style["cursor"]            = "pointer";

  const play_button = document.createElement("img");
  play_button.id                            = "play_button";
  play_button.src                           = "lifecast_res/play_button.png";
  play_button.draggable                     = false;
  play_button.style.display                 = "none";
  play_button.style.width                   = sz;

  const pause_button = document.createElement("img");
  pause_button.id                           = "pause_button";
  pause_button.src                          = "lifecast_res/pause_button.png";
  pause_button.draggable                    = false;
  pause_button.style.display                = "none";
  pause_button.style.width                  = sz;

  const rewind_button = document.createElement("img");
  rewind_button.id                          = "rewind_button";
  rewind_button.src                         = "lifecast_res/rewind_button.png";
  rewind_button.draggable                   = false;
  rewind_button.style.display               = "none";
  rewind_button.style.width                 = sz;

  const buffering_button = document.createElement("img");
  buffering_button.id                       = "buffering_button";
  buffering_button.src                      = "lifecast_res/spinner.png";
  buffering_button.draggable                = false;
  buffering_button.style.display            = "none";
  buffering_button.style.opacity            = 0.5;
  buffering_button.style.width              = sz;


  var spinner_rotation_angle = 0;
  setInterval(function() {
    buffering_button.style.transform = "rotate(" + spinner_rotation_angle + "deg)";
    spinner_rotation_angle += 5;
  }, 16);

  makeUnselectable(nonvr_controls);
  nonvr_controls.appendChild(play_button);
  nonvr_controls.appendChild(pause_button);
  nonvr_controls.appendChild(rewind_button);
  nonvr_controls.appendChild(buffering_button);

  document.body.appendChild(nonvr_controls);
}

function debugLog(message) {
  ++debug_msg_count;
  if (debug_msg_count > 30) {
    //return; // HACK: stop adding new messages once we reach a limit
    debug_log = "";
    debug_msg_count = 0;
  }
  debug_log += message + "<br>";
  setDebugText(debug_log);
}

function setDebugText(message) {
  if (!enable_debug_text) return;
  debug_text_div.innerHTML = message;
}

function handleGenericButtonPress() {
  if (photo_mode) {
    resetVRToCenter();
  } else {
    toggleVideoPlayPause();
  }
}

function resetVRToCenter() {
  if (!renderer.xr.isPresenting) return;
  delay1frame_reset = false;

  // Sadly, the code below is close but not quite right (it doesn't get 0 when the Oculus
  // reset button is pressed). Whatever is in renderer.xr.getCamera() isn't the position
  // we need.
  //var p = renderer.xr.getCamera().position;
  //world_group.position.set(p.x, p.y, p.z);

  // Instead, we need to find the average point between the left and right camera.
  if (renderer.xr.getCamera().cameras.length == 2) {
    // The position of the left or right camera in the world coordinate frame can be
    // found by multiplying the 0 vector by transform to world from camera.
    var p0 = new THREE.Vector3(0, 0, 0);
    var p1 = new THREE.Vector3(0, 0, 0);
    p0.applyMatrix4(renderer.xr.getCamera().cameras[0].matrix);
    p1.applyMatrix4(renderer.xr.getCamera().cameras[1].matrix);

    // Find the point half way between the left and right camera.
    var avg_x = (p0.x + p1.x) / 2.0;
    var avg_y = (p0.y + p1.y) / 2.0;
    var avg_z = (p0.z + p1.z) / 2.0;

    world_group.position.set(avg_x, avg_y, avg_z);
  }
}

function playVideoIfReady() {
  if (!metadata && metadata_url != "") {
    // TODO: we should do a better job of showing this status to the user (similar to buffering)
    console.log("Can't play because metadata not yet loaded.");
    return;
  }
  if (!video) return;

  video.play();
  has_played_video = true;
  updateFthetaRotationUniforms(video.currentTime);
}

function toggleVideoPlayPause() {
  if (photo_mode || embed_mode) return;

  nonvr_menu_fade_counter = 60;
  const video_is_playing = !!(video.currentTime > 0 && !video.paused && !video.ended && video.readyState > 2);
  if (video_is_playing || video_is_buffering) {
    video_is_buffering = false;
    video.pause();
  } else {
    resetVRToCenter();
    playVideoIfReady();
  }
}

function handleNonVrPlayButton() {
  playVideoIfReady();
}

function handleNonVrPauseButton() {
  video.pause();
}


function verticalScrollCameraHandler() {
  var h = window.innerHeight;
  var top = window.pageYOffset || document.documentElement.scrollTop;
  var container_rect = container.getBoundingClientRect();
  var y0 = container_rect.top;
  var y1 = container_rect.bottom;
  var y_mid = (y0 + y1) * 0.5;
  var y_frac = 2.0 * y_mid / h - 1.0; // Ranges from -1 to 1 as the middle of the container moves from top to bottom of the window.
  camera.position.set(0, -y_frac * 0.2 + vscroll_bias, 0.01);
}

function onWindowResize() {
  // In embed mode, use the width and height of the container div.
  let width = embed_mode ? container.clientWidth : window.innerWidth;
  let height = embed_mode ? container.clientHeight : window.innerHeight;
  camera.aspect = width / height;
  renderer.setSize(width, height);
  camera.updateProjectionMatrix();

  if (cam_mode == "vscroll") { verticalScrollCameraHandler(); }
}

function updateControlsAndButtons() {
  if (photo_mode || embed_mode) return;

  const video_is_playing = !!(video.currentTime > 0 && !video.paused && !video.ended && video.readyState >= 2);

  // Fade out but only if the mouse is not over a button
  if (!nonvr_controls.mouse_is_over) {
    --nonvr_menu_fade_counter;
  }

  var opacity = video.ended || video_is_buffering ? 1.0 : Math.min(1.0, nonvr_menu_fade_counter / 30.0);
  opacity *= nonvr_controls.mouse_is_over || video_is_buffering ? 1.0 : 0.7;

  if (!video_is_playing) {
    opacity = 1.0; // always show controls before playing. This is important for iOS where the video won't load without an interaction!
  }

  nonvr_controls.style.opacity = opacity;

  nonvr_menu_fade_counter = Math.max(-60, nonvr_menu_fade_counter); // Allowing this to go negative means it takes a couple of frames of motion for it to become visible.

  if (!has_played_video && is_ios) {
    byId("play_button").style.display   = "inline";
    if(next_video_button) next_video_button.style.display = "inline";
    if(create_button) create_button.style.display = "inline";
    byId("pause_button").style.display  = "none";
    byId("rewind_button").style.display = "none";
    byId("buffering_button").style.display = "none";
    vrbutton3d.visible = false;
    return;
  }

  if (video_is_buffering) {
    vrbutton3d.rotateZ(-0.1);
    vrbutton_material.map = vrbutton_texture_buffering;
    byId("play_button").style.display   = "none";
    if(next_video_button) next_video_button.style.display = "none";
    if(create_button) create_button.style.display = "none";
    byId("pause_button").style.display  = "none";
    byId("rewind_button").style.display = "none";
    byId("buffering_button").style.display = "inline";
    vrbutton3d.visible = vr_session_active && (!looking_glass_config); // Only show if we are in VR and its not Looking Glass
    return;
  } else {
    vrbutton3d.rotation.set(0, 0, 0);
    vrbutton_material.map = vrbutton_texture_rewind;
  }

  if (video.ended) {
    byId("play_button").style.display   = "none";
    if(next_video_button) next_video_button.style.display = "none";
    if(create_button) create_button.style.display = "none";
    byId("pause_button").style.display  = "none";
    byId("buffering_button").style.display = "none";
    byId("rewind_button").style.display = "inline";
    vrbutton3d.visible = vr_session_active && (!looking_glass_config); // Only show if we are in VR and its not Looking Glass
    return;
  }

  if (!video || nonvr_menu_fade_counter <= 0) {
    byId("play_button").style.display   = "none";
    if(next_video_button) next_video_button.style.display = "none";
    if(create_button) create_button.style.display = "none";
    byId("pause_button").style.display  = "none";
    byId("rewind_button").style.display = "none";
    byId("buffering_button").style.display = "none";
    vrbutton3d.visible = false;
    return;
  }

  if (video_is_playing) {
    byId("play_button").style.display   = "none";
    if(next_video_button) next_video_button.style.display = "none";
    if(create_button) create_button.style.display = "none";
    byId("pause_button").style.display  = "inline";
    byId("rewind_button").style.display = "none";
    byId("buffering_button").style.display = "none";
    vrbutton3d.visible = false;
    return;
  }

  if (!video_is_playing && video.readyState >= 2) {
    byId("play_button").style.display   = "inline";
    if(next_video_button) next_video_button.style.display = "inline";
    if(create_button) create_button.style.display = "inline";
    byId("pause_button").style.display  = "none";
    byId("rewind_button").style.display = "none";
    byId("buffering_button").style.display = "none";
    vrbutton3d.visible = false;
    return;
  }
}


function render() {
  ldi_ftheta_mesh.num_patches_not_culled = 0;
  // The fragment shader uses the distance from the camera to the origin to determine how
  // aggressively to fade out fragments that might be part of a streaky triangle. We need
  // to compute that distance differently depending on whether we are in VR or not.
  var novr_camera_position = camera.position;

  var vr_camera_position = renderer.xr.getCamera().position.clone();

  // During the first few frames of VR, the camera position from head tracking is often
  // unreliable. For example, on the Vision Pro, it usually teleports ~1 meter after 1, 2
  // or sometimes 3 frames (its random). Instead of handling this with a timer, we'll just
  // detect any time the tracking jumps by an unreasonable amount (0.25m in 1 frame).
  if (prev_vr_camera_position) {
    const TRACKING_JUMP_THRESHOLD_SQ = 0.25 * 0.25;
    if (vr_camera_position.distanceToSquared(prev_vr_camera_position) > TRACKING_JUMP_THRESHOLD_SQ) {
      resetVRToCenter();
    }
  }
  prev_vr_camera_position = vr_camera_position.clone();

  vr_camera_position.sub(world_group.position); // Subtract this to account for shifts in the world_group for view resets.
  var cam_position_for_shader =
    renderer.xr.isPresenting ? vr_camera_position : novr_camera_position;
  ldi_ftheta_mesh.uniforms.uDistCamFromOrigin.value = cam_position_for_shader.length();

  updateGamepad(vr_controller0);
  updateGamepad(vr_controller1);

  updateControlsAndButtons();
  if (lock_position) { resetVRToCenter(); }

  if (handsAvailable()) {
    const indexFingerTipPosL = hand0.joints['index-finger-tip'].position;
    const indexFingerTipPosR = hand1.joints['index-finger-tip'].position;
    gesture_control.updateLeftHand(indexFingerTipPosL.x, indexFingerTipPosL.y, indexFingerTipPosL.z);
    gesture_control.updateRightHand(indexFingerTipPosR.x, indexFingerTipPosR.y, indexFingerTipPosR.z);
    gesture_control.updateTransformation(debugLog, world_group.position, ldi_ftheta_mesh.position);
    gesture_control.prevRightHandPosition.set(indexFingerTipPosR.x, indexFingerTipPosR.y, indexFingerTipPosR.z);
    gesture_control.prevLeftHandPosition.set(indexFingerTipPosL.x, indexFingerTipPosL.y, indexFingerTipPosL.z);
    left_finger_indicator.position.set(indexFingerTipPosL.x, indexFingerTipPosL.y, indexFingerTipPosL.z);
    right_finger_indicator.position.set(indexFingerTipPosR.x, indexFingerTipPosR.y, indexFingerTipPosR.z);
  } else if (vr_controller0 && vr_controller1) {
    gesture_control.updateLeftHand(vr_controller0.position.x, vr_controller0.position.y, vr_controller0.position.z);
    gesture_control.updateRightHand(vr_controller1.position.x, vr_controller1.position.y, vr_controller1.position.z);
    gesture_control.updateTransformation(debugLog, world_group.position, ldi_ftheta_mesh.position);
    gesture_control.prevRightHandPosition.set(vr_controller1.position.x, vr_controller1.position.y, vr_controller1.position.z);
    gesture_control.prevLeftHandPosition.set(vr_controller0.position.x, vr_controller0.position.y, vr_controller0.position.z);
    left_finger_indicator.position.set(vr_controller0.position.x, vr_controller0.position.y, vr_controller0.position.z);
    right_finger_indicator.position.set(vr_controller1.position.x, vr_controller1.position.y, vr_controller1.position.z);
  }


  // If in non-VR and not moving the mouse, show that it's 3D using a nice gentle rotation
  // This also enables programmatic pan, zoom, and dolly effects via updateEmbedControls
  if (cam_mode == "default" && (!is_ios || is_ios && embed_mode)) {
    if (Date.now() - mouse_last_moved_time > AUTO_CAM_MOVE_TIME) {
      let fov = anim_fov_offset + anim_fov * Math.sin(Date.now() / anim_fov_speed * Math.PI) * 0.5;
      camera.fov = fov;
      let x = anim_x_offset + anim_x * Math.sin(Date.now() / anim_x_speed * Math.PI) * 0.5;
      let y = anim_y_offset + anim_y * Math.sin(Date.now() / anim_y_speed * Math.PI) * 0.5;
      let z = anim_z_offset + anim_z * Math.sin(Date.now() / anim_z_speed * Math.PI) * 0.5;
      camera.position.set(x, y, z);
      let u = anim_u_offset + anim_u * Math.sin(Date.now() / anim_u_speed * Math.PI) * 0.5;
      let v = anim_v_offset + anim_v * Math.sin(Date.now() / anim_v_speed * Math.PI) * 0.5;
      camera.lookAt(u, v, -4.0);
      camera.updateProjectionMatrix();
    } else {
      if (!right_mouse_is_down) {
        cam_drag_u *= 0.97;
        cam_drag_v *= 0.97;
      }
      camera.position.set(-prev_mouse_u * 0.2 + cam_drag_u, prev_mouse_v * 0.2 + cam_drag_v, 0.0);
      camera.lookAt(cam_drag_u, cam_drag_v, -0.3);
    }
  }

  ldi_ftheta_mesh.matrix = gesture_control.getCurrentTransformation();
  ldi_ftheta_mesh.matrix.decompose(ldi_ftheta_mesh.position, ldi_ftheta_mesh.quaternion, ldi_ftheta_mesh.scale);

  // HACK: The video texture doesn't update as it should on Vision Pro, so here' well force it.
  if (is_safari && video != undefined) {
    texture.needsUpdate = true;
  }

  renderer.render(scene, camera);

  // Reset the view center if we started a VR session 1 frame earlier (we have to wait 1
  // frame to get correct data).
  if (delay1frame_reset) { resetVRToCenter(); }

  // console.log("num_patches_not_culled=", ldi_ftheta_mesh.num_patches_not_culled);
}

function handsAvailable() {
  return hand0 && hand1 && hand0.joints && hand1.joints && hand0.joints['index-finger-tip'] && hand1.joints['index-finger-tip'];
}

function animate() {
  renderer.setAnimationLoop( render );
}

function initVrController(vr_controller) {
  if (!vr_controller) { return; }
  if (vr_controller.lockout_timer === undefined) {
    return;
  }

  // This is used to prevent the same button press from being handled multiple times.
  vr_controller.lockout_timer = 0;

  vr_controller.addEventListener('select', handleGenericButtonPress);

  vr_controller.addEventListener('connected', function(e) {
    vr_controller.gamepad = e.data.gamepad;
    //console.log("gamepad connected", e.data.handedness, vr_controller.gamepad);
  });

  vr_controller.button_A = false;
  vr_controller.button_B = false;
}

// See https://fossies.org/linux/three.js/examples/webxr_vr_handinput_cubes.html
function initHandControllers(handleft, handright) {
  if (!handleft) { return; }
  if (!handright) { return; }

  handright.addEventListener('pinchstart', function() {
    debugLog("Right pinchstart");
    gesture_control.rightPinchStart();
    right_finger_indicator.scale.set(2.0, 2.0, 2.0);
  });
  handright.addEventListener('pinchend', function() {
    debugLog("Right pinchend");
    gesture_control.rightPinchEnd();
    right_finger_indicator.scale.set(1.0, 1.0, 1.0);
    playVideoIfReady();
  });

  handleft.addEventListener('pinchstart', function() {
    debugLog("Left pinchstart");
    gesture_control.leftPinchStart();
    left_finger_indicator.scale.set(2.0, 2.0, 2.0);
  });
  handleft.addEventListener('pinchend', function() {
    debugLog("Left pinchend");
    gesture_control.leftPinchEnd();
    left_finger_indicator.scale.set(1.0, 1.0, 1.0);
    playVideoIfReady();
  });
}

function updateGamepad(vr_controller) {
  //debugLog("updateGamepad for controller: ", vr_controller);
  if (!vr_controller) return;
  if (!vr_controller.gamepad) return;

  // Uncomment to show button state
  //console.log("buttons=" + JSON.stringify(vr_controller.gamepad.buttons.map((b) => b.value)));
  //setDebugText("buttons=" + JSON.stringify(vr_controller.gamepad.buttons.map((b) => b.value)));

  var prev_button_A = vr_controller.button_A;
  var prev_button_B = vr_controller.button_B;

  vr_controller.button_A = vr_controller.gamepad.buttons[4].value > 0;
  vr_controller.button_B = vr_controller.gamepad.buttons[5].value > 0;

  // Handle the left controller button press
  if (vr_controller.button_A && !prev_button_A) {
    debugLog("CONTROLLER: Left pinchstart");
    gesture_control.leftPinchStart();
    left_finger_indicator.scale.set(2.0, 2.0, 2.0);
  } else if (!vr_controller.button_A && prev_button_A) {
    debugLog("CONTROLLER: Left pinchend");
    gesture_control.leftPinchEnd();
    left_finger_indicator.scale.set(1.0, 1.0, 1.0);
  }

  if (vr_controller.button_B && !prev_button_B) {
    debugLog("CONTROLLER: Right pinchstart");
    gesture_control.rightPinchStart();
    right_finger_indicator.scale.set(2.0, 2.0, 2.0);
  } else if (!vr_controller.button_B && prev_button_B) {
    debugLog("CONTROLLER: Right pinchend");
    gesture_control.rightPinchEnd();
    right_finger_indicator.scale.set(1.0, 1.0, 1.0);
  }
  debugLog("CONTROLLER: " + JSON.stringify(vr_controller.gamepad.buttons.map((b) => b.value)));

  vr_controller.lockout_timer = Math.max(0, vr_controller.lockout_timer - 1);
  if (vr_controller.lockout_timer == 0) {
    // Check for A or B button release.
    if (!vr_controller.button_A && prev_button_A) {
      vr_controller.lockout_timer = 10;
      handleGenericButtonPress();
    }
    // Check for B button release.
    if (!vr_controller.button_B && prev_button_B) {
      vr_controller.lockout_timer = 10;
      handleGenericButtonPress();
    }
  }
}

// Lifecast's rendering pipeline uses a different coordinate system than THREE.js, so we
// need to convert.
function convertRotationMatrixLifecastToThreeJs(R) {
  return [
    +R[0], +R[3], -R[6],
    +R[1], +R[4], -R[7],
    -R[2], -R[5], +R[8],
  ];
}

// TODO/BUG: with new 2 layer representation, we might also have to rotate the BG layer patches

function rotateFthetaMeshBoundingSpheres(R) {
  // Update bounding spheres for patches.
  var Rm = new THREE.Matrix3().fromArray(R);
  for (var p of ldi_ftheta_mesh.ftheta_fg_geoms) {
    p.boundingSphere.center.copy(p.originalBoundingSphere.center);
    p.boundingSphere.center.applyMatrix3(Rm);
  }
  for (var p of ldi_ftheta_mesh.ftheta_bg_geoms) {
    p.boundingSphere.center.copy(p.originalBoundingSphere.center);
    p.boundingSphere.center.applyMatrix3(Rm);
  }
}

function updateFthetaRotationUniforms(video_time) {
  if (!metadata) return;
  if (is_chrome) return; // For chrome well do a different way that is more efficient.

  // "This should work as long as video_time is never ahead of the true time"
  // ^-- original comment... so wishful, so naive. Of course Firefox fucks this up.
  // Lets try adding a fudge time offset in case this doesn't happen.
  let firefox_fudge_time = 0.3; // This should be enough for 4hz updates
  var start_time = video_time - firefox_fudge_time;
  var start_frame = Math.max(0, Math.floor(start_time * metadata.fps));

  ldi_ftheta_mesh.uniforms.uFirstFrameInFthetaTable.value = start_frame;
  var i = 0;
  for (var frame_index = start_frame; frame_index < start_frame + FTHETA_UNIFORM_ROTATION_BUFFER_SIZE; ++frame_index) {
    var clamp_frame_index = Math.min(frame_index, metadata.frame_to_rotation.length - 1); // Repeat the last frame's data if we go past the end.
    var R = convertRotationMatrixLifecastToThreeJs(metadata.frame_to_rotation[clamp_frame_index]);
    ldi_ftheta_mesh.uniforms.uFrameIndexToFthetaRotation.value[i].fromArray(R);
    ++i;
  }
}


//https://www.w3.org/TR/2016/CR-orientation-event-20160818/#worked-example-2
function getRotationMatrix( alpha, beta, gamma ) {
  var degtorad = Math.PI / 180; // Degree-to-Radian conversion

  var _x = beta  ? beta  * degtorad : 0; // beta value
  var _y = gamma ? gamma * degtorad : 0; // gamma value
  var _z = alpha ? alpha * degtorad : 0; // alpha value

  var cX = Math.cos( _x );
  var cY = Math.cos( _y );
  var cZ = Math.cos( _z );
  var sX = Math.sin( _x );
  var sY = Math.sin( _y );
  var sZ = Math.sin( _z );

  var m11 = cZ * cY - sZ * sX * sY;
  var m12 = - cX * sZ;
  var m13 = cY * sZ * sX + cZ * sY;

  var m21 = cY * sZ + cZ * sX * sY;
  var m22 = cZ * cX;
  var m23 = sZ * sY - cZ * cY * sX;

  var m31 = - cX * sY;
  var m32 = sX;
  var m33 = cX * cY;

  return [
    m11,    m12,    m13,
    m21,    m22,    m23,
    m31,    m32,    m33
  ];
};

function createFingertipIndicator(color) {
  const geometry = new THREE.SphereGeometry(0.005, 8, 8);
  const material = new THREE.MeshBasicMaterial({ color: color });
  const sphere = new THREE.Mesh(geometry, material);
  return sphere;
}

function createSceneIndicator() {
  const geometry = new THREE.BoxGeometry(.05, .02, .02);
  const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
  const cube = new THREE.Mesh(geometry, material);
  return cube;
}

function createWorldIndicator() {
  const geometry = new THREE.BoxGeometry(.015, .015, .05);
  const material = new THREE.MeshBasicMaterial({ color: 0x0000ff });
  const cube = new THREE.Mesh(geometry, material);
  return cube;
}

function createMeshPositionIndicator() {
  const geometry = new THREE.BoxGeometry(.01, .06, .01);
  const material = new THREE.MeshBasicMaterial({ color: 0xff0000 });
  const cube = new THREE.Mesh(geometry, material);
  return cube;
}

export function updateEmbedControls(
    _fov, _x, _y, _z, _u, _v,
    _anim_fov, _anim_x, _anim_y, _anim_z, _anim_u, _anim_v,
    _anim_fov_speed, _anim_x_speed, _anim_y_speed, _anim_z_speed, _anim_u_speed, _anim_v_speed,
) {
  anim_fov_offset = _fov;
  anim_x_offset = _x;
  anim_y_offset = _y;
  anim_z_offset = _z;
  anim_u_offset = _u;
  anim_v_offset = _v;
  anim_fov = _anim_fov;
  anim_x = _anim_x;
  anim_y = _anim_y;
  anim_z = _anim_z;
  anim_u = _anim_u;
  anim_v = _anim_v;
  anim_fov_speed = _anim_fov_speed;
  anim_x_speed = _anim_x_speed;
  anim_y_speed = _anim_y_speed;
  anim_z_speed = _anim_z_speed;
  anim_u_speed = _anim_u_speed;
  anim_v_speed = _anim_v_speed;
  onWindowResize();
  playVideoIfReady();
}

export function init({
  _format = "ldi2", // ldi2 or ldi3
  _media_url = "",        // this should be high-res, but h264 for compatibility
  _media_url_oculus = "", // use this URL when playing in oculus browser (which might support h265)
  _media_url_mobile = "", // a fallback video file for mobile devices that can't play higher res video
  _media_url_safari = "", // a fallback video file for safari (i.e. Vision Pro) [not mobile]
  _metadata_url = "", // required for ftheta projection
  _force_recenter_frames = [], // (If supported), VR coordinate frame is reset on these frames.
  _embed_in_div = "",
  _cam_mode="default",
  _hfov = 80,
  _vscroll_bias = 0.0,
  _framerate = 30,
  _ftheta_scale = null,
  _slideshow = [], // If there is a list of media files here, we can cycle through them
  _next_video_url = "",
  _next_video_thumbnail = "",
  _lock_position = false,
  _create_button_url = "",
  _decode_12bit = true,
  _enable_pinch_world_drag = false,
  _looking_glass_config = null
}={}) {
  if (_media_url.includes("ldi3") || _media_url_oculus.includes("ldi3") || _media_url_mobile.includes("ldi3")) {
    _format = "ldi3";
    console.log("Inferred format 'ldi3' from filename");
  }

  cam_mode        = _cam_mode;
  vscroll_bias    = _vscroll_bias;
  vid_framerate   = _framerate;
  metadata_url    = _metadata_url;
  next_video_url  = _next_video_url;
  next_video_thumbnail  = _next_video_thumbnail;
  slideshow       = _slideshow;
  lock_position   = _lock_position;
  create_button_url = _create_button_url;

  looking_glass_config = _looking_glass_config;
  let enter_xr_button_title = "ENTER VR";
  let exit_xr_button_title = "EXIT VR";
  if(looking_glass_config) {
    enter_xr_button_title = "START LOOKING GLASS";
    exit_xr_button_title =  "EXIT LOOKING GLASS";
  }

  if (is_ios) {
    if (window.innerHeight > window.innerWidth) { // portrait
      _hfov = 120;
    } else {
      _hfov = 90;
    }
  }
  anim_fov_offset = _hfov;

  if (slideshow.length > 0) {
    photo_mode = true;
    _media_url = slideshow[slideshow_index];
  }

  if (_metadata_url != "") {
    // Load the metadata json file which contains camera poses for each frame.
    loadJSON(_metadata_url, function(json) {
      metadata = json;
      console.log("Loaded ", _metadata_url);
      console.log("Title: ", metadata.title);
      console.log("FPS:", metadata.fps);
      console.log("# Frames: ", metadata.frame_to_rotation.length);

      // Set uniforms from metadata
      if (photo_mode) {
        var R = convertRotationMatrixLifecastToThreeJs(metadata.frame_to_rotation[0]);
        ldi_ftheta_mesh.uniforms.uFthetaRotation.value = R;
        rotateFthetaMeshBoundingSpheres(R);
      } else {
        updateFthetaRotationUniforms(0.0);
      }
    });
  }

  if (_embed_in_div == "") {
    setBodyStyle();
    container = document.createElement("div");
    container.style.margin = "0px";
    container.style.border = "0px";
    container.style.padding = "0px";
    document.body.appendChild(container);
  } else {
    embed_mode = true;
    container = byId(_embed_in_div);
  }

  if (new URLSearchParams(window.location.search).get('embed')) {
    embed_mode = true;
  }

  if (cam_mode == "default") {
    container.style.cursor = "move";
  }

  var ext = filenameExtension(_media_url);
  if (ext == "png" || ext == "jpg") {
    photo_mode = true;
    texture = new THREE.TextureLoader().load(
      _media_url,
      function(texture) {// onLoad callback
        // Nothing to do
      },
      function(xhr) { // Progress callback
        //const percentage = (xhr.loaded / xhr.total) * 100;
      },
      function(error) { // error callback
        document.documentElement.innerHTML = "Error loading texture: "  + _media_url;
      }
    );
    // Some of this isn't necessary, but makes the texture consistent between Photo/Video.
    texture.format = THREE.RGBAFormat;
    texture.type = THREE.UnsignedByteType;
    texture.minFilter = THREE.LinearFilter; // This matters! Fixes a rendering glitch.
    texture.magFilter = THREE.LinearFilter;
    texture.generateMipmaps = false;
  } else {
    photo_mode = false;
    video = document.createElement('video');
    video.setAttribute("crossorigin", "anonymous");
    video.setAttribute("type", "video/mp4");
    video.setAttribute("playsinline", true);

    // Select the best URL based on browser
    let best_media_url = _media_url;
    if (_media_url_oculus != "" && is_oculus) {
      best_media_url = _media_url_oculus;
    }
    if (_media_url_mobile != "" && is_ios) {
      best_media_url = _media_url_mobile;
    }
    if (_media_url_safari != "" && is_safari && !is_ios) {
      best_media_url = _media_url_safari;
    }
    video.src = best_media_url;


    video.style.display = "none";
    video.preload = "auto";
    video.addEventListener("waiting", function() { video_is_buffering = true; });
    video.addEventListener("playing", function() { video_is_buffering = false; });
    video.addEventListener("error",     function() {
      document.documentElement.innerHTML = "Error loading video URL: "  + best_media_url;
    });
    document.body.appendChild(video);

    var frame_callback = function() {};
    if (is_chrome) {
      frame_callback = function(frame_index) {
        // This fixes a weird bug in Chrome. Seriously WTF. When the video has an
        // audio track, the callback's time is shifted by 1 frame.
        if (hasAudio(video)) frame_index -= 1;
        if (vid_framerate == 60) frame_index -= 1; // TODO: this is just a hack, its not fixing a bug in chrome. It is working around a quirk in upscaling 30->60FPS. Not tested with 60fps vids that have audio!
        if (frame_index < 0) frame_index = 0;

        if (metadata) {
          if (frame_index >= metadata.frame_to_rotation.length) frame_index = metadata.frame_to_rotation.length - 1;
          var R = convertRotationMatrixLifecastToThreeJs(metadata.frame_to_rotation[frame_index]);
          ldi_ftheta_mesh.uniforms.uFthetaRotation.value = R;
          rotateFthetaMeshBoundingSpheres(R);

          // Force view recenter on specified frames
          if (_force_recenter_frames.includes(frame_index)) {
            console.log("forced recenter on frame", frame_index);
            resetVRToCenter();
          }
        }
      };
    }

    // Firefox gets very slow unless we give it RGBA format (see https://threejs.org/docs/#api/en/textures/VideoTexture).
    // We want to use THREE.FloatType textures here so we can benefit from 10 bit video,
    // but it causes Firefox, Safari and Oculus browsers to be slow, so for these we need
    // to use 8 bit textures :(. TODO: revisit this.
    texture = new TimedVideoTexture(
      video,
      THREE.RGBAFormat,
      THREE.UnsignedByteType,
      frame_callback,
      vid_framerate);
  }

  makeNonVrControls();

  camera = new THREE.PerspectiveCamera(_hfov, window.innerWidth / window.innerHeight, 0.1, 110);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);

  world_group = new THREE.Group();
  scene.add(world_group);

  left_finger_indicator = createFingertipIndicator(0x00008f);
  right_finger_indicator = createFingertipIndicator(0x8f0000);
  scene.add(left_finger_indicator);
  scene.add(right_finger_indicator);

  scene.add(createSceneIndicator());

  world_group.add(createWorldIndicator());

  ldi_ftheta_mesh = new LdiFthetaMesh(_format, is_chrome, photo_mode, _metadata_url, _decode_12bit, texture, _ftheta_scale)
  world_group.add(ldi_ftheta_mesh)
  ldi_ftheta_mesh.add(createMeshPositionIndicator());

  // Make the point sprite for VR buttons.
  const vrbutton_geometry = new THREE.PlaneGeometry(0.1, 0.1);
  vrbutton_texture_rewind = new THREE.TextureLoader().load('./lifecast_res/rewind_button.png');
  vrbutton_texture_buffering = new THREE.TextureLoader().load('./lifecast_res/spinner.png');
  vrbutton_material = new THREE.MeshBasicMaterial({map: vrbutton_texture_buffering, transparent: true});
  vrbutton3d = new THREE.Mesh(vrbutton_geometry, vrbutton_material);
  vrbutton3d.visible = false;
  vrbutton3d.position.set(0, 0, -1);
  world_group.add(vrbutton3d);

  // See https://github.com/mrdoob/three.js/blob/dev/examples/webxr_vr_sandbox.html
  // for more examples of using HTMLMesh.
  if (enable_debug_text) {
    debug_text_div = document.createElement("debug_text_div");
    debug_text_div.innerHTML = "";
    debug_text_div.style.width = '400px';
    debug_text_div.style.height = '600px';
    debug_text_div.style.backgroundColor = 'rgba(128, 128, 128, 0.9)';
    debug_text_div.style.fontFamily = 'Arial';
    debug_text_div.style.fontSize = '14px';
    debug_text_div.style.padding = '10px';
    // We have to add the div to the document.body or it wont render.
    // But to keep it out of view (in 2D), move it far offscreen.
    debug_text_div.style.position = 'absolute';
    debug_text_div.style.left = '-1000px';
    debug_text_div.style.top = '-1000px';
    document.body.appendChild(debug_text_div);

    debug_text_mesh = new HTMLMesh(debug_text_div);
    debug_text_mesh.position.x = -0.5;
    debug_text_mesh.position.y = 0.25;
    debug_text_mesh.position.z = -1.0;
    debug_text_mesh.rotation.y = Math.PI / 9;
    debug_text_mesh.scale.setScalar(1.0);
    world_group.add(debug_text_mesh);
  }

  renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: "high-performance",
    depth: true
  });
  renderer.setPixelRatio(window.devicePixelRatio);

  renderer.xr.enabled = true;
  // Tradeoff quality vs runtime for VR. We are fragment shader limited so this matters.
  if (_format == "ldi2") {
    renderer.xr.setFramebufferScaleFactor(1.0);
    renderer.xr.setFoveation(0.5);
  } else if (_format == "ldi3") {
    // TODO: these don't seem to work on Vision Pro, but we want to reduce the framebuffer
    renderer.xr.setFramebufferScaleFactor(0.95);
    renderer.xr.setFoveation(0.9);
  }
  renderer.xr.setReferenceSpaceType('local');

  //renderer.outputColorSpace = THREE.sRGBEncoding; // TODO: I dont know if this is correct or even does anything. TODO: check Vision Pro
  container.appendChild(renderer.domElement);
  window.addEventListener('resize', onWindowResize);

  if (embed_mode) {
    onWindowResize();
  } else {
    renderer.setSize(window.innerWidth, window.innerHeight);
  }

  container.style.position = 'relative';
  if (is_ios) {
    // Wait a second before asking for device orientation permissions (we might already
    // have permissions and can tell if this is the case because we will have some data)
    setTimeout(function() {
      if (!got_orientation_data) {
        get_vr_button = HelpGetVR.createBanner(renderer, enter_xr_button_title, exit_xr_button_title);
        container.appendChild(get_vr_button);
      }
    }, 1000);

  } else {
    get_vr_button = HelpGetVR.createBanner(renderer, enter_xr_button_title, exit_xr_button_title);
    container.appendChild(get_vr_button);
  }

  // Non_VR mouse camera controls.
  if (cam_mode == "default" && !is_ios) {
    container.addEventListener('mousemove', function(e) {
      var rect = container.getBoundingClientRect();
      prev_mouse_u = ((e.clientX - rect.left) / rect.width) - 0.5;
      prev_mouse_v = ((e.clientY - rect.top) / rect.height) - 0.5;

      mouse_last_moved_time = Date.now();
    });
  }

  if (cam_mode == "orbit" && !is_ios) {
    camera.position.set(0, 0, 2.0);
    orbit_controls = new OrbitControls(camera, renderer.domElement);
    orbit_controls.panSpeed = 20.0;
    orbit_controls.target.set(0, 0, -2.0); // NOTE the 2 here is half the octree size of 4 meters^3
    orbit_controls.enableDamping = true;
    orbit_controls.enableZoom = true; // TODO: this is cool but needs some tweaking
    orbit_controls.dampingFactor = 0.3;
    orbit_controls.saveState();
  }

  if (is_ios && !embed_mode) {
    document.body.style["touch-action"] = "none";
    document.addEventListener('touchmove', function(e) {
      e.preventDefault();
      var touch = e.touches[0];

      let u = (touch.pageX / window.innerWidth - 0.5) * 2.0;
      let v = (touch.pageY / window.innerHeight - 0.5) * 2.0;
      mobile_drag_u = mobile_drag_u * 0.8 + u * 0.2;
      mobile_drag_v = mobile_drag_v * 0.8 + v * 0.2;
    }, false);
  }

  // VR trigger button to play/pause (or pinch gesture)
  vr_controller0 = renderer.xr.getController(0);
  vr_controller1 = renderer.xr.getController(1);
  initVrController(vr_controller0);
  initVrController(vr_controller1);

  hand0 = renderer.xr.getHand(0);
  hand1 = renderer.xr.getHand(1);
  initHandControllers(hand0, hand1);

  // Disable right click on play/pause button
  const images = document.getElementsByTagName('img');
  for (let i = 0; i < images.length; i++) {
    images[i].addEventListener('contextmenu', event => event.preventDefault());
  }
  container.addEventListener('contextmenu', event => event.preventDefault());
  if (!(photo_mode || embed_mode)) {
    nonvr_controls.addEventListener('contextmenu', event => event.preventDefault());
    trackMouseStatus(nonvr_controls);

    // Setup button handles for non-VR interface
    byId("play_button").addEventListener('click', handleNonVrPlayButton);
    byId("rewind_button").addEventListener('click', handleNonVrPlayButton);
    byId("pause_button").addEventListener('click', handleNonVrPauseButton);
    byId("buffering_button").addEventListener('click', function() {
      video_is_buffering = false;
      handleNonVrPauseButton();
    });

    document.addEventListener('mousemove', e => {
      if (!mouse_is_down) nonvr_menu_fade_counter = Math.min(60, nonvr_menu_fade_counter + 5);
    });
  }

  document.addEventListener('mousedown', e => {
    mouse_is_down = true;
    if (e.button == 2) right_mouse_is_down = true;
  });
  document.addEventListener('mouseup', e => {
    mouse_is_down = false;
    if (e.button == 2) right_mouse_is_down = false;
  });
  document.addEventListener('mousemove', e => {
    if(right_mouse_is_down) {
      cam_drag_u -= event.movementX / 2000.0;
      cam_drag_v += event.movementY / 2000.0;
    }
  });

  document.addEventListener('keydown', function(event) {
    const key = event.key;
    if (key == " ") {
      toggleVideoPlayPause();
    }
    if (key == "s" && slideshow.length > 0) {
      slideshow_index = (slideshow_index + 1) % slideshow.length;
      ldi_ftheta_mesh.uniforms.uTexture.value = new THREE.TextureLoader().load(slideshow[slideshow_index]);

      ldi_ftheta_mesh.uniforms.uTexture.value.format = THREE.RGBAFormat;
      ldi_ftheta_mesh.uniforms.uTexture.value.type = THREE.UnsignedByteType;
      ldi_ftheta_mesh.uniforms.uTexture.value.minFilter = THREE.LinearFilter; // This matters! Fixes a rendering glitch.
      ldi_ftheta_mesh.uniforms.uTexture.value.magFilter = THREE.LinearFilter;
      ldi_ftheta_mesh.uniforms.uTexture.value.generateMipmaps = false;

      if (_format == "ldi2") {
        ldi_ftheta_mesh.ldi2_fg_material.needsUpdate = true;
        ldi_ftheta_mesh.ldi2_bg_material.needsUpdate = true;
      }
      if (_format == "ldi3") {
        ldi_ftheta_mesh.ldi3_layer0_material.needsUpdate = true;
        ldi_ftheta_mesh.ldi3_layer1_material.needsUpdate = true;
        ldi_ftheta_mesh.ldi3_layer2_material.needsUpdate = true;
      }
    }

    if (_format == "ldi2"){
      if (key == "x") { for (var m of ldi_ftheta_mesh.ftheta_fg_meshes) { m.visible = !m.visible; } }
    }
    if (_format == "ldi3") {
      if (key == "z") { for (var m of ldi_ftheta_mesh.ftheta_bg_meshes) { m.visible = !m.visible; } }
      if (key == "x") { for (var m of ldi_ftheta_mesh.ftheta_mid_meshes) { m.visible = !m.visible; } }
      if (key == "c") { for (var m of ldi_ftheta_mesh.ftheta_fg_meshes) { m.visible = !m.visible; } }
    }

  });

  if (cam_mode == "vscroll") {
    window.addEventListener("scroll", verticalScrollCameraHandler);
  }

  if (is_ios) { // TODO: or android?

    window.addEventListener('orientationchange', function() {
      // reset the "home" angle
      got_orientation_data = false;
    });

    addEventListener('deviceorientation', function(e) {

      // if we got device orientation data, it means we don't need to request it
      if (get_vr_button) {
        get_vr_button.style.display = "none";
      }

      let R = getRotationMatrix(e.alpha, e.beta, e.gamma);
      //console.log("\n" +
      //  R[0].toFixed(2) + " " + R[1].toFixed(2) + " " + R[2].toFixed(2) + "\n" +
      //  R[3].toFixed(2) + " " + R[4].toFixed(2) + " " + R[5].toFixed(2) + "\n" +
      //  R[6].toFixed(2) + " " + R[7].toFixed(2) + " " + R[8].toFixed(2)
      //);

      // Note, below we use the values R[2] and R[8] to determine the motion of the camera
      // in response to the mobile device orientation. Why elements 2 and 8? Answer:
      // trial and error. Intuition: dot products between basis vector and rows or columns
      // of the rotation matrix.

      if (!got_orientation_data) {
        got_orientation_data = true;

        init_orientation_a = R[2];
        init_orientation_b = R[8];
      }

      // Gradually decay the "initial" angle toward whatever the current angle is.
      // This gives it a chance to eventually recover if it gets crooked.
      init_orientation_a = init_orientation_a * 0.995 + R[2] * 0.005;
      init_orientation_b = init_orientation_b * 0.995 + R[8] * 0.005;

      let diff_orientation_a = init_orientation_a - R[2];
      let diff_orientation_b = init_orientation_b - R[8];

      let p = -diff_orientation_b + mobile_drag_v;
      let q = diff_orientation_a + mobile_drag_u;

      if (window.innerHeight > window.innerWidth) { // portrait
        _hfov = 120;
      } else {
        _hfov = 90;
      }
      camera.fov = _hfov;
      camera.updateProjectionMatrix();

      camera.position.set(-q * 1.0, p * 1.0, 0.0);
      camera.lookAt(0, 0, -1);
    });


  }

  container.addEventListener('mousedown', function() {
    maybe_click = true;
    setTimeout(function() { maybe_click = false; }, 200);
  });
  container.addEventListener('mouseup', function() {
    if (maybe_click) {
      toggleVideoPlayPause();
    }
  });

  // If the Oculus button is held to reset the view center, we need to move the
  // world_group back to 0.
  var reset_event_handler = function(event) {
    world_group.position.set(0, 0, 0);
  };

  let xr_ref_space;
  renderer.xr.addEventListener('sessionstart', function(event) {
    // Wait 1 second for looking glass's popup window to open before trying to install an
    // event handler in it, so we can play/pause while that screen is selected.
    if (_looking_glass_config) {
      setTimeout(() => {
        //  Looking glass creates a separate window. We need to add event handlers to that window
        // to be able to play/pause.
        if(_looking_glass_config.popup) {
          _looking_glass_config.popup.addEventListener('keydown', function(e) {
            if (e.key == " ") toggleVideoPlayPause();
          });
        }
      }, 1000);
    }

    // Start the video playing automatically if the user enters VR.
    if (!photo_mode) {
      debugLog("playVideoIfReady in sessionstart");

      playVideoIfReady();
    }

    // When we enter VR, toggle on the VR-only 3d buttons.
    vr_session_active = true;

    // Create an event handler for the Oculus reset to center button. We have to wait to
    // construct the handler here to get a non-null XReferenceSpace.
    xr_ref_space = renderer.xr.getReferenceSpace();
    if(xr_ref_space.addEventListener) xr_ref_space.addEventListener("reset", reset_event_handler);

    // Move the world_group back to the origin 1 frame from now (doing it now wont work).
    delay1frame_reset = true; // Calls resetVRToCenter(); 1 frame from now.
  });

  renderer.xr.addEventListener('sessionend', function(event) {
    // Destroy the handler we created on sessionstart. This way we don't get multiple
    // handlers if the user goes back and forth between VR and non.
    if(xr_ref_space.removeEventListener) xr_ref_space.removeEventListener("reset", reset_event_handler);

    // When we enter VR, toggle on the VR-only 3d buttons.
    vr_session_active = false;
    vrbutton3d.visible = false;

    // When we exit VR mode on Oculus Browser it messes up the camera, so lets reset it.
    world_group.position.set(0, 0, 0);
  });

  // For f-theta projection, whenever a timeupdate event occurs, we will update the
  // uniforms storing a lookup table of frame index to rotation. The timeupdate events
  // might only happen at 4Hz, but it's OK because we bring in a 2 second block of
  // uniforms, then let the shader pick the right one.
  if (!is_chrome && video) {
    video.addEventListener('timeupdate', function(event) {
      updateFthetaRotationUniforms(video.currentTime);
    });
  }

  animate();
} // end init()
