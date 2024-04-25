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

import {LdiFthetaMesh} from "./LdiFthetaMesh11.js";
import * as THREE from './three152.module.min.js';
import {OrbitControls} from "./OrbitControls.js";
import {HTMLMesh} from './HTMLMesh.js';
import {HelpGetVR} from './HelpGetVR11.js';
import {GestureControlModule} from './GestureControlModule.js';
import {XRControllerModelFactory} from './XRControllerModelFactory.js';
import {XRHandModelFactory} from './XRHandModelFactory.js';

const gesture_control = new GestureControlModule();

let enable_debug_text = false; // Turn this on if you want to use debugLog() or setDebugText().
let debug_text_mesh, debug_text_div;
let debug_log = "";
let debug_msg_count = 0;

let container, camera, scene, renderer;
let vr_controller0, vr_controller1; // used for getting controller state, including buttons
let controller_grip0, controller_grip1; // used for rendering controller models
let hand0, hand1, hand_model0, hand_model1; // for XR hand-tracking

let ldi_ftheta_mesh;
let world_group; // A THREE.Group that stores all of the meshes (foreground and background), so they can be transformed together by modifying the group.
let interface_group; // A separate Group for 3D interface components
let prev_vr_camera_position;

let video;
let texture;
let nonvr_menu_fade_counter = 1;
let mouse_is_down = false;

let toggle_layer0 = true;
let toggle_layer1 = true;
let toggle_layer2 = true;

let prev_mouse_u = 0.5;
let prev_mouse_v = 0.5;
let cam_drag_u = 0.0;
let cam_drag_v = 0.0;
let right_mouse_is_down = false;

let nonvr_controls;
let is_buffering = true;
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

let lock_position = false;
let orbit_controls;
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
let anim_x_offset = 0;
let anim_y_offset = 0;
let anim_z_offset = 0;
let anim_u_offset = 0;
let anim_v_offset = 0;
let anim_x = 0.15;
let anim_y = 0.10;
let anim_z = 0.05;
let anim_u = 0.15;
let anim_v = 0.10;
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

  container.appendChild(nonvr_controls);
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
    // TODO: Decide on a way to reset view that doesn't interfere with gesture controls
    //resetVRToCenter();
  } else {
    toggleVideoPlayPause();
  }
}

function resetVRToCenter() {
  // Reset the gesture_control
  gesture_control.reset();
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
  if (!video) return;

  video.play();
  has_played_video = true;
}

function toggleVideoPlayPause() {
  if (photo_mode) return;

  nonvr_menu_fade_counter = 60;

  const video_is_playing = !!(video.currentTime > 0 && !video.paused && !video.ended && video.readyState > 2);
  if (video_is_playing || is_buffering) {
    is_buffering = false;
    video.pause();
  } else {
    playVideoIfReady();
  }
}

function handleNonVrPlayButton() {
  playVideoIfReady();
}

function handleNonVrPauseButton() {
  video.pause();
}


function onWindowResize() {
  // In embed mode, use the width and height of the container div.
  let width = embed_mode ? container.clientWidth : window.innerWidth;
  let height = embed_mode ? container.clientHeight : window.innerHeight;
  camera.aspect = width / height;
  renderer.setSize(width, height);
  camera.updateProjectionMatrix();
}

function updateControlsAndButtons() {
  if (!nonvr_controls) return;

  const video_is_playing = video && !!(video.currentTime > 0 && !video.paused && !video.ended && video.readyState >= 2);
  if (video) {
    // Fade out but only if the mouse is not over a button
    if (!nonvr_controls.mouse_is_over) {
      --nonvr_menu_fade_counter;
    }
    nonvr_menu_fade_counter = Math.max(-60, nonvr_menu_fade_counter); // Allowing this to go negative means it takes a couple of frames of motion for it to become visible.


    var opacity = video.ended || is_buffering ? 1.0 : Math.min(1.0, nonvr_menu_fade_counter / 30.0);
    opacity *= nonvr_controls.mouse_is_over || is_buffering ? 1.0 : 0.7;

    if (!video_is_playing) {
      opacity = 1.0; // always show controls before playing. This is important for iOS where the video won't load without an interaction!
    }
  }

  nonvr_controls.style.opacity = opacity;

  if (video && !has_played_video && is_ios) {
    byId("play_button").style.display   = "inline";
    byId("pause_button").style.display  = "none";
    byId("rewind_button").style.display = "none";
    byId("buffering_button").style.display = "none";
    vrbutton3d.visible = false;
    return;
  }

  if (is_buffering) {
    vrbutton3d.rotateZ(-0.1);
    vrbutton_material.map = vrbutton_texture_buffering;
    byId("play_button").style.display   = "none";
    byId("pause_button").style.display  = "none";
    byId("rewind_button").style.display = "none";
    byId("buffering_button").style.display = "inline";
    vrbutton3d.visible = vr_session_active && (!looking_glass_config); // Only show if we are in VR and its not Looking Glass
    return;
  } else {
    vrbutton3d.rotation.set(0, 0, 0);
    vrbutton_material.map = vrbutton_texture_rewind;
  }

  if (video && video.ended) {
    byId("play_button").style.display   = "none";
    byId("pause_button").style.display  = "none";
    byId("buffering_button").style.display = "none";
    byId("rewind_button").style.display = "inline";
    vrbutton3d.visible = vr_session_active && (!looking_glass_config); // Only show if we are in VR and its not Looking Glass
    return;
  }

  if (!video || nonvr_menu_fade_counter <= 0) {
    byId("play_button").style.display   = "none";
    byId("pause_button").style.display  = "none";
    byId("rewind_button").style.display = "none";
    byId("buffering_button").style.display = "none";
    vrbutton3d.visible = false;
    return;
  }

  if (video_is_playing) {
    byId("play_button").style.display   = "none";
    byId("pause_button").style.display  = "inline";
    byId("rewind_button").style.display = "none";
    byId("buffering_button").style.display = "none";
    vrbutton3d.visible = false;
    return;
  }

  if (!video_is_playing && video.readyState >= 2) {
    byId("play_button").style.display   = "inline";
    byId("pause_button").style.display  = "none";
    byId("rewind_button").style.display = "none";
    byId("buffering_button").style.display = "none";
    vrbutton3d.visible = false;
    return;
  }
}

function setVisibilityForLayerMeshes(l, v) {
  for (var m of ldi_ftheta_mesh.layer_to_meshes[l]) { m.visible = v; }
}

function render() {
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


  updateControlsAndButtons();
  if (lock_position) { resetVRToCenter(); }

  if (handsAvailable()) {
    const indexFingerTipPosL = hand0.joints['index-finger-tip'].position;
    const indexFingerTipPosR = hand1.joints['index-finger-tip'].position;
    gesture_control.updateLeftHand(indexFingerTipPosL);
    gesture_control.updateRightHand(indexFingerTipPosR);
    gesture_control.updateTransformation(world_group.position, ldi_ftheta_mesh.position);
  } else if (vr_controller0 && vr_controller1) {
    updateGamepad(vr_controller0, "left");
    updateGamepad(vr_controller1, "right");
    gesture_control.updateLeftHand(vr_controller0.position);
    gesture_control.updateRightHand(vr_controller1.position);
    gesture_control.updateTransformation(world_group.position, ldi_ftheta_mesh.position);
  }


  // If in non-VR and not moving the mouse, show that it's 3D using a nice gentle rotation
  if (cam_mode == "default" && !got_orientation_data) {
    if (Date.now() - mouse_last_moved_time > AUTO_CAM_MOVE_TIME) {
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

  // Render each layer in order, clearing the depth buffer between. This is important
  // to get alpha blending right.
  renderer.clearColor();
  renderer.clearDepth();
  world_group.visible = true;
  interface_group.visible = false;
  if (toggle_layer0) {
    setVisibilityForLayerMeshes(0, true);
    setVisibilityForLayerMeshes(1, false);
    setVisibilityForLayerMeshes(2, false);
    renderer.render(scene, camera);
  }
  if (toggle_layer1) {
    setVisibilityForLayerMeshes(0, false);
    setVisibilityForLayerMeshes(1, true);
    setVisibilityForLayerMeshes(2, false);
    //renderer.clearDepth(); // TODO: not sure if we still need this with normalized depth
    renderer.render(scene, camera);
  }
  if (toggle_layer2) {
    setVisibilityForLayerMeshes(0, false);
    setVisibilityForLayerMeshes(1, false);
    setVisibilityForLayerMeshes(2, true);
    //renderer.clearDepth(); // TODO: not sure if we still need this with normalized depth
    renderer.render(scene, camera);
  }

  // In a final pass, render the interface.
  world_group.visible = false;
  interface_group.visible = true;
  renderer.render(scene, camera);  // clears depth automatically (unwanted but unavoidable without warnings from THREE.js and hack workarounds).

  // Reset the view center if we started a VR session 1 frame earlier (we have to wait 1
  // frame to get correct data).
  if (delay1frame_reset) { resetVRToCenter(); }
}

function handsAvailable() {
  return hand0 && hand1 && hand0.joints && hand1.joints && hand0.joints['index-finger-tip'] && hand1.joints['index-finger-tip'];
}

function animate() {
  renderer.setAnimationLoop( render );
}

function initVrController(vr_controller) {
  debugLog("initVrController for controller: " + vr_controller);
  if (!vr_controller) {
    debugLog("initVrController: no controller found");
    return;
  }

  // This is used to prevent the same button press from being handled multiple times.
  vr_controller.lockout_timer = 0;

  vr_controller.addEventListener('select', handleGenericButtonPress);

  vr_controller.addEventListener('connected', function(e) {
    vr_controller.gamepad = e.data.gamepad;
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
  });
  handright.addEventListener('pinchend', function() {
    debugLog("Right pinchend");
    gesture_control.rightPinchEnd();
    playVideoIfReady();
  });

  handleft.addEventListener('pinchstart', function() {
    debugLog("Left pinchstart");
    gesture_control.leftPinchStart();
  });
  handleft.addEventListener('pinchend', function() {
    debugLog("Left pinchend");
    gesture_control.leftPinchEnd();
    playVideoIfReady();
  });
}

function updateGamepad(vr_controller, hand) {
  if (!vr_controller) {
    return;
  }
  if (!vr_controller.gamepad) {
    return;
  }

  // Uncomment to show button state
  //console.log("buttons=" + JSON.stringify(vr_controller.gamepad.buttons.map((b) => b.value)));
  //setDebugText("buttons=" + JSON.stringify(vr_controller.gamepad.buttons.map((b) => b.value)));

  var prev_button_A = vr_controller.button_A;
  var prev_button_B = vr_controller.button_B;

  // Quest 3 Controller Buttons
  // Left Hand
  // Main Trigger
  // vr_controller.gamepad.buttons[0].value > 0;
  // Secondary Trigger
  // vr_controller.gamepad.buttons[1].value > 0;
  // X Button
  // vr_controller.gamepad.buttons[4].value > 0;
  // Y Button
  // vr_controller.gamepad.buttons[5].value > 0;

  // Right Hand
  // Main Trigger
  // vr_controller.gamepad.buttons[1].value > 0;
  // Secondary Trigger
  // vr_controller.gamepad.buttons[1].value > 0;
  // A Button
  // vr_controller.gamepad.buttons[4].value > 0;
  // B Button
  // vr_controller.gamepad.buttons[5].value > 0;

  vr_controller.button_A = vr_controller.gamepad.buttons[0].value > 0;
  vr_controller.button_B = vr_controller.gamepad.buttons[1].value > 0;

  // Handle the left controller button press
  if (vr_controller.button_A && !prev_button_A) {
    debugLog("Controller button A start hand " + hand);
    if (hand == "left") {
      gesture_control.leftPinchStart();
    } else {
      gesture_control.rightPinchStart();
    }
  } else if (!vr_controller.button_A && prev_button_A) {
    debugLog("Controller button A end hand " + hand);
    if (hand == "left") {
      gesture_control.leftPinchEnd();
    } else {
      gesture_control.rightPinchEnd();
    }
  }

  if (vr_controller.button_B && !prev_button_B) {
    debugLog("Controller button B start hand " + hand);
    if (hand == "left") {
      gesture_control.leftPinchStart();
    } else {
      gesture_control.rightPinchStart();
    }
  } else if (!vr_controller.button_B && prev_button_B) {
    debugLog("Controller button B end hand " + hand);
    if (hand == "left") {
      gesture_control.leftPinchEnd();
    } else {
      gesture_control.rightPinchEnd();
    }
  }

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

function applyHandMaterialRecursive(object, material) {
  object.traverse((child) => {
    if (child.isMesh) {
      child.material = material;
      child.renderOrder = 10; // HACK: draw hands last for transparency without writing to depth
    }
  });
}

function setupHandAndControllerModels() {
  const controllerModelFactory = new XRControllerModelFactory();
  const handModelFactory = new XRHandModelFactory();

  vr_controller0 = renderer.xr.getController(0);
  vr_controller1 = renderer.xr.getController(1);
  controller_grip0 = renderer.xr.getControllerGrip(0);
  controller_grip1 = renderer.xr.getControllerGrip(1);
  hand0 = renderer.xr.getHand(0);
  hand1 = renderer.xr.getHand(1);

  debugLog("calling initVrController...");
  initVrController(vr_controller0);
  initVrController(vr_controller1);
  debugLog("calling initHandControllers...");
  initHandControllers(hand0, hand1);

  const hand_material = new THREE.MeshPhongMaterial({
    color: 0x8cc6ff,
    transparent: true,
    opacity: 0.33,
    depthTest: true,
    depthWrite: false,
    side: THREE.DoubleSide
  });
  // Wait until hand models load, then overwrite their material
  hand_model0 = handModelFactory.createHandModel(hand0, "mesh", function() {
    applyHandMaterialRecursive(hand_model0, hand_material);
  });
  hand_model1 = handModelFactory.createHandModel(hand1, "mesh", function() {
    applyHandMaterialRecursive(hand_model1, hand_material);
  });

  controller_grip0.add(controllerModelFactory.createControllerModel(controller_grip0));
  controller_grip1.add(controllerModelFactory.createControllerModel(controller_grip1));
  hand1.add(hand_model0);
  hand0.add(hand_model1);
  interface_group.add(vr_controller0); // TODO: is this needed?
  interface_group.add(vr_controller1);
  interface_group.add(controller_grip0);
  interface_group.add(controller_grip1);
  interface_group.add(hand0);
  interface_group.add(hand1);

  // We need to add some light for the hand material to be anything other than black
  scene.add(new THREE.HemisphereLight( 0xbcbcbc, 0xa5a5a5, 3));
  scene.add(new THREE.DirectionalLight( 0xffffff, 3));
}


export function init({
  _format = "ldi3", // ldi3 only for now.. maybe add VR180 and other formats later?
  _media_url = "",        // this should be high-res, but h264 for compatibility
  _media_url_oculus = "", // use this URL when playing in oculus browser (which might support h265)
  _media_url_mobile = "", // a fallback video file for mobile devices that can't play higher res video
  _media_url_safari = "", // a fallback video file for safari (i.e. Vision Pro) [not mobile]
  _embed_in_div = "",
  _cam_mode="default",
  _vfov = 80,
  _ftheta_scale = null,
  _lock_position = false,
  _decode_12bit = true,
  _looking_glass_config = null,
  _autoplay_muted = false, // If this is a video, try to start playing immediately (muting is required)
  _loop = false,
  _transparent_bg = false, //  If you don't need transparency, it is faster to set this to false
}={}) {
  if (_media_url.includes("ldi3") || _media_url_oculus.includes("ldi3") || _media_url_mobile.includes("ldi3")) {
    _format = "ldi3";
    console.log("Inferred format 'ldi3' from filename");
  }

  cam_mode        = _cam_mode;
  lock_position   = _lock_position;

  looking_glass_config = _looking_glass_config;
  let enter_xr_button_title = "ENTER VR";
  let exit_xr_button_title = "EXIT VR";
  if(looking_glass_config) {
    enter_xr_button_title = "START LOOKING GLASS";
    exit_xr_button_title =  "EXIT LOOKING GLASS";
  }

  if (_embed_in_div == "") {
    setBodyStyle();
    container = document.body;
    container.style.margin = "0px";
    container.style.border = "0px";
    container.style.padding = "0px";
  } else {
    embed_mode = true;
    container = byId(_embed_in_div);
  }

  // Remove any existing children of the container (eg. loading spinner)
  while (container.firstChild) {
    container.removeChild(container.firstChild);
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
        is_buffering = false;
      },
      function(xhr) { // Progress callback
        //const percentage = (xhr.loaded / xhr.total) * 100;
      },
      function(error) { // error callback
        container.innerHTML = "Error loading texture: "  + _media_url;
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
    video.loop = _loop;

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
    video.addEventListener("waiting", function() { is_buffering = true; });
    video.addEventListener("playing", function() { is_buffering = false; });
    video.addEventListener("canplay", function() { is_buffering = false; });
    video.addEventListener("error",     function() {
      container.innerHTML = "Error loading video URL: "  + best_media_url;
    });

    if(_autoplay_muted) {
      video.muted = true;
      video.play();
    }

    document.body.appendChild(video);

    texture = new THREE.VideoTexture(video)
    texture.minFilter = THREE.LinearFilter;
    texture.magFilter = THREE.LinearFilter;
    texture.format = THREE.RGBAFormat;
    texture.type = THREE.UnsignedByteType;
  }

  makeNonVrControls();

  camera = new THREE.PerspectiveCamera(_vfov, window.innerWidth / window.innerHeight, 0.1, 110);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x000000);

  world_group = new THREE.Group();
  interface_group = new THREE.Group();
  scene.add(world_group);
  scene.add(interface_group);

  ldi_ftheta_mesh = new LdiFthetaMesh(_format, _decode_12bit, texture, _ftheta_scale)
  world_group.add(ldi_ftheta_mesh)

  // Make the point sprite for VR buttons.
  const vrbutton_geometry = new THREE.PlaneGeometry(0.1, 0.1);
  vrbutton_texture_rewind = new THREE.TextureLoader().load('./lifecast_res/rewind_button.png');
  vrbutton_texture_buffering = new THREE.TextureLoader().load('./lifecast_res/spinner.png');
  vrbutton_material = new THREE.MeshBasicMaterial({map: vrbutton_texture_buffering, transparent: true});
  vrbutton3d = new THREE.Mesh(vrbutton_geometry, vrbutton_material);
  vrbutton3d.visible = false;
  vrbutton3d.position.set(0, 0, -1);
  interface_group.add(vrbutton3d);

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
    debug_text_div.style.color = 'black';

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
    interface_group.add(debug_text_mesh);
  }

  renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: "high-performance",
    preserveDrawingBuffer: true,
    alpha: _transparent_bg
  });
  renderer.autoClear = false;
  renderer.autoClearColor = false;
  renderer.autoClearDepth = true;
  renderer.autoClearStencil = false;
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.xr.enabled = true;
  if (_transparent_bg) {
    renderer.setClearColor(0xffffff, 0.0);
    scene.background = null;
  }
  if (_format == "ldi3") {
    // TODO: these don't seem to work on Vision Pro, but we want to reduce the framebuffer
    renderer.xr.setFramebufferScaleFactor(0.95);
    renderer.xr.setFoveation(0.9);
  } else {
    console.log("ERROR: unknown _format:", _format);
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
        get_vr_button = HelpGetVR.createBanner(renderer, enter_xr_button_title, exit_xr_button_title, debugLog);
        container.appendChild(get_vr_button);
      }
    }, 1000);

  } else {
    get_vr_button = HelpGetVR.createBanner(renderer, enter_xr_button_title, exit_xr_button_title, debugLog);
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

  // Setup hand/controller models and initialize stuff related to user input from controllers or hands
  setupHandAndControllerModels();

  // Disable right click on play/pause button
  const images = document.getElementsByTagName('img');
  for (let i = 0; i < images.length; i++) {
    images[i].addEventListener('contextmenu', event => event.preventDefault());
  }
  container.addEventListener('contextmenu', event => event.preventDefault());
  if (!photo_mode) {
    nonvr_controls.addEventListener('contextmenu', event => event.preventDefault());
    trackMouseStatus(nonvr_controls);

    // Setup button handles for non-VR interface
    byId("play_button").addEventListener('click', handleNonVrPlayButton);
    byId("rewind_button").addEventListener('click', handleNonVrPlayButton);
    byId("pause_button").addEventListener('click', handleNonVrPauseButton);
    byId("buffering_button").addEventListener('click', function() {
      is_buffering = false;
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
    if (_format == "ldi3") {
      if (key == "z") { toggle_layer0 = !toggle_layer0; }
      if (key == "x") { toggle_layer1 = !toggle_layer1; }
      if (key == "c") { toggle_layer2 = !toggle_layer2; }
    }

  });

  renderer.domElement.addEventListener('wheel', function(event) {
    event.preventDefault();
    const MIN_FOV = 30;
    const MAX_FOV = 120;
    // Note: event.deltaY is typically +100 or -100 per wheel click
    const FOV_CHANGE_SPEED = 0.01;
    camera.fov += event.deltaY * FOV_CHANGE_SPEED;
    camera.fov = Math.max(MIN_FOV, Math.min(camera.fov, MAX_FOV));
    camera.updateProjectionMatrix();
  }, false);

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

      camera.fov = _vfov;
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
    gesture_control.reset();
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
      playVideoIfReady();

      // Unmute for immersive experience!
      if (_autoplay_muted) {
        video.muted = false;
      }
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
    gesture_control.reset();
  });

  // Remove any redundant loading indicator (from LifecastVideoPlayerPreloader)
  let preload_indicators = container.getElementsByClassName("lifecast_preload_indicator");
  for (let i = 0; i < preload_indicators.length; i++) {
    preload_indicators[i].style.display = "none";
  }

  animate();
} // end init()
