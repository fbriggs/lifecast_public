import * as THREE from './three149.module.min.js';

/*  WebXR Gesture Control
  - Clicking and dragging with the left hand should result in translation so that the object follows the hand
  - Clicking and dragging with the right hand, same thing
  - Clicking with both left and right hands and dragging them together to one half their initial distance should set scale to 0.5
  - Clicking with both left and right hands and pulling them apart to twice their initial distance should set scale to 2.0
  - Clicking with both hands, holding the left hand still, and moving the right hand in the x-z plane while keeping it equidistant from the left hand should result in rotation about the y axis
  - Clicking with both hands, pulling them apart while translating should result in both translation and scale
*/
class GestureControlModule {
  constructor() {
    this.leftHandPosition = new THREE.Vector3();
    this.rightHandPosition = new THREE.Vector3();
    this.leftPinchStartPosition = null;
    this.rightPinchStartPosition = null;

    this.prevScale = 1.0;
    this.currentScale = 1.0;
    this.pinchDistanceInitial = 1.0;

    this.prevRotY = 0;
    this.currentRotY = 0;
    this.pinchRotYInitial = 0;

    this.currentTranslation = new THREE.Vector3();
    this.transformationMatrix = new THREE.Matrix4();
  }

  updateLeftHand(x, y, z) {
    this.leftHandPosition.set(x, y, z);
    this.updateTransformation();
  }

  updateRightHand(x, y, z) {
    this.rightHandPosition.set(x, y, z);
    this.updateTransformation();
  }

  leftPinchStart() {
    this.leftPinchStartPosition = this.leftHandPosition.clone().sub(this.currentTranslation);
  }

  leftPinchEnd() {
    this.leftPinchStartPosition = null;
  }

  rightPinchStart() {
    this.pinchDistanceInitial = this.rightHandPosition.distanceTo(this.leftHandPosition);
    this.rightPinchStartPosition = this.rightHandPosition.clone();
    this.pinchRotYInitial = this.calcPinchRotationY();
  }

  rightPinchEnd() {
    this.rightPinchStartPosition = null;
    this.prevScale = this.currentScale;
    this.prevRotY = this.currentRotY;
  }

  calcPinchRotationY() {
    let dz = this.leftHandPosition.z - this.rightHandPosition.z;
    let dx = this.leftHandPosition.x - this.rightHandPosition.x;
    if (dx === 0 && dz === 0) {
      return 0;
    }
    return -Math.atan2(dz, dx);
  }

  getCurrentTransformation() {
    return this.transformationMatrix;
  }

  updateTransformation() {
    if (this.leftPinchStartPosition) {
      // Translation
      this.currentTranslation = this.leftHandPosition.clone().sub(this.leftPinchStartPosition);
    }
    if (this.leftPinchStartPosition && this.rightPinchStartPosition) {
      // Scale
      let pinchDistanceCurrent = this.rightHandPosition.distanceTo(this.leftHandPosition);
      this.currentScale = this.prevScale * pinchDistanceCurrent / this.pinchDistanceInitial;
      // Rotation
      let currentRotY = this.calcPinchRotationY();
      this.currentRotY = this.prevRotY + currentRotY - this.pinchRotYInitial;
    }

    this.transformationMatrix.identity();
    // Rotate
    this.transformationMatrix.multiply(new THREE.Matrix4().makeRotationY(this.currentRotY));
    // Scale
    this.transformationMatrix.scale(new THREE.Vector3(this.currentScale, this.currentScale, this.currentScale));
    // Translate
    this.transformationMatrix.setPosition(this.currentTranslation);
  }
}

export {GestureControlModule};
