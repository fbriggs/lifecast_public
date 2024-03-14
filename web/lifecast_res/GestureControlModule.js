import * as THREE from './three149.module.min.js';

class GestureControlModule {
  constructor() {
    this.leftHandPosition = new THREE.Vector3();
    this.rightHandPosition = new THREE.Vector3();
    this.prevLeftHandPosition = new THREE.Vector3();
    this.prevRightHandPosition = new THREE.Vector3();
    this.isLeftPinching = false;
    this.isRightPinching = false;

    this.currentScale = 1.0;

    this.currentRotY = 0;

    this.currentTranslation = new THREE.Vector3();

    this.transformationMatrix = new THREE.Matrix4();
    this.transformationMatrix.identity();
  }

  updateLeftHand(x, y, z) {
    this.leftHandPosition.set(x, y, z);
    this.updateTransformation();
    this.prevLeftHandPosition.set(x, y, z);
  }

  updateRightHand(x, y, z) {
    this.rightHandPosition.set(x, y, z);
    this.updateTransformation();
    this.prevRightHandPosition.set(x, y, z);
  }

  leftPinchStart() {
    this.isLeftPinching = true;
  }

  leftPinchEnd() {
    this.isLeftPinching = false;
  }

  rightPinchStart() {
    this.isRightPinching = true;
  }

  rightPinchEnd() {
    this.isRightPinching = false;
  }

  getCurrentTransformation() {
    return this.transformationMatrix;
  }

  updateTransformation() {
    if (this.isLeftPinching) {
      this.currentTranslation.add(this.leftHandPosition).sub(this.prevLeftHandPosition);
    }
    this.transformationMatrix.identity();
    // TODO: Transform the world to track the hands as the user "drags" two points in 3D
    this.transformationMatrix.setPosition(this.currentTranslation);
  }
}

export {GestureControlModule};
