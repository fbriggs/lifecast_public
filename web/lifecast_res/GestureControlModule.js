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
  }

  updateRightHand(x, y, z) {
    this.rightHandPosition.set(x, y, z);
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
      let translationDelta = this.leftHandPosition.clone().sub(this.prevLeftHandPosition);
      this.currentTranslation.add(translationDelta);
      this.transformationMatrix.multiply(new THREE.Matrix4().makeTranslation(
        this.leftHandPosition.x - this.prevLeftHandPosition.x,
        this.leftHandPosition.y - this.prevLeftHandPosition.y,
        this.leftHandPosition.z - this.prevLeftHandPosition.z));
    } else if (this.isRightPinching) {
      let translationDelta = this.rightHandPosition.clone().sub(this.prevRightHandPosition);
      this.currentTranslation.add(translationDelta);
      this.transformationMatrix.multiply(new THREE.Matrix4().makeTranslation(
        this.rightHandPosition.x - this.prevRightHandPosition.x,
        this.rightHandPosition.y - this.prevRightHandPosition.y,
        this.rightHandPosition.z - this.prevRightHandPosition.z));
    }

    if (this.isLeftPinching && this.isRightPinching) {
      const prevDistance = this.prevLeftHandPosition.distanceTo(this.prevRightHandPosition);
      const currentDistance = this.leftHandPosition.distanceTo(this.rightHandPosition);
      let scaleDelta = currentDistance / prevDistance;
      // Translate so that the center point of the hands plus currentTranslation is at the origin
      let center = this.leftHandPosition.clone().add(this.rightHandPosition).multiplyScalar(0.5).add(this.currentTranslation);
      this.transformationMatrix.multiply(new THREE.Matrix4().makeTranslation(-center.x, -center.y, -center.z));
      // Scale
      this.transformationMatrix.multiply(new THREE.Matrix4().makeScale(scaleDelta, scaleDelta, scaleDelta));
      // Translate back
      this.transformationMatrix.multiply(new THREE.Matrix4().makeTranslation(center.x, center.y, center.z));
    }
  }
}

export {GestureControlModule};
