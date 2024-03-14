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

  updateTransformation(offset) {
    if (this.isLeftPinching) {
      let translationDelta = this.leftHandPosition.clone().sub(this.prevLeftHandPosition);
      /*
      this.currentTranslation.add(translationDelta);
      this.transformationMatrix.multiply(new THREE.Matrix4().makeTranslation(
        (1.0 / this.currentScale) * (this.leftHandPosition.x - this.prevLeftHandPosition.x),
        (1.0 / this.currentScale) * (this.leftHandPosition.y - this.prevLeftHandPosition.y),
        (1.0 / this.currentScale) * (this.leftHandPosition.z - this.prevLeftHandPosition.z)
      ));
      */
    } else if (this.isRightPinching) {
      let translationDelta = this.rightHandPosition.clone().sub(this.prevRightHandPosition);
      /*
      this.currentTranslation.add(translationDelta);
      this.transformationMatrix.multiply(new THREE.Matrix4().makeTranslation(
        (1.0 / this.currentScale) * (this.rightHandPosition.x - this.prevRightHandPosition.x),
        (1.0 / this.currentScale) * (this.rightHandPosition.y - this.prevRightHandPosition.y),
        (1.0 / this.currentScale) * (this.rightHandPosition.z - this.prevRightHandPosition.z)
      ));
      */
    }

    if (this.isLeftPinching && this.isRightPinching) {
      const prevDistance = this.prevLeftHandPosition.distanceTo(this.prevRightHandPosition);
      const currentDistance = this.leftHandPosition.distanceTo(this.rightHandPosition);
      let scaleDelta = currentDistance / prevDistance;
      this.currentScale *= scaleDelta;
      // Translate so that the center of scaling is currentTranslation + (left + right) / 2
      //let center = this.leftHandPosition.clone().add(this.rightHandPosition).multiplyScalar(0.5).add(this.currentTranslation);
      //let center = this.leftHandPosition.clone().add(this.rightHandPosition).multiplyScalar(0.5);
      //center.sub(offset);

      let center = new THREE.Vector3(0, 0, 0);
      center.add(offset);
      center.sub(this.leftHandPosition.clone().add(this.rightHandPosition).multiplyScalar(0.5));
      // Translate
      this.transformationMatrix.multiply(new THREE.Matrix4().makeTranslation(
        -center.x,
        -center.y,
        -center.z
      ));
      // Scale
      this.transformationMatrix.multiply(new THREE.Matrix4().makeScale(scaleDelta, scaleDelta, scaleDelta));
      // Translate back
      this.transformationMatrix.multiply(new THREE.Matrix4().makeTranslation(
        center.x,
        center.y,
        center.z
      ));
    }
  }
}

export {GestureControlModule};
