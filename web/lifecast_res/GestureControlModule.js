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

  updateTransformation(logFn, offset) {
    if (this.isLeftPinching && !this.isRightPinching) {
      let translationDelta = this.leftHandPosition.clone().sub(this.prevLeftHandPosition);
      this.currentTranslation.add(translationDelta);
      this.transformationMatrix.multiply(new THREE.Matrix4().makeTranslation(
        (1.0 / this.currentScale) * (this.leftHandPosition.x - this.prevLeftHandPosition.x),
        (1.0 / this.currentScale) * (this.leftHandPosition.y - this.prevLeftHandPosition.y),
        (1.0 / this.currentScale) * (this.leftHandPosition.z - this.prevLeftHandPosition.z)
      ));
    } else if (this.isRightPinching && !this.isLeftPinching) {
      let translationDelta = this.rightHandPosition.clone().sub(this.prevRightHandPosition);
      this.currentTranslation.add(translationDelta);
      this.transformationMatrix.multiply(new THREE.Matrix4().makeTranslation(
        (1.0 / this.currentScale) * (this.rightHandPosition.x - this.prevRightHandPosition.x),
        (1.0 / this.currentScale) * (this.rightHandPosition.y - this.prevRightHandPosition.y),
        (1.0 / this.currentScale) * (this.rightHandPosition.z - this.prevRightHandPosition.z)
      ));
    }
    else if (this.isLeftPinching && this.isRightPinching) {
      const prevDistance = this.prevLeftHandPosition.distanceTo(this.prevRightHandPosition);
      const currentDistance = this.leftHandPosition.distanceTo(this.rightHandPosition);
      let scaleDelta = currentDistance / prevDistance;
      this.currentScale *= scaleDelta;

      // Get the midpoint of the two fingers
      let midpoint = this.leftHandPosition.clone().add(this.rightHandPosition).multiplyScalar(0.5);
      //logFn("Midpoint " + midpoint.x.toFixed(2) + ", " + midpoint.y.toFixed(2) + ", " + midpoint.z.toFixed(2));

      // Decompose the matrix to get the current translation
      let translation = new THREE.Vector3();
      let rotation = new THREE.Quaternion();
      let scale = new THREE.Vector3();
      this.transformationMatrix.decompose(translation, rotation, scale);

      // Extract the translation component of the transformation matrix
      let center = new THREE.Vector3();
      center.add(midpoint);
      center.sub(offset);
      //center.sub(translation);
      center.sub(this.currentTranslation);

      //center.sub(offset);
      // Print the center rounded to 3 decimal places
      logFn("Center " + center.x.toFixed(2) + ", " + center.y.toFixed(2) + ", " + center.z.toFixed(2));

      // Translate
      this.transformationMatrix.multiply(new THREE.Matrix4().makeTranslation(
        center.x,
        center.y,
        center.z
      ));
      // Scale
      this.transformationMatrix.multiply(new THREE.Matrix4().makeScale(scaleDelta, scaleDelta, scaleDelta));
      // Translate back
      this.transformationMatrix.multiply(new THREE.Matrix4().makeTranslation(
        -center.x,
        -center.y,
        -center.z
      ));
    }
  }
}

export {GestureControlModule};
