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
    let transformationMatrix = new THREE.Matrix4();
    // TODO: Use this.currentTranslation and this.currentScale to create a matrix
    // ...
  }

  getCurrentTransformation() {
    let transformationMatrix = new THREE.Matrix4();

    // Create a scaling matrix
    let scaleMatrix = new THREE.Matrix4().makeScale(this.currentScale, this.currentScale, this.currentScale);

    // Create a rotation matrix
    let rotationMatrix = new THREE.Matrix4().makeRotationY(this.currentRotY);

    // Create a translation matrix
    let translationMatrix = new THREE.Matrix4().makeTranslation(
      this.currentTranslation.x,
      this.currentTranslation.y,
      this.currentTranslation.z
    );

    // Combine them by first scaling, then rotating, and finally translating
    transformationMatrix.multiply(rotationMatrix);
    transformationMatrix.multiply(translationMatrix);
    transformationMatrix.multiply(scaleMatrix);

    // Alternatively, you can pre-multiply, which is equivalent to multiplying in the reverse order:
    // transformationMatrix.premultiply(translationMatrix).premultiply(rotationMatrix).premultiply(scaleMatrix);

    return transformationMatrix;
  }

  updateTransformation(logFn, world_group_position, mesh_position) {
    if (this.isLeftPinching && !this.isRightPinching) {
      let translationDelta = this.leftHandPosition.clone().sub(this.prevLeftHandPosition);
      //this.currentTranslation.add(translationDelta.multiplyScalar(1.0 / this.currentScale));
      this.currentTranslation.add(translationDelta);
    }
    else if (this.isRightPinching && !this.isLeftPinching) {
      let translationDelta = this.rightHandPosition.clone().sub(this.prevRightHandPosition);
      //this.currentTranslation.add(translationDelta.multiplyScalar(1.0 / this.currentScale));
      this.currentTranslation.add(translationDelta);
    }
    else if (this.isLeftPinching && this.isRightPinching) {
      // Use the average of both left and right translation
      let translationDeltaLeft = this.leftHandPosition.clone().sub(this.prevLeftHandPosition);
      let translationDeltaRight = this.rightHandPosition.clone().sub(this.prevRightHandPosition);
      let translationDelta = translationDeltaLeft.add(translationDeltaRight).multiplyScalar(0.5);
      //this.currentTranslation.add(translationDelta.multiplyScalar(1.0 / this.currentScale));
      this.currentTranslation.add(translationDelta);

      let prevDistance = this.prevLeftHandPosition.distanceTo(this.prevRightHandPosition);
      let currentDistance = this.leftHandPosition.distanceTo(this.rightHandPosition);
      let scaleDelta = currentDistance / prevDistance;
      this.currentScale *= scaleDelta;

      // Scaling the mesh down moves the grasp point toward the mesh center
      // To compensate, translate the mesh toward the grasp point (if scaleDelta < 1) or away from the grasp point (if scaleDelta > 1)
      let grasp_point = this.leftHandPosition.clone().add(this.rightHandPosition).multiplyScalar(0.5);
      grasp_point.sub(world_group_position);
      grasp_point.sub(mesh_position);
      this.currentTranslation.add(grasp_point.multiplyScalar(1.0 - scaleDelta));

      logFn("Grasp point: " + grasp_point.x.toFixed(2) + ", " + grasp_point.y.toFixed(2) + ", " + grasp_point.z.toFixed(2));
      logFn("Mesh position: " + mesh_position.x.toFixed(2) + ", " + mesh_position.y.toFixed(2) + ", " + mesh_position.z.toFixed(2));
    }

    //logFn("x " + this.currentTranslation.x.toFixed(2) + " y " + this.currentTranslation.y.toFixed(2) + " z " + this.currentTranslation.z.toFixed(2) + " scale " + this.currentScale.toFixed(2));
  }
}

export {GestureControlModule};
