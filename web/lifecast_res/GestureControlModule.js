import * as THREE from './three152.module.min.js';

class GestureControlModule {
  constructor() {
    this.leftHandPosition = new THREE.Vector3();
    this.rightHandPosition = new THREE.Vector3();
    this.prevLeftHandPosition = new THREE.Vector3();
    this.prevRightHandPosition = new THREE.Vector3();
    this.isLeftPinching = false;
    this.isRightPinching = false;

    this.currentScale = 1.0;

    this.currentRotX = 0;
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

  getHandAngleY(left, right) {
    let dz = left.z - right.z;
    let dx = left.x - right.x;
    if (dx === 0 && dz === 0) {
      return 0;
    }
    return -Math.atan2(dz, dx);
  }

  getHandAngleX(left, right) {
    let dy = left.y - right.y;
    let dz = left.z - right.z;
    if (dz === 0 && dy === 0) {
      return 0;
    }
    return -Math.atan2(dy, dz);
  }

  normalizeAngle(angle) {
    if (angle > Math.PI) {
      angle -= 2 * Math.PI;
    } else if (angle < -Math.PI) {
      angle += 2 * Math.PI;
    }
    return angle;
  }

  getCurrentTransformation() {
    let transformationMatrix = new THREE.Matrix4();

    // Create a scaling matrix
    let scaleMatrix = new THREE.Matrix4().makeScale(this.currentScale, this.currentScale, this.currentScale);

    // Create a rotation matrix
    let rotationMatrixY = new THREE.Matrix4().makeRotationY(this.currentRotY);
    let rotationMatrixX = new THREE.Matrix4().makeRotationX(this.currentRotX);

    // Create a translation matrix
    let translationMatrix = new THREE.Matrix4().makeTranslation(
      this.currentTranslation.x,
      this.currentTranslation.y,
      this.currentTranslation.z
    );

    transformationMatrix.multiply(translationMatrix);
    transformationMatrix.multiply(scaleMatrix);
    transformationMatrix.multiply(rotationMatrixX);
    transformationMatrix.multiply(rotationMatrixY);

    return transformationMatrix;
  }

  updateTransformation(logFn, world_group_position, mesh_position) {
    if (this.isLeftPinching && !this.isRightPinching) {
      let translationDelta = this.leftHandPosition.clone().sub(this.prevLeftHandPosition);
      this.currentTranslation.add(translationDelta);
    }
    else if (this.isRightPinching && !this.isLeftPinching) {
      let translationDelta = this.rightHandPosition.clone().sub(this.prevRightHandPosition);
      this.currentTranslation.add(translationDelta);
    }
    else if (this.isLeftPinching && this.isRightPinching) {
      // Use the average of both left and right translation
      let translationDeltaLeft = this.leftHandPosition.clone().sub(this.prevLeftHandPosition);
      let translationDeltaRight = this.rightHandPosition.clone().sub(this.prevRightHandPosition);
      let translationDelta = translationDeltaLeft.add(translationDeltaRight).multiplyScalar(0.5);
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
      this.currentTranslation.add(grasp_point.clone().multiplyScalar(1.0 - scaleDelta));

      // Rotate about the Y axis
      let rotationDeltaY = this.normalizeAngle(this.getHandAngleY(this.leftHandPosition, this.rightHandPosition) - this.getHandAngleY(this.prevLeftHandPosition, this.prevRightHandPosition));
      this.currentRotY += rotationDeltaY;

      let rotation_motion_y = new THREE.Vector3(-grasp_point.z, 0, grasp_point.x);
      rotation_motion_y.multiplyScalar(Math.max(Math.min(rotationDeltaY, 0.1), -0.1));
      this.currentTranslation.add(rotation_motion_y);

      // Rotate about the X axis
      let rotationDeltaX = this.normalizeAngle(this.getHandAngleX(this.leftHandPosition, this.rightHandPosition) - this.getHandAngleX(this.prevLeftHandPosition, this.prevRightHandPosition));
      this.currentRotX += rotationDeltaX;

      let rotation_motion_x = new THREE.Vector3(0, grasp_point.y, -grasp_point.z);
      rotation_motion_x.multiplyScalar(Math.max(Math.min(rotationDeltaX, 0.1), -0.1));
      this.currentTranslation.add(rotation_motion_x);

    }
  }
}

export {GestureControlModule};
