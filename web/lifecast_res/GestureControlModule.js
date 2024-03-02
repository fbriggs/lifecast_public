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
    this.isLeftPinching = false;
    this.isRightPinching = false;
    this.initialDistance = null;
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
    this.isLeftPinching = true;
  }

  leftPinchEnd() {
    this.isLeftPinching = false;
    this.initialDistance = null; // Reset initial distance on releasing pinch
  }

  rightPinchStart() {
    this.isRightPinching = true;
  }

  rightPinchEnd() {
    this.isRightPinching = false;
    this.initialDistance = null; // Reset initial distance on releasing pinch
  }

  getCurrentTransformation() {
    return this.transformationMatrix;
  }

  updateTransformation() {
    if (this.isLeftPinching && this.isRightPinching) {
      // Both hands are pinching - handle scaling and rotation
      const currentDistance = this.leftHandPosition.distanceTo(this.rightHandPosition);
      if (this.initialDistance === null) {
        this.initialDistance = currentDistance;
      } else {
        const scale = currentDistance / this.initialDistance;
        this.transformationMatrix.makeScale(scale, scale, scale);
        
        // Rotation - simplification: compute only if hands maintain initial distance
        if (Math.abs(currentDistance - this.initialDistance) < 0.1) { // threshold to avoid simultaneous scale and rotate
          const direction = new THREE.Vector3().subVectors(this.rightHandPosition, this.leftHandPosition).normalize();
          const angle = Math.atan2(direction.x, direction.z);
          this.transformationMatrix.makeRotationY(angle);
        }
      }
    } else if (this.isLeftPinching || this.isRightPinching) {
      // Single hand pinching - handle translation
      const translation = this.isLeftPinching ? this.leftHandPosition : this.rightHandPosition;
      this.transformationMatrix.makeTranslation(translation.x, translation.y, translation.z);
    }
  }
}

export {GestureControlModule};
