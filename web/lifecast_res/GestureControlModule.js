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
    this.translation = new THREE.Vector3();
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
    this.leftPinchStartPosition = this.leftHandPosition.clone().sub(this.translation);
  }

  leftPinchEnd() {
    this.leftPinchStartPosition = null;
  }

  rightPinchStart() {
    this.rightPinchStartPosition = this.rightHandPosition.clone();
  }

  rightPinchEnd() {
    this.rightPinchStartPosition = null;
  }

  getCurrentTransformation() {
    return this.transformationMatrix;
  }

  updateTransformation() {
    if (this.leftPinchStartPosition) {
      this.translation = this.leftHandPosition.clone().sub(this.leftPinchStartPosition);
    }
    this.transformationMatrix.makeTranslation(this.translation.x, this.translation.y, this.translation.z);
  }
}

export {GestureControlModule};
