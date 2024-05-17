import * as THREE from './three152.module.min.js';
import {
    VR180_FragmentShader,
    VR180_VertexShader
} from "./LifecastVideoPlayerShaders11.js";

export class Vr180Mesh extends THREE.Object3D {
    constructor(texture) {
        super();

        this.uniforms = {
            uTexture: { value: texture },
            uEffectRadius: { value: 0 },
        };

        const material = new THREE.ShaderMaterial({
            vertexShader: VR180_VertexShader,
            fragmentShader: VR180_FragmentShader,
            uniforms: this.uniforms,
            side: THREE.DoubleSide,
            transparent: true
        });

        // Create left eye mesh
        const leftMesh = this.createVr180Mesh(material, true);
        leftMesh.layers.set(1);  // Use layer 1 for left eye
        this.add(leftMesh);

        // Create right eye mesh
        const rightMesh = this.createVr180Mesh(material, false);
        rightMesh.layers.set(2);  // Use layer 2 for right eye
        this.add(rightMesh);
    }

    createVr180Mesh(material, isLeftEye) {
        // A half-sphere from angle 180 to 360 degrees, 1000 meter radius
        const geometry = new THREE.SphereGeometry(1000, 64, 64, Math.PI, Math.PI);

        // Modify UVs for stereo view
        const uvs = geometry.attributes.uv.array;
        for (let i = 0; i < uvs.length; i += 2) {
            if (isLeftEye) {
                uvs[i] = (1.0 - uvs[i]) * 0.5;  // Left eye
            } else {
                uvs[i] = 1.0 - (uvs[i] * 0.5);  // Right eye
            }
        }

        geometry.attributes.uv.needsUpdate = true;        

        return new THREE.Mesh(geometry, material);
    }
}
