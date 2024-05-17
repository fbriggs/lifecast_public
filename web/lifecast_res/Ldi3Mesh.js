import {
    LDI3_fthetaBgFragmentShader,
    LDI3_fthetaBgVertexShader,
    LDI3_fthetaFgFragmentShader,
    LDI3_fthetaFgVertexShader
} from "./LifecastVideoPlayerShaders11.js";
import * as THREE from './three152.module.min.js';

export const NUM_LAYERS = 3;

/*
 That class is a THREE Object3D displaying a LDI.
 */
export class Ldi3Mesh extends THREE.Object3D {
    layer_to_meshes = Array.from({ length: NUM_LAYERS }, () => []);

    ftheta_scale = null

    constructor(_decode_12bit, texture, _ftheta_scale = null) {

        super()
        this.ftheta_scale = 1.15;
        console.log("_ftheta_scale=", this.ftheta_scale);

        // Make the initial shader uniforms.
        this.uniforms = {
            uTexture: { value: texture },
            uEffectRadius: { value: 100.0 },  // Set to a lower value to play intro animation
        };

        // Make the foreground mesh material.
        var shader_prefix = "";
        if (_decode_12bit) shader_prefix += "#define DECODE_12BIT\n";

        //// LDI3 materials ////

        const ldi3_layer0_material = this.ldi3_layer0_material = new THREE.ShaderMaterial({
            vertexShader:   shader_prefix + LDI3_fthetaBgVertexShader,
            fragmentShader: shader_prefix + LDI3_fthetaBgFragmentShader,
            uniforms: this.uniforms,
            depthTest: true,
            depthWrite: true,
            transparent: true,
            wireframe: false
        });
        ldi3_layer0_material.side = THREE.BackSide;
        ldi3_layer0_material.depthFunc = THREE.LessDepth;

        const ldi3_layer1_material = this.ldi3_layer1_material = new THREE.ShaderMaterial({
            vertexShader:   shader_prefix + LDI3_fthetaFgVertexShader,
            fragmentShader: shader_prefix + LDI3_fthetaFgFragmentShader,
            uniforms: this.uniforms,
            depthTest: true,
            depthWrite: true,
            transparent: true,
            wireframe: false
        });
        ldi3_layer1_material.side = THREE.BackSide;
        ldi3_layer1_material.depthFunc = THREE.LessEqualDepth;

        const ldi3_layer2_material = this.ldi3_layer2_material = new THREE.ShaderMaterial({
            vertexShader:    "#define LAYER2\n" + shader_prefix + LDI3_fthetaFgVertexShader,
            fragmentShader:  "#define LAYER2\n" + shader_prefix + LDI3_fthetaFgFragmentShader,
            uniforms: this.uniforms,
            depthTest: true,
            depthWrite: true,
            transparent: true,
            wireframe: false
        });
        ldi3_layer2_material.side = THREE.BackSide;
        ldi3_layer2_material.depthFunc = THREE.LessEqualDepth;

        const inflation = 3.0;
        this.makeEquiangularMesh(ldi3_layer0_material, 128, 4, 0, inflation);
        this.makeEquiangularMesh(ldi3_layer1_material, 128, 4, 1, inflation);
        this.makeEquiangularMesh(ldi3_layer2_material, 96, 4, 2, inflation); // HACK: a few less triangles here to give some overhead on Quest Pro to not exceed triangle limit when displaying extra UI elements.
    }

    makeEquiangularMesh(material, GRID_SIZE, NUM_PATCHES, order, ftheta_inflation, is_oculus) {
        const NUM_QUADS_PER_SIDE = NUM_PATCHES * GRID_SIZE;
        const MARGIN = 2;

        for (var patch_j = 0; patch_j < NUM_PATCHES; ++patch_j) {
            for (var patch_i = 0; patch_i < NUM_PATCHES; ++patch_i) {
                const verts   = [];
                const indices = [];
                const uvs     = [];

                for (var j = 0; j <= GRID_SIZE; ++j) {
                    for (var i = 0; i <= GRID_SIZE; ++i) {
                        const ii = i + patch_i * GRID_SIZE;
                        const jj = j + patch_j * GRID_SIZE;
                        const u  = ii / NUM_QUADS_PER_SIDE;
                        const v  = jj / NUM_QUADS_PER_SIDE;

                        const a = 2.0 * (u - 0.5);
                        const b = 2.0 * (v - 0.5);
                        const theta = Math.atan2(b, a);
                        var r = Math.sqrt(a * a + b * b) / this.ftheta_scale;
                        r = 0.5 * r + 0.5 * Math.pow(r, ftheta_inflation);
                        const phi = r * Math.PI / 2.0;

                        const x = Math.cos(theta) * Math.sin(phi);
                        const y = Math.sin(theta) * Math.sin(phi);
                        const z = -Math.cos(phi);

                        verts.push(x, y, z);
                        uvs.push(u, v);
                    }
                }

                for (var j = 0; j < GRID_SIZE; ++j) {
                    for (var i = 0; i < GRID_SIZE; ++i) {
                        // Skip quads outside the image circle.
                        const ii = i + patch_i * GRID_SIZE;
                        const jj = j + patch_j * GRID_SIZE;
                        const di = ii - NUM_QUADS_PER_SIDE / 2;
                        const dj = jj - NUM_QUADS_PER_SIDE / 2;
                        if (di * di + dj * dj > (NUM_QUADS_PER_SIDE+MARGIN) * (NUM_QUADS_PER_SIDE+MARGIN) / 4) continue;

                        const a = i + (GRID_SIZE + 1) * j;
                        const b = a + 1;
                        const c = a + (GRID_SIZE + 1);
                        const d = c + 1;
                        indices.push(a, c, b);
                        indices.push(c, d, b);
                    }
                }

                if (indices.length > 0) {
                    const geometry = new THREE.BufferGeometry();
                    geometry.setIndex(indices);
                    geometry.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
                    geometry.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));

                    const mesh = new THREE.Mesh(geometry, material);

                    this.layer_to_meshes[order].push(mesh);

                    mesh.frustumCulled = false;

                    mesh.renderOrder = order;
                    this.add(mesh);
                }
            }
        }

    }
}

