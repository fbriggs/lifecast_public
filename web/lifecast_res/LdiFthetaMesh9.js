import {
    LDI2_fthetaBgFragmentShader,
    LDI2_fthetaBgVertexShader,
    LDI2_fthetaFgFragmentShader,
    LDI2_fthetaFgVertexShader,
    LDI3_fthetaBgFragmentShader,
    LDI3_fthetaBgVertexShader,
    LDI3_fthetaFgFragmentShader,
    LDI3_fthetaFgVertexShader
} from "./LifecastVideoPlayerShaders9.js";
import * as THREE from './three152.module.min.js';

export const FTHETA_UNIFORM_ROTATION_BUFFER_SIZE = 60; // This MUST match the uniform array size in fgVertexShader


/*
 That class is a THREE Object3D displaying a LDI.
 */
export class LdiFthetaMesh extends THREE.Object3D {
    ftheta_fg_geoms = []; // references to all of the patches in the f-theta mesh (foreground)
    ftheta_bg_geoms = []; // references to all of the patches in the f-theta mesh (background)
    ftheta_fg_meshes = []; // above is for the geometry (so we can rotate). this one is for the Mesh so we can toggle visibility for debug purposes
    ftheta_mid_meshes = []; // toggle visibility for debug purposes
    ftheta_bg_meshes = []; // toggle visibility for debug purposes
    num_patches_not_culled = 0; // Used for performance stats (want to know how many patches are being draw in various scenes).
    ftheta_scale = null

    constructor(_format, is_chrome, photo_mode, _metadata_url, _decode_12bit, texture, _ftheta_scale = null) {

        super()

        if (_ftheta_scale == null) {
            if (_format == "ldi2") this.ftheta_scale = 1.2;
            else if (_format == "ldi3") this.ftheta_scale = 1.15;
            else { console.log("Error, unknown format: ", _format); }
        } else {
            this.ftheta_scale = _ftheta_scale
        }
        console.log("_format=", _format);
        console.log("_ftheta_scale=", this.ftheta_scale);


        // Make the initial shader uniforms.
        var placeholder_ftheta_rotation_arr = new Array(FTHETA_UNIFORM_ROTATION_BUFFER_SIZE);
        for (var i = 0; i < FTHETA_UNIFORM_ROTATION_BUFFER_SIZE; ++i) {
            placeholder_ftheta_rotation_arr[i] = new THREE.Matrix3();
        }
        this.uniforms = {
            uTexture: { value: texture },
            uDistCamFromOrigin: { value: 0.0 },
            uFthetaRotation: { value: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] },
            uFirstFrameInFthetaTable: { value: 0 },
            uFrameIndexToFthetaRotation: { value: placeholder_ftheta_rotation_arr },
            uCurrFrameChrome: { value: 0 },
        };

        // Make the foreground mesh material.
        var shader_prefix = "";
        if (is_chrome)  shader_prefix += "#define CHROME\n";
        if (photo_mode) shader_prefix += "#define PHOTO\n";
        if (_metadata_url == "") shader_prefix += "#define NO_METADATA\n";
        if (_decode_12bit) shader_prefix += "#define DECODE_12BIT\n";

        //// LDI2 materials ////

        const ldi2_fg_material = this.ldi2_fg_material = new THREE.ShaderMaterial({
            vertexShader:   shader_prefix + LDI2_fthetaFgVertexShader,
            fragmentShader: shader_prefix + LDI2_fthetaFgFragmentShader,
            uniforms: this.uniforms,
            depthTest: true,
            depthWrite: true,
            transparent: false,
            wireframe: false
        });
        ldi2_fg_material.side = THREE.DoubleSide;

        const ldi2_bg_material = this.ldi2_bg_material = new THREE.ShaderMaterial({
            vertexShader:   shader_prefix + LDI2_fthetaBgVertexShader,
            fragmentShader: shader_prefix + LDI2_fthetaBgFragmentShader,
            uniforms: this.uniforms,
            depthTest: true,
            depthWrite: true,
            transparent: false,
            wireframe: false
        });
        ldi2_bg_material.side = THREE.DoubleSide;
        ldi2_bg_material.depthFunc = THREE.LessDepth;

        //// LDI3 materials ////

        const ldi3_layer0_material = this.ldi3_layer0_material = new THREE.ShaderMaterial({
            vertexShader:   shader_prefix + LDI3_fthetaBgVertexShader,
            fragmentShader: shader_prefix + LDI3_fthetaBgFragmentShader,
            uniforms: this.uniforms,
            depthTest: true,
            depthWrite: true,
            transparent: false,
            wireframe: false
        });
        ldi3_layer0_material.side = THREE.DoubleSide;
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
        ldi3_layer1_material.side = THREE.DoubleSide;
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
        ldi3_layer2_material.side = THREE.DoubleSide;
        ldi3_layer2_material.depthFunc = THREE.LessEqualDepth;

        if (_format == "ldi2") {
            const inflation = 1.0;
            this.makeFthetaMesh(_format, ldi2_fg_material, 32, 14, 0, inflation);
            this.makeFthetaMesh(_format, ldi2_bg_material, 32, 14, 1, inflation);
        } else if (_format == "ldi3") {
            const inflation = 3.0;
            this.makeFthetaMesh(_format, ldi3_layer0_material, 32, 16, 0, inflation);
            this.makeFthetaMesh(_format, ldi3_layer1_material, 32, 16, 1, inflation);
            this.makeFthetaMesh(_format, ldi3_layer2_material, 32, 16, 2, inflation);
        } else {
            console.log("Unrecognized format: ", _format);
        }
    }

    makeFthetaMesh(format, material, GRID_SIZE, NUM_PATCHES, order, ftheta_inflation, is_oculus) {
        const NUM_QUADS_PER_SIDE = NUM_PATCHES * GRID_SIZE;
        const MARGIN = 3;

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
                        if (format == "ldi3") r = 0.5 * r + 0.5 * Math.pow(r, ftheta_inflation);
                        const phi = r * Math.PI / 2.0;

                        const x = Math.cos(theta) * Math.sin(phi);
                        const y = Math.sin(theta) * Math.sin(phi);
                        const z = -Math.cos(phi);

                        // TODO: the 10's here are a hack for frustum culling. 10 might not be optimal
                        verts.push(x * 10, y * 10, z * 10);
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
                        if (di * di + dj * dj > (NUM_QUADS_PER_SIDE-MARGIN) * (NUM_QUADS_PER_SIDE-MARGIN) / 4) continue;

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
                    geometry.computeBoundingSphere();
                    geometry.originalBoundingSphere = geometry.boundingSphere.clone();
                    geometry.boundingSphere.radius *= 1.1;

                    const mesh = new THREE.Mesh(geometry, material);

                    if (format == "ldi2") {
                        if (order == 0) {
                            this.ftheta_fg_geoms.push(geometry);
                            this.ftheta_fg_meshes.push(mesh);
                        } else if (order == 1) {
                            this.ftheta_bg_geoms.push(geometry);
                        }
                    } else if (format == "ldi3") {
                        if (order == 0) {
                            this.ftheta_bg_geoms.push(geometry);
                            this.ftheta_bg_meshes.push(mesh);
                        } else { // this puts L1 and L2 together
                            this.ftheta_fg_geoms.push(geometry);
                        }
                        if (order == 1) {
                            this.ftheta_mid_meshes.push(mesh);
                        }
                        if (order == 2) {
                            this.ftheta_fg_meshes.push(mesh);
                        }
                    }

                    // We can only do frustum culling in Chrome with ftheta projection, because we
                    // need to be able to update the bounding sphere centers with the current
                    // frame's rotation matrix outside the shader. Oculus is a flavor of Chrome.
                    // We don't really need to do this on desktop / mobile as an optimization, and it
                    // can sometimes cause artifacts, so now its only on Oculus.
                    mesh.frustumCulled = is_oculus;
                    mesh.onBeforeRender = () => { this.num_patches_not_culled += 1; };

                    //const color = (patch_i + 3542) * (patch_j + 3444) * 329482983;
                    //const wireframe_material = new THREE.MeshBasicMaterial({color: color, side: THREE.DoubleSide, depthTest: false, transparent: false, wireframe:true});
                    //const mesh = new THREE.Mesh(geometry, wireframe_material);

                    mesh.renderOrder = order;
                    this.add(mesh); //world_group
                }
            }
        }

    }
}

