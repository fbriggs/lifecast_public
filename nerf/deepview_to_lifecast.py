# MIT License. Copyright (c) 2024 Lifecast Incorporated. Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import time
import copy
import argparse
import json
import os
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from numpy_encoder import NumpyEncoder


def save_tinynerf_dataset(data, filepath):
    print("Saving TinyNeRF dataset to", filepath)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, cls=NumpyEncoder)


def load_deepview_dataset(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    data.sort(key=lambda x: x['name'])
    return data


def convert_deepview_video_to_tinynerf(deepview_json_path, output_json_path, num_cameras):
    dataset = load_deepview_dataset(deepview_json_path,)

    output_data = {'frames_data': []}
    cam_num = 0
    for camera in dataset:
        if cam_num >= num_cameras: break
        
        # Prepare world_from_camera matrix for TinyNerf
        R_mat = Rotation.from_rotvec(camera["orientation"]).as_matrix()
        t = np.array(camera['position'])[:, np.newaxis]
        extrinsics = np.concatenate((R_mat, -np.dot(R_mat, t)), axis=1)
        cam_from_world = np.vstack((extrinsics, np.array([0, 0, 0, 1])))
        world_from_cam = np.linalg.inv(cam_from_world)

        # Adjust coordinate system to match TinyNerf
        mirror_and_flip = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0,  1, 0],
            [0,  0,  0, 1]
        ])
        world_from_cam = np.dot(mirror_and_flip, world_from_cam)

        world_from_cam[:3, 1] = -world_from_cam[:3, 1]

        output_data['frames_data'].append({
            "cam_model": "rectilinear",
            "cx": camera["principal_point"][0],
            "cy": camera["principal_point"][1],
            "fx": camera["focal_length"],
            "fy": camera["focal_length"],
            "radial_distortion": camera["radial_distortion"],
            "width": int(camera["width"]),
            "height": int(camera["height"]),
            "image_filename": camera["name"],
            "timestamp": 0,
            "world_from_cam": world_from_cam.flatten(order='F').tolist(),
        })
        cam_num += 1

    save_tinynerf_dataset(output_data, output_json_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert a DeepView directory to TinyNerf format for video.')
    parser.add_argument('--deepview_json_path', required=True, help='Path to input DeepView json data')
    parser.add_argument('--output_json_path', required=True, help='Path to write lifecast format json data')
    parser.add_argument('--num_cameras', default=20, required=False, help='ok to only use a subset of cameras from the dataset')
    
    args = parser.parse_args()
    
    convert_deepview_video_to_tinynerf(args.deepview_json_path, args.output_json_path, args.num_cameras)
