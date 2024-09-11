# Copyright 2022 DeepScenario GmbH

import os
import warnings
import shutil
import json
import contextlib
import tempfile
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from matplotlib import cm as colormaps
from tqdm import tqdm
from typing import ClassVar, ContextManager, Optional


@dataclass
class Object:
    # yapf: disable
    translation: np.ndarray         # [m] center coordinates of the bottom of the object [x, y, z]
    rotation: np.ndarray            # [-] rotation vector of the object
    dimension: np.ndarray           # [m] dimensions of the object [l, w, h]
    velocity: np.ndarray            # [m/s] velocity of the object [v_x, v_y, v_z]
    angular_velocity: np.ndarray    # [rad/s] angular velocity of the object [w_x, w_y, w_z]
    acceleration: np.ndarray        # [m/s^2] acceleration of the object [a_x, a_y, a_z]
    category_id: int                # [-] id of the category the object belongs to
    track_id: int                   # [-] id of the track the object belongs to
    road_position: Optional[dict] = None  # [-] road position of the object

    box_idxs: ClassVar[list] = [0, 1, 2, 3, 0,  # base
                                4, 7, 3, 7,     # right
                                6, 2, 6,        # back
                                5,              # left
                                4, 1, 5, 0]     # front
    tip_idxs: ClassVar[list] = [0, 1, 2, 0]
    
    # Add additonal attribute into the class
    ego_vehicle: int = 0            # [-] 1 if object is ego vehicle, 0 otherwise
    # yapf: enable

    @classmethod
    def deserialize(cls, ann: dict) -> 'Object':
        road_position = ann.get('road_position')  # get the road_position dictionary if it exists, otherwise set it to None
        if road_position is not None:
            road_position = dict(road_position)  # convert the road_position dictionary to a Python dictionary
        return cls(np.array(ann['translation']), np.array(ann['rotation']), np.array(ann['dimension']),
                np.array(ann['velocity']), np.array(ann['angular_velocity']), np.array(ann['acceleration']),
                ann['category_id'], ann['track_id'], road_position, ann['ego_vehicle'])

    def get_box_corners(self) -> np.ndarray:
        # corners in local coordinate frame
        l, w, h = self.dimension
        corners_x = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        corners_y = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        corners_z = [0, 0, 0, 0, h, h, h, h]

        # rotate and translate
        transform = get_transform(self.translation, self.rotation)
        return np.dot(transform[:3, :3], np.array([corners_x, corners_y, corners_z])).T + transform[:3, 3].reshape(1, 3)

    def get_tip_corners(self) -> np.ndarray:
        # corners in local coordinate frame
        l, w, h = self.dimension
        corners_x = [0, l / 2, 0]
        corners_y = [w / 2, 0, -w / 2]
        corners_z = [h, h, h]

        # rotate and translate
        transform = get_transform(self.translation, self.rotation)
        return np.dot(transform[:3, :3], np.array([corners_x, corners_y, corners_z])).T + transform[:3, 3].reshape(1, 3)

    def get_color(self) -> tuple:
        return colormaps.get_cmap('Set1')(self.category_id)[:3]


def load_json(json_file: str) -> dict:
    with open(json_file) as file:
        return json.load(file)


def get_item_with_location_id(items: list, location_id: int) -> dict:
    for item in items:
        if item['location_id'] == location_id:
            return item
    raise RuntimeError(f'No item with location id {location_id} found in {items}')


def get_transform(translation: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    transform[:3, 3] = translation
    return transform


def load_frames(ann_dir: str, ann_dicts: list) -> list:
    frames = []
    ann_zip_file = os.path.join(ann_dir, 'annotations.zip')
    with tempfile.TemporaryDirectory() as tmp_dir:
        shutil.unpack_archive(ann_zip_file, tmp_dir)
        for ann_dict in tqdm(ann_dicts, desc='Loading frames'):
            frame_id = ann_dict['frame_id']
            ann_fname = ann_dict['file_name']
            ann_file = os.path.join(tmp_dir, ann_fname)
            if not os.path.isfile(ann_file):
                warnings.warn(f'Skipping frame {frame_id} as file {ann_file} cannot be found')
                continue
            anns = load_json(ann_file)['annotations']        
            # Add ego_vehicle attribute to the annotation
            for ann in anns:
                ann['ego_vehicle'] = 0
            objs = [Object.deserialize(ann) for ann in anns]
            frames.append({'ann_fname': ann_fname, 'frame_id': frame_id, 'objects': objs})
    return sorted(frames, key=lambda k: k['frame_id'])



# Load single frame as a dictionary
def load_frame(ann_dir: str, ann_dict: dict) -> dict:
    frames = []
    frame_id = ann_dict['frame_id']
    ann_zip_file = os.path.join(ann_dir, 'annotations.zip')
    with tempfile.TemporaryDirectory() as tmp_dir:
        shutil.unpack_archive(ann_zip_file, tmp_dir)
        ann_fname = ann_dict['file_name']
        ann_file = os.path.join(tmp_dir, ann_fname)
        if not os.path.isfile(ann_file):
            warnings.warn(f'Skipping frame {frame_id} as file {ann_file} cannot be found')
            return None
        anns = load_json(ann_file)['annotations']
        # Add ego_vehicle attribute to the annotation
        for ann in anns:
            ann['ego_vehicle'] = 0
        objs = [Object.deserialize(ann) for ann in anns]
        frames.append({'ann_fname': ann_fname, 'frame_id': frame_id, 'objects': objs})
        return sorted(frames, key=lambda k: k['frame_id'])