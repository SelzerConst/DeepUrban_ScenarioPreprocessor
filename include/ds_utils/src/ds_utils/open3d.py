# Copyright 2022 DeepScenario GmbH

from PyQt5.QtWidgets import QDesktopWidget, QApplication
import numpy as np
import open3d as o3d

from ds_utils.generic import Object, get_transform


class Spectator:
    app = QApplication([])

    def __init__(
        self, extrinsic_matrix: np.ndarray, width: int = QDesktopWidget().screenGeometry().width(),
        height: int = QDesktopWidget().screenGeometry().height()
    ) -> None:
        self.extrinsic_matrix = extrinsic_matrix
        self.width = width
        self.height = height
        self.intrinsic = get_pinhole_camera_intrinsic_from_fov(fovx=90, img_size=(width, height))


def get_pinhole_camera_intrinsic_from_fov(fovx: float, img_size: tuple) -> o3d.camera.PinholeCameraIntrinsic:
    w, h = img_size
    fx = 0.5 * w / np.tan(np.deg2rad(0.5 * fovx))  # fovx must be given in degrees
    fy = fx
    cx, cy = w / 2 - 0.5, h / 2 - 0.5
    return o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)


def get_object_geometries(obj: Object) -> dict:
    geometries = {}
    transform = get_transform(obj.translation, obj.rotation)

    # box
    l, w, h = obj.dimension
    color = obj.get_color()
    box = o3d.geometry.TriangleMesh.create_box(width=l - 1e-2, height=w - 1e-2, depth=h - 1e-2)
    box.translate((-l / 2, -w / 2, 0)).transform(transform)
    box.paint_uniform_color(color)
    geometries[f'Object {obj.track_id} Box'] = box

    # lines
    box_corners = obj.get_box_corners()[obj.box_idxs]
    tip_corners = obj.get_tip_corners()[obj.tip_idxs]
    edges = [[i, i + 1] for i in range(len(box_corners) - 1)] + \
            [[j + len(box_corners), j + 1 + len(box_corners)] for j in range(len(tip_corners) - 1)]
    colors = [[0, 0, 0] for _ in range(len(edges))]
    lines = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(np.vstack([box_corners, tip_corners])),
                                 lines=o3d.utility.Vector2iVector(edges))
    lines.colors = o3d.utility.Vector3dVector(colors)
    geometries[f'Object {obj.track_id} Lines'] = lines

    # arrow
    v_mag = np.linalg.norm(obj.velocity)
    if not np.isclose(v_mag, 0):
        size = 5
        length = v_mag / 5
        arrow = o3d.geometry.TriangleMesh.create_arrow(cone_radius=0.06 * size, cone_height=0.2 * length,
                                                       cylinder_radius=0.035 * size, cylinder_height=0.8 * length)
        R = get_arrow_rotation_matrix(obj.velocity)
        arrow.rotate(R, center=(0, 0, 0))
        arrow.translate(obj.translation + h * transform[:3, 2])
        arrow.paint_uniform_color([1, 1, 0])
        geometries[f'Object {obj.track_id} Arrow'] = arrow

    return geometries


def get_object_text(obj: Object) -> tuple:
    v_mag = np.linalg.norm(obj.velocity)
    text_pos = obj.translation
    text_str = f'ID {obj.track_id}; {int(3.6 * v_mag)} km/h'
    return text_pos, text_str


def get_arrow_rotation_matrix(vector: np.ndarray) -> np.ndarray:
    v_norm = np.linalg.norm(vector)
    assert not np.isclose(v_norm, 0)
    v_unit = vector / v_norm
    mat = np.array([[0, 0, v_unit[0]], [0, 0, v_unit[1]], [-v_unit[0], -v_unit[1], 0]])
    return np.eye(3) + mat + mat.dot(mat) / (1 + v_unit[2])
