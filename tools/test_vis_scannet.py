import open3d as o3d
from pathlib import Path
import torch
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
# from matplotlib import pyplot as plt
# import argparse

_label_to_color_uint8 = {
    -1: (0., 0., 0.),
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    13: (100., 85., 144.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (140., 153., 101.),
    28: (158., 218., 229.),
    29: (100., 125., 154.),
    30: (178., 127., 135.),
    32: (146., 111., 194.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    35: (96., 207., 209.),
    36: (227., 119., 194.),
    37: (213., 92., 176.),
    38: (94., 106., 211.),
    39: (82., 84., 163.),
}

_label_to_color = dict([
    (label, (np.array(color_uint8).astype(np.float64) / 255.0).tolist())
    for label, color_uint8 in _label_to_color_uint8.items()
])

def load_real_data(pth_path):
    """
    Args:
        pth_path: Path to the .pth file.
    Returns:
        points: (N, 3), float64
        colors: (N, 3), float64, 0-1
        labels: (N, ), int64, {1, 2, ..., 36, 39, 255}.
    """
    # - points: (N, 3), float32           -> (N, 3), float64
    # - colors: (N, 3), float32, 0-255    -> (N, 3), float64, 0-1
    # - labels: (N, 1), float64, 0-19,255 -> (N,  ), int64, 0-19,255
    data = torch.load(pth_path)
    points = data["coord"]
    colors = data["color"]
    labels = data["semantic_gt20"].squeeze()
    points = points.astype(np.float64)
    colors = colors.astype(np.float64) / 255.0
    assert len(points) == len(colors) == len(labels)

    labels = labels.astype(np.int64).squeeze()
    return points, colors, labels


def load_pred_labels(label_path):
    """
    Args:
        label_path: Path to the .txt file.
    Returns:
        labels: (N, ), int64, {1, 2, ..., 36, 39}.
    """
    def read_labels(label_path):
        labels = []
        with open(label_path, "r") as f:
            for line in f:
                labels.append(int(line.strip()))
        return np.array(labels)

    return np.array(read_labels(label_path))


def render_to_image(pcd, save_path):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save_path)


def visualize_scene_by_path(scene_path, rr_time, save_as_image=False):
    rr.set_time_seconds("time", rr_time)
    #label_dir = Path("data/scannet/val")
    #label_dir = Path("/home/yang/argos/pointcept/exp/scannet/semseg-pt-v3m1-0-base/result")
    label_dir = Path("/home/yang/argos/pointcept/exp/scannet/pretrain-msc-v2m1-4-noc-normal-pt-v3m1-0f-base/result")
    print(f"Visualizing {scene_path}")
    label_path = label_dir / f"{scene_path.stem}_pred.npy"

    # Load pcd and real labels.
    points, colors, real_labels = load_real_data(scene_path)

    # Visualize rgb colors
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if save_as_image:
        render_to_image(pcd, f"image/{scene_path.stem}_rgb.png")
    else:
        #o3d.visualization.draw_geometries([pcd], window_name="RGB colors")
        rr.log("world/rgb_pc", rr.Points3D(points, colors=colors))

    # Visualize real labels
    real_label_colors = np.array([_label_to_color[l] for l in real_labels])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(real_label_colors)
    if save_as_image:
        render_to_image(pcd, f"image/{scene_path.stem}_real.png")
    else:
        #o3d.visualization.draw_geometries([pcd], window_name="Real labels")
        rr.log("world/semseg_gt", rr.Points3D(points, colors=real_label_colors))

    # Load predicted labels
    pred_labels = np.load(label_path)
    pred_label_colors = np.array([_label_to_color[l] for l in pred_labels])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(pred_label_colors)
    if save_as_image:
        render_to_image(pcd, f"image/{scene_path.stem}_pred.png")
    else:
        #o3d.visualization.draw_geometries([pcd], window_name="Pred labels")
        rr.log("world/semseg_pred", rr.Points3D(points, colors=pred_label_colors))

def visualize_scene_by_name(scene_name, rr_time, save_as_image=False):
    data_root = Path("data") / "scannet" / "val"
    scene_paths = sorted(list(data_root.glob("*.pth")))

    found = False
    for scene_path in scene_paths:
        if scene_path.stem == scene_name:
            found = True
            visualize_scene_by_path(scene_path, rr_time, save_as_image=save_as_image)
            break

    if not found:
        raise ValueError(f"Scene {scene_name} not found.")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # rr.script_add_args(parser)
    # args = parser.parse_args()
    # rr.script_setup(args, "pointcept_visualization")
    blueprint = rrb.Blueprint(
        rrb.Grid(
            rrb.Spatial3DView(
                origin = "/world",
                contents= [
                    "+ $origin/semseg_gt"
                ],
                name = "semseg groundtruth"
            ),
            rrb.Spatial3DView(
                origin = "/world",
                contents = [
                    "+ $origin/semseg_pred"
                ],
                name = "semseg prediction"
            ),
            rrb.Spatial3DView(
                origin = "/world",
                contents = [
                    "+ $origin/rgb_pc"
                ],
                name = "rgb pointcloud"
            ),
            grid_columns = 2
        )

    )
    rr.init("scannet_rerun", spawn = True)
    rr.send_blueprint(blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, timeless=True)
    data_root = Path("data") / "scannet" / "val"
    scene_paths = sorted(list(data_root.glob("*.pth")))
    print(scene_paths)
    scene_names = [p.stem for p in scene_paths]
    rr_time = 0
    for scene_name in scene_names:
        print(scene_name)
        visualize_scene_by_name(scene_name, rr_time, save_as_image=True)
        rr_time += 1
    #rr.script_teardown(args)