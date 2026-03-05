from typing import Dict, List, Sequence, Tuple, Union

import mujoco
import numpy as np
from spatialmath import SE3

from .improved_adapter import get_improved_modules

GEOM_SPHERE = int(mujoco.mjtGeom.mjGEOM_SPHERE)
GEOM_BOX = int(mujoco.mjtGeom.mjGEOM_BOX)


def _name_selected(name: str, prefixes: Sequence[str], exact_names: Sequence[str]) -> bool:
    if name in exact_names:
        return True
    return any(name.startswith(prefix) for prefix in prefixes)


def _normalize_rotation(rot: np.ndarray) -> np.ndarray:
    """
    Project a near-rotation matrix to SO(3) to satisfy strict spatialmath checks.
    """
    r = np.asarray(rot, dtype=float).reshape(3, 3)
    if not np.isfinite(r).all():
        return np.eye(3)
    u, _, vt = np.linalg.svd(r)
    rn = u @ vt
    if np.linalg.det(rn) < 0:
        u[:, -1] *= -1.0
        rn = u @ vt
    return rn


def _world_to_base_pose(base_se3: SE3, world_pos: np.ndarray, world_rot: np.ndarray) -> SE3:
    r_bw = base_se3.R
    t_bw = base_se3.t
    pos_b = r_bw.T @ (np.asarray(world_pos) - t_bw)
    rot_b_raw = r_bw.T @ np.asarray(world_rot, dtype=float).reshape(3, 3)
    rot_b = _normalize_rotation(rot_b_raw)
    return SE3.Rt(rot_b, pos_b)


def _geom_world_aabb(center_w: np.ndarray, rot_w: np.ndarray, halfsize: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # AABB extents for oriented box in world frame: abs(R) @ halfsize
    ext = np.abs(rot_w) @ np.asarray(halfsize, dtype=float)
    return center_w - ext, center_w + ext


def _build_combined_shelf_obstacle(env, inflation: float):
    mods = get_improved_modules()
    ImprovedBrick = mods["ImprovedBrick"]

    model = env.mj_model
    data = env.mj_data
    base_se3 = env.robot.base

    mins = []
    maxs = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if not name or not name.startswith("shelf_layer_"):
            continue
        center_w = np.array(data.geom_xpos[gid], dtype=float)
        rot_w = _normalize_rotation(np.array(data.geom_xmat[gid], dtype=float).reshape(3, 3))
        # geom_size for box is half-size
        half = np.array(model.geom_size[gid], dtype=float)
        mn, mx = _geom_world_aabb(center_w, rot_w, half)
        mins.append(mn)
        maxs.append(mx)

    if not mins:
        return None

    mn = np.min(np.vstack(mins), axis=0)
    mx = np.max(np.vstack(maxs), axis=0)
    center_w = (mn + mx) * 0.5
    dims_w = (mx - mn) + 2.0 * float(inflation)
    dims_w = np.maximum(dims_w, np.array([0.02, 0.02, 0.02]))

    # Combined shelf uses world-axis box orientation.
    pose_b = _world_to_base_pose(base_se3, center_w, np.eye(3))
    return ImprovedBrick(pose_b, dims_w)


def build_rrt_obstacles(
    env,
    inflation: float = 0.03,
    combine_shelf_layers: bool = True,
    return_debug: bool = False,
) -> Union[List, Tuple[List, Dict]]:
    """
    Build improved_rrt_robot obstacle geometries from MuJoCo runtime geoms.

    Obstacles include:
    - Custom obstacle geoms: obstacle_*
    - Microscope collision geom: microscope_col
    - Shelf obstacle: merged shelf collision volume (default), or per-layer boxes
    """
    mods = get_improved_modules()
    ImprovedSphere = mods["ImprovedSphere"]
    ImprovedBrick = mods["ImprovedBrick"]

    model = env.mj_model
    data = env.mj_data
    base_se3 = env.robot.base

    include_prefixes = ("obstacle_",)
    include_exact = ("microscope_col",)

    obstacles = []
    debug = {
        "obstacle_geoms": [],
        "count_obstacle": 0,
        "count_microscope": 0,
        "count_shelf_layers": 0,
        "count_shelf_combined": 0,
    }
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if combine_shelf_layers and name and name.startswith("shelf_layer_"):
            debug["count_shelf_layers"] += 1
            continue
        if not name or not _name_selected(name, include_prefixes, include_exact):
            continue

        debug["obstacle_geoms"].append(name)
        if name.startswith("obstacle_"):
            debug["count_obstacle"] += 1
        elif name == "microscope_col":
            debug["count_microscope"] += 1

        gtype = int(model.geom_type[gid])
        pos_w = np.array(data.geom_xpos[gid], dtype=float)
        rot_w = np.array(data.geom_xmat[gid], dtype=float).reshape(3, 3)
        base_pose = _world_to_base_pose(base_se3, pos_w, rot_w)
        size = np.array(model.geom_size[gid], dtype=float)

        if gtype == GEOM_SPHERE:
            radius = max(0.001, float(size[0]) + inflation)
            obstacles.append(ImprovedSphere(base_pose, radius))
            continue

        if gtype == GEOM_BOX:
            dims = 2.0 * size + 2.0 * inflation
            dims = np.maximum(dims, np.array([0.002, 0.002, 0.002]))
            obstacles.append(ImprovedBrick(base_pose, dims))
            continue

        # For mesh/capsule/cylinder fallback to conservative bounding sphere.
        rbound = float(model.geom_rbound[gid])
        radius = max(0.001, rbound + inflation)
        obstacles.append(ImprovedSphere(base_pose, radius))

    if combine_shelf_layers:
        shelf = _build_combined_shelf_obstacle(env, inflation=inflation)
        if shelf is not None:
            obstacles.append(shelf)
            debug["count_shelf_combined"] = 1
    else:
        # Backward-compatible option: include every shelf layer as its own box.
        for gid in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            if not name or not name.startswith("shelf_layer_"):
                continue
            debug["count_shelf_layers"] += 1
            pos_w = np.array(data.geom_xpos[gid], dtype=float)
            rot_w = np.array(data.geom_xmat[gid], dtype=float).reshape(3, 3)
            base_pose = _world_to_base_pose(base_se3, pos_w, rot_w)
            size = np.array(model.geom_size[gid], dtype=float)
            dims = 2.0 * size + 2.0 * inflation
            dims = np.maximum(dims, np.array([0.002, 0.002, 0.002]))
            obstacles.append(ImprovedBrick(base_pose, dims))

    if return_debug:
        return obstacles, debug
    return obstacles


def get_joint_limits_from_env(env) -> List:
    limits = []
    for joint_name in getattr(env, "joint_names", []):
        jid = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            limits.append((-np.pi, np.pi))
            continue
        low, high = env.mj_model.jnt_range[jid]
        if not np.isfinite(low) or not np.isfinite(high):
            low, high = -np.pi, np.pi
        limits.append((float(low), float(high)))

    if not limits:
        limits = [(-np.pi, np.pi)] * 6
    return limits
