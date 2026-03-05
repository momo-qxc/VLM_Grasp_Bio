import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import spatialmath as sm

# Ensure "arm.*" imports work even when called outside GUI bootstrap.
_ROOT = Path(__file__).resolve().parents[1]
_MANIPULATOR_ROOT = _ROOT / "manipulator_grasp"
if str(_MANIPULATOR_ROOT) not in sys.path:
    sys.path.append(str(_MANIPULATOR_ROOT))

import grasp_process_optimized
from grasp_process_optimized import (
    _execute_planner_sequence,
    execute_grasp,
    record_object_origin,
    record_place,
)

from arm.motion_planning.trajectory_planning.trajectory_planner import TrajectoryPlanner
from arm.motion_planning.trajectory_planning.trajectory_parameter import TrajectoryParameter
from arm.motion_planning.trajectory_planning.velocity_planning.quintic_velocity_planning.quintic_velocity_parameter import (
    QuinticVelocityParameter,
)
from arm.motion_planning.trajectory_planning.path_planning.joint_planning.joint_parameter import (
    JointParameter,
)
from arm.motion_planning.trajectory_planning.path_planning.cartesian_planning.cartesian_parameter import (
    CartesianParameter,
)
from arm.motion_planning.trajectory_planning.path_planning.cartesian_planning.position_planning.line_position_planning.line_position_parameter import (
    LinePositionParameter,
)
from arm.motion_planning.trajectory_planning.path_planning.cartesian_planning.attitude_planning.one_attitude_planning.one_attitude_parameter import (
    OneAttitudeParameter,
)

from .improved_adapter import get_improved_modules
from .obstacle_extractor import build_rrt_obstacles, get_joint_limits_from_env

# Shelf constants aligned with current project logic.
SHELF_LAYER_HEIGHTS = [0.09, 0.414, 0.738, 1.053, 1.377]
SHELF_LAYER_TOLERANCE = 0.15
SHELF_X_MIN = 1.61
SHELF_X_MAX = 1.97
SHELF_APPROACH_OFFSET = 0.35

DEFAULT_RRT_CFG = {
    # Smaller expand step gives denser collision sampling and smoother RRT growth.
    "expand_dis": float(np.pi / 12),
    "goal_sample_rate": 45.0,
    "max_iter": 140,
    "max_iter_growth": 30,
    "max_cycles": 32,
    "execute_segments_per_cycle": 1,
    "segment_time": 0.55,
    "goal_tolerance": 0.06,
    "obstacle_inflation": 0.03,
    # Use a denser checker than planning expand_dis to reduce false negatives.
    "collision_check_expand_dis": float(np.pi / 18),
    "combine_shelf_layers": True,
    "subgoal_step_norm": 0.90,
    "local_joint_pad": 0.55,
    "local_joint_min_span": 0.70,
    "local_joint_max_span": 2.40,
    "failure_pad_gain": 0.25,
    "failure_subgoal_decay": 0.12,
    "failure_iter_gain": 25,
    "failure_global_after": 3,
    "enable_transit_waypoint": True,
    "transit_trigger_q_delta_max": 1.80,
    "transit_min_z": 1.55,
    # Direct-line fast path execution tuning.
    "direct_duration_per_rad": 1.25,
    "direct_min_duration": 0.60,
    "direct_max_duration": 2.80,
    # Keep fast path for short moves only; large moves should still use RRT.
    "direct_max_norm": 1.00,
    # Retry with a stronger global-search profile before declaring failure.
    "enable_rescue_replan": True,
    "rescue_cycles": 16,
    "rescue_max_iter": 320,
    "rescue_max_iter_growth": 60,
    "rescue_goal_sample_rate": 70.0,
    # Safety: avoid selecting IK targets known to be in collision.
    "allow_risky_goal_fallback": False,
    # Final approach/descend strategy near placement area.
    "enable_rrt_descent": True,
    "descent_step_height": 0.08,
    "descent_min_steps": 4,
    "descent_xy_offset_primary": 0.035,
    "descent_xy_offset_secondary": 0.060,
    "descent_xy_offset_tertiary": 0.090,
    "descent_max_joint_jump": 1.35,
    "descent_retry_levels": 3,
    "descent_retry_scale_step": 0.50,
    "descent_follow_reachable_xy": True,
    "descent_max_xy_drift": 0.18,
    "descent_relax_wrist3_on_fail": True,
    # Post-release retreat from cluttered target area.
    "enable_rrt_post_release_retreat": True,
    "post_release_raise_z": 0.25,
    # Return-home policy after task completion.
    "enable_rrt_return_home": True,
    # Wrist-3 lock policy for spill-sensitive transport (e.g., Petri dish with liquid).
    "enable_wrist3_lock": True,
    "wrist3_lock_deg": 180.0,
    "wrist3_align_before_rrt": True,
    "wrist3_align_tol_deg": 2.0,
    "wrist3_lock_align_duration": 0.7,
}


def _log(log_fn: Optional[Callable], level: str, msg: str) -> None:
    if log_fn is None:
        print(f"[{level}] {msg}")
        return
    try:
        log_fn(level, msg)
    except TypeError:
        log_fn(f"[{level}] {msg}")


def _as_so3(rot: sm.SO3) -> sm.SO3:
    if isinstance(rot, sm.SO3):
        return rot
    return sm.SO3(np.array(rot))


def _collision_expand_dis(cfg: Dict) -> float:
    expand_dis = float(cfg["expand_dis"])
    custom = cfg.get("collision_check_expand_dis", None)
    if custom is None:
        return expand_dis
    return max(1e-4, min(float(custom), expand_dis))


def _direct_move_duration(q0: np.ndarray, q1: np.ndarray, cfg: Dict) -> float:
    dist = float(np.linalg.norm(np.asarray(q1, dtype=float) - np.asarray(q0, dtype=float)))
    duration = dist * float(cfg.get("direct_duration_per_rad", 1.0))
    duration = max(duration, float(cfg.get("direct_min_duration", cfg["segment_time"])))
    duration = min(duration, float(cfg.get("direct_max_duration", 3.0)))
    return float(duration)


def _align_joint_angle_near_reference(angle: float, ref: float, low: float, high: float) -> float:
    best = float(np.clip(angle, low, high))
    best_dist = abs(best - ref)
    two_pi = 2.0 * np.pi
    if (high - low) >= (two_pi - 1e-3):
        for k in range(-5, 6):
            cand = angle + k * two_pi
            if cand < low or cand > high:
                continue
            d = abs(cand - ref)
            if d < best_dist:
                best_dist = d
                best = cand
    return float(np.clip(best, low, high))


def _apply_wrist3_lock(q: np.ndarray, wrist3_lock: Optional[float], area_limits) -> np.ndarray:
    q_locked = np.asarray(q, dtype=float).copy()
    if wrist3_lock is None:
        return q_locked
    if len(q_locked) < 6 or len(area_limits) < 6:
        return q_locked
    low, high = area_limits[5]
    q_locked[5] = _align_joint_angle_near_reference(float(wrist3_lock), float(q_locked[5]), low, high)
    return q_locked


def _resolve_wrist3_lock(robot, cfg: Dict, area_limits):
    if not bool(cfg.get("enable_wrist3_lock", False)):
        return None
    q_cur = np.asarray(robot.get_joint(), dtype=float)
    if len(q_cur) < 6 or len(area_limits) < 6:
        return None
    desired = np.deg2rad(float(cfg.get("wrist3_lock_deg", 180.0)))
    low, high = area_limits[5]
    return _align_joint_angle_near_reference(desired, float(q_cur[5]), low, high)


def _move_joint(
    env,
    robot,
    q_start: Sequence[float],
    q_end: Sequence[float],
    duration: float,
    gripper_ctrl: Optional[float] = None,
) -> None:
    planner = TrajectoryPlanner(
        TrajectoryParameter(
            JointParameter(np.asarray(q_start, dtype=float), np.asarray(q_end, dtype=float)),
            QuinticVelocityParameter(float(duration)),
        )
    )
    _execute_planner_sequence(env, robot, [planner], [0.0, float(duration)], gripper_ctrl=gripper_ctrl)


def _move_cartesian_line(
    env,
    robot,
    start_xyz: Sequence[float],
    end_xyz: Sequence[float],
    keep_rotation: sm.SO3,
    duration: float,
    gripper_ctrl: Optional[float] = None,
) -> None:
    keep_rotation = _as_so3(keep_rotation)
    planner = TrajectoryPlanner(
        TrajectoryParameter(
            CartesianParameter(
                LinePositionParameter(np.asarray(start_xyz, dtype=float), np.asarray(end_xyz, dtype=float)),
                OneAttitudeParameter(keep_rotation, keep_rotation),
            ),
            QuinticVelocityParameter(float(duration)),
        )
    )
    _execute_planner_sequence(env, robot, [planner], [0.0, float(duration)], gripper_ctrl=gripper_ctrl)


def _gripper_close(env, robot, steps: int = 1500) -> None:
    action = np.zeros(7)
    grip = 0.0
    for i in range(steps):
        action[:6] = robot.get_joint()
        grip = min(255.0, grip + 0.2)
        action[-1] = grip
        env.step(action)
        if grasp_process_optimized._render_callback is not None and i % grasp_process_optimized._RENDER_INTERVAL == 0:
            grasp_process_optimized._render_callback()


def _gripper_open(env, robot, steps: int = 2000) -> None:
    action = np.zeros(7)
    grip = 255.0
    for i in range(steps):
        action[:6] = robot.get_joint()
        grip = max(0.0, grip - 0.1)
        action[-1] = grip
        env.step(action)
        if grasp_process_optimized._render_callback is not None and i % grasp_process_optimized._RENDER_INTERVAL == 0:
            grasp_process_optimized._render_callback()


def _is_shelf_grasp(grasp_pos: Sequence[float]) -> Tuple[bool, int]:
    x, _, z = grasp_pos
    if x < SHELF_X_MIN or x > SHELF_X_MAX:
        return False, -1
    for i, layer_z in enumerate(SHELF_LAYER_HEIGHTS):
        if abs(z - layer_z) < SHELF_LAYER_TOLERANCE:
            return True, i
    return False, -1


def _compute_grasp_pose(gg, T_wc: Optional[sm.SE3]) -> sm.SE3:
    if T_wc is None:
        n_wc = np.array([0.0, -1.0, 0.0])
        o_wc = np.array([-1.0, 0.0, -0.5])
        t_wc = np.array([0.85, 0.8, 1.6])
        T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))
    T_co = sm.SE3.Trans(gg.translations[0]) * sm.SE3(
        sm.SO3.TwoVectors(x=gg.rotation_matrices[0][:, 0], y=gg.rotation_matrices[0][:, 1])
    )
    return T_wc * T_co


def _solve_ik_with_offsets(robot, base_pose: sm.SE3, offsets: Sequence[Sequence[float]]):
    for off in offsets:
        target = sm.SE3.Trans(base_pose.t + np.asarray(off, dtype=float)) * sm.SE3(sm.SO3(base_pose.R))
        q = robot.ikine(target)
        if len(q) > 0:
            return q, target
    return np.array([]), None


def _place_orientation_candidates(target_x: float, grasp_rotation: sm.SO3):
    # Reuse the project's existing placement orientation policy:
    # x < 0.7 => face-back vertical-down (Rz(pi)*Rx(pi))
    # x >= 0.7 => front vertical-down (Rx(pi))
    if target_x < 0.7:
        primary = sm.SO3((sm.SE3.Rz(np.pi) * sm.SE3.Rx(np.pi)).R)
        secondary = sm.SO3((sm.SE3.Rx(np.pi)).R)
    else:
        primary = sm.SO3((sm.SE3.Rx(np.pi)).R)
        secondary = sm.SO3((sm.SE3.Rz(np.pi) * sm.SE3.Rx(np.pi)).R)
    return [
        primary,
        secondary,
        sm.SO3((sm.SE3.Rz(np.pi / 2) * sm.SE3.Rx(np.pi)).R),
        sm.SO3((sm.SE3.Rz(-np.pi / 2) * sm.SE3.Rx(np.pi)).R),
        _as_so3(grasp_rotation),
    ]


def _goal_z_candidates(safe_z: float, target_z: float):
    z_candidates = []
    for z in [
        safe_z + 0.30,
        safe_z + 0.20,
        safe_z + 0.12,
        safe_z + 0.08,
        safe_z,
        max(safe_z - 0.08, target_z + 0.30),
        max(target_z + 0.45, 1.15),
        max(target_z + 0.35, 1.05),
        max(target_z + 0.28, 0.98),
    ]:
        if z > target_z + 0.18:
            z_candidates.append(float(z))

    uniq_z = []
    for z in z_candidates:
        if all(abs(z - uz) > 1e-6 for uz in uniq_z):
            uniq_z.append(z)
    return uniq_z


def _find_goal_joint(
    robot,
    target_pos: Sequence[float],
    safe_z: float,
    grasp_rotation: sm.SO3,
):
    tx = float(target_pos[0])
    ty = float(target_pos[1])
    tz = float(target_pos[2])

    uniq_z = _goal_z_candidates(safe_z=safe_z, target_z=tz)

    for rot in _place_orientation_candidates(tx, grasp_rotation):
        for z in uniq_z:
            T_goal = sm.SE3.Trans(tx, ty, z) * sm.SE3(rot)
            q_goal = robot.ikine(T_goal)
            if len(q_goal) > 0:
                return np.asarray(q_goal, dtype=float), T_goal, rot

    return np.array([]), None, None


def _legacy_shelf_grasp_and_retreat(
    env,
    robot,
    grasp_world_pos: np.ndarray,
    log_fn: Optional[Callable] = None,
):
    """
    Replicate the legacy shelf-grasp stages (approach/grasp/lift/retreat) from
    grasp_process_optimized as closely as possible, so grasp behavior matches
    the previously stable pipeline.
    Returns:
        q_home: initial joint at function entry
        grasp_rotation: SO3 at grasp moment
        T_retreat: SE3 pose after shelf retreat
    """
    q_home = np.asarray(robot.get_joint(), dtype=float)
    action = np.zeros(7)

    # Stage 1: pre-grasp joint posture.
    _log(log_fn, "STEP", "[RRT] 阶段1: 进入货架预抓取姿态（复用原抓取逻辑）")
    q1 = np.array([np.pi / 2, -np.pi / 4, np.pi / 2, -np.pi / 4, -np.pi / 2, 0.0])
    planner1 = TrajectoryPlanner(
        TrajectoryParameter(
            JointParameter(q_home, q1),
            QuinticVelocityParameter(1.0),
        )
    )
    _execute_planner_sequence(env, robot, [planner1], [0.0, 1.0])

    # Stage 2A: joint move to shelf align point.
    _log(log_fn, "STEP", "[RRT] 阶段2: 货架对准并水平伸入抓取（复用原抓取逻辑）")
    robot.set_joint(q1)
    approach_dir = np.array([1.0, 0.0, 0.0])
    side_dir = np.array([0.0, 1.0, 0.0])
    r_horizontal = sm.SO3.TwoVectors(x=approach_dir, y=side_dir)

    align_x = SHELF_X_MIN - SHELF_APPROACH_OFFSET
    align_point = np.array([align_x, grasp_world_pos[1], grasp_world_pos[2]])
    T_align = sm.SE3.Trans(align_point) * sm.SE3(r_horizontal)

    q_align = robot.ikine(T_align)
    if len(q_align) == 0:
        align_point_adjusted = align_point + np.array([0.1, 0.0, -0.05])
        T_align_adj = sm.SE3.Trans(align_point_adjusted) * sm.SE3(r_horizontal)
        q_align = robot.ikine(T_align_adj)
        if len(q_align) == 0:
            raise RuntimeError("RRT模式: 原逻辑货架对准点IK失败")
        T_align = T_align_adj

    planner_2a = TrajectoryPlanner(
        TrajectoryParameter(
            JointParameter(np.asarray(q1, dtype=float), np.asarray(q_align, dtype=float)),
            QuinticVelocityParameter(1.5),
        )
    )
    _execute_planner_sequence(env, robot, [planner_2a], [0.0, 1.5])
    robot.set_joint(q_align)

    # Stage 2B: Cartesian insertion to pre-grasp.
    T2_pos = grasp_world_pos - np.array([0.01, 0.0, 0.0])
    T2 = sm.SE3.Trans(T2_pos) * sm.SE3(r_horizontal)
    planner_2b = TrajectoryPlanner(
        TrajectoryParameter(
            CartesianParameter(
                LinePositionParameter(T_align.t, T2.t),
                OneAttitudeParameter(r_horizontal, r_horizontal),
            ),
            QuinticVelocityParameter(1.5),
        )
    )
    _execute_planner_sequence(env, robot, [planner_2b], [0.0, 1.5])

    # Stage 3: final insertion and grasp.
    T3_pos = grasp_world_pos + np.array([0.01, 0.0, 0.0])
    T3 = sm.SE3.Trans(T3_pos) * sm.SE3(r_horizontal)
    planner_3 = TrajectoryPlanner(
        TrajectoryParameter(
            CartesianParameter(
                LinePositionParameter(T2.t, T3.t),
                OneAttitudeParameter(r_horizontal, r_horizontal),
            ),
            QuinticVelocityParameter(1.0),
        )
    )
    _execute_planner_sequence(env, robot, [planner_3], [0.0, 1.0])

    _log(log_fn, "STEP", "[RRT] 阶段3: 闭合夹爪抓取（复用原抓取逻辑）")
    for i in range(1500):
        action[:6] = robot.get_joint()
        action[-1] += 0.2
        action[-1] = np.min([action[-1], 255.0])
        env.step(action)
        if grasp_process_optimized._render_callback is not None and i % grasp_process_optimized._RENDER_INTERVAL == 0:
            grasp_process_optimized._render_callback()

    # Stage 4: vertical lift then horizontal retreat.
    _log(log_fn, "STEP", "[RRT] 阶段4: 先垂直抬起，再水平退出到安全距离（复用原抓取逻辑）")
    T_grasp = robot.get_cartesian()
    grasp_rotation = sm.SO3(T_grasp.R)

    T4_up = sm.SE3.Trans(T3.t[0], T3.t[1], T3.t[2] + 0.12) * sm.SE3(grasp_rotation)
    planner_4_up = TrajectoryPlanner(
        TrajectoryParameter(
            CartesianParameter(
                LinePositionParameter(T3.t, T4_up.t),
                OneAttitudeParameter(grasp_rotation, grasp_rotation),
            ),
            QuinticVelocityParameter(1.0),
        )
    )
    _execute_planner_sequence(env, robot, [planner_4_up], [0.0, 1.0], gripper_ctrl=255.0)

    T4_retreat_pos = T4_up.t - np.array([0.3, 0.0, 0.0])
    T4 = sm.SE3.Trans(T4_retreat_pos) * sm.SE3(grasp_rotation)
    planner_4 = TrajectoryPlanner(
        TrajectoryParameter(
            CartesianParameter(
                LinePositionParameter(T4_up.t, T4.t),
                OneAttitudeParameter(grasp_rotation, grasp_rotation),
            ),
            QuinticVelocityParameter(1.0),
        )
    )
    _execute_planner_sequence(env, robot, [planner_4], [0.0, 1.0], gripper_ctrl=255.0)

    return q_home, grasp_rotation, T4


def _align_goal_near_current(q_cur: np.ndarray, q_goal: np.ndarray, area_limits):
    q_aligned = np.asarray(q_goal, dtype=float).copy()
    two_pi = 2.0 * np.pi
    for i, (low, high) in enumerate(area_limits):
        best = q_aligned[i]
        best_dist = abs(best - q_cur[i])
        # For joints with >= 2pi span, prefer the equivalent angle nearest current joint.
        if (high - low) >= (two_pi - 1e-3):
            for k in range(-3, 4):
                cand = q_goal[i] + k * two_pi
                if cand < low or cand > high:
                    continue
                d = abs(cand - q_cur[i])
                if d < best_dist:
                    best_dist = d
                    best = cand
        # Conservative clamp for finite-range joints.
        q_aligned[i] = float(np.clip(best, low, high))
    return q_aligned


def _compute_cycle_goal(q_cur: np.ndarray, q_goal: np.ndarray, subgoal_step_norm: float):
    if subgoal_step_norm <= 0:
        return q_goal, False
    diff = q_goal - q_cur
    dist = float(np.linalg.norm(diff))
    if dist <= subgoal_step_norm:
        return q_goal, False
    ratio = float(subgoal_step_norm / max(dist, 1e-9))
    q_sub = q_cur + ratio * diff
    return q_sub, True


def _build_local_joint_area(
    q_cur: np.ndarray,
    q_goal: np.ndarray,
    area_limits,
    pad: float,
    min_span: float,
    max_span: float,
):
    area = []
    for i, (low_lim, high_lim) in enumerate(area_limits):
        lo = max(float(low_lim), float(min(q_cur[i], q_goal[i]) - pad))
        hi = min(float(high_lim), float(max(q_cur[i], q_goal[i]) + pad))

        if hi - lo < min_span:
            center = 0.5 * (lo + hi)
            lo = max(float(low_lim), center - 0.5 * min_span)
            hi = min(float(high_lim), center + 0.5 * min_span)

        if hi - lo > max_span:
            center = 0.5 * (q_cur[i] + q_goal[i])
            lo = max(float(low_lim), center - 0.5 * max_span)
            hi = min(float(high_lim), center + 0.5 * max_span)

        if hi <= lo:
            lo, hi = float(low_lim), float(high_lim)
        area.append((float(lo), float(hi)))
    return area


def _execute_joint_segments(
    env,
    robot,
    planner_robot,
    obstacles,
    path_params,
    seg_n: int,
    segment_time: float,
    cfg: Dict,
    gripper_ctrl: float = 255.0,
    wrist3_lock: Optional[float] = None,
    log_fn: Optional[Callable] = None,
) -> bool:
    area_limits = get_joint_limits_from_env(env)
    checker = _make_collision_checker(planner_robot, obstacles, _collision_expand_dis(cfg))
    for idx in range(seg_n):
        seg = path_params[idx]
        q0 = np.asarray(seg.get_q0(), dtype=float)
        q1 = np.asarray(seg.get_q1(), dtype=float)
        if wrist3_lock is not None:
            q0 = _apply_wrist3_lock(q0, wrist3_lock, area_limits)
            q1 = _apply_wrist3_lock(q1, wrist3_lock, area_limits)
        if not _is_segment_collision_free(checker, q0, q1):
            _log(log_fn, "WARN", f"[RRT] 执行段 {idx + 1}/{seg_n} 前检测到碰撞，放弃本轮路径并重规划")
            return False
        _move_joint(
            env,
            robot,
            q0,
            q1,
            duration=segment_time,
            gripper_ctrl=float(gripper_ctrl),
        )
        q_now = np.asarray(robot.get_joint(), dtype=float)
        if not _is_segment_collision_free(checker, q_now, q_now):
            _log(log_fn, "WARN", f"[RRT] 执行段 {idx + 1}/{seg_n} 后姿态处于碰撞状态，下一轮立即重规划")
            return False
    return True


def _try_direct_step(
    env,
    robot,
    planner_robot,
    obstacles,
    q_cur: np.ndarray,
    q_goal_cycle: np.ndarray,
    cfg: Dict,
    gripper_ctrl: float = 255.0,
    wrist3_lock: Optional[float] = None,
    area_limits=None,
) -> bool:
    if area_limits is None:
        area_limits = get_joint_limits_from_env(env)
    q_goal_cycle = _apply_wrist3_lock(np.asarray(q_goal_cycle, dtype=float), wrist3_lock, area_limits)
    checker = _make_collision_checker(planner_robot, obstacles, _collision_expand_dis(cfg))

    dist = float(np.linalg.norm(q_goal_cycle - q_cur))
    if dist > float(cfg.get("direct_max_norm", 1.0)):
        return False

    LineSegment = get_improved_modules()["LineSegment"]
    if checker.check_collision(LineSegment(q_cur, q_goal_cycle)):
        return False

    # Execute as one smooth segment to avoid stop-go jitter that can shake loose held objects.
    _move_joint(
        env,
        robot,
        q_cur,
        q_goal_cycle,
        duration=_direct_move_duration(q_cur, q_goal_cycle, cfg),
        gripper_ctrl=float(gripper_ctrl),
    )
    return True


def _is_joint_collision_free(planner_robot, obstacles, q: np.ndarray, expand_dis: float) -> bool:
    mods = get_improved_modules()
    CheckCollisionRobot = mods["CheckCollisionRobot"]
    LineSegment = mods["LineSegment"]
    checker = CheckCollisionRobot(obstacles, float(expand_dis), planner_robot)
    return not checker.check_collision(LineSegment(q, q))


def _make_collision_checker(planner_robot, obstacles, expand_dis: float):
    mods = get_improved_modules()
    CheckCollisionRobot = mods["CheckCollisionRobot"]
    return CheckCollisionRobot(obstacles, float(expand_dis), planner_robot)


def _is_segment_collision_free(checker, q0: np.ndarray, q1: np.ndarray) -> bool:
    LineSegment = get_improved_modules()["LineSegment"]
    return not checker.check_collision(LineSegment(q0, q1))


def _find_collision_free_goal_joint(
    env,
    robot,
    target_pos: Sequence[float],
    safe_z: float,
    grasp_rotation: sm.SO3,
    cfg: Dict,
    wrist3_lock: Optional[float] = None,
    log_fn: Optional[Callable] = None,
):
    tx = float(target_pos[0])
    ty = float(target_pos[1])
    tz = float(target_pos[2])

    obstacles = build_rrt_obstacles(
        env,
        inflation=float(cfg["obstacle_inflation"]),
        combine_shelf_layers=bool(cfg.get("combine_shelf_layers", True)),
        return_debug=False,
    )
    planner_robot = get_improved_modules()["ImprovedRobot"]()
    checker = _make_collision_checker(planner_robot, obstacles, _collision_expand_dis(cfg))
    area_limits = get_joint_limits_from_env(env)
    q_ref = np.asarray(robot.get_joint(), dtype=float)

    fallback = None
    checked = 0
    collision_rejected = 0
    segment_rejected = 0
    candidates_segment_free = []
    candidates_point_free = []
    z_candidates = _goal_z_candidates(safe_z=safe_z, target_z=tz)
    for rot in _place_orientation_candidates(tx, grasp_rotation):
        for z in z_candidates:
            T_goal = sm.SE3.Trans(tx, ty, z) * sm.SE3(rot)
            q_goal = robot.ikine(T_goal)
            if len(q_goal) == 0:
                continue
            q_goal = _align_goal_near_current(q_ref, np.asarray(q_goal, dtype=float), area_limits)
            q_goal = _apply_wrist3_lock(q_goal, wrist3_lock, area_limits)
            checked += 1
            if fallback is None:
                fallback = (q_goal, T_goal, rot)
            if not _is_segment_collision_free(checker, q_goal, q_goal):
                collision_rejected += 1
                continue
            dq_norm = float(np.linalg.norm(q_goal - q_ref))
            dq_max = float(np.max(np.abs(q_goal - q_ref)))
            candidates_point_free.append((dq_max, dq_norm, q_goal, T_goal, rot))
            if _is_segment_collision_free(checker, q_ref, q_goal):
                candidates_segment_free.append((dq_max, dq_norm, q_goal, T_goal, rot))
            else:
                segment_rejected += 1

    if candidates_segment_free:
        candidates_segment_free.sort(key=lambda x: (x[0], x[1]))
        _, _, q_goal, T_goal, rot = candidates_segment_free[0]
        _log(
            log_fn,
            "INFO",
            f"[RRT] 放置上方候选筛选: 检查{checked}个候选,"
            f" 点碰撞通过{len(candidates_point_free)}个, 直连通过{len(candidates_segment_free)}个, 选最近候选",
        )
        return q_goal, T_goal, rot

    if candidates_point_free:
        candidates_point_free.sort(key=lambda x: (x[0], x[1]))
        _, _, q_goal, T_goal, rot = candidates_point_free[0]
        _log(
            log_fn,
            "WARN",
            f"[RRT] 放置上方候选筛选: 检查{checked}个候选,"
            f" 点碰撞通过{len(candidates_point_free)}个但直连均不通(拒绝{segment_rejected}个),"
            " 回退到最近点可行候选",
        )
        return q_goal, T_goal, rot

    if fallback is not None and bool(cfg.get("allow_risky_goal_fallback", False)):
        _log(
            log_fn,
            "WARN",
            f"[RRT] 放置上方候选筛选: {collision_rejected}个候选点均碰撞，回退到首个IK可达目标（高风险）",
        )
        return fallback

    if fallback is not None:
        _log(
            log_fn,
            "WARN",
            f"[RRT] 放置上方候选筛选: {collision_rejected}个候选点均碰撞，已禁用高风险回退，请调整目标点或重试",
        )

    return np.array([]), None, None


def _find_transit_joint(
    env,
    robot,
    target_pos: Sequence[float],
    grasp_rotation: sm.SO3,
    cfg: Dict,
    wrist3_lock: Optional[float] = None,
    log_fn: Optional[Callable] = None,
):
    obstacles = build_rrt_obstacles(
        env,
        inflation=float(cfg["obstacle_inflation"]),
        combine_shelf_layers=bool(cfg.get("combine_shelf_layers", True)),
        return_debug=False,
    )
    planner_robot = get_improved_modules()["ImprovedRobot"]()
    checker = _make_collision_checker(planner_robot, obstacles, _collision_expand_dis(cfg))
    area_limits = get_joint_limits_from_env(env)
    q_ref = np.asarray(robot.get_joint(), dtype=float)

    cur_pose = robot.get_cartesian()
    z_mid = max(
        float(cur_pose.t[2]) + 0.05,
        float(target_pos[2]) + 0.70,
        float(cfg.get("transit_min_z", 1.50)),
    )
    x_candidates = [1.25, 1.10, 0.98]
    y_candidates = [0.35, 0.45, 0.55]
    rot_candidates = [
        _as_so3(grasp_rotation),
        sm.SO3((sm.SE3.Rx(np.pi)).R),
        sm.SO3((sm.SE3.Rz(np.pi) * sm.SE3.Rx(np.pi)).R),
    ]

    seg_free = []
    point_free = []
    checked = 0
    for x in x_candidates:
        for y in y_candidates:
            for rot in rot_candidates:
                T_mid = sm.SE3.Trans(float(x), float(y), float(z_mid)) * sm.SE3(rot)
                q_mid = robot.ikine(T_mid)
                if len(q_mid) == 0:
                    continue
                q_mid = _align_goal_near_current(q_ref, np.asarray(q_mid, dtype=float), area_limits)
                q_mid = _apply_wrist3_lock(q_mid, wrist3_lock, area_limits)
                checked += 1
                if not _is_segment_collision_free(checker, q_mid, q_mid):
                    continue
                dq_norm = float(np.linalg.norm(q_mid - q_ref))
                dq_max = float(np.max(np.abs(q_mid - q_ref)))
                point_free.append((dq_max, dq_norm, q_mid, T_mid))
                if _is_segment_collision_free(checker, q_ref, q_mid):
                    seg_free.append((dq_max, dq_norm, q_mid, T_mid))

    if seg_free:
        seg_free.sort(key=lambda x: (x[0], x[1]))
        _, _, q_mid, T_mid = seg_free[0]
        _log(log_fn, "INFO", f"[RRT] 中转点筛选: 检查{checked}个候选，选中直连可行中转点")
        return q_mid, T_mid
    if point_free:
        point_free.sort(key=lambda x: (x[0], x[1]))
        _, _, q_mid, T_mid = point_free[0]
        _log(log_fn, "WARN", f"[RRT] 中转点筛选: 仅找到点可行中转点（无直连），将尝试RRT到中转点")
        return q_mid, T_mid
    _log(log_fn, "WARN", "[RRT] 中转点筛选: 未找到可用中转点")
    return np.array([]), None


def _descent_xy_offsets(cfg: Dict, scale: float = 1.0):
    scale = max(0.0, float(scale))
    radii = []
    for key in ("descent_xy_offset_primary", "descent_xy_offset_secondary", "descent_xy_offset_tertiary"):
        r = max(0.0, float(cfg.get(key, 0.0))) * scale
        if r <= 1e-9:
            continue
        if any(abs(r - existed) < 1e-9 for existed in radii):
            continue
        radii.append(r)

    raw = [(0.0, 0.0)]
    for r in radii:
        raw.extend(
            [
                (r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r),
                (r, r), (r, -r), (-r, r), (-r, -r),
            ]
        )

    offsets = []
    for dx, dy in raw:
        if any(abs(dx - ox) < 1e-9 and abs(dy - oy) < 1e-9 for ox, oy in offsets):
            continue
        offsets.append((float(dx), float(dy)))
    return offsets


def _find_rrt_reachable_pose_joint(
    env,
    robot,
    base_xy: Sequence[float],
    target_z: float,
    target_rot: sm.SO3,
    cfg: Dict,
    wrist3_lock: Optional[float] = None,
    offset_scale: float = 1.0,
    *,
    log_tag: str = "接近",
):
    obstacles = build_rrt_obstacles(
        env,
        inflation=float(cfg["obstacle_inflation"]),
        combine_shelf_layers=bool(cfg.get("combine_shelf_layers", True)),
        return_debug=False,
    )
    planner_robot = get_improved_modules()["ImprovedRobot"]()
    checker = _make_collision_checker(planner_robot, obstacles, _collision_expand_dis(cfg))
    area_limits = get_joint_limits_from_env(env)
    q_ref = np.asarray(robot.get_joint(), dtype=float)
    max_jump = float(cfg.get("descent_max_joint_jump", 10.0))

    seg_free = []
    point_free = []
    checked = 0
    rejected_jump = 0
    rejected_collision = 0
    rejected_ik = 0
    for dx, dy in _descent_xy_offsets(cfg, scale=offset_scale):
        tx = float(base_xy[0]) + dx
        ty = float(base_xy[1]) + dy
        T_try = sm.SE3.Trans(tx, ty, float(target_z)) * sm.SE3(target_rot)
        q_try = robot.ikine(T_try)
        if len(q_try) == 0:
            rejected_ik += 1
            continue

        q_try = _align_goal_near_current(q_ref, np.asarray(q_try, dtype=float), area_limits)
        q_try = _apply_wrist3_lock(q_try, wrist3_lock, area_limits)
        checked += 1

        dq_norm = float(np.linalg.norm(q_try - q_ref))
        dq_max = float(np.max(np.abs(q_try - q_ref)))
        if dq_max > max_jump:
            rejected_jump += 1
            continue

        if not _is_segment_collision_free(checker, q_try, q_try):
            rejected_collision += 1
            continue

        offset_norm = float(np.hypot(dx, dy))
        point_free.append((offset_norm, dq_max, dq_norm, q_try, T_try, (dx, dy)))
        if _is_segment_collision_free(checker, q_ref, q_try):
            seg_free.append((offset_norm, dq_max, dq_norm, q_try, T_try, (dx, dy)))

    if seg_free:
        seg_free.sort(key=lambda x: (x[0], x[1], x[2]))
        off_norm, _, _, q_goal, T_goal, offset_xy = seg_free[0]
        return q_goal, T_goal, offset_xy, "segment_free", checked, off_norm
    if point_free:
        point_free.sort(key=lambda x: (x[0], x[1], x[2]))
        off_norm, _, _, q_goal, T_goal, offset_xy = point_free[0]
        return q_goal, T_goal, offset_xy, "point_free", checked, off_norm

    return (
        np.array([]),
        None,
        None,
        (
            "none("
            f"scale={float(offset_scale):.2f}, "
            f"checked={checked}, ik_rej={rejected_ik}, jump_rej={rejected_jump}, col_rej={rejected_collision}"
            ")"
        ),
        checked,
        float("inf"),
    )


def _rrt_descend_to_place(
    env,
    robot,
    target_pos: Sequence[float],
    drop_rot: sm.SO3,
    cfg: Dict,
    wrist3_lock: Optional[float] = None,
    log_fn: Optional[Callable] = None,
):
    T_now = robot.get_cartesian()
    z_start = float(T_now.t[2])
    z_goal = float(target_pos[2])
    if z_start <= z_goal + 1e-5:
        return

    dz = z_start - z_goal
    min_steps = max(1, int(cfg.get("descent_min_steps", 3)))
    step_height = max(1e-3, float(cfg.get("descent_step_height", 0.08)))
    n_steps = max(min_steps, int(np.ceil(dz / step_height)))
    z_levels = np.linspace(z_start, z_goal, n_steps + 1)[1:]
    target_xy = np.array([float(target_pos[0]), float(target_pos[1])], dtype=float)
    base_xy = target_xy.copy()
    retry_levels = max(1, int(cfg.get("descent_retry_levels", 3)))
    retry_scale_step = max(0.0, float(cfg.get("descent_retry_scale_step", 0.5)))
    follow_reachable_xy = bool(cfg.get("descent_follow_reachable_xy", True))
    max_xy_drift = max(0.0, float(cfg.get("descent_max_xy_drift", 0.18)))
    relax_wrist3_on_fail = bool(cfg.get("descent_relax_wrist3_on_fail", True))
    active_wrist3_lock = wrist3_lock

    for i, zi in enumerate(z_levels, start=1):
        q_step = np.array([])
        T_step = None
        offset_xy = (0.0, 0.0)
        mode = "none(uninitialized)"
        checked = 0
        off_norm = float("inf")
        used_scale = 1.0
        used_center = base_xy.copy()
        center_candidates = [base_xy]
        if (
            follow_reachable_xy
            and float(np.linalg.norm(base_xy - target_xy)) > 1e-6
            and i == len(z_levels)
        ):
            center_candidates = [target_xy, base_xy]

        # Primary pass: keep current wrist_3 lock and expand XY offset radius on retries.
        for center_xy in center_candidates:
            for retry_idx in range(retry_levels):
                used_scale = 1.0 + retry_idx * retry_scale_step
                q_step, T_step, offset_xy, mode, checked, off_norm = _find_rrt_reachable_pose_joint(
                    env=env,
                    robot=robot,
                    base_xy=(float(center_xy[0]), float(center_xy[1])),
                    target_z=float(zi),
                    target_rot=drop_rot,
                    cfg=cfg,
                    wrist3_lock=active_wrist3_lock,
                    offset_scale=used_scale,
                    log_tag=f"下降层{i}",
                )
                if len(q_step) > 0:
                    used_center = np.array(center_xy, dtype=float)
                    break
            if len(q_step) > 0:
                break

        # Fallback pass: if still blocked, temporarily relax wrist_3 lock for descent only.
        if len(q_step) == 0 and active_wrist3_lock is not None and relax_wrist3_on_fail:
            _log(
                log_fn,
                "WARN",
                f"[RRT] 阶段6.{i}: 当前wrist_3锁定下降失败，尝试临时放宽锁定后重试",
            )
            for center_xy in center_candidates:
                for retry_idx in range(retry_levels):
                    used_scale = 1.0 + retry_idx * retry_scale_step
                    q_step, T_step, offset_xy, mode, checked, off_norm = _find_rrt_reachable_pose_joint(
                        env=env,
                        robot=robot,
                        base_xy=(float(center_xy[0]), float(center_xy[1])),
                        target_z=float(zi),
                        target_rot=drop_rot,
                        cfg=cfg,
                        wrist3_lock=None,
                        offset_scale=used_scale,
                        log_tag=f"下降层{i}",
                    )
                    if len(q_step) > 0:
                        used_center = np.array(center_xy, dtype=float)
                        active_wrist3_lock = None
                        _log(log_fn, "WARN", f"[RRT] 阶段6.{i}: 已临时放宽wrist_3锁定以完成下降")
                        break
                if len(q_step) > 0:
                    break

        if len(q_step) == 0:
            raise RuntimeError(
                "RRT模式: 阶段6下降"
                f"第{i}/{len(z_levels)}层未找到可行无碰撞姿态 | z={float(zi):.3f}"
                f" | base_xy=({float(base_xy[0]):.3f}, {float(base_xy[1]):.3f})"
                f" | retry={retry_levels}, scale_step={retry_scale_step:.2f}"
                f" | {mode}"
            )

        center_shift = used_center - target_xy
        center_shift_norm = float(np.linalg.norm(center_shift))
        if mode == "point_free":
            _log(
                log_fn,
                "WARN",
                f"[RRT] 阶段6.{i}: 候选点仅点可行(checked={checked})，"
                f"将通过RRT绕行到 z={float(zi):.3f} (offset_scale={used_scale:.2f}, "
                f"center_shift={center_shift_norm:.3f})",
            )
        else:
            lock_desc = "unlock" if active_wrist3_lock is None else "lock"
            _log(
                log_fn,
                "INFO",
                f"[RRT] 阶段6.{i}: 选中{mode}候选(checked={checked}), "
                f"z={float(zi):.3f}, xy偏移=({offset_xy[0]:+.3f},{offset_xy[1]:+.3f}), "
                f"|off|={off_norm:.3f}, offset_scale={used_scale:.2f}, wrist3={lock_desc}, "
                f"center_shift={center_shift_norm:.3f}",
            )
        _online_rrt_move_to_goal(
            env,
            q_step,
            cfg,
            gripper_ctrl=255.0,
            wrist3_lock=active_wrist3_lock,
            log_fn=log_fn,
        )

        # Follow the actually reachable XY to avoid being repeatedly pulled back
        # into a blocked center at lower Z levels.
        if follow_reachable_xy and T_step is not None:
            next_xy = np.array([float(T_step.t[0]), float(T_step.t[1])], dtype=float)
            drift = next_xy - target_xy
            drift_norm = float(np.linalg.norm(drift))
            if max_xy_drift > 0.0 and drift_norm > max_xy_drift:
                next_xy = target_xy + drift * (max_xy_drift / max(drift_norm, 1e-9))
                _log(
                    log_fn,
                    "WARN",
                    f"[RRT] 阶段6.{i}: 可达点偏移过大，已限制到max_xy_drift={max_xy_drift:.3f}",
                )
            base_xy = next_xy


def _online_rrt_move_to_goal(
    env,
    q_goal: Sequence[float],
    cfg: Dict,
    gripper_ctrl: float = 255.0,
    wrist3_lock: Optional[float] = None,
    log_fn: Optional[Callable] = None,
) -> None:
    mods = get_improved_modules()
    ImprovedRobot = mods["ImprovedRobot"]
    RRTMap = mods["RRTMap"]
    RobotRRTParameter = mods["RobotRRTParameter"]
    RRTPlanner = mods["RRTPlanner"]

    area_limits = get_joint_limits_from_env(env)
    planner_robot = ImprovedRobot()
    robot = env.robot
    q_goal = np.asarray(q_goal, dtype=float)
    fail_streak = 0

    for cycle in range(1, int(cfg["max_cycles"]) + 1):
        cycle_t0 = time.perf_counter()
        q_cur = np.asarray(robot.get_joint(), dtype=float)
        q_goal = _align_goal_near_current(q_cur, q_goal, area_limits)
        q_goal = _apply_wrist3_lock(q_goal, wrist3_lock, area_limits)
        q_err = float(np.max(np.abs(q_cur - q_goal)))
        if q_err < float(cfg["goal_tolerance"]):
            _log(log_fn, "SUCCESS", f"[RRT] 已到达目标，关节误差={q_err:.4f}")
            return

        # Adaptive strategy after consecutive failures:
        # increase exploration window and reduce subgoal length.
        effective_subgoal = float(cfg["subgoal_step_norm"]) * (
            1.0 - fail_streak * float(cfg.get("failure_subgoal_decay", 0.0))
        )
        effective_subgoal = max(0.35, effective_subgoal)
        q_goal_cycle, use_subgoal = _compute_cycle_goal(q_cur, q_goal, effective_subgoal)

        pad_gain = float(cfg.get("failure_pad_gain", 0.0))
        effective_pad = float(cfg["local_joint_pad"]) * (1.0 + pad_gain * fail_streak)
        use_global_area = fail_streak >= int(cfg.get("failure_global_after", 99))
        if use_global_area:
            area = area_limits
        else:
            area = _build_local_joint_area(
                q_cur=q_cur,
                q_goal=q_goal_cycle,
                area_limits=area_limits,
                pad=effective_pad,
                min_span=float(cfg["local_joint_min_span"]),
                max_span=float(cfg["local_joint_max_span"]),
            )

        obstacles, obs_debug = build_rrt_obstacles(
            env,
            inflation=float(cfg["obstacle_inflation"]),
            combine_shelf_layers=bool(cfg.get("combine_shelf_layers", True)),
            return_debug=True,
        )
        shelf_desc = (
            "合并货架=1"
            if obs_debug["count_shelf_combined"] > 0
            else f"货架层={obs_debug['count_shelf_layers']}"
        )
        _log(
            log_fn,
            "INFO",
            f"[RRT] Cycle {cycle}: 障碍物数量={len(obstacles)}"
            f" (球体={obs_debug['count_obstacle']}, 显微镜={obs_debug['count_microscope']}, {shelf_desc})"
            f" | {'子目标' if use_subgoal else '目标'}规划"
            f" | fail_streak={fail_streak}"
            f" | q_err={q_err:.3f}"
            f"{' | 全局搜索域' if use_global_area else ''}",
        )

        # Fast path: for short goals, execute one smooth direct segment if collision-free.
        if _try_direct_step(
            env=env,
            robot=robot,
            planner_robot=planner_robot,
            obstacles=obstacles,
            q_cur=q_cur,
            q_goal_cycle=q_goal_cycle,
            cfg=cfg,
            gripper_ctrl=float(gripper_ctrl),
            wrist3_lock=wrist3_lock,
            area_limits=area_limits,
        ):
            fail_streak = 0
            dt = time.perf_counter() - cycle_t0
            _log(log_fn, "INFO", f"[RRT] Cycle {cycle}: 直连无碰撞，已执行平滑直连运动（耗时{dt:.2f}s）")
            continue

        cycle_max_iter = (
            int(cfg["max_iter"])
            + (cycle - 1) * int(cfg.get("max_iter_growth", 0))
            + fail_streak * int(cfg.get("failure_iter_gain", 0))
        )
        rrt_map = RRTMap(area=area, obstacles=obstacles)
        param = RobotRRTParameter(
            start=q_cur,
            goal=q_goal_cycle,
            robot=planner_robot,
            expand_dis=float(cfg["expand_dis"]),
            goal_sample_rate=float(cfg["goal_sample_rate"]),
            max_iter=cycle_max_iter,
        )
        planner = RRTPlanner(rrt_map, param)
        if not planner.success:
            fail_streak += 1
            dt = time.perf_counter() - cycle_t0
            _log(log_fn, "WARN", f"[RRT] Cycle {cycle}: 规划失败（耗时{dt:.2f}s），准备下一轮重规划")
            continue

        path_params = planner.get_path_parameters()
        if not path_params:
            fail_streak += 1
            dt = time.perf_counter() - cycle_t0
            _log(log_fn, "WARN", f"[RRT] Cycle {cycle}: 空路径（耗时{dt:.2f}s），准备下一轮重规划")
            continue

        seg_n = min(int(cfg["execute_segments_per_cycle"]), len(path_params))
        _log(
            log_fn,
            "INFO",
            f"[RRT] Cycle {cycle}: 规划到 {len(path_params)} 段，本轮执行 {seg_n} 段（边规划边运动）",
        )
        executed_ok = _execute_joint_segments(
            env=env,
            robot=robot,
            planner_robot=planner_robot,
            obstacles=obstacles,
            path_params=path_params,
            seg_n=seg_n,
            segment_time=float(cfg["segment_time"]),
            cfg=cfg,
            gripper_ctrl=float(gripper_ctrl),
            wrist3_lock=wrist3_lock,
            log_fn=log_fn,
        )
        if not executed_ok:
            fail_streak += 1
            dt = time.perf_counter() - cycle_t0
            _log(log_fn, "WARN", f"[RRT] Cycle {cycle}: 执行阶段触发安全重规划（耗时{dt:.2f}s）")
            continue
        fail_streak = 0
        dt = time.perf_counter() - cycle_t0
        _log(log_fn, "INFO", f"[RRT] Cycle {cycle}: 本轮完成（耗时{dt:.2f}s）")

    q_cur = np.asarray(robot.get_joint(), dtype=float)
    q_err = float(np.max(np.abs(q_cur - q_goal)))
    if bool(cfg.get("enable_rescue_replan", True)):
        rescue_cfg = dict(cfg)
        rescue_cfg.update(
            {
                "enable_rescue_replan": False,
                "max_cycles": int(cfg.get("rescue_cycles", 12)),
                "max_iter": max(int(cfg["max_iter"]), int(cfg.get("rescue_max_iter", 280))),
                "max_iter_growth": max(
                    int(cfg.get("max_iter_growth", 0)),
                    int(cfg.get("rescue_max_iter_growth", 40)),
                ),
                "goal_sample_rate": max(
                    float(cfg["goal_sample_rate"]),
                    float(cfg.get("rescue_goal_sample_rate", 65.0)),
                ),
                "failure_global_after": 0,
                "subgoal_step_norm": 0.0,
                # Rescue uses full RRT to escape local minima; keep direct fast path disabled.
                "direct_max_norm": 0.0,
            }
        )
        _log(
            log_fn,
            "WARN",
            f"[RRT] 主循环未收敛（q_err={q_err:.4f}），启动救援重规划 "
            f"(cycles={rescue_cfg['max_cycles']}, max_iter={rescue_cfg['max_iter']})",
        )
        _online_rrt_move_to_goal(
            env,
            q_goal,
            rescue_cfg,
            gripper_ctrl=float(gripper_ctrl),
            wrist3_lock=wrist3_lock,
            log_fn=log_fn,
        )
        return

    raise RuntimeError(
        f"在线RRT在最大循环次数内未收敛（max_cycles={cfg['max_cycles']}，最终误差={q_err:.4f}）"
    )


def execute_grasp_with_online_rrt(
    env,
    gg,
    T_wc: Optional[sm.SE3] = None,
    target_pos: Optional[Sequence[float]] = None,
    object_name: Optional[str] = None,
    log_fn: Optional[Callable] = None,
    rrt_cfg: Optional[Dict] = None,
) -> None:
    """
    Obstacle-scene executor using improved_rrt_robot with online planning:
    plan a short segment -> move a short segment -> replan.

    This function intentionally lives outside existing grasp_process_optimized code path.
    """
    cfg = dict(DEFAULT_RRT_CFG)
    if rrt_cfg:
        cfg.update(rrt_cfg)

    if target_pos is None:
        target_pos = [0.2, 0.2, 0.92]
        _log(log_fn, "WARN", f"[RRT] 未检测到放置点，使用默认目标: {target_pos}")

    robot = env.robot
    q_home = np.asarray(robot.get_joint(), dtype=float)

    T_wo = _compute_grasp_pose(gg, T_wc)
    grasp_world_pos = np.asarray(T_wo.t, dtype=float)
    is_shelf, shelf_layer = _is_shelf_grasp(grasp_world_pos)
    _log(
        log_fn,
        "INFO",
        f"[RRT] 抓取点: ({grasp_world_pos[0]:.3f}, {grasp_world_pos[1]:.3f}, {grasp_world_pos[2]:.3f})"
        f" | shelf={is_shelf} layer={shelf_layer + 1 if shelf_layer >= 0 else '-'}",
    )

    # Fallback for non-shelf grasps. The user's requested path is shelf grasp.
    if not is_shelf:
        _log(log_fn, "WARN", "[RRT] 当前不是货架抓取，回退到原始执行器")
        execute_grasp(env, gg, T_wc=T_wc, target_pos=list(target_pos), object_name=object_name, desktop_fusion=False)
        return

    if object_name:
        record_object_origin(object_name, grasp_world_pos, is_shelf, shelf_layer)

    q_home, grasp_rotation, T_retreat = _legacy_shelf_grasp_and_retreat(
        env=env,
        robot=robot,
        grasp_world_pos=grasp_world_pos,
        log_fn=log_fn,
    )

    area_limits = get_joint_limits_from_env(env)
    wrist3_lock = _resolve_wrist3_lock(robot, cfg, area_limits)
    if wrist3_lock is not None:
        _log(log_fn, "INFO", f"[RRT] 启用wrist_3锁定: {np.degrees(wrist3_lock):.1f}°")
        if bool(cfg.get("wrist3_align_before_rrt", True)):
            q_cur_lock = np.asarray(robot.get_joint(), dtype=float)
            q_align_lock = _apply_wrist3_lock(q_cur_lock, wrist3_lock, area_limits)
            align_tol = np.deg2rad(float(cfg.get("wrist3_align_tol_deg", 2.0)))
            if abs(float(q_cur_lock[5] - q_align_lock[5])) > align_tol:
                _log(
                    log_fn,
                    "STEP",
                    f"[RRT] 阶段4A: 先对齐wrist_3到锁定角 ({np.degrees(q_align_lock[5]):.1f}°)",
                )
                _move_joint(
                    env,
                    robot,
                    q_cur_lock,
                    q_align_lock,
                    duration=float(cfg.get("wrist3_lock_align_duration", 0.7)),
                    gripper_ctrl=255.0,
                )

    # 5) Online RRT transit to target-above pose.
    _log(log_fn, "STEP", "[RRT] 阶段5: 开始在线RRT（规划一段，执行一段）")
    safe_z = max(float(T_retreat.t[2]), float(target_pos[2]) + 0.30, 1.20)
    q_goal, T_goal_high, place_rot = _find_collision_free_goal_joint(
        env=env,
        robot=robot,
        target_pos=target_pos,
        safe_z=safe_z,
        grasp_rotation=grasp_rotation,
        cfg=cfg,
        wrist3_lock=wrist3_lock,
        log_fn=log_fn,
    )
    if len(q_goal) == 0:
        raise RuntimeError(
            f"RRT模式: 放置上方目标点IK失败 | target=({float(target_pos[0]):.3f}, {float(target_pos[1]):.3f}, {float(target_pos[2]):.3f})"
        )
    _log(
        log_fn,
        "INFO",
        f"[RRT] 放置上方IK成功: ({T_goal_high.t[0]:.3f}, {T_goal_high.t[1]:.3f}, {T_goal_high.t[2]:.3f})",
    )
    q_now = np.asarray(robot.get_joint(), dtype=float)
    q_delta_max = float(np.max(np.abs(q_goal - q_now)))
    q_delta_norm = float(np.linalg.norm(q_goal - q_now))
    _log(log_fn, "INFO", f"[RRT] 目标关节差值: max={q_delta_max:.3f}, norm={q_delta_norm:.3f}")

    # If target joint gap is too large, insert a transit waypoint to avoid getting stuck in one narrow corridor.
    if bool(cfg.get("enable_transit_waypoint", True)) and q_delta_max > float(cfg.get("transit_trigger_q_delta_max", 2.2)):
        _log(log_fn, "STEP", "[RRT] 阶段5A: 目标关节跨度较大，先规划到中转安全点")
        q_mid, T_mid = _find_transit_joint(
            env=env,
            robot=robot,
            target_pos=target_pos,
            grasp_rotation=grasp_rotation,
            cfg=cfg,
            wrist3_lock=wrist3_lock,
            log_fn=log_fn,
        )
        if len(q_mid) > 0:
            _log(
                log_fn,
                "INFO",
                f"[RRT] 中转点: ({T_mid.t[0]:.3f}, {T_mid.t[1]:.3f}, {T_mid.t[2]:.3f})，先执行在线RRT中转",
            )
            _online_rrt_move_to_goal(
                env,
                q_mid,
                cfg,
                gripper_ctrl=255.0,
                wrist3_lock=wrist3_lock,
                log_fn=log_fn,
            )
            # Re-select final goal from the transit posture.
            q_goal, T_goal_high, place_rot = _find_collision_free_goal_joint(
                env=env,
                robot=robot,
                target_pos=target_pos,
                safe_z=safe_z,
                grasp_rotation=grasp_rotation,
                cfg=cfg,
                wrist3_lock=wrist3_lock,
                log_fn=log_fn,
            )
            if len(q_goal) == 0:
                raise RuntimeError("RRT模式: 中转后未找到可行放置上方目标")
            q_now = np.asarray(robot.get_joint(), dtype=float)
            q_delta_max = float(np.max(np.abs(q_goal - q_now)))
            q_delta_norm = float(np.linalg.norm(q_goal - q_now))
            _log(log_fn, "INFO", f"[RRT] 中转后目标关节差值: max={q_delta_max:.3f}, norm={q_delta_norm:.3f}")
        else:
            _log(log_fn, "WARN", "[RRT] 未找到可用中转点，直接尝试到放置上方目标")

    _online_rrt_move_to_goal(
        env,
        q_goal,
        cfg,
        gripper_ctrl=255.0,
        wrist3_lock=wrist3_lock,
        log_fn=log_fn,
    )

    # 6) Lower to place point (RRT-aware descent).
    _log(log_fn, "STEP", "[RRT] 阶段6: 下降放置")
    T_now = robot.get_cartesian()
    # Keep the orientation used by the high-point IK to avoid a sudden rotation at drop stage.
    drop_rot = place_rot if place_rot is not None else grasp_rotation
    T_drop = sm.SE3.Trans(float(target_pos[0]), float(target_pos[1]), float(target_pos[2])) * sm.SE3(drop_rot)
    if bool(cfg.get("enable_rrt_descent", True)):
        _rrt_descend_to_place(
            env=env,
            robot=robot,
            target_pos=target_pos,
            drop_rot=drop_rot,
            cfg=cfg,
            wrist3_lock=wrist3_lock,
            log_fn=log_fn,
        )
    else:
        try:
            _move_cartesian_line(
                env,
                robot,
                T_now.t,
                T_drop.t,
                keep_rotation=sm.SO3(T_now.R),
                duration=2.2,
                gripper_ctrl=255.0,
            )
        except Exception:
            q_drop = robot.ikine(T_drop)
            if len(q_drop) == 0:
                raise RuntimeError("RRT模式: 放置下降IK失败")
            _move_joint(env, robot, robot.get_joint(), q_drop, duration=1.8, gripper_ctrl=255.0)

    # 7) Open gripper and retreat.
    _log(log_fn, "STEP", "[RRT] 阶段7: 松开夹爪并后退")
    _gripper_open(env, robot, steps=2000)
    T_after_drop = robot.get_cartesian()
    raise_z = float(T_after_drop.t[2]) + float(cfg.get("post_release_raise_z", 0.25))
    if bool(cfg.get("enable_rrt_post_release_retreat", True)):
        q_raise, T_raise, offset_xy, mode, checked, _ = _find_rrt_reachable_pose_joint(
            env=env,
            robot=robot,
            base_xy=(float(T_after_drop.t[0]), float(T_after_drop.t[1])),
            target_z=raise_z,
            target_rot=sm.SO3(T_after_drop.R),
            cfg=cfg,
            wrist3_lock=wrist3_lock,
            log_tag="放置后抬升",
        )
        if len(q_raise) > 0:
            _log(
                log_fn,
                "INFO",
                f"[RRT] 阶段7: 选中{mode}抬升候选(checked={checked}), "
                f"xy偏移=({offset_xy[0]:+.3f},{offset_xy[1]:+.3f}), z={raise_z:.3f}",
            )
            _online_rrt_move_to_goal(
                env,
                q_raise,
                cfg,
                gripper_ctrl=0.0,
                wrist3_lock=wrist3_lock,
                log_fn=log_fn,
            )
        else:
            _log(log_fn, "WARN", "[RRT] 阶段7: 未找到可行RRT抬升候选，回退笛卡尔直线上抬")
            T_raise = sm.SE3.Trans(
                T_after_drop.t[0],
                T_after_drop.t[1],
                raise_z,
            ) * sm.SE3(sm.SO3(T_after_drop.R))
            _move_cartesian_line(
                env,
                robot,
                T_after_drop.t,
                T_raise.t,
                keep_rotation=sm.SO3(T_after_drop.R),
                duration=1.0,
                gripper_ctrl=0.0,
            )
    else:
        T_raise = sm.SE3.Trans(T_after_drop.t[0], T_after_drop.t[1], raise_z) * sm.SE3(sm.SO3(T_after_drop.R))
        _move_cartesian_line(
            env,
            robot,
            T_after_drop.t,
            T_raise.t,
            keep_rotation=sm.SO3(T_after_drop.R),
            duration=1.0,
            gripper_ctrl=0.0,
        )

    # 8) Return home for consistency.
    if bool(cfg.get("enable_rrt_return_home", True)):
        _log(log_fn, "STEP", "[RRT] 阶段8: 使用在线RRT回到初始姿态")
        _online_rrt_move_to_goal(
            env,
            q_home,
            cfg,
            gripper_ctrl=0.0,
            wrist3_lock=wrist3_lock,
            log_fn=log_fn,
        )
    else:
        _log(log_fn, "STEP", "[RRT] 阶段8: 回到初始姿态")
        _move_joint(env, robot, robot.get_joint(), q_home, duration=1.5, gripper_ctrl=0.0)

    if object_name:
        record_place(object_name, list(target_pos))
    _log(log_fn, "SUCCESS", "[RRT] 避障抓取放置流程完成")
