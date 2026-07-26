"""
Autonomous continuous pick-and-place with scanning and rover simulation.

Workflow:
  1. Arm init → ready pose
  2. SCANNING: joint1 sweeps -pi/2 to +pi/2 with YOLO detection
  3. CONFIRM: stop, wait 0.3s, re-detect (false-positive filter)
  4. REACH_CHECK: IK check → GRASP or WAIT_ROVER
  5. WAIT_ROVER: wait for any key (simulate rover arrival) → small re-scan
  6. GRASP: set5 dual-place logic (left/right joint1)
  7. POST_GRASP: check if more minerals in view → skip scan if yes

Exit: Esc key or Ctrl+C
Rover signal: any key press during WAIT_ROVER state
"""

from __future__ import annotations

import enum
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from drivers.camera import make_camera
from drivers.robot.grasp_driver import GraspDriver, selected_arm_config
from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose
from utils.camera_utils import compose_cam_to_base_transform, load_config, load_hand_eye
from utils.ordinary_grasp import GraspPose, estimate_grasps
from utils.transforms import transform_grasp_pose_to_base
from utils.yolo_utils import load_yolo


# ============================================================
# 放置位姿 — 沿用 set5 的双向逻辑
# ============================================================
_PLACE_JOINTS_2_6 = (-0.57, -0.81, 0.98, 0.0, 0.0)
PLACE_JOINTS_LEFT  = (+2.61,) + _PLACE_JOINTS_2_6
PLACE_JOINTS_RIGHT = (-2.61,) + _PLACE_JOINTS_2_6

# 扫描参数
SCAN_LIMIT = np.pi / 2          # joint1 扫描范围 ±90°
SCAN_SPEED = 0.2                # 扫描转速 rad/s (安全低速)
SCAN_INFER_EVERY = 3            # 每 N 帧做一次 YOLO 推理
SCAN_CONF_THRESHOLD = 0.5       # 扫描时检测置信度阈值
SMALL_SCAN_RANGE = np.pi / 6    # 小车到达后小范围重扫 ±30°
CONFIRM_WAIT = 0.3              # 确认阶段等待画面稳定时间 (s)
POST_GRASP_WAIT = 0.5           # 抓取后回预备位等待稳定时间 (s)


# ============================================================
# 状态机
# ============================================================
class State(enum.Enum):
    SCANNING = enum.auto()
    CONFIRM = enum.auto()
    REACH_CHECK = enum.auto()
    GRASP = enum.auto()
    WAIT_ROVER = enum.auto()
    POST_GRASP = enum.auto()


# ============================================================
# 工具函数 (来自 set5)
# ============================================================
def _wait_motion(controller: RebotArmEndPose, duration: float, extra: float = 0.6) -> None:
    thread = getattr(controller, "_send_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=duration + extra + 2.0)
    else:
        time.sleep(duration + extra)


def _move_ready(controller: RebotArmEndPose, ready_cfg: dict[str, Any]) -> None:
    duration = float(ready_cfg.get("duration", 3.0))
    controller.move_to_traj(
        x=float(ready_cfg.get("x", 0.25)),
        y=float(ready_cfg.get("y", 0.0)),
        z=float(ready_cfg.get("z", 0.35)),
        roll=float(ready_cfg.get("roll", 0.0)),
        pitch=float(ready_cfg.get("pitch", 1.2)),
        yaw=float(ready_cfg.get("yaw", 0.0)),
        duration=duration,
    )
    _wait_motion(controller, duration)


def _move_to_joints(
    controller: RebotArmEndPose,
    target_joints: tuple[float, ...],
    duration: float = 3.0,
    max_vel: float = 2.0,
    send_freq: float = 50.0,
    settle_thresh: float = 0.02,
) -> None:
    """Joint-space minimum-jerk trajectory to target joint angles."""
    n = controller._n
    q_target = np.array(target_joints, dtype=np.float64)[:n]
    q_now = controller.rebotarm.get_state()[0][:n]
    q_start = q_now.copy()

    q_err = np.abs(q_target - q_start)
    max_err = float(np.max(q_err))
    if max_err < 0.01:
        return

    t_ramp = max_err / max_vel
    t_total = max(t_ramp * 2.0, duration * 0.5)
    dt_send = 1.0 / send_freq
    num_steps = max(2, int(t_total / dt_send))

    t = np.linspace(0, t_total, num_steps)
    traj = np.zeros((num_steps, n))
    for i in range(n):
        err_i = q_target[i] - q_start[i]
        s = t / t_total
        traj[:, i] = q_start[i] + err_i * (10.0 * s ** 3 - 15.0 * s ** 4 + 6.0 * s ** 5)

    interval = t_total / num_steps if num_steps > 0 else dt_send
    timeout = duration + 10.0
    deadline = time.monotonic() + timeout
    controller._vlim_override = np.full(n, max_vel, dtype=np.float64)

    for i in range(num_steps):
        if time.monotonic() > deadline:
            break
        controller._q_target[:] = traj[i]
        time.sleep(interval)

    controller._q_target[:] = q_target
    settle_deadline = time.monotonic() + 3.0
    while time.monotonic() < settle_deadline:
        q_now, _, _ = controller.rebotarm.get_state()
        if np.max(np.abs(q_now[:n] - q_target)) < settle_thresh:
            break
        time.sleep(0.02)
    controller._vlim_override = None


def _read_joint1(controller: RebotArmEndPose) -> float:
    return float(controller.rebotarm.get_state()[0][0])


def _read_all_joints(controller: RebotArmEndPose) -> np.ndarray:
    return controller.rebotarm.get_state()[0][:controller._n].copy()


def _cam_to_base(T_hand_eye: np.ndarray, grasp_driver: GraspDriver, cfg: dict[str, Any]) -> np.ndarray:
    return compose_cam_to_base_transform(grasp_driver.get_tcp_pose(), T_hand_eye, cfg)


# ============================================================
# 扫描运动 (非阻塞, 在主循环每帧调用一次)
# ============================================================
def _scan_step(
    controller: RebotArmEndPose,
    scan_direction: int,
    dt: float,
) -> tuple[int, bool]:
    """Advance joint1 sweep by one step. Return (direction, hit_limit)."""
    n = controller._n
    q_now = _read_all_joints(controller)
    j1 = q_now[0]

    dq = scan_direction * SCAN_SPEED * dt
    j1_new = j1 + dq

    hit_limit = False
    if scan_direction > 0 and j1_new >= +SCAN_LIMIT:
        j1_new = +SCAN_LIMIT
        scan_direction = -1
        hit_limit = True
        print(f"[scan] hit +90°, reversing  j1={j1_new:+.3f}")
    elif scan_direction < 0 and j1_new <= -SCAN_LIMIT:
        j1_new = -SCAN_LIMIT
        scan_direction = +1
        hit_limit = True
        print(f"[scan] hit -90°, reversing  j1={j1_new:+.3f}")

    q_target = q_now.copy()
    q_target[0] = j1_new
    controller._q_target[:] = q_target

    return scan_direction, hit_limit


def _scan_stop(controller: RebotArmEndPose) -> None:
    """Hold current joint positions (stop sweep)."""
    q_now = _read_all_joints(controller)
    controller._q_target[:] = q_now


# ============================================================
# YOLO 检测辅助
# ============================================================
def _find_best_target(
    model,
    color_bgr: np.ndarray,
    depth_mm: np.ndarray,
    K: np.ndarray,
    yolo_opts: dict[str, Any],
    target_name: str,
) -> Optional[GraspPose]:
    """Run YOLO on a single frame and return the best valid target grasp."""
    results = model.predict(
        color_bgr, verbose=False,
        device=yolo_opts.get("device", "cpu"),
        conf=float(yolo_opts.get("conf", 0.25)),
        iou=float(yolo_opts.get("iou", 0.45)),
    )
    grasps = estimate_grasps(results, depth_mm, K)

    best = None
    for g in grasps:
        if not g.is_valid:
            continue
        if target_name in g.class_name.lower():
            if best is None or g.conf > best.conf:
                best = g
    return best


# ============================================================
# 渲染
# ============================================================
def _render_display(
    image: np.ndarray,
    grasps: list[GraspPose],
    state: State,
    target_name: str,
    scan_direction: int,
    elapsed: float,
) -> np.ndarray:
    display = image.copy()
    for grasp in grasps:
        color = (0, 255, 0) if grasp.is_valid else (0, 165, 255)
        cv2.rectangle(display,
                      (int(grasp.bbox_xyxy[0]), int(grasp.bbox_xyxy[1])),
                      (int(grasp.bbox_xyxy[2]), int(grasp.bbox_xyxy[3])),
                      color, 2)
        cv2.putText(display, f"{grasp.class_name} {grasp.conf:.2f}",
                    (int(grasp.bbox_xyxy[0]), int(grasp.bbox_xyxy[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    dir_str = ">>" if scan_direction > 0 else "<<"
    cv2.putText(display, f"[{state.name}] {dir_str}  t={elapsed:.0f}s  Esc=exit",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return display


# ============================================================
# 抓取执行 (沿用 set5 双向逻辑)
# ============================================================
def _execute_grasp_and_place(
    controller: RebotArmEndPose,
    grasp_driver: GraspDriver,
    grasp6d: tuple[float, ...],
    pre6d: tuple[float, ...],
    ready_cfg: dict[str, Any],
    target_name: str,
) -> bool:
    """set5-style pick with left/right place decision."""
    xg, yg, zg, rxg, ryg, rzg = grasp6d
    xp, yp, zp, rxp, ryp, rzp = pre6d

    print(f"[grasp] pregrasp  xyz=({xp:+.3f},{yp:+.3f},{zp:+.3f})  rpy=({rxp:+.3f},{ryp:+.3f},{rzp:+.3f})")
    print(f"[grasp] grasp     xyz=({xg:+.3f},{yg:+.3f},{zg:+.3f})  rpy=({rxg:+.3f},{ryg:+.3f},{rzg:+.3f})")

    print("[grasp] Open gripper")
    grasp_driver.open_gripper()

    print("[grasp] Move to pregrasp")
    if not controller.move_to_traj(xp, yp, zp, rxp, ryp, rzp, duration=2.0):
        print("[grasp] Pregrasp IK failed")
        return False
    _wait_motion(controller, 2.0)

    print("[grasp] Move to grasp")
    if not controller.move_to_traj(xg, yg, zg, rxg, ryg, rzg, duration=1.5):
        print("[grasp] Grasp IK failed")
        return False
    _wait_motion(controller, 1.5)

    print("[grasp] Closing gripper")
    if not grasp_driver.grasp():
        print("[grasp] Empty grasp")
        return False
    print("[grasp] Holding object")

    # --- 读取 joint1, 决定左转还是右转 ---
    j1 = _read_joint1(controller)
    if j1 > 0:
        place_joints = PLACE_JOINTS_LEFT
        side = "LEFT"
    else:
        place_joints = PLACE_JOINTS_RIGHT
        side = "RIGHT"

    print(f"[grasp] joint1 at grasp = {j1:+.3f} rad -> {side} place (j1={place_joints[0]:+.3f})")

    print(f"[grasp] Lift to ready with {target_name}")
    _move_ready(controller, ready_cfg)

    print(f"[grasp] Joint-space move to {side} place joints: {[f'{v:+.3f}' for v in place_joints]}")
    _move_to_joints(controller, place_joints, duration=3.0)

    print(f"[grasp] Release gripper - {target_name} dropped")
    grasp_driver.open_gripper(timeout=0.5)
    time.sleep(0.8)

    print("[grasp] Return to ready")
    _move_ready(controller, ready_cfg)

    return True


# ============================================================
# 主程序
# ============================================================
def main() -> int:
    cfg = load_config(PROJECT_ROOT / "config/default.yaml")

    target_name = "bottle"
    robot_cfg = cfg.get("robot", {})
    ready_cfg = robot_cfg.get(
        "ready_pose",
        {"x": 0.25, "y": 0.0, "z": 0.35, "roll": 0.0, "pitch": 1.2, "yaw": 0.0, "duration": 3.0},
    )
    cam_cfg = cfg.get("camera", {})
    print(f"=== Camera: {cam_cfg.get('type')} ===")
    print(f"=== Target: {target_name} ===")
    print(f"=== Place LEFT  joints: {[f'{v:+.3f}' for v in PLACE_JOINTS_LEFT]} ===")
    print(f"=== Place RIGHT joints: {[f'{v:+.3f}' for v in PLACE_JOINTS_RIGHT]} ===")
    print(f"=== Scan range: ±{np.degrees(SCAN_LIMIT):.0f}° at {SCAN_SPEED:.1f} rad/s ===")
    cam = make_camera(cfg)

    last_grasps: list[GraspPose] = []
    frame_index = 0
    fps_counter = 0
    fps_timer = time.perf_counter()
    fps_value = 0.0

    window_name = "SetAuto - Autonomous Scan & Pick"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print(f"\n=== Autonomous mode ===")
    print(f"  Scanning: joint1 sweeps ±90°")
    print(f"  Rover signal: press Shift during WAIT_ROVER state")
    print(f"  Exit: press Esc or Ctrl+C\n")

    controller: Optional[RebotArmEndPose] = None
    rebotarm = None
    grasp_driver: Optional[GraspDriver] = None
    T_hand_eye: Optional[np.ndarray] = None
    yolo_opts: dict[str, Any] = {}
    robot_ready = False

    try:
        cam.open()
        cam.warm_up(10)
        K = cam.K.astype(np.float32)

        cam_type = str(cam_cfg.get("type", "")).lower()
        T_hand_eye, hand_eye_mode = load_hand_eye(PROJECT_ROOT, cam_type)
        if T_hand_eye is None or hand_eye_mode != "eye_in_hand":
            print("[WARN] Hand-eye calibration unavailable")
            T_hand_eye = None

        yolo_cfg = cfg.get("yolo", {})
        gp_cfg = cfg.get("grasp_pipeline", {})
        grasp_cfg = gp_cfg.get("grasp", {})

        model_name = yolo_cfg.get("model_name", "yoloe-26s-seg.pt")
        pregrasp_offset_m = float(grasp_cfg.get("pregrasp_offset_m", 0.08))

        print(f"=== Load YOLO: {model_name} ===")
        model, yolo_opts = load_yolo(cfg, project_root=PROJECT_ROOT)

        print("=== Init robot ===")
        selected = selected_arm_config(robot_cfg.get("repo_root"))
        rebotarm = RebotArm()
        controller = RebotArmEndPose(rebotarm, arm_control_mode=selected.controller_mode)
        grasp_driver = GraspDriver(
            rebotarm, controller,
            gripper_config=robot_cfg.get("gripper"),
            repo_root=robot_cfg.get("repo_root"),
        )
        grasp_driver.start()
        robot_ready = True
        print(f"[Robot] mode: {selected.controller_mode}")

        print("[Robot] Move ready")
        _move_ready(controller, ready_cfg)

        # 设置扫描时 joint1 的速度限制
        vlim = np.array([0.25, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float64)
        controller._vlim_override = vlim

        # ============================================================
        # 状态机初始化
        # ============================================================
        state = State.SCANNING
        scan_direction = 1  # 1: 向正方向扫, -1: 向负方向扫
        last_scan_time = time.monotonic()
        confirm_deadline = 0.0
        post_grasp_deadline = 0.0
        grasp6d: Optional[tuple[float, ...]] = None
        pre6d: Optional[tuple[float, ...]] = None
        start_time = time.monotonic()
        lap_count = 0  # 扫描来回次数

        while True:
            loop_start = time.monotonic()
            color_bgr, depth_mm = cam.get_frame()
            if color_bgr is None or depth_mm is None:
                time.sleep(0.05)
                continue

            now = time.monotonic()
            dt = now - last_scan_time
            last_scan_time = now
            elapsed = now - start_time

            frame_index += 1
            fps_counter += 1
            if now - fps_timer >= 1.0:
                fps_value = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            # ========================================================
            # STATE: SCANNING
            # ========================================================
            if state == State.SCANNING:
                scan_direction, hit_limit = _scan_step(controller, scan_direction, max(dt, 0.02))
                if hit_limit:
                    lap_count += 1

                if frame_index % SCAN_INFER_EVERY == 0:
                    target = _find_best_target(model, color_bgr, depth_mm, K, yolo_opts, target_name)
                    if target is not None and target.conf >= SCAN_CONF_THRESHOLD:
                        print(f"\n[scan] DETECTED  conf={target.conf:.3f}  pos={target.position.tolist()}  j1={_read_joint1(controller):+.3f}")
                        _scan_stop(controller)
                        state = State.CONFIRM
                        confirm_deadline = time.monotonic() + CONFIRM_WAIT
                        # 更新最后一次检测结果用于后续 display
                        last_results = model.predict(
                            color_bgr, verbose=False,
                            device=yolo_opts.get("device", "cpu"),
                            conf=float(yolo_opts.get("conf", 0.25)),
                            iou=float(yolo_opts.get("iou", 0.45)),
                        )
                        last_grasps = estimate_grasps(last_results, depth_mm, K)

            # ========================================================
            # STATE: CONFIRM
            # ========================================================
            elif state == State.CONFIRM:
                if time.monotonic() >= confirm_deadline:
                    snap_color, snap_depth = cam.get_frame()
                    if snap_color is None or snap_depth is None:
                        print("[confirm] frame capture failed, back to scanning")
                        state = State.SCANNING
                        continue

                    target = _find_best_target(model, snap_color, snap_depth, K, yolo_opts, target_name)
                    if target is not None and target.conf >= SCAN_CONF_THRESHOLD:
                        print(f"[confirm] CONFIRMED  conf={target.conf:.3f}  pos={target.position.tolist()}")

                        if T_hand_eye is None:
                            print("[confirm] No hand-eye calibration, back to scanning")
                            state = State.SCANNING
                            continue

                        T_cam2base = _cam_to_base(T_hand_eye, grasp_driver, cfg)
                        g6d, p6d = transform_grasp_pose_to_base(
                            target.position, target.tcp_rotation, T_cam2base, pregrasp_offset_m,
                        )
                        grasp6d = g6d
                        pre6d = p6d

                        state = State.REACH_CHECK
                    else:
                        print(f"[confirm] FALSE POSITIVE, back to scanning  j1={_read_joint1(controller):+.3f}")
                        state = State.SCANNING

            # ========================================================
            # STATE: REACH_CHECK
            # ========================================================
            elif state == State.REACH_CHECK:
                assert grasp6d is not None
                assert pre6d is not None

                xg, yg, zg, rxg, ryg, rzg = grasp6d
                ik_ok = controller.move_to_ik(xg, yg, zg, rxg, ryg, rzg)
                if ik_ok:
                    print(f"[reach] IK OK -> execute grasp")
                    # 恢复正常速度限制
                    controller._vlim_override = None
                    state = State.GRASP
                else:
                    print(f"[reach] IK FAILED -> waiting for rover  (j1={_read_joint1(controller):+.3f})")
                    state = State.WAIT_ROVER

            # ========================================================
            # STATE: WAIT_ROVER
            # ========================================================
            elif state == State.WAIT_ROVER:
                # 显示提示覆盖在画面底部 (等待底部 key 检测触发)
                cv2.putText(
                    color_bgr,
                    "Mineral out of reach - push bench closer, then press SHIFT",
                    (10, color_bgr.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2,
                )
                cv2.putText(
                    color_bgr,
                    f"Last mineral pos: {grasp6d[0]:+.3f}, {grasp6d[1]:+.3f}, {grasp6d[2]:+.3f}",
                    (10, color_bgr.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2,
                )
                # key detection moved to bottom of loop

            # ========================================================
            # 渲染显示
            # ========================================================
            display = _render_display(color_bgr, last_grasps, state, target_name, scan_direction, elapsed)
            cv2.imshow(window_name, display)

            key = cv2.waitKeyEx(1)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 27:  # Esc
                print("\n[Exit] Esc pressed")
                break

            # WAIT_ROVER: 按 Shift = 小车到达信号, 触发小范围重扫
            _SHIFT_KEYS = (0xFFE1, 0xFFE2)  # Left Shift, Right Shift (Linux GTK)
            if state == State.WAIT_ROVER and key in _SHIFT_KEYS:
                print(f"\n[rover] Rover arrived, small re-scan...")
                j1_now = _read_joint1(controller)
                j1_min = max(-SCAN_LIMIT, j1_now - SMALL_SCAN_RANGE)
                j1_max = min(+SCAN_LIMIT, j1_now + SMALL_SCAN_RANGE)
                print(f"[rover] Small scan: {np.degrees(j1_min):+.0f}° → {np.degrees(j1_max):+.0f}°")

                found = False
                for target_j1 in np.linspace(j1_now, j1_max, 10):
                    q_temp = _read_all_joints(controller)
                    q_temp[0] = target_j1
                    controller._q_target[:] = q_temp
                    time.sleep(0.6)
                    snap_c, snap_d = cam.get_frame()
                    if snap_c is not None:
                        t = _find_best_target(model, snap_c, snap_d, K, yolo_opts, target_name)
                        if t is not None and t.conf >= SCAN_CONF_THRESHOLD:
                            print(f"[rover] Found at j1={target_j1:+.3f}, conf={t.conf:.3f}")
                            T = _cam_to_base(T_hand_eye, grasp_driver, cfg)
                            g6d, p6d = transform_grasp_pose_to_base(
                                t.position, t.tcp_rotation, T, pregrasp_offset_m,
                            )
                            grasp6d, pre6d = g6d, p6d
                            found = True
                            break

                if not found:
                    for target_j1 in np.linspace(j1_now, j1_min, 10):
                        q_temp = _read_all_joints(controller)
                        q_temp[0] = target_j1
                        controller._q_target[:] = q_temp
                        time.sleep(0.6)
                        snap_c, snap_d = cam.get_frame()
                        if snap_c is not None:
                            t = _find_best_target(model, snap_c, snap_d, K, yolo_opts, target_name)
                            if t is not None and t.conf >= SCAN_CONF_THRESHOLD:
                                print(f"[rover] Found at j1={target_j1:+.3f}, conf={t.conf:.3f}")
                                T = _cam_to_base(T_hand_eye, grasp_driver, cfg)
                                g6d, p6d = transform_grasp_pose_to_base(
                                    t.position, t.tcp_rotation, T, pregrasp_offset_m,
                                )
                                grasp6d, pre6d = g6d, p6d
                                found = True
                                break

                if found:
                    state = State.REACH_CHECK
                else:
                    print("[rover] Still not found, push closer and press Shift again")

            # ========================================================
            # STATE: GRASP (独立 if, 不在 elif 链中)
            # ========================================================
            if state == State.GRASP:
                assert grasp6d is not None
                assert pre6d is not None
                ok = _execute_grasp_and_place(
                    controller, grasp_driver, grasp6d, pre6d, ready_cfg, target_name,
                )
                if ok:
                    print("[grasp] Pick-and-place OK")
                else:
                    print("[grasp] Pick-and-place FAILED")
                state = State.POST_GRASP
                post_grasp_deadline = time.monotonic() + POST_GRASP_WAIT

            # ========================================================
            # STATE: POST_GRASP
            # ========================================================
            elif state == State.POST_GRASP:
                if time.monotonic() >= post_grasp_deadline:
                    snap_c, snap_d = cam.get_frame()
                    residual = None
                    if snap_c is not None:
                        residual = _find_best_target(model, snap_c, snap_d, K, yolo_opts, target_name)

                    if residual is not None and residual.conf >= SCAN_CONF_THRESHOLD:
                        print(f"[post] More {target_name} in view, skip scan -> grasp directly")
                        T_cam2base = _cam_to_base(T_hand_eye, grasp_driver, cfg)
                        g6d, p6d = transform_grasp_pose_to_base(
                            residual.position, residual.tcp_rotation, T_cam2base, pregrasp_offset_m,
                        )
                        grasp6d = g6d
                        pre6d = p6d
                        state = State.REACH_CHECK
                    else:
                        print(f"[post] No more {target_name} in view, resume scanning\n")
                        controller._vlim_override = vlim  # 恢复扫描速度限制
                        state = State.SCANNING

            # ========================================================
            # 渲染显示
            # ========================================================
            display = _render_display(color_bgr, last_grasps, state, target_name, scan_direction, elapsed)
            cv2.imshow(window_name, display)

            key = cv2.waitKeyEx(1)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == 27:  # Esc
                print("\n[Exit] Esc pressed")
                break

    finally:
        print("\n[Exit] Release and disconnect")
        controller._vlim_override = None
        try:
            if robot_ready and grasp_driver is not None and controller is not None and getattr(controller, "_running", False):
                grasp_driver.release_gripper()
        except Exception as exc:
            print(f"[Exit] {exc}")
        try:
            if controller is not None and getattr(controller, "_running", False):
                controller.end()
            elif rebotarm is not None:
                rebotarm.disconnect()
        except Exception as exc:
            print(f"[Exit] {exc}")
        try:
            cam.close()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Done.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
