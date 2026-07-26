"""
Pick fruit and place with auto-repeat after first G press.

新模型 (yolo11n-seg.pt) 识别 5 类水果: banana, grape, peach, pineapple, carambola.
首次按 G 触发抓取, 放置回预备位后自动检测画面, 有目标就继续抓, 没有则返回手动等待.

Workflow:
  1. Init → ready pose, wait for G
  2. G: detect → pick → place → return to ready
  3. Auto-detect: if target in view → auto pick (loop)
  4. If no target → stop, wait for next G

Keys:
  G: start pick-and-place (first time) / manual override
  R: resume live preview / stop auto mode
  Q/Esc: quit

Usage:
    python scripts/set6.py
    python scripts/set6.py --target banana
    python scripts/set6.py --target all        # 检测所有 5 类
    python scripts/set6.py --dry-run
"""

from __future__ import annotations

import argparse
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


# 放置位姿
_PLACE_JOINTS_2_6 = (-0.57, -0.81, 0.98, 0.0, 0.0)
PLACE_JOINTS_LEFT  = (+2.61,) + _PLACE_JOINTS_2_6
PLACE_JOINTS_RIGHT = (-2.61,) + _PLACE_JOINTS_2_6

# 新模型支持的所有类别
ALL_TARGETS = ["banana", "grape", "peach", "pineapple", "carambola"]

# 柔性抓取参数 (水果不会被夹爆)
SOFT_CLOSE_TORQUE = 1.0    # 闭合阶段扭矩 (N·m), 原始值
SOFT_HOLD_FORCE  = 0.30   # 夹住后保持力 (N·m), 原始值


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


def _cam_to_base(T_hand_eye: np.ndarray, grasp_driver: GraspDriver, cfg: dict[str, Any]) -> np.ndarray:
    return compose_cam_to_base_transform(grasp_driver.get_tcp_pose(), T_hand_eye, cfg)


def _find_target(
    grasps: list[GraspPose],
    target_names: list[str],
) -> Optional[GraspPose]:
    """Return highest-confidence valid grasp among matching target classes."""
    best = None
    for g in grasps:
        if not g.is_valid:
            continue
        for name in target_names:
            if name in g.class_name.lower():
                if best is None or g.conf > best.conf:
                    best = g
                break
    return best


def _execute_pick_dual_place(
    controller: RebotArmEndPose,
    grasp_driver: GraspDriver,
    grasp6d: tuple[float, ...],
    pre6d: tuple[float, ...],
    place_joints_left: tuple[float, ...],
    place_joints_right: tuple[float, ...],
    ready_cfg: dict[str, Any],
    dry_run: bool,
    target_name: str,
) -> bool:
    xg, yg, zg, rxg, ryg, rzg = grasp6d
    xp, yp, zp, rxp, ryp, rzp = pre6d

    print(f"[pick] pregrasp  xyz=({xp:+.3f},{yp:+.3f},{zp:+.3f})  rpy=({rxp:+.3f},{ryp:+.3f},{rzp:+.3f})")
    print(f"[pick] grasp     xyz=({xg:+.3f},{yg:+.3f},{zg:+.3f})  rpy=({rxg:+.3f},{ryg:+.3f},{rzg:+.3f})")

    if dry_run:
        print("[pick] dry run; skip motion")
        return True

    print("[pick] Open gripper")
    grasp_driver.open_gripper()

    print("[pick] Move to pregrasp")
    if not controller.move_to_traj(xp, yp, zp, rxp, ryp, rzp, duration=2.0):
        print("[pick] Pregrasp IK failed")
        return False
    _wait_motion(controller, 2.0)

    print("[pick] Move to grasp")
    if not controller.move_to_traj(xg, yg, zg, rxg, ryg, rzg, duration=1.5):
        print("[pick] Grasp IK failed")
        return False
    _wait_motion(controller, 1.5)

    print("[pick] Closing gripper (soft grasp)")
    # 柔性抓取: 临时降低扭矩, 感知到接触即停, 不夹爆水果
    _orig_close_torque = grasp_driver._close_torque
    _orig_default_force = grasp_driver._default_force
    grasp_driver._close_torque = grasp_driver._close_sign * SOFT_CLOSE_TORQUE
    grasp_driver._default_force = grasp_driver._close_sign * SOFT_HOLD_FORCE
    ok = grasp_driver.grasp(force=SOFT_HOLD_FORCE, timeout=3.0)
    grasp_driver._close_torque = _orig_close_torque
    grasp_driver._default_force = _orig_default_force
    print("[pick] Holding object" if ok else "[pick] Empty grasp")
    if not ok:
        return False

    j1 = _read_joint1(controller)
    if j1 > 0:
        place_joints = place_joints_left
        side = "LEFT"
    else:
        place_joints = place_joints_right
        side = "RIGHT"

    print(f"[pick] joint1 at grasp = {j1:+.3f} rad -> {side} place (j1={place_joints[0]:+.3f})")

    print(f"[pick] Lift to ready with {target_name}")
    _move_ready(controller, ready_cfg)

    print(f"[pick] Joint-space move to {side} place: {[f'{v:+.3f}' for v in place_joints]}")
    _move_to_joints(controller, place_joints, duration=3.0)

    print(f"[pick] Release gripper - {target_name} dropped")
    grasp_driver.open_gripper(timeout=0.5)
    time.sleep(0.8)

    print("[pick] Return to ready")
    _move_ready(controller, ready_cfg)

    return True


def _render_display(
    image: np.ndarray,
    grasps: list[GraspPose],
    best_target: Optional[GraspPose],
    status_text: str,
    auto_mode: bool,
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

    cv2.putText(display, status_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

    if auto_mode:
        cv2.putText(display, "[AUTO]", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)

    if best_target is not None:
        x_m, y_m, z_m = best_target.position.tolist()
        cv2.putText(display, f"{best_target.class_name}: xyz=({x_m:+.3f},{y_m:+.3f},{z_m:+.3f})",
                    (10, display.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 140), 2)

    return display


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auto pick-and-place with new fruit model")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--target", default="banana", help=f"target class or 'all' for: {', '.join(ALL_TARGETS)}")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(PROJECT_ROOT / args.config)

    # 目标类别
    if args.target.lower() == "all":
        target_names = list(ALL_TARGETS)
    else:
        target_names = [args.target.lower()]

    robot_cfg = cfg.get("robot", {})
    ready_cfg = robot_cfg.get(
        "ready_pose",
        {"x": 0.25, "y": 0.0, "z": 0.35, "roll": 0.0, "pitch": 1.2, "yaw": 0.0, "duration": 3.0},
    )
    cam_cfg = cfg.get("camera", {})
    print(f"=== Camera: {cam_cfg.get('type')} ===")
    print(f"=== Model: yolo11n-seg.pt ===")
    print(f"=== Target(s): {target_names} ===")
    print(f"=== Place LEFT  joints: {[f'{v:+.3f}' for v in PLACE_JOINTS_LEFT]} ===")
    print(f"=== Place RIGHT joints: {[f'{v:+.3f}' for v in PLACE_JOINTS_RIGHT]} ===")
    cam = make_camera(cfg)

    last_results: list[Any] = []
    last_grasps: list[GraspPose] = []
    frozen = False
    last_display: Optional[np.ndarray] = None
    auto_mode = False           # 首次按 G 后自动连续抓取
    auto_trigger = False        # auto_mode 下触发新一轮检测
    frame_index = 0
    fps_counter = 0
    fps_timer = time.perf_counter()
    fps_value = 0.0

    window_name = "Set6 - Auto Pick Fruit"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print(f"\n[Keys]  G=start  R=stop auto  Q/ESC=quit\n")

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

        # 强制使用新模型
        model_name = "yolov11s.pt"
        pregrasp_offset_m = float(grasp_cfg.get("pregrasp_offset_m", 0.08))
        infer_every = max(1, int(gp_cfg.get("infer_every_live", 2)))

        print(f"=== Load YOLO: {model_name} ===")
        # 覆盖 config 中的 model_name 以加载新模型
        yolo_cfg["model_name"] = model_name
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

        while True:
            # --- 自动模式: 回预备位后检测并触发下一轮 ---
            if auto_trigger:
                time.sleep(0.5)  # 等画面稳定
                snap_color, snap_depth = cam.get_frame()
                if snap_color is not None:
                    snap_results = model.predict(
                        snap_color, verbose=False,
                        device=yolo_opts.get("device", "cpu"),
                        conf=float(yolo_opts.get("conf", 0.25)),
                        iou=float(yolo_opts.get("iou", 0.45)),
                    )
                    snap_grasps = estimate_grasps(snap_results, snap_depth, K)
                    snap_target = _find_target(snap_grasps, target_names)

                    if snap_target is not None:
                        print(f"\n[AUTO] Found {snap_target.class_name} conf={snap_target.conf:.3f}")
                        print(f"  position_xyz={snap_target.position.tolist()}")

                        snap_display = _render_display(snap_color, snap_grasps, snap_target, "AUTO-PICK", True)
                        frozen = True
                        last_display = snap_display
                        last_results = snap_results
                        last_grasps = snap_grasps

                        if T_hand_eye is None:
                            print("[AUTO] No hand-eye calibration, stop auto")
                            auto_mode = False
                            auto_trigger = False
                            frozen = False
                            last_display = None
                            continue

                        T_cam2base = _cam_to_base(T_hand_eye, grasp_driver, cfg)
                        grasp6d, pre6d = transform_grasp_pose_to_base(
                            snap_target.position, snap_target.tcp_rotation,
                            T_cam2base, pregrasp_offset_m,
                        )

                        ok = _execute_pick_dual_place(
                            controller, grasp_driver,
                            grasp6d, pre6d,
                            PLACE_JOINTS_LEFT, PLACE_JOINTS_RIGHT,
                            ready_cfg,
                            dry_run=args.dry_run,
                            target_name=snap_target.class_name,
                        )
                        if ok:
                            print("[AUTO] Pick-and-place OK")
                            auto_trigger = True  # 继续下一轮
                        else:
                            print("[AUTO] Pick-and-place FAILED, stop auto")
                            auto_mode = False
                            auto_trigger = False
                            frozen = False
                            last_display = None
                        continue
                    else:
                        print("[AUTO] No more targets in view, returning to manual mode\n")
                        auto_mode = False
                        auto_trigger = False
                        frozen = False
                        last_display = None
                else:
                    auto_trigger = False

            # --- 正常帧循环 ---
            color_bgr, depth_mm = cam.get_frame()
            if color_bgr is None or depth_mm is None:
                continue

            frame_index += 1
            fps_counter += 1
            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                fps_value = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            if not frozen and (frame_index % infer_every == 0 or not last_results):
                last_results = model.predict(
                    color_bgr, verbose=False,
                    device=yolo_opts.get("device", "cpu"),
                    conf=float(yolo_opts.get("conf", 0.25)),
                    iou=float(yolo_opts.get("iou", 0.45)),
                )
                last_grasps = estimate_grasps(last_results, depth_mm, K)

            best_target = _find_target(last_grasps, target_names)

            mode_tag = "[AUTO]" if auto_mode else ""
            status = f"{'FROZEN' if frozen else 'LIVE'} {fps_value:.1f}fps {mode_tag} | G=start R=stop Q=quit"
            if frozen and last_display is not None:
                display = last_display.copy()
                cv2.putText(display, "[FROZEN]", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
            else:
                display = _render_display(color_bgr, last_grasps, best_target, status, auto_mode)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                auto_mode = False
                auto_trigger = False
                frozen = False
                last_display = None
                print("[Key] Auto mode stopped, back to manual")
                continue

            if key in (ord("g"), ord("G")):
                if auto_mode:
                    print("[G] Already in auto mode, ignoring")
                    continue

                print(f"\n[G] Start pick-and-place")
                auto_mode = True
                snap_color, snap_depth = cam.get_frame()
                if snap_color is None or snap_depth is None:
                    print("[G] Frame capture failed")
                    auto_mode = False
                    continue

                snap_results = model.predict(
                    snap_color, verbose=False,
                    device=yolo_opts.get("device", "cpu"),
                    conf=float(yolo_opts.get("conf", 0.25)),
                    iou=float(yolo_opts.get("iou", 0.45)),
                )
                snap_grasps = estimate_grasps(snap_results, snap_depth, K)
                snap_target = _find_target(snap_grasps, target_names)

                if snap_target is None:
                    print(f"[G] No target detected, waiting...")
                    auto_mode = False
                    continue

                print(f"[G] {snap_target.class_name} conf={snap_target.conf:.3f}")
                print(f"  position_xyz={snap_target.position.tolist()}")

                snap_display = _render_display(snap_color, snap_grasps, snap_target, "SNAPSHOT", auto_mode)
                frozen = True
                last_display = snap_display
                last_results = snap_results
                last_grasps = snap_grasps

                if T_hand_eye is None:
                    print("[G] Hand-eye calibration unavailable")
                    auto_mode = False
                    frozen = False
                    continue

                T_cam2base = _cam_to_base(T_hand_eye, grasp_driver, cfg)
                grasp6d, pre6d = transform_grasp_pose_to_base(
                    snap_target.position, snap_target.tcp_rotation,
                    T_cam2base, pregrasp_offset_m,
                )

                ok = _execute_pick_dual_place(
                    controller, grasp_driver,
                    grasp6d, pre6d,
                    PLACE_JOINTS_LEFT, PLACE_JOINTS_RIGHT,
                    ready_cfg,
                    dry_run=args.dry_run,
                    target_name=snap_target.class_name,
                )
                if ok:
                    print("[G] Pick-and-place OK, entering auto mode")
                    auto_trigger = True  # 触发 auto 循环
                else:
                    print("[G] Pick-and-place FAILED")
                    auto_mode = False
                    frozen = False
                    last_display = None

    finally:
        print("\n[Exit] Release and disconnect")
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
