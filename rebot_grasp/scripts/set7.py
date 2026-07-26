"""
Pick banana and place at dual fixed joint angles (left/right).

Gripper opens at startup.

Workflow:
  1. Detect banana via YOLO
  2. Go to pregrasp -> grasp banana -> lift to ready
  3. Read joint1 -> choose left or right place joints
  4. Joint-space move to place -> release -> return to ready

Keys:
  G: capture and execute pick-and-place
  R: resume live preview
  Q/Esc: release gripper, home, and exit

Usage:
    python scripts/set7.py
    python scripts/set7.py --dry-run
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


_PLACE_JOINTS_2_6 = (-0.57, -0.81, 0.98, 0.0, 0.0)
PLACE_JOINTS_LEFT  = (+2.61,) + _PLACE_JOINTS_2_6
PLACE_JOINTS_RIGHT = (-2.61,) + _PLACE_JOINTS_2_6


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


def _blind_grasp(grasp_driver: GraspDriver, timeout: float = 1.2) -> None:
    """Close blindly at max speed, ignore force feedback."""
    print(f"[blind] Closing ({timeout:.1f}s)...")
    _orig_torque = grasp_driver._close_torque
    _orig_kd = grasp_driver._kd_close
    grasp_driver._close_torque = grasp_driver._close_sign * grasp_driver._tau_max
    grasp_driver._kd_close = 0.0
    grasp_driver.grasp(timeout=timeout)
    grasp_driver._close_torque = _orig_torque
    grasp_driver._kd_close = _orig_kd
    print("[blind] Done — assume grasped")


def _cam_to_base(T_hand_eye: np.ndarray, grasp_driver: GraspDriver, cfg: dict[str, Any]) -> np.ndarray:
    return compose_cam_to_base_transform(grasp_driver.get_tcp_pose(), T_hand_eye, cfg)


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

    _blind_grasp(grasp_driver)

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
    best_banana: Optional[GraspPose],
    status_text: str,
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

    if best_banana is not None:
        x_m, y_m, z_m = best_banana.position.tolist()
        cv2.putText(display, f"banana: xyz=({x_m:+.3f},{y_m:+.3f},{z_m:+.3f})",
                    (10, display.shape[0] - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 140), 2)

    return display


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pick banana and place at dual fixed joint angles")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dry-run", action="store_true", help="estimate only; do not move the arm")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(PROJECT_ROOT / args.config)

    robot_cfg = cfg.get("robot", {})
    ready_cfg = robot_cfg.get(
        "ready_pose",
        {"x": 0.25, "y": 0.0, "z": 0.35, "roll": 0.0, "pitch": 1.2, "yaw": 0.0, "duration": 3.0},
    )
    cam_cfg = cfg.get("camera", {})
    print(f"=== Camera: {cam_cfg.get('type')} ===")
    cam = make_camera(cfg)

    last_results: list[Any] = []
    last_grasps: list[GraspPose] = []
    frozen = False
    last_display: Optional[np.ndarray] = None
    frame_index = 0
    fps_counter = 0
    fps_timer = time.perf_counter()
    fps_value = 0.0

    window_name = "Set7 - Pick Banana -> Dual Joint Place"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print("\n[Keys]  G=pick+place  R=resume  Q/ESC=quit\n")

    controller: Optional[RebotArmEndPose] = None
    rebotarm: Optional[RebotArm] = None
    grasp_driver: Optional[GraspDriver] = None
    T_hand_eye: Optional[np.ndarray] = None
    yolo_opts: dict[str, Any] = {}
    robot_ready = False

    try:
        cam.open()
        cam.warm_up(15)
        K = cam.K.astype(np.float32)

        cam_type = str(cam_cfg.get("type", "")).lower()
        T_hand_eye, hand_eye_mode = load_hand_eye(PROJECT_ROOT, cam_type)
        if T_hand_eye is None or hand_eye_mode != "eye_in_hand":
            print("[WARN] Hand-eye calibration unavailable; grasp execution disabled")
            T_hand_eye = None

        yolo_cfg = cfg.get("yolo", {})
        gp_cfg = cfg.get("grasp_pipeline", {})
        grasp_cfg = gp_cfg.get("grasp", {})

        model_name = "yolo11s-seg.pt"
        pregrasp_offset_m = float(grasp_cfg.get("pregrasp_offset_m", 0.08))
        depth_quantile = float(grasp_cfg.get("depth_quantile", 0.75))
        infer_every = max(1, int(gp_cfg.get("infer_every_live", 2)))

        print(f"=== Load YOLO: {model_name} ===")
        yolo_cfg["model_name"] = model_name
        model, yolo_opts = load_yolo(cfg, project_root=PROJECT_ROOT)

        print("=== Init robot ===")
        selected = selected_arm_config(robot_cfg.get("repo_root"))
        rebotarm = RebotArm()
        controller = RebotArmEndPose(rebotarm, arm_control_mode=selected.controller_mode)
        grasp_driver = GraspDriver(
            rebotarm,
            controller,
            gripper_config=robot_cfg.get("gripper"),
            repo_root=robot_cfg.get("repo_root"),
        )
        grasp_driver.start()
        robot_ready = True
        print(f"[Robot] mode: {selected.controller_mode}")

        print("[Robot] Move ready")
        _move_ready(controller, ready_cfg)
        print("[Robot] Open gripper at startup")
        grasp_driver.open_gripper()

        while True:
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
                    color_bgr,
                    verbose=False,
                    device=yolo_opts.get("device", "cpu"),
                    conf=float(yolo_opts.get("conf", 0.25)),
                    iou=float(yolo_opts.get("iou", 0.45)),
                )
                last_grasps = estimate_grasps(last_results, depth_mm, K, depth_quantile=depth_quantile)

            best_banana = None
            for grasp in last_grasps:
                if not grasp.is_valid:
                    continue
                if "banana" in grasp.class_name.lower():
                    if best_banana is None or grasp.conf > best_banana.conf:
                        best_banana = grasp

            status = f"{'FROZEN' if frozen else 'LIVE'} {fps_value:.1f}fps | G=pick+place R=resume Q=quit"
            if frozen and last_display is not None:
                display = last_display.copy()
                cv2.putText(display, "[FROZEN]", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
            else:
                display = _render_display(color_bgr, last_grasps, best_banana, status)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("r"), ord("R")):
                frozen = False
                last_display = None
                continue

            if key in (ord("g"), ord("G")):
                print("\n[G] Capture and execute pick-and-place")
                print("[Step1] Detect banana")
                snap_color, snap_depth = cam.get_frame()
                if snap_color is None or snap_depth is None:
                    print("[G] Frame capture failed")
                    continue

                snap_results = model.predict(
                    snap_color,
                    verbose=False,
                    device=yolo_opts.get("device", "cpu"),
                    conf=float(yolo_opts.get("conf", 0.25)),
                    iou=float(yolo_opts.get("iou", 0.45)),
                )
                snap_grasps = estimate_grasps(snap_results, snap_depth, K, depth_quantile=depth_quantile)

                snap_banana = None
                for grasp in snap_grasps:
                    if not grasp.is_valid:
                        continue
                    if "banana" in grasp.class_name.lower():
                        if snap_banana is None or grasp.conf > snap_banana.conf:
                            snap_banana = grasp

                if snap_banana is None:
                    print("[G] No valid banana detected")
                    continue

                print(f"\n[G] Banana: class={snap_banana.class_name} conf={snap_banana.conf:.3f}")
                print(f"  position_xyz={snap_banana.position.tolist()}")

                snap_display = _render_display(snap_color, snap_grasps, snap_banana, "SNAPSHOT")
                frozen = True
                last_display = snap_display
                last_results = snap_results
                last_grasps = snap_grasps

                if T_hand_eye is None:
                    print("[G] Hand-eye calibration unavailable")
                    continue

                T_cam2base = _cam_to_base(T_hand_eye, grasp_driver, cfg)

                grasp6d, pre6d = transform_grasp_pose_to_base(
                    snap_banana.position,
                    snap_banana.tcp_rotation,
                    T_cam2base,
                    pregrasp_offset_m,
                )

                ok = _execute_pick_dual_place(
                    controller,
                    grasp_driver,
                    grasp6d,
                    pre6d,
                    PLACE_JOINTS_LEFT,
                    PLACE_JOINTS_RIGHT,
                    ready_cfg,
                    dry_run=args.dry_run,
                    target_name="banana",
                )

                if ok:
                    print("[G] Pick-and-place completed successfully!")
                else:
                    print("[G] Pick-and-place failed")

    finally:
        print("\n[Exit] Release gripper and home")
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
