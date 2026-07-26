"""
Grasp cube demo based on YOLO.

Workflow:
  1. Detect cube
  2. Go to grasp cube
  3. Grasp cube and lift to ready position
  4. Auto: move arm to placement joints (left/right), open gripper to release

Keys:
  G: capture, grasp, and auto-place cube
  R: resume live preview
  Q/Esc: release gripper, home, and exit

Usage:
    python scripts/set.py
    python scripts/set.py --dry-run
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import threading
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

# 将项目根目录加入 sys.path，以便导入项目内的自定义模块
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


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _wait_motion(controller: RebotArmEndPose, duration: float, extra: float = 0.6) -> None:
    """等待轨迹运动完成。
    
    controller 内部可能有一个 _send_thread 线程正在发送轨迹点，
    如果有则等待线程结束，否则按 duration + extra 睡眠等待。
    """
    thread = getattr(controller, "_send_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=duration + extra + 2.0)
    else:
        time.sleep(duration + extra)


def _move_ready(controller: RebotArmEndPose, ready_cfg: dict[str, Any]) -> None:
    """将机械臂末端移动到配置文件定义的预备位姿 (Cartesian 空间)。"""
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


def _reset_wrist_joints(controller: RebotArmEndPose) -> None:
    """将关节5和关节6复位为0 (速度1.0)，用于IK失败后恢复。"""
    q_now = controller.rebotarm.arm.get_positions()
    q_reset = q_now.copy()
    q_reset[4] = 0.0
    q_reset[5] = 0.0
    print(f"[Reset] Wrist joints 5/6 -> 0 from {q_now[[4,5]].tolist()}")
    controller._vlim_override = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    controller._q_target[:] = q_reset
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 4.0:
        current_q = controller.rebotarm.arm.get_positions()
        err = max(abs(current_q[4] - q_reset[4]), abs(current_q[5] - q_reset[5]))
        if err < 0.05:
            break
        time.sleep(0.1)
    controller._vlim_override = None


def _cam_to_base(T_hand_eye: np.ndarray, grasp_driver: GraspDriver, cfg: dict[str, Any]) -> np.ndarray:
    """计算相机坐标系到机器人基座坐标系的齐次变换矩阵。"""
    return compose_cam_to_base_transform(grasp_driver.get_tcp_pose(), T_hand_eye, cfg)


def _read_joint1(controller: RebotArmEndPose) -> float:
    """读取当前 joint1 角度。"""
    return float(controller.rebotarm.get_state()[0][0])


def _visual_servo_center(
    cam,
    model: Any,
    yolo_opts: dict[str, Any],
    controller: RebotArmEndPose,
    target_name: str,
    K: np.ndarray,
    cfg: dict[str, Any],
    *,
    deadzone_ratio: float = 0.10,
    max_step_rad: float = 0.08,
    gain_p: float = 0.8,
    max_iter: int = 30,
) -> None:
    """旋转 base joint (j1) 使目标在画面中居中 — 非阻塞 P 控制。"""
    W = int(cfg.get("camera", {}).get("color_width", 1280))
    cx_center = W / 2.0
    deadzone = W * deadzone_ratio
    fov_h_rad = 2.0 * math.atan(W / (2.0 * max(float(K[0, 0]), 1e-6)))

    print(f"[Servo] W={W} centre={cx_center:.0f} deadzone=±{deadzone:.0f}px  FOVh={math.degrees(fov_h_rad):.1f}deg")

    for iteration in range(1, max_iter + 1):
        color, _ = cam.get_frame()
        if color is None:
            continue

        results = model.predict(
            color, verbose=False,
            device=yolo_opts.get("device", "cpu"),
            conf=float(yolo_opts.get("conf", 0.25)),
            iou=float(yolo_opts.get("iou", 0.45)),
        )

        # 找出最左侧的目标 box
        target_cx: Optional[float] = None
        best_x1 = float("inf")
        for result in results:
            for box in result.boxes:
                if target_name in str(result.names.get(int(box.cls[0]), "")).lower():
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    if x1 < best_x1:
                        best_x1 = float(x1)
                        target_cx = float((x1 + x2) / 2.0)

        if target_cx is None:
            controller._q_target[0] = float(controller.rebotarm.get_state()[0][0])
            print(f"[Servo] iter {iteration}/{max_iter}: target lost, braking")
            continue

        error = cx_center - target_cx
        print(f"[Servo] iter {iteration}/{max_iter}: cx_obj={target_cx:.0f} error={error:+.0f}px")

        if abs(error) <= deadzone:
            controller._q_target[0] = float(controller.rebotarm.get_state()[0][0])
            print(f"[Servo] Centred — braking & settling")
            time.sleep(0.3)
            return

        rad_per_px = fov_h_rad / W
        delta = float(np.clip(error * gain_p * rad_per_px, -max_step_rad, max_step_rad))
        target_j1 = float(controller.rebotarm.get_state()[0][0]) + delta
        print(f"[Servo]   j1 -> {target_j1:+.4f} rad  (step={delta:+.4f})")
        controller._q_target[0] = target_j1
        time.sleep(0.05)

    print(f"[Servo] Max iterations reached, proceeding")


# ---------------------------------------------------------------------------
# 抓取动作序列
# ---------------------------------------------------------------------------

def _blind_grasp(grasp_driver: GraspDriver, timeout: float = 1.2) -> None:
    """全扭矩闭合，忽略力反馈 — 直接判定抓取成功。"""
    print(f"[blind] Closing ({timeout:.1f}s)...")
    _orig_torque = grasp_driver._close_torque
    _orig_kd = grasp_driver._kd_close
    grasp_driver._close_torque = grasp_driver._close_sign * grasp_driver._tau_max
    grasp_driver._kd_close = 0.0
    grasp_driver.grasp(timeout=timeout)
    grasp_driver._close_torque = _orig_torque
    grasp_driver._kd_close = _orig_kd
    print("[blind] Done — assume grasped")


def _execute_grasp_sequence(
    controller: RebotArmEndPose,
    grasp_driver: GraspDriver,
    grasp6d: tuple[float, ...],
    pre6d: tuple[float, ...],
    ready_cfg: dict[str, Any],
    dry_run: bool,
) -> bool:
    """执行完整抓取序列: 开爪 -> 预抓取位 -> 抓取位 -> 合爪 -> 抬升到预备位。
    
    grasp6d: (x, y, z, roll, pitch, yaw) 抓取目标位姿 (基座坐标系)
    pre6d:   预抓取位姿 (抓取位上方偏移)
    ready_cfg: 预备位配置
    返回 True 表示抓取成功，False 表示失败。
    """
    xg, yg, zg, rxg, ryg, rzg = grasp6d
    xp, yp, zp, rxp, ryp, rzp = pre6d

    print(f"[Step2] pregrasp  xyz=({xp:+.3f},{yp:+.3f},{zp:+.3f})  rpy=({rxp:+.3f},{ryp:+.3f},{rzp:+.3f})")
    print(f"[Step2] grasp     xyz=({xg:+.3f},{yg:+.3f},{zg:+.3f})  rpy=({rxg:+.3f},{ryg:+.3f},{rzg:+.3f})")

    if dry_run:
        print("[Step2] dry run; skip motion")
        return True

    # 步骤 2a: 张开夹爪，避免碰撞物体
    print("[Step2] Open gripper")
    grasp_driver.open_gripper()
    pos, _, _ = grasp_driver.get_gripper_state()
    if pos < 1.0:
        print(f"[Step2] WARNING: Gripper may not be fully open! pos={pos:.2f}")

    # 步骤 2b: 移动到物体上方的预抓取位
    print("[Step2] Move to pregrasp")
    if not controller.move_to_traj(xp, yp, zp, rxp, ryp, rzp, duration=2.0):
        print("[Step2] Pregrasp IK failed")
        return False
    _wait_motion(controller, 2.0)

    # 步骤 2c: 下降到抓取位
    print("[Step2] Move to grasp")
    if not controller.move_to_traj(xg, yg, zg, rxg, ryg, rzg, duration=1.5):
        print("[Step2] Grasp IK failed")
        return False
    _wait_motion(controller, 1.5)

    # 步骤 2d: 盲抓（全扭矩闭合，不依赖力反馈）
    print("[Step2] Blind grasp")
    _blind_grasp(grasp_driver)

    # 步骤 3: Cartesian 抬升 Z 到 0.15, 再调整关节 2/3
    print("[Step3] Cartesian lift Z to 0.20")
    if not controller.move_to_traj(xg, yg, 0.20, rxg, ryg, rzg, duration=2.0):
        print("[Step3] Cartesian lift IK failed")
        return False
    _wait_motion(controller, 2.0)

    print("[Step3] Set joints 2/3 to lift position")
    controller._vlim_override = np.array([1.2, 1.2, 1.2, 0.5, 0.5, 0.5])
    q_now = controller.rebotarm.arm.get_positions()
    q_lift = q_now.copy()
    q_lift[1] = -0.57
    q_lift[2] = -0.81
    controller._q_target[:] = q_lift
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 3.0:
        current_q = controller.rebotarm.arm.get_positions()
        err = max(abs(current_q[1] - q_lift[1]), abs(current_q[2] - q_lift[2]))
        if err < 0.05:
            break
        time.sleep(0.1)
    controller._vlim_override = None
    print(f"[Step3] Lift done, joints: {q_lift.tolist()}")

    return True


# ---------------------------------------------------------------------------
# 画面渲染
# ---------------------------------------------------------------------------

def _render_display(
    image: np.ndarray,
    grasps: list[GraspPose],
    best_cube: Optional[GraspPose],
    status_text: str,
) -> np.ndarray:
    """在图像上叠加 YOLO 检测框、状态文本和最优cube位置信息。"""
    display = image.copy()

    # 绘制所有检测到的物体的边界框与类别/置信度
    for grasp in grasps:
        color = (0, 255, 0) if grasp.is_valid else (0, 165, 255)
        bx, by = int(grasp.bbox_xyxy[0]), int(grasp.bbox_xyxy[1])
        bx2, by2 = int(grasp.bbox_xyxy[2]), int(grasp.bbox_xyxy[3])
        cv2.rectangle(display, (bx, by), (bx2, by2), color, 2)
        cv2.putText(display, f"{grasp.class_name} {grasp.conf:.2f}",
                    (bx, by - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # 顶部状态栏
    cv2.putText(display, status_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

    # 底部显示最优cube的相机坐标系 xyz
    if best_cube is not None:
        x_m, y_m, z_m = best_cube.position.tolist()
        cv2.putText(display,
                    f"cube: xyz=({x_m:+.3f},{y_m:+.3f},{z_m:+.3f})",
                    (10, display.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 140), 2)

    return display


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Grasp cube and position arm demo")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dry-run", action="store_true", help="estimate only; do not move the arm")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    cfg = load_config(PROJECT_ROOT / args.config)

    # ---- 读取配置 ----
    robot_cfg = cfg.get("robot", {})
    ready_cfg = robot_cfg.get(
        "ready_pose",
        {"x": 0.25, "y": 0.0, "z": 0.35, "roll": 0.0, "pitch": 1.2, "yaw": 0.0, "duration": 3.0},
    )
    cam_cfg = cfg.get("camera", {})
    print(f"=== Camera: {cam_cfg.get('type')} ===")
    cam = make_camera(cfg)

    # ---- 状态变量 ----
    last_results: list[Any] = []          # 最近一次 YOLO 检测原始结果
    last_grasps: list[GraspPose] = []     # 从检测结果解析出的抓取位姿
    frame_index = 0
    fps_counter = 0
    fps_timer = time.perf_counter()
    fps_value = 0.0

    # ---- OpenCV 窗口 ----
    window_name = "Set - Grasp Cube"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    print("\n[Keys]  G=grasp+place  R=resume  Q/ESC=quit\n")

    # ---- 机器人相关变量 (延迟初始化) ----
    controller: Optional[RebotArmEndPose] = None
    rebotarm: Optional[RebotArm] = None
    grasp_driver: Optional[GraspDriver] = None
    T_hand_eye: Optional[np.ndarray] = None   # 手眼标定矩阵
    yolo_opts: dict[str, Any] = {}
    robot_ready = False                       # 机器人是否已成功启动
    grasp_mode = False                        # G 开关: True=持续抓取, False=关闭
    grasp_mode_lock = threading.Lock()        # 保护 grasp_mode 的线程锁
    processing_thread: Optional[threading.Thread] = None  # 后台抓取线程引用

    try:
        # ================================================================
        # 初始化阶段
        # ================================================================

        # 打开相机并预热若干帧，获取相机内参矩阵 K
        cam.open()
        cam.warm_up(15)
        K = cam.K.astype(np.float32)

        # 加载手眼标定 (eye-in-hand)
        cam_type = str(cam_cfg.get("type", "")).lower()
        T_hand_eye, hand_eye_mode = load_hand_eye(PROJECT_ROOT, cam_type)
        if T_hand_eye is None or hand_eye_mode != "eye_in_hand":
            print("[WARN] Hand-eye calibration unavailable; grasp execution disabled")
            T_hand_eye = None

        # 读取 YOLO 和抓取管线参数
        yolo_cfg = cfg.get("yolo", {})
        gp_cfg = cfg.get("grasp_pipeline", {})
        grasp_cfg = gp_cfg.get("grasp", {})

        model_name = yolo_cfg.get("model_name", "yoloe-26s-seg.pt")
        pregrasp_offset_m = float(grasp_cfg.get("pregrasp_offset_m", 0.08))
        depth_quantile = float(grasp_cfg.get("depth_quantile", 0.75))
        infer_every = max(1, int(gp_cfg.get("infer_every_live", 2)))

        # 加载 YOLO 模型
        print(f"=== Load YOLO: {model_name} ===")
        model, yolo_opts = load_yolo(cfg, project_root=PROJECT_ROOT)

        # 初始化机器人: 连接机械臂、启动控制器和夹爪驱动
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

        # 机械臂移动到预备位
        print("[Robot] Move ready")
        print(f"[Robot] Initial joints: {rebotarm.arm.get_positions().tolist()}")
        _move_ready(controller, ready_cfg)
        grasp_driver.open_gripper()
        print("[Robot] Gripper opened")

        # ---- 后台抓取线程 ----
        def _auto_process_loop() -> None:
            """持续检测并抓取 cube, 直到 grasp_mode 被关闭."""
            nonlocal last_results, last_grasps
            cube_count = 0

            while True:
                with grasp_mode_lock:
                    if not grasp_mode:
                        break

                # ---- 先检查画面中是否有 cube ----
                snap_color, snap_depth = cam.get_frame()
                if snap_color is None or snap_depth is None:
                    print("[G] ERROR: frame capture failed, aborting")
                    print(f"[G] Joints before reset: {rebotarm.arm.get_positions().tolist()}")
                    _move_ready(controller, ready_cfg)
                    break

                # 快速 YOLO 检测是否有 cube
                quick_results = model.predict(
                    snap_color,
                    verbose=False,
                    device=yolo_opts.get("device", "cpu"),
                    conf=float(yolo_opts.get("conf", 0.25)),
                    iou=float(yolo_opts.get("iou", 0.45)),
                )
                found = False
                for result in quick_results:
                    for box in result.boxes:
                        if "cube" in str(result.names.get(int(box.cls[0]), "")).lower():
                            found = True
                            break
                if not found:
                    print(f"[G] No cube found, waiting... (processed {cube_count})")
                    time.sleep(1.0)
                    continue

                # ---- Visual servoing: 旋转 joint1 使目标居中 ----
                _visual_servo_center(cam, model, yolo_opts, controller, "cube", K, cfg)

                # ---- 居中后拍照进行 3D 抓取估计 ----
                snap_color, snap_depth = cam.get_frame()
                if snap_color is None or snap_depth is None:
                    print("[G] ERROR: frame capture after servo failed, aborting")
                    print(f"[G] Joints before reset: {rebotarm.arm.get_positions().tolist()}")
                    _move_ready(controller, ready_cfg)
                    break

                snap_results = model.predict(
                    snap_color,
                    verbose=False,
                    device=yolo_opts.get("device", "cpu"),
                    conf=float(yolo_opts.get("conf", 0.25)),
                    iou=float(yolo_opts.get("iou", 0.45)),
                )
                snap_grasps = estimate_grasps(snap_results, snap_depth, K, depth_quantile=depth_quantile)

                # 找出最左侧的 cube（深度 < 0.5m）
                snap_cube = None
                for grasp in snap_grasps:
                    if not grasp.is_valid:
                        continue
                    if "cube" in grasp.class_name.lower():
                        if grasp.position[2] > 0.5:
                            continue
                        if snap_cube is None or grasp.position[0] < snap_cube.position[0]:
                            snap_cube = grasp

                if snap_cube is None:
                    print(f"[G] No valid cube after servo, waiting... (processed {cube_count})")
                    time.sleep(1.0)
                    continue

                cube_count += 1
                print(f"\n[G] === Cube #{cube_count}: class={snap_cube.class_name} conf={snap_cube.conf:.3f}")
                print(f"  position_xyz={snap_cube.position.tolist()}")

                cube_cam_x = float(snap_cube.position[0])
                print(f"  cam_x={cube_cam_x:+.3f}")

                # 更新冻结画面
                last_results = snap_results
                last_grasps = snap_grasps

                # 相机 -> 基座坐标变换
                T_cam2base = _cam_to_base(T_hand_eye, grasp_driver, cfg)

                grasp6d, pre6d = transform_grasp_pose_to_base(
                    snap_cube.position,
                    snap_cube.tcp_rotation,
                    T_cam2base,
                    pregrasp_offset_m,
                )

                # 执行抓取序列
                ok = _execute_grasp_sequence(
                    controller,
                    grasp_driver,
                    grasp6d,
                    pre6d,
                    ready_cfg,
                    dry_run=args.dry_run,
                )

                if not ok:
                    print(f"[G] ERROR: cube #{cube_count} grasp failed (IK or empty), returning to ready")
                    print(f"[G] Joints before reset: {rebotarm.arm.get_positions().tolist()}")
                    _move_ready(controller, ready_cfg)
                    continue

                print(f"[G] Cube #{cube_count} grasped!")

                # ---- 放置: 根据 joint1 角度决定左/右 ----
                rotate_q = rebotarm.arm.get_positions()
                j1_at_grasp = float(rotate_q[0])
                side = "left" if j1_at_grasp > 0 else "right"
                rotate_q[0] = 2.61 if j1_at_grasp > 0 else -2.61
                print(f"[G] Joints after lift, before rotate: {rebotarm.arm.get_positions().tolist()}")
                print(f"[G] joint1 at grasp = {j1_at_grasp:+.3f} -> {side} place (j1={rotate_q[0]:+.3f})")

                controller._vlim_override = np.array([2.0, 2.0, 2.0, 1.6, 1.6, 1.6])
                controller._q_target[:] = rotate_q
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < 8.0:
                    if np.max(np.abs(rebotarm.arm.get_positions() - rotate_q)) < 0.05:
                        print("[G] Arm reached placement position")
                        break
                    time.sleep(0.1)
                else:
                    print("[G] ERROR: arm rotation timeout, returning to ready")
                    print(f"[G] Joints before reset: {rebotarm.arm.get_positions().tolist()}")
                    controller._vlim_override = None
                    _move_ready(controller, ready_cfg)
                    continue

                controller._vlim_override = None

                # 释放 cube: 用 grasp_driver 方法张开夹爪
                print("[G] Open gripper to release cube")
                grasp_driver.open_gripper()
                time.sleep(0.3)

                # 闭合夹爪: 用盲抓闭合
                print("[G] Close gripper")
                _blind_grasp(grasp_driver, timeout=0.5)

                # 回到预备位
                print("[G] Return arm to ready position")
                print(f"[G] Joints before reset: {rebotarm.arm.get_positions().tolist()}")
                duration = float(ready_cfg.get("duration", 3.0))
                if not controller.move_to_traj(
                    x=float(ready_cfg.get("x", 0.25)),
                    y=float(ready_cfg.get("y", 0.0)),
                    z=float(ready_cfg.get("z", 0.35)),
                    roll=float(ready_cfg.get("roll", 0.0)),
                    pitch=float(ready_cfg.get("pitch", 1.2)),
                    yaw=float(ready_cfg.get("yaw", 0.0)),
                    duration=duration,
                ):
                    print("[G] Ready IK failed, reset wrist joints 5/6 to 0")
                    _reset_wrist_joints(controller)
                    if not controller.move_to_traj(
                        x=float(ready_cfg.get("x", 0.25)),
                        y=float(ready_cfg.get("y", 0.0)),
                        z=float(ready_cfg.get("z", 0.35)),
                        roll=float(ready_cfg.get("roll", 0.0)),
                        pitch=float(ready_cfg.get("pitch", 1.2)),
                        yaw=float(ready_cfg.get("yaw", 0.0)),
                        duration=duration,
                    ):
                        print("[G] Still failed after wrist reset, safe_home then retry")
                        controller.safe_home()
                        _move_ready(controller, ready_cfg)
                    else:
                        _wait_motion(controller, duration)
                else:
                    _wait_motion(controller, duration)

                print(f"[G] Cube #{cube_count} placed!")
                grasp_driver.open_gripper()

            # 模式关闭, 复位
            print(f"[G] Grasp mode off, processed {cube_count} cubes")
            print(f"[G] Joints before final reset: {rebotarm.arm.get_positions().tolist()}")
            servoing = True
            _move_ready(controller, ready_cfg)
            servoing = False
            grasp_driver.open_gripper()

        # ================================================================
        # 主循环: 实时预览 + 键盘交互
        # ================================================================
        while True:
            # ---- 获取相机帧 ----
            color_bgr, depth_mm = cam.get_frame()
            if color_bgr is None or depth_mm is None:
                continue

            # ---- FPS 统计 ----
            frame_index += 1
            fps_counter += 1
            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                fps_value = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            # ---- YOLO 实时检测 (每 infer_every 帧执行一次) ----
            if frame_index % infer_every == 0 or not last_results:
                last_results = model.predict(
                    color_bgr,
                    verbose=False,
                    device=yolo_opts.get("device", "cpu"),
                    conf=float(yolo_opts.get("conf", 0.25)),
                    iou=float(yolo_opts.get("iou", 0.45)),
                )
                last_grasps = estimate_grasps(last_results, depth_mm, K, depth_quantile=depth_quantile)

            # ---- 从检测结果中找出置信度最高的 cube ----
            best_cube = None
            for grasp in last_grasps:
                if not grasp.is_valid:
                    continue
                if "cube" in grasp.class_name.lower():
                    if best_cube is None or grasp.conf > best_cube.conf:
                        best_cube = grasp

            # ---- 状态栏文本 ----
            status = f"LIVE  {fps_value:.1f}fps | G=grasp+place R=resume Q=quit"

            # ---- 渲染并显示 ----
            display = _render_display(color_bgr, last_grasps, best_cube, status)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            # ---- 窗口关闭检测 ----
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            # ---- Q / Esc: 关闭抓取模式并退出 ----
            if key in (ord("q"), ord("Q"), 27):
                with grasp_mode_lock:
                    grasp_mode = False
                if processing_thread is not None and processing_thread.is_alive():
                    print("[Q] Waiting for grasp thread to finish...")
                    processing_thread.join(timeout=10.0)
                break

            # ---- R: 恢复实时预览 ----
            if key in (ord("r"), ord("R")):
                continue

            # ============================================================
            # G 键: 开关持续抓取模式
            # ============================================================
            if key in (ord("g"), ord("G")):
                if T_hand_eye is None:
                    print("[G] Hand-eye calibration unavailable")
                    continue

                with grasp_mode_lock:
                    grasp_mode = not grasp_mode

                if grasp_mode:
                    print("\n[G] === Grasp mode ON ===")
                    processing_thread = threading.Thread(target=_auto_process_loop, daemon=True)
                    processing_thread.start()
                else:
                    print("\n[G] === Grasp mode OFF (resetting after current cube) ===")

    # ================================================================
    # 清理阶段
    # ================================================================
    finally:
        print("\n[Exit] Blind close gripper and home")
        # 盲抓闭合夹爪
        try:
            if robot_ready and grasp_driver is not None and controller is not None and getattr(controller, "_running", False):
                _blind_grasp(grasp_driver, timeout=1.0)
        except Exception as exc:
            print(f"[Exit] {exc}")
        # 机械臂回零并断开连接
        try:
            if controller is not None and getattr(controller, "_running", False):
                controller.end()
            elif rebotarm is not None:
                rebotarm.disconnect()
        except Exception as exc:
            print(f"[Exit] {exc}")
        # 关闭相机
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
