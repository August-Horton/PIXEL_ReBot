"""
Grasp banana demo based on YOLO.

Workflow:
  1. Detect banana
  2. Go to grasp banana
  3. Grasp banana and lift to ready position
  4. Auto: move arm to placement joints (left/right), open gripper to release

Keys:
  G: capture, grasp, and auto-place banana
  R: resume live preview
  Q/Esc: release gripper, home, and exit

Usage:
    python scripts/set.py
    python scripts/set.py --dry-run
"""

from __future__ import annotations

import argparse
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
    """将关节5和关节6复位为0 (速度0.3)，用于IK失败后恢复。"""
    q_now = controller.rebotarm.arm.get_positions()
    q_reset = q_now.copy()
    q_reset[4] = 0.0
    q_reset[5] = 0.0
    print(f"[Reset] Wrist joints 5/6 -> 0 from {q_now[[4,5]].tolist()}")
    controller._vlim_override = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
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


# ---------------------------------------------------------------------------
# 抓取动作序列
# ---------------------------------------------------------------------------

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

    # 步骤 2d: 闭合夹爪抓取物体
    print("[Step2] Closing gripper")
    ok = grasp_driver.grasp()
    print("[Step2] Holding object" if ok else "[Step2] Empty grasp")
    if not ok:
        return False

    # 步骤 3: Cartesian 抬升 Z 到 0.15, 再调整关节 2/3
    print("[Step3] Cartesian lift Z to 0.20")
    if not controller.move_to_traj(xg, yg, 0.20, rxg, ryg, rzg, duration=2.0):
        print("[Step3] Cartesian lift IK failed")
        return False
    _wait_motion(controller, 2.0)

    print("[Step3] Set joints 2/3 to lift position")
    controller._vlim_override = np.array([0.8, 0.8, 0.8, 0.5, 0.5, 0.5])
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
    best_banana: Optional[GraspPose],
    status_text: str,
) -> np.ndarray:
    """在图像上叠加 YOLO 检测框、OBB方向、状态文本和最优香蕉位置信息。"""
    display = image.copy()

    # 绘制所有检测到的物体的边界框与类别/置信度
    for grasp in grasps:
        is_banana = "banana" in grasp.class_name.lower()
        color = (0, 255, 0) if grasp.is_valid else (0, 165, 255)

        # OBB 方向可视化 + 长轴/短轴 (仅香蕉)
        if is_banana and grasp.is_valid:
            rect_pts = getattr(grasp, "rect_points", None)
            has_obb = rect_pts is not None and rect_pts.shape == (4, 2)
            if has_obb:
                pts_i = rect_pts.astype(np.int32)
                cv2.polylines(display, [pts_i], True, (255, 200, 0), 2)

            # 短轴线 (蓝色, 夹爪抓取方向) 和 长轴线 (红色, 物体纵轴)
            short_pts = getattr(grasp, "short_edge_points", None)
            if short_pts is not None and short_pts.shape == (2, 2):
                sp = short_pts.astype(np.int32)
                mid = (sp[0] + sp[1]) // 2
                long_angle_rad = np.deg2rad(grasp.long_axis_angle_deg)
                long_len = grasp.long_edge_px * 0.5
                long_end = np.int32([mid[0] + long_len * np.cos(long_angle_rad),
                                     mid[1] + long_len * np.sin(long_angle_rad)])
                cv2.line(display, tuple(mid), tuple(long_end), (0, 0, 255), 2)
                cv2.line(display, tuple(sp[0]), tuple(sp[1]), (255, 0, 0), 2)
                cv2.circle(display, tuple(sp[0]), 3, (255, 100, 0), -1)
                cv2.circle(display, tuple(sp[1]), 3, (255, 100, 0), -1)

        # 轴对齐 bbox (香蕉用虚线以示区分)
        bx, by = int(grasp.bbox_xyxy[0]), int(grasp.bbox_xyxy[1])
        bx2, by2 = int(grasp.bbox_xyxy[2]), int(grasp.bbox_xyxy[3])
        if is_banana and grasp.is_valid:
            # 虚线 bbox
            for i in range(0, by2 - by, 6):
                cv2.line(display, (bx, by + i), (bx, min(by + i + 3, by2)), color, 1)
                cv2.line(display, (bx2, by + i), (bx2, min(by + i + 3, by2)), color, 1)
            for i in range(0, bx2 - bx, 6):
                cv2.line(display, (bx + i, by), (min(bx + i + 3, bx2), by), color, 1)
                cv2.line(display, (bx + i, by2), (min(bx + i + 3, bx2), by2), color, 1)
        else:
            cv2.rectangle(display, (bx, by), (bx2, by2), color, 2)
        cv2.putText(display, f"{grasp.class_name} {grasp.conf:.2f}",
                    (bx, by - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # 顶部状态栏
    cv2.putText(display, status_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

    # 底部显示最优香蕉的相机坐标系 xyz + OBB角度
    if best_banana is not None:
        x_m, y_m, z_m = best_banana.position.tolist()
        long_deg = best_banana.long_axis_angle_deg
        short_deg = best_banana.angle_deg
        cv2.putText(display,
                    f"xyz=({x_m:+.3f},{y_m:+.3f},{z_m:+.3f})  long={long_deg:.0f}deg  grip={short_deg:.0f}deg",
                    (10, display.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (120, 255, 140), 2)

    return display


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Grasp banana and position arm demo")
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
    window_name = "Set - Grasp Banana"
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

        # ---- 后台抓取线程 ----
        def _auto_process_loop() -> None:
            """持续检测并抓取香蕉, 直到 grasp_mode 被关闭."""
            nonlocal last_results, last_grasps
            banana_count = 0

            while True:
                with grasp_mode_lock:
                    if not grasp_mode:
                        break

                # 拍一张快照
                snap_color, snap_depth = cam.get_frame()
                if snap_color is None or snap_depth is None:
                    print("[G] ERROR: frame capture failed, aborting")
                    print(f"[G] Joints before reset: {rebotarm.arm.get_positions().tolist()}")
                    _move_ready(controller, ready_cfg)
                    break

                # YOLO 检测
                snap_results = model.predict(
                    snap_color,
                    verbose=False,
                    device=yolo_opts.get("device", "cpu"),
                    conf=float(yolo_opts.get("conf", 0.25)),
                    iou=float(yolo_opts.get("iou", 0.45)),
                )
                snap_grasps = estimate_grasps(snap_results, snap_depth, K, depth_quantile=depth_quantile)

                # 找出最左侧的香蕉
                snap_banana = None
                for grasp in snap_grasps:
                    if not grasp.is_valid:
                        continue
                    if "banana" in grasp.class_name.lower():
                        if snap_banana is None or grasp.bbox_xyxy[0] < snap_banana.bbox_xyxy[0]:
                            snap_banana = grasp

                if snap_banana is None:
                    print(f"[G] No banana found, waiting... (processed {banana_count})")
                    time.sleep(1.0)
                    continue

                banana_count += 1
                print(f"\n[G] === Banana #{banana_count}: class={snap_banana.class_name} conf={snap_banana.conf:.3f}")
                print(f"  position_xyz={snap_banana.position.tolist()}")

                banana_cam_x = float(snap_banana.position[0])
                side = "right" if banana_cam_x > 0 else "left"
                print(f"  cam_x={banana_cam_x:+.3f} -> {side} side")

                # 更新冻结画面
                last_results = snap_results
                last_grasps = snap_grasps

                # 相机 -> 基座坐标变换
                T_cam2base = _cam_to_base(T_hand_eye, grasp_driver, cfg)

                grasp6d, pre6d = transform_grasp_pose_to_base(
                    snap_banana.position,
                    snap_banana.tcp_rotation,
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
                    print(f"[G] ERROR: banana #{banana_count} grasp failed (IK or empty), returning to ready")
                    print(f"[G] Joints before reset: {rebotarm.arm.get_positions().tolist()}")
                    _move_ready(controller, ready_cfg)
                    continue

                print(f"[G] Banana #{banana_count} grasped!")

                # ---- 放置: 旋转 joint1 ----
                rotate_q = rebotarm.arm.get_positions()
                print(f"[G] Joints after lift, before rotate: {rotate_q.tolist()}")
                rotate_q[0] = -2.61 if banana_cam_x > 0 else 2.61
                print(f"[G] Rotate joint1 to: {rotate_q.tolist()}")

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

                # 释放香蕉
                with grasp_driver._state_lock:
                    grasp_driver._state = grasp_driver._STATE_IDLE

                print("[G] Open gripper to release banana")
                rebotarm.gripper.send_mit(np.array([-5.500]), kp=np.array([8.0]), kd=np.array([1.0]))
                time.sleep(0.3)

                with grasp_driver._state_lock:
                    grasp_driver._state = grasp_driver._STATE_IDLE
                    grasp_driver._target_pos = 0.0

                print("[G] Close gripper")
                rebotarm.gripper.send_mit(np.array([0.0]), kp=np.array([8.0]), kd=np.array([1.0]))
                time.sleep(0.3)

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

                print(f"[G] Banana #{banana_count} placed!")

            # 模式关闭, 复位
            print(f"[G] Grasp mode off, processed {banana_count} bananas")
            print(f"[G] Joints before final reset: {rebotarm.arm.get_positions().tolist()}")
            _move_ready(controller, ready_cfg)

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

            # ---- 从检测结果中找出置信度最高的 banana ----
            best_banana = None
            for grasp in last_grasps:
                if not grasp.is_valid:
                    continue
                if "banana" in grasp.class_name.lower():
                    if best_banana is None or grasp.conf > best_banana.conf:
                        best_banana = grasp

            # ---- 状态栏文本 ----
            status = f"LIVE  {fps_value:.1f}fps | G=grasp+place R=resume Q=quit"

            # ---- 渲染并显示 ----
            display = _render_display(color_bgr, last_grasps, best_banana, status)

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
                    print("\n[G] === Grasp mode OFF (resetting after current banana) ===")

    # ================================================================
    # 清理阶段
    # ================================================================
    finally:
        print("\n[Exit] Release gripper and home")
        # 释放夹爪
        try:
            if robot_ready and grasp_driver is not None and controller is not None and getattr(controller, "_running", False):
                grasp_driver.release_gripper()
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
