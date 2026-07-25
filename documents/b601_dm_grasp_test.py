"""
B601-DM 抓握能力测试 & 问题诊断 Demo

测试范围：
  1. 机械臂连接确认 & DM 模式验证
  2. Cartesian 运动基础能力
  3. 夹爪开合测试 (不同距离)
  4. 力控抓取测试 (不同力度)
  5. 空夹/实夹 判定准确性
  6. 连续抓取-释放循环
  7. 实时状态监控 (位置/速度/力矩/夹爪状态)

键盘：
  O: 张开夹爪到最大
  C: 闭合夹爪抓取 (默认力)
  F: 力控抓取 (逐步加力: 0.1 → 0.2 → 0.3 → 0.4 → 0.5)
  L: 释放夹爪
  1-3: 移动到预设测试位置 (1=近, 2=中, 3=远)
  R: 回到预备位
  T: 运行完整自检套件
  S: 打印当前状态快照
  Q/Esc: 释放夹爪, 回零, 退出

Usage:
    python documents/b601_dm_grasp_test.py
    python documents/b601_dm_grasp_test.py --dry-run
    python documents/b601_dm_grasp_test.py --config config/default.yaml
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from drivers.robot.grasp_driver import (
    GRIPPER_MAX_DISTANCE_M,
    GraspDriver,
    selected_arm_config,
    selected_hardware_yaml,
)
from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose
from utils.camera_utils import load_config


# ---------------------------------------------------------------------------
# 测试结果记录
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str = ""
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestReport:
    results: list[TestResult] = field(default_factory=list)
    arm_type: str = ""
    controller_mode: str = ""
    hardware_yaml: str = ""

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    @property
    def total(self) -> int:
        return len(self.results)

    def print_summary(self) -> None:
        print("\n" + "=" * 60)
        print(f"  B601-DM 抓握测试报告")
        print(f"  硬件配置: {self.hardware_yaml}")
        print(f"  机械臂类型: {self.arm_type}")
        print(f"  控制模式: {self.controller_mode}")
        print(f"  测试结果: {self.passed_count}/{self.total} 通过")
        print("=" * 60)
        for r in self.results:
            icon = "[PASS]" if r.passed else "[FAIL]"
            print(f"  {icon} {r.name}")
            if r.detail:
                print(f"       {r.detail}")
            if r.values:
                for k, v in r.values.items():
                    print(f"       {k}: {v}")
        print("=" * 60)


report = TestReport()


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _wait_motion(controller: RebotArmEndPose, duration: float, extra: float = 0.6) -> None:
    thread = getattr(controller, "_send_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=duration + extra + 2.0)
    else:
        time.sleep(duration + extra)


def _move_to(
    controller: RebotArmEndPose,
    grasp_driver: GraspDriver,
    x: float,
    y: float,
    z: float,
    roll: float = 0.0,
    pitch: float = 0.7,
    yaw: float = 0.0,
    duration: float = 2.0,
    label: str = "",
) -> bool:
    """移动到目标位姿并打印结果。"""
    prefix = f"[{label}] " if label else ""
    tcp_before = grasp_driver.get_tcp_pose()
    print(f"{prefix}Target  xyz=({x:+.3f},{y:+.3f},{z:+.3f})  rpy=({roll:+.3f},{pitch:+.3f},{yaw:+.3f})")
    print(f"{prefix}TCP before: xyz=({tcp_before[0,3]:+.3f},{tcp_before[1,3]:+.3f},{tcp_before[2,3]:+.3f})")

    ok = controller.move_to_traj(x, y, z, roll, pitch, yaw, duration=duration)
    if not ok:
        print(f"{prefix}IK failed")
        return False

    _wait_motion(controller, duration)

    tcp_after = grasp_driver.get_tcp_pose()
    err_x = abs(tcp_after[0, 3] - x)
    err_y = abs(tcp_after[1, 3] - y)
    err_z = abs(tcp_after[2, 3] - z)
    print(f"{prefix}TCP after:  xyz=({tcp_after[0,3]:+.3f},{tcp_after[1,3]:+.3f},{tcp_after[2,3]:+.3f})")
    print(f"{prefix}Position error: x={err_x:.4f}m  y={err_y:.4f}m  z={err_z:.4f}m")
    return True


def _move_ready(controller: RebotArmEndPose, ready_cfg: dict[str, Any]) -> None:
    duration = float(ready_cfg.get("duration", 3.0))
    controller.move_to_traj(
        x=float(ready_cfg.get("x", 0.3)),
        y=float(ready_cfg.get("y", 0.0)),
        z=float(ready_cfg.get("z", 0.3)),
        roll=float(ready_cfg.get("roll", 0.0)),
        pitch=float(ready_cfg.get("pitch", 0.7)),
        yaw=float(ready_cfg.get("yaw", 0.0)),
        duration=duration,
    )
    _wait_motion(controller, duration)


def _print_gripper_state(grasp_driver: GraspDriver, label: str = "") -> None:
    """打印夹爪当前状态。"""
    try:
        pos, vel, torq = grasp_driver.get_gripper_state()
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}Gripper  pos={pos:.4f}  vel={vel:.4f}  torque={torq:.4f}")
    except RuntimeError as e:
        print(f"[{label}] Gripper state unavailable: {e}")


def _print_tcp_pose(grasp_driver: GraspDriver, label: str = "") -> None:
    """打印 TCP 位姿。"""
    T = grasp_driver.get_tcp_pose()
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}TCP  xyz=({T[0,3]:+.4f},{T[1,3]:+.4f},{T[2,3]:+.4f})")
    print(f"{prefix}     rot=[{T[0,0]:+.4f} {T[0,1]:+.4f} {T[0,2]:+.4f}]")
    print(f"{prefix}         [{T[1,0]:+.4f} {T[1,1]:+.4f} {T[1,2]:+.4f}]")
    print(f"{prefix}         [{T[2,0]:+.4f} {T[2,1]:+.4f} {T[2,2]:+.4f}]")


def _print_joints(rebotarm: RebotArm) -> None:
    """打印当前关节角度。"""
    try:
        q = rebotarm.arm.get_positions()
        print(f"Joints: {[f'{v:+.4f}' for v in q.tolist()]}")
    except Exception as e:
        print(f"Joints unavailable: {e}")


# ---------------------------------------------------------------------------
# 测试套件
# ---------------------------------------------------------------------------

def test_arm_identification(rebotarm: RebotArm, repo_root: Optional[str]) -> None:
    """测试1: 确认机械臂类型为 DM。"""
    print("\n--- Test 1: Arm Identification ---")
    try:
        hw_yaml = selected_hardware_yaml(repo_root)
        selected = selected_arm_config(repo_root)
        report.hardware_yaml = str(hw_yaml.name)
        report.arm_type = selected.arm_type
        report.controller_mode = selected.controller_mode

        print(f"  Hardware YAML: {hw_yaml}")
        print(f"  Arm type: {selected.arm_type}")
        print(f"  Controller mode: {selected.controller_mode}")

        if selected.arm_type == "dm":
            report.results.append(TestResult("Arm type (DM)", True,
                                             f"controller_mode={selected.controller_mode}"))
        else:
            report.results.append(TestResult("Arm type (DM)", False,
                                             f"Expected 'dm', got '{selected.arm_type}'"))

        # 打印关节数量
        q = rebotarm.arm.get_positions()
        print(f"  Joint count: {len(q)}")
        print(f"  Joint positions: {[f'{v:+.4f}' for v in q.tolist()]}")

        # 检查夹爪
        has_gripper = getattr(rebotarm, "has_gripper", False)
        print(f"  Has gripper: {has_gripper}")
        if has_gripper:
            report.results.append(TestResult("Gripper detected", True))
        else:
            report.results.append(TestResult("Gripper detected", False,
                                             "Gripper not found in hardware config"))

    except Exception as e:
        report.results.append(TestResult("Arm identification", False, str(e)))


def test_basic_motion(
    controller: RebotArmEndPose,
    grasp_driver: GraspDriver,
    ready_cfg: dict[str, Any],
    dry_run: bool,
) -> None:
    """测试2: Cartesian 基础运动能力。"""
    print("\n--- Test 2: Basic Cartesian Motion ---")
    if dry_run:
        print("  Dry run; skipping")
        report.results.append(TestResult("Basic motion", True, "dry-run skipped"))
        return

    # 回到预备位
    print("  Moving to ready pose...")
    _move_ready(controller, ready_cfg)
    time.sleep(0.5)

    # 测试 Z 轴升降
    tcp = grasp_driver.get_tcp_pose()
    x0, y0 = float(tcp[0, 3]), float(tcp[1, 3])

    all_ok = True

    # 升到 z=0.35
    if not _move_to(controller, grasp_driver, x0, y0, 0.35, label="Z-up"):
        all_ok = False

    time.sleep(0.3)

    # 降到 z=0.20
    if not _move_to(controller, grasp_driver, x0, y0, 0.20, label="Z-down"):
        all_ok = False

    time.sleep(0.3)

    # 回到预备位
    _move_ready(controller, ready_cfg)

    report.results.append(TestResult(
        "Basic Cartesian motion",
        all_ok,
        "Z up/down test" if all_ok else "IK failure on Z motion",
    ))


def test_gripper_open_close(
    grasp_driver: GraspDriver,
    dry_run: bool,
) -> None:
    """测试3: 夹爪开合基本功能。"""
    print("\n--- Test 3: Gripper Open/Close ---")
    if dry_run:
        print("  Dry run; skipping")
        report.results.append(TestResult("Gripper open/close", True, "dry-run skipped"))
        return

    all_ok = True

    # 完全张开
    print("  Opening fully (0.09m)...")
    try:
        grasp_driver.open_gripper(distance_m=GRIPPER_MAX_DISTANCE_M, timeout=5.0)
        pos, vel, torq = grasp_driver.get_gripper_state()
        print(f"  State after open: pos={pos:.4f} vel={vel:.4f} torque={torq:.4f}")
        if abs(pos) < 0.5:  # 夹爪基本没有打开
            all_ok = False
            print("  WARNING: Gripper position barely changed after open command!")
    except Exception as e:
        all_ok = False
        print(f"  ERROR opening gripper: {e}")

    time.sleep(0.5)

    # 半开 (约 4.5cm)
    print("  Opening half (0.045m)...")
    try:
        grasp_driver.open_gripper(distance_m=0.045, timeout=3.0)
        pos, vel, torq = grasp_driver.get_gripper_state()
        print(f"  State after half-open: pos={pos:.4f} vel={vel:.4f} torque={torq:.4f}")
    except Exception as e:
        all_ok = False
        print(f"  ERROR half-opening gripper: {e}")

    time.sleep(0.5)

    # 闭合 (force grasp with no object = should report empty)
    print("  Closing gripper (empty, expect empty grasp)...")
    try:
        ok = grasp_driver.grasp(force=0.15, timeout=5.0)
        pos, vel, torq = grasp_driver.get_gripper_state()
        print(f"  Grasp result: {'HOLDING' if ok else 'EMPTY (correct, no object)'}")
        print(f"  State after close: pos={pos:.4f} vel={vel:.4f} torque={torq:.4f}")

        if ok:
            print("  WARNING: Empty grasp returned True! Check stall_vel or hard_stop_angle.")
            report.results.append(TestResult(
                "Empty grasp detection", False,
                f"Empty grasp returned True; pos={pos:.4f}, vel={vel:.4f}",
                {"stall_vel_threshold": 0.05, "startup_dist": 0.30, "hard_stop_angle": 0.05}
            ))
        else:
            report.results.append(TestResult("Empty grasp detection", True,
                                             f"Correctly detected empty; pos={pos:.4f}"))
    except Exception as e:
        all_ok = False
        print(f"  ERROR closing gripper: {e}")

    time.sleep(0.5)

    # 释放
    print("  Releasing gripper...")
    try:
        grasp_driver.release_gripper(timeout=4.0)
        pos, _, _ = grasp_driver.get_gripper_state()
        print(f"  State after release: pos={pos:.4f}")
    except Exception as e:
        all_ok = False
        print(f"  ERROR releasing gripper: {e}")

    report.results.append(TestResult(
        "Gripper open/close cycle",
        all_ok,
    ))


def test_grasp_force_levels(
    grasp_driver: GraspDriver,
    dry_run: bool,
) -> None:
    """测试4: 不同力度的力控抓取 (需要放置物体在夹爪之间)。"""
    print("\n--- Test 4: Grasp Force Levels ---")
    if dry_run:
        print("  Dry run; skipping")
        report.results.append(TestResult("Grasp force levels", True, "dry-run skipped"))
        return

    force_levels = [0.10, 0.15, 0.20, 0.30, 0.40]
    results_table = []

    for force in force_levels:
        print(f"\n  Testing force={force:.2f}...")
        print(f"  >>> PLACE an object between the gripper jaws, then this test will grasp it. <<<")
        print(f"  Waiting 3 seconds...")
        time.sleep(3.0)

        # 先张开
        grasp_driver.open_gripper(distance_m=0.07, timeout=3.0)
        time.sleep(0.5)

        # 力控抓取
        try:
            ok = grasp_driver.grasp(force=force, timeout=5.0)
            pos, vel, torq = grasp_driver.get_gripper_state()
            status = "HOLDING" if ok else "EMPTY"
            results_table.append({
                "force": force,
                "result": status,
                "pos": round(pos, 4),
                "vel": round(vel, 4),
                "torque": round(torq, 4),
            })
            print(f"  force={force:.2f}: {status}  pos={pos:.4f}  torque={torq:.4f}")
        except Exception as e:
            print(f"  force={force:.2f}: ERROR - {e}")
            results_table.append({"force": force, "result": "ERROR", "pos": 0, "vel": 0, "torque": 0})

        # 释放
        grasp_driver.release_gripper(timeout=3.0)
        time.sleep(0.5)

    # 汇总
    print("\n  Force test results:")
    print(f"  {'Force':>8}  {'Result':>8}  {'Pos':>8}  {'Torque':>8}")
    for r in results_table:
        print(f"  {r['force']:>8.2f}  {r['result']:>8}  {r['pos']:>8.4f}  {r['torque']:>8.4f}")

    holding_count = sum(1 for r in results_table if r["result"] == "HOLDING")
    report.results.append(TestResult(
        "Multi-force grasp test",
        True,
        f"{holding_count}/{len(force_levels)} force levels held object",
        {"results": str(results_table)},
    ))


def test_grasp_repeatability(
    grasp_driver: GraspDriver,
    dry_run: bool,
    cycles: int = 3,
) -> None:
    """测试5: 连续抓取-释放循环的重复性。"""
    print(f"\n--- Test 5: Grasp/Release Repeatability ({cycles} cycles) ---")
    if dry_run:
        print("  Dry run; skipping")
        report.results.append(TestResult("Grasp repeatability", True, "dry-run skipped"))
        return

    times_grasp: list[float] = []
    times_release: list[float] = []
    positions: list[float] = []

    for i in range(cycles):
        print(f"\n  Cycle {i + 1}/{cycles}")
        print(f"  >>> PLACE object, test will auto-run in 2 seconds <<<")
        time.sleep(2.0)

        # 张开
        t0 = time.perf_counter()
        grasp_driver.open_gripper(distance_m=0.07, timeout=3.0)
        t_open = time.perf_counter() - t0

        time.sleep(0.3)

        # 抓取
        t0 = time.perf_counter()
        ok = grasp_driver.grasp(timeout=5.0)
        t_grasp = time.perf_counter() - t0
        times_grasp.append(t_grasp)

        if ok:
            pos, vel, torq = grasp_driver.get_gripper_state()
            positions.append(pos)
            print(f"  Cycle {i + 1}: GRASPED  pos={pos:.4f}  time={t_grasp:.3f}s")
        else:
            print(f"  Cycle {i + 1}: EMPTY  time={t_grasp:.3f}s")

        # 释放
        t0 = time.perf_counter()
        grasp_driver.release_gripper(timeout=3.0)
        t_release = time.perf_counter() - t0
        times_release.append(t_release)

        time.sleep(0.5)

    # 统计
    if times_grasp:
        print(f"\n  Grasp timing: avg={np.mean(times_grasp):.3f}s  min={np.min(times_grasp):.3f}s  max={np.max(times_grasp):.3f}s")
    if times_release:
        print(f"  Release timing: avg={np.mean(times_release):.3f}s  min={np.min(times_release):.3f}s  max={np.max(times_release):.3f}s")
    if positions:
        print(f"  Contact positions: {[f'{p:.4f}' for p in positions]}")
        print(f"  Position std: {np.std(positions):.6f}")

    report.results.append(TestResult(
        "Grasp repeatability",
        len(times_grasp) >= cycles,
        f"{len(times_grasp)}/{cycles} successful grasps",
        {
            "grasp_avg_s": round(np.mean(times_grasp), 3) if times_grasp else 0,
            "pos_std": round(float(np.std(positions)), 6) if len(positions) > 1 else 0,
        },
    ))


def test_positioned_grasp(
    controller: RebotArmEndPose,
    grasp_driver: GraspDriver,
    ready_cfg: dict[str, Any],
    dry_run: bool,
) -> None:
    """测试6: 在预设位置执行抓取 (需要提前把物体放在目标位置)。"""
    print("\n--- Test 6: Positioned Grasp ---")
    if dry_run:
        print("  Dry run; skipping")
        report.results.append(TestResult("Positioned grasp", True, "dry-run skipped"))
        return

    # 测试位置: 三个不同的 (y, z)
    test_positions = [
        {"x": 0.35, "y": 0.05, "z": 0.12, "name": "Front-center"},
        {"x": 0.35, "y": 0.12, "z": 0.12, "name": "Front-right"},
        {"x": 0.35, "y": -0.08, "z": 0.12, "name": "Front-left"},
    ]

    positions_ok = 0

    for tp in test_positions:
        name = tp["name"]
        print(f"\n  >>> PLACE object at {name} position, then press Enter... <<<")
        input(f"  [{name}] Press Enter when ready...")

        _move_ready(controller, ready_cfg)
        time.sleep(0.5)

        # 开爪
        grasp_driver.open_gripper(distance_m=0.06, timeout=3.0)
        time.sleep(0.3)

        # 预抓取位 (目标上方)
        pre_z = tp["z"] + 0.10
        print(f"  [{name}] Moving to pregrasp z={pre_z:.3f}...")
        if not _move_to(controller, grasp_driver, tp["x"], tp["y"], pre_z, label=name):
            continue

        time.sleep(0.3)

        # 抓取位
        print(f"  [{name}] Moving to grasp z={tp['z']:.3f}...")
        if not _move_to(controller, grasp_driver, tp["x"], tp["y"], tp["z"],
                        duration=1.5, label=f"{name}-grasp"):
            _move_ready(controller, ready_cfg)
            continue

        time.sleep(0.3)

        # 抓取
        ok = grasp_driver.grasp(force=0.25, timeout=5.0)
        if ok:
            pos, _, torq = grasp_driver.get_gripper_state()
            print(f"  [{name}] GRASPED! pos={pos:.4f} torque={torq:.4f}")
            positions_ok += 1
        else:
            pos, vel, torq = grasp_driver.get_gripper_state()
            print(f"  [{name}] EMPTY GRASP - pos={pos:.4f} vel={vel:.4f} torque={torq:.4f}")

        # 抬升
        _move_to(controller, grasp_driver, tp["x"], tp["y"], 0.28, duration=2.0, label=f"{name}-lift")

        time.sleep(0.3)

        # 释放
        grasp_driver.release_gripper(timeout=3.0)

        # 回到预备位
        _move_ready(controller, ready_cfg)
        time.sleep(0.5)

    report.results.append(TestResult(
        "Positioned grasp",
        positions_ok > 0,
        f"{positions_ok}/{len(test_positions)} positions grasped successfully",
    ))


def test_continuous_monitoring(
    grasp_driver: GraspDriver,
    rebotarm: RebotArm,
    duration: float = 10.0,
) -> None:
    """测试7: 连续状态监控 (观察夹爪状态随时间变化)。"""
    print(f"\n--- Test 7: Continuous Monitoring ({duration}s) ---")
    print("  Recording gripper state at 10Hz...")

    samples: list[dict] = []
    deadline = time.monotonic() + duration

    while time.monotonic() < deadline:
        try:
            pos, vel, torq = grasp_driver.get_gripper_state()
            samples.append({
                "t": round(time.monotonic(), 3),
                "pos": round(float(pos), 4),
                "vel": round(float(vel), 4),
                "torque": round(float(torq), 4),
            })
        except RuntimeError:
            pass
        time.sleep(0.1)

    if samples:
        positions = [s["pos"] for s in samples]
        velocities = [s["vel"] for s in samples]
        torques = [s["torque"] for s in samples]

        print(f"  Samples: {len(samples)}")
        print(f"  Position: min={min(positions):.4f}  max={max(positions):.4f}  std={np.std(positions):.6f}")
        print(f"  Velocity: min={min(velocities):.4f}  max={max(velocities):.4f}  std={np.std(velocities):.6f}")
        print(f"  Torque:   min={min(torques):.4f}  max={max(torques):.4f}  std={np.std(torques):.6f}")

        # 检测异常抖动
        vel_high = sum(1 for v in velocities if abs(v) > 0.1)
        if vel_high > 0:
            print(f"  WARNING: {vel_high}/{len(samples)} samples with velocity > 0.1 (possible jitter)")

        report.results.append(TestResult(
            "Continuous monitoring",
            True,
            f"{len(samples)} samples collected",
            {
                "pos_range": f"[{min(positions):.4f}, {max(positions):.4f}]",
                "vel_max": round(max(abs(v) for v in velocities), 4),
                "high_vel_count": vel_high,
            },
        ))
    else:
        report.results.append(TestResult("Continuous monitoring", False, "No samples collected"))


# ---------------------------------------------------------------------------
# 完整自检
# ---------------------------------------------------------------------------

def run_full_self_test(
    rebotarm: RebotArm,
    controller: RebotArmEndPose,
    grasp_driver: GraspDriver,
    ready_cfg: dict[str, Any],
    repo_root: Optional[str],
    dry_run: bool,
) -> None:
    """运行完整的 B601-DM 自检套件。"""
    print("\n" + "=" * 60)
    print("  B601-DM 完整自检套件")
    print("=" * 60)

    test_arm_identification(rebotarm, repo_root)
    test_basic_motion(controller, grasp_driver, ready_cfg, dry_run)
    test_gripper_open_close(grasp_driver, dry_run)

    # 力控测试需要人工放置物体
    print("\n  >>> The following tests require placing objects. Continue? [Y/n] <<<")
    # 非交互式环境下默认跳过
    # test_grasp_force_levels(grasp_driver, dry_run)
    # test_grasp_repeatability(grasp_driver, dry_run)
    # test_positioned_grasp(controller, grasp_driver, ready_cfg, dry_run)

    test_continuous_monitoring(grasp_driver, rebotarm, duration=5.0)

    report.print_summary()


# ---------------------------------------------------------------------------
# 命令行参数
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B601-DM Grasp Capability Test Demo")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Estimate only; do not move the arm")
    parser.add_argument("--self-test", action="store_true", help="Run full self-test suite automatically")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    cfg = load_config(PROJECT_ROOT / args.config)

    robot_cfg = cfg.get("robot", {})
    ready_cfg = robot_cfg.get(
        "ready_pose",
        {"x": 0.3, "y": 0.0, "z": 0.3, "roll": 0.0, "pitch": 0.7, "yaw": 0.0, "duration": 3.0},
    )
    repo_root = robot_cfg.get("repo_root")

    print("=" * 60)
    print("  B601-DM 抓握能力测试 Demo")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 60)
    print()
    print("  按键说明:")
    print("    O: 张开夹爪到最大 (0.09m)")
    print("    5: 张开夹爪一半 (0.045m)")
    print("    C: 闭合抓取 (默认力 0.30)")
    print("    F: 力控抓取循环 (0.10→0.15→0.20→0.30→0.40)")
    print("    L: 释放夹爪")
    print("    1: 移动到测试位 1 (x=0.35, z=0.10)")
    print("    2: 移动到测试位 2 (x=0.35, z=0.18)")
    print("    3: 移动到测试位 3 (x=0.30, z=0.25)")
    print("    R: 回到预备位")
    print("    S: 打印当前状态快照")
    print("    T: 运行完整自检套件")
    print("    M: 连续监控夹爪状态 (5秒)")
    print("    Q/Esc: 退出")
    print()

    controller: Optional[RebotArmEndPose] = None
    rebotarm: Optional[RebotArm] = None
    grasp_driver: Optional[GraspDriver] = None
    robot_ready = False
    monitor_thread: Optional[threading.Thread] = None
    monitor_stop = threading.Event()

    try:
        # ---- 初始化 ----
        if not args.dry_run:
            print("=== Init robot ===")
            selected = selected_arm_config(repo_root)
            print(f"  Arm type: {selected.arm_type}")
            print(f"  Controller mode: {selected.controller_mode}")

            if selected.arm_type != "dm":
                print(f"  WARNING: Expected arm_type='dm', got '{selected.arm_type}'")
                print("  This demo is designed for B601-DM. Results may vary.")

            rebotarm = RebotArm()
            controller = RebotArmEndPose(rebotarm, arm_control_mode=selected.controller_mode)
            grasp_driver = GraspDriver(
                rebotarm,
                controller,
                gripper_config=robot_cfg.get("gripper"),
                repo_root=repo_root,
            )

            print("  Starting GraspDriver...")
            grasp_driver.start()
            robot_ready = True
            print("  Robot ready!")

            _print_joints(rebotarm)
            _print_gripper_state(grasp_driver, "Init")
            _print_tcp_pose(grasp_driver, "Init")

        else:
            print("=== Dry run mode: robot will not be initialized ===")

        # ---- 自检模式 ----
        if args.self_test and robot_ready:
            assert rebotarm is not None
            assert controller is not None
            assert grasp_driver is not None
            run_full_self_test(rebotarm, controller, grasp_driver, ready_cfg, repo_root, args.dry_run)
            return 0

        # ---- 交互式主循环 ----
        print("\nReady for commands. Press a key...\n")

        force_index = 0
        force_sequence = [0.10, 0.15, 0.20, 0.30, 0.40]

        while True:
            # 读取键盘输入
            try:
                import termios
                import tty

                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                tty.setraw(fd)
                key = sys.stdin.read(1)
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            except Exception:
                key = "q"

            if key in ("q", "Q", "\x1b"):
                break

            if not robot_ready and not args.dry_run:
                print("Robot not initialized")
                continue

            # ---- O: 完全张开 ----
            if key in ("o", "O"):
                print("\n[O] Opening gripper fully (0.09m)...")
                if not args.dry_run:
                    grasp_driver.open_gripper(distance_m=GRIPPER_MAX_DISTANCE_M, timeout=3.0)
                    _print_gripper_state(grasp_driver, "O")

            # ---- 5: 半开 ----
            elif key == "5":
                print("\n[5] Opening gripper half (0.045m)...")
                if not args.dry_run:
                    grasp_driver.open_gripper(distance_m=0.045, timeout=3.0)
                    _print_gripper_state(grasp_driver, "5")

            # ---- C: 闭合抓取 ----
            elif key in ("c", "C"):
                print("\n[C] Grasping with default force (0.30)...")
                if not args.dry_run:
                    ok = grasp_driver.grasp(timeout=5.0)
                    print(f"  Result: {'HOLDING' if ok else 'EMPTY'}")
                    _print_gripper_state(grasp_driver, "C")

            # ---- F: 力控抓取循环 ----
            elif key in ("f", "F"):
                force = force_sequence[force_index % len(force_sequence)]
                force_index += 1
                print(f"\n[F] Grasping with force={force:.2f}...")
                if not args.dry_run:
                    grasp_driver.open_gripper(distance_m=0.06, timeout=3.0)
                    time.sleep(0.3)
                    ok = grasp_driver.grasp(force=force, timeout=5.0)
                    print(f"  Result: {'HOLDING' if ok else 'EMPTY'}")
                    _print_gripper_state(grasp_driver, "F")

            # ---- L: 释放 ----
            elif key in ("l", "L"):
                print("\n[L] Releasing gripper...")
                if not args.dry_run:
                    grasp_driver.release_gripper(timeout=4.0)
                    _print_gripper_state(grasp_driver, "L")

            # ---- 1/2/3: 测试位置 ----
            elif key in ("1", "2", "3"):
                positions = {
                    "1": {"x": 0.35, "z": 0.10, "name": "Low"},
                    "2": {"x": 0.35, "z": 0.18, "name": "Mid"},
                    "3": {"x": 0.30, "z": 0.25, "name": "High"},
                }
                p = positions[key]
                print(f"\n[{key}] Moving to {p['name']} position: x={p['x']}, z={p['z']}...")
                if not args.dry_run:
                    _move_to(controller, grasp_driver, p["x"], 0.0, p["z"], label=p["name"])
                    _print_tcp_pose(grasp_driver, key)

            # ---- R: 预备位 ----
            elif key in ("r", "R"):
                print("\n[R] Moving to ready pose...")
                if not args.dry_run:
                    _move_ready(controller, ready_cfg)
                    _print_joints(rebotarm)
                    _print_tcp_pose(grasp_driver, "Ready")

            # ---- S: 状态快照 ----
            elif key in ("s", "S"):
                print("\n[S] State Snapshot:")
                if not args.dry_run:
                    _print_joints(rebotarm)
                    _print_gripper_state(grasp_driver, "S")
                    _print_tcp_pose(grasp_driver, "S")

            # ---- M: 连续监控 5s ----
            elif key in ("m", "M"):
                print("\n[M] Monitoring gripper for 5 seconds...")
                if not args.dry_run:
                    print("  t       pos        vel        torque")
                    print("  ------  ---------  ---------  ---------")
                    deadline = time.monotonic() + 5.0
                    while time.monotonic() < deadline:
                        try:
                            pos, vel, torq = grasp_driver.get_gripper_state()
                            print(f"  {time.monotonic():>6.1f}  {pos:>9.4f}  {vel:>9.4f}  {torq:>9.4f}")
                        except RuntimeError:
                            pass
                        time.sleep(0.2)
                    print("  Monitoring complete.")

            # ---- T: 完整自检 ----
            elif key in ("t", "T"):
                if not args.dry_run and robot_ready:
                    run_full_self_test(rebotarm, controller, grasp_driver, ready_cfg, repo_root, args.dry_run)
                else:
                    print("\n[T] Self-test requires robot to be initialized")

            else:
                if key.isprintable():
                    print(f"\n  Unknown key: '{key}'")

    # ================================================================
    # 清理
    # ================================================================
    finally:
        monitor_stop.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=2.0)

        print("\n[Exit] Releasing gripper and homing...")
        try:
            if robot_ready and grasp_driver is not None and controller is not None:
                if getattr(controller, "_running", False):
                    grasp_driver.release_gripper(timeout=3.0)
        except Exception as exc:
            print(f"[Exit] Release error: {exc}")

        try:
            if controller is not None and getattr(controller, "_running", False):
                controller.end()
            elif rebotarm is not None:
                rebotarm.disconnect()
        except Exception as exc:
            print(f"[Exit] Disconnect error: {exc}")

        print("Done.")
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
