"""
Minimal script: open gripper only.
Usage: python scripts/test_open_gripper.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from drivers.robot.grasp_driver import GraspDriver, selected_arm_config
from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose


def main() -> int:
    selected = selected_arm_config()
    rebotarm = RebotArm()
    controller = RebotArmEndPose(rebotarm, arm_control_mode=selected.controller_mode)
    grasp_driver = GraspDriver(rebotarm, controller)

    try:
        grasp_driver.start()
        print("Opening gripper...")
        grasp_driver.open_gripper()
        print("Gripper opened.")

    finally:
        try:
            if getattr(controller, "_running", False):
                grasp_driver.release_gripper()
                controller.end()
        except Exception:
            pass
        print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
