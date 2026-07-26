"""
Ordinary grasp test.

Keys:
  Left click: sample depth at a pixel.
  G: print the current best grasp.
  Q/Esc: exit.

Usage:
    python scripts/ordinary_grasp_pipeline.py
"""

import os
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _path in (PROJECT_ROOT,):
    path_str = str(_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from drivers.camera import make_camera
from utils.camera_utils import load_config
from utils.ordinary_grasp import draw_grasp, estimate_grasps, get_depth_mm, select_best_grasp
from utils.transforms import canonicalize_parallel_gripper_tcp_rotation, rotation_matrix_to_euler_zyx


clicked_point = {"u": -1, "v": -1}


def mouse_callback(event, x, y, flags, param):
    del flags, param
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point["u"] = x
        clicked_point["v"] = y
        print(f"[Test] Clicked pixel: (u={x}, v={y})")


def print_best_grasp(grasp) -> None:
    tcp_rotation = canonicalize_parallel_gripper_tcp_rotation(grasp.tcp_rotation)
    print("\n[G] Best grasp:")
    print(f"  class={grasp.class_name} conf={grasp.conf:.3f}")
    print(f"  center_px={grasp.center_px} angle_deg={grasp.angle_deg:.2f}")
    print(f"  jaw_width_m={grasp.jaw_width_m:.4f} object_length_m={grasp.object_length_m:.4f}")
    print(f"  position_xyz={grasp.position.tolist()}")
    print(f"  grasp_rpy={rotation_matrix_to_euler_zyx(grasp.rotation).tolist()}")
    print(f"  tcp_rpy={rotation_matrix_to_euler_zyx(tcp_rotation).tolist()}")


def main():
    config_path = PROJECT_ROOT / "config" / "default.yaml"
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)
    cam_type = str(cfg.get("camera", {}).get("type", "")).lower()
    yolo_cfg = cfg.get("yolo", {})
    det_cfg = cfg.get("detection", {})
    grasp_cfg = cfg.get("grasp_pipeline", {}).get("grasp", {})

    model_name = yolo_cfg.get("model_name", "yoloe-26s-seg.pt")
    device = yolo_cfg.get("device", "cpu")
    use_world = bool(yolo_cfg.get("use_world", False))
    custom_classes = list(yolo_cfg.get("custom_classes", ["cup"]))
    conf_thres = float(det_cfg.get("conf_threshold", 0.25))
    iou_thres = float(det_cfg.get("iou_threshold", 0.45))
    depth_quantile = float(grasp_cfg.get("depth_quantile", 0.75))

    print("=== Init YOLO ===")
    model_path = models_dir / model_name
    print(f"Load model: {model_path}")
    model = YOLO(str(model_path))
    if use_world and ("world" in model_name.lower() or "yoloe" in model_name.lower()):
        model.set_classes(custom_classes)
        print(f"Open-vocabulary classes: {custom_classes}")

    print(f"\n=== Init camera: {cam_type} ===")
    cam = make_camera(cfg)
    cam.open()
    cam.warm_up(10)
    K = cam.K.astype("float32")
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    print(f"[Camera] fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    window_name = f"Ordinary Grasp Test ({cam_type})"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)
    print("\n[Keys] Left click=sample depth  G=print best  Q=quit")

    try:
        while True:
            color_image, depth_mm = cam.get_frame()
            if color_image is None or depth_mm is None:
                continue

            results = model.predict(
                color_image,
                verbose=False,
                device=device,
                conf=conf_thres,
                iou=iou_thres,
            )

            grasps = estimate_grasps(results, depth_mm, K, depth_quantile=depth_quantile)
            for grasp in grasps:
                draw_grasp(color_image, grasp)

            best = select_best_grasp(grasps)
            if best is not None:
                x_m, y_m, z_m = best.position.tolist()
                best_text = (
                    f"best={best.class_name} conf={best.conf:.2f} "
                    f"xyz=({x_m:+.3f},{y_m:+.3f},{z_m:+.3f}) jaw={best.jaw_width_m * 100:.1f}cm"
                )
                cv2.putText(
                    color_image,
                    best_text,
                    (10, color_image.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (120, 255, 140),
                    2,
                )

            cu, cv_y = clicked_point["u"], clicked_point["v"]
            if cu >= 0 and cv_y >= 0:
                z_mm = get_depth_mm(depth_mm, cu, cv_y, 5)
                if z_mm > 0:
                    z_m = z_mm / 1000.0
                    x_m = (cu - cx) * z_m / fx
                    y_m = (cv_y - cy) * z_m / fy
                    cv2.drawMarker(color_image, (cu, cv_y), (255, 0, 255), cv2.MARKER_CROSS, 18, 2)
                    label = f"TEST X:{x_m:.3f} Y:{y_m:.3f} Z:{z_m:.3f} m"
                    cv2.rectangle(color_image, (cu + 5, cv_y - 25), (cu + 320, cv_y + 5), (0, 0, 0), -1)
                    cv2.putText(
                        color_image,
                        label,
                        (cu + 8, cv_y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 0, 255),
                        2,
                    )

            title = f"{cam_type.upper()} | {model_name} | ordinary grasp"
            cv2.putText(color_image, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
            cv2.imshow(window_name, color_image)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("g"), ord("G")) and best is not None:
                print_best_grasp(best)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
