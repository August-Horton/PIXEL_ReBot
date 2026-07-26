"""
3D object detection preview.

Keys:
  Left click: sample depth at a pixel.
  Q/Esc: exit.

Usage:
    python scripts/object_detection.py
"""

import os
os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype")

import sys
import cv2
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

from drivers.camera import make_camera
from utils.camera_utils import load_config
from utils.ordinary_grasp import get_depth_mm

# ==========================================
# Mouse state
# ==========================================
clicked_point = {"u": -1, "v": -1}

def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point["u"] = x
        clicked_point["v"] = y
        print(f"[Test] Clicked pixel: (u={x}, v={y})")

# ==========================================
# Main
# ==========================================
def main():
    config_path  = PROJECT_ROOT / "config" / "default.yaml"
    models_dir   = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)

    cam_type = cfg.get("camera", {}).get("type", "").lower()

    yolo_cfg     = cfg.get("yolo", {})
    model_name   = "yolov11s.pt"
    device       = yolo_cfg.get("device", "cpu")
    use_world    = yolo_cfg.get("use_world", False)
    custom_classes = yolo_cfg.get("custom_classes", ["person", "cup", "cell phone"])

    model_path = models_dir / model_name

    print(f"=== Init YOLO ===")
    print(f"Load model: {model_path}")
    model = YOLO(str(model_path))

    is_open_vocab = use_world and ("world" in model_name.lower() or "yoloe" in model_name.lower())
    if is_open_vocab:
        print(f"Enable open vocabulary: {len(custom_classes)} classes")
        model.set_classes(custom_classes)

    print(f"YOLO ready on {device.upper()}")
    if "26" in model_name:
        print(f"[Info] YOLO26 end-to-end inference enabled.")

    # Camera
    print(f"\n=== Init camera: {cam_type} ===")
    try:
        cam = make_camera(cfg)
    except ValueError as e:
        print(f"\n[Fatal] {e}")
        sys.exit(1)

    try:
        cam.open()
    except RuntimeError as e:
        print(f"\n[Fatal] {e}")
        sys.exit(1)

    K  = cam.K
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    print(f"[Camera] {cam_type} (fx:{fx:.2f}, cx:{cx:.2f})")

    print("\n[Keys] Left click=sample point  Q=quit")
    window_name = f"Unified 3D Vision ({cam_type})"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, mouse_callback)

    try:
        while True:
            color_image, depth_mm = cam.get_frame()
            if color_image is None:
                continue

            results = model.predict(color_image, verbose=False, device=device)

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    cls_id, conf = int(box.cls[0]), float(box.conf[0])
                    class_name = model.names[cls_id]

                    u, v = (x1 + x2) // 2, (y1 + y2) // 2
                    x_m, y_m, z_m = 0.0, 0.0, 0.0
                    valid_depth = False

                    if depth_mm is not None:
                        z_mm = get_depth_mm(depth_mm, u, v, 5)
                        if z_mm > 0:
                            z_m = z_mm / 1000.0
                            x_m = (u - cx) * z_m / fx
                            y_m = (v - cy) * z_m / fy
                            valid_depth = True

                    if valid_depth:
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(color_image, (u, v), 5, (0, 0, 255), -1)
                        text_label  = f"{class_name} {conf:.2f}"
                        coord_label = f"X:{x_m:.2f} Y:{y_m:.2f} Z:{z_m:.2f} (m)"
                        bg_w = max(len(text_label), len(coord_label)) * 10
                        cv2.rectangle(color_image, (x1, y1 - 40), (x1 + bg_w, y1), (0, 0, 0), -1)
                        cv2.putText(color_image, text_label,  (x1 + 5, y1 - 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),   2)
                        cv2.putText(color_image, coord_label, (x1 + 5, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        cv2.putText(color_image, f"{class_name} (No Depth)",
                                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            # Point sampling
            cu, cv_y = clicked_point["u"], clicked_point["v"]
            if cu != -1 and cv_y != -1 and depth_mm is not None:
                h_dm, w_dm = depth_mm.shape
                if 0 <= cu < w_dm and 0 <= cv_y < h_dm:
                    cz_mm = get_depth_mm(depth_mm, cu, cv_y)
                    if cz_mm > 0:
                        cz_m = cz_mm / 1000.0
                        cx_m = (cu - cx) * cz_m / fx
                        cy_m = (cv_y - cy) * cz_m / fy
                        cv2.drawMarker(color_image, (cu, cv_y), (255, 0, 255),
                                       cv2.MARKER_CROSS, 20, 2)
                        test_label = f"TEST -> X:{cx_m:.3f} Y:{cy_m:.3f} Z:{cz_m:.3f} m"
                        cv2.rectangle(color_image, (cu + 5, cv_y - 25),
                                      (cu + 320, cv_y + 5), (0, 0, 0), -1)
                        cv2.putText(color_image, test_label, (cu + 10, cv_y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            cv2.putText(color_image, f"{cam_type.upper()} | {model_name.upper()}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(window_name, color_image)

            key = cv2.waitKey(1) & 0xFF
            if key in [ord('q'), ord('Q'), 27]:
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

    finally:
        cam.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
