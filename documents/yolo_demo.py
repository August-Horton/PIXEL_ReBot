"""
YOLO inference demo — test model loading, GPU, and detection speed.

Usage:
    python yolo_demo.py
    python yolo_demo.py --model yoloe-26l-seg.pt --device cpu
    python yolo_demo.py --camera           # use realsense camera
    python yolo_demo.py --loops 50         # stress test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.yolo_utils import load_yolo
from utils.camera_utils import load_config


def gpu_info() -> dict:
    """Gather CUDA/GPU info."""
    info: dict = {}
    try:
        import torch
        info["pytorch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["device_count"] = torch.cuda.device_count()
            info["device_name"] = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
            info["total_mem_gb"] = round(mem, 1)
    except Exception as e:
        info["error"] = str(e)
    return info


def create_test_image(w: int = 640, h: int = 480) -> np.ndarray:
    """Create a synthetic test image with some shapes."""
    img = np.random.randint(60, 180, (h, w, 3), dtype=np.uint8)
    # add a rectangle (simulating box)
    cv2.rectangle(img, (200, 150), (400, 350), (100, 200, 100), -1)
    cv2.rectangle(img, (200, 150), (400, 350), (0, 255, 0), 2)
    # add a circle (simulating bottle)
    cv2.circle(img, (500, 300), 60, (200, 150, 100), -1)
    cv2.circle(img, (500, 300), 60, (255, 0, 0), 2)
    return img


def demo_image(model, yolo_opts: dict, img: np.ndarray) -> float:
    """Run inference on a single image, return elapsed seconds."""
    device = yolo_opts.get("device", "cpu")
    conf = float(yolo_opts.get("conf", 0.25))
    iou = float(yolo_opts.get("iou", 0.45))

    t0 = time.perf_counter()
    results = model.predict(img, verbose=False, device=device, conf=conf, iou=iou)
    elapsed = time.perf_counter() - t0

    print(f"\n  Inference: {elapsed*1000:.1f} ms  |  {1/elapsed:.1f} FPS")
    for r in results:
        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            print(f"  Detections: {len(boxes)}")
            for i, box in enumerate(boxes):
                cls_id = int(box.cls[0].item())
                name = r.names.get(cls_id, str(cls_id))
                conf_val = float(box.conf[0].item())
                print(f"    [{i}] {name}  conf={conf_val:.3f}")
        else:
            print("  No detections")
    return elapsed


def main():
    parser = argparse.ArgumentParser(description="YOLO Demo")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model", default=None, help="override model name (e.g. yoloe-26s-seg.pt)")
    parser.add_argument("--device", default=None, help="override device (cuda / cpu)")
    parser.add_argument("--camera", action="store_true", help="use live camera")
    parser.add_argument("--loops", type=int, default=1, help="number of inference runs")
    parser.add_argument("--image", default=None, help="path to test image (jpg/png)")
    args = parser.parse_args()

    # ---- GPU info ----
    print("=" * 55)
    print("GPU / CUDA Info")
    print("=" * 55)
    for k, v in gpu_info().items():
        print(f"  {k}: {v}")

    # ---- Load config + model ----
    cfg = load_config(PROJECT_ROOT / args.config)
    yolo_cfg = cfg.get("yolo", {})

    model_name = args.model or yolo_cfg.get("model_name", "yoloe-26s-seg.pt")
    device = args.device or yolo_cfg.get("device", "cpu")

    print(f"\n{'='*55}")
    print(f"Loading model: {model_name}  |  device: {device}")
    print("=" * 55)

    t0 = time.perf_counter()
    model, yolo_opts = load_yolo(
        cfg,
        project_root=PROJECT_ROOT,
        model_override=model_name,
        device_override=device,
    )
    print(f"  Model loaded in {time.perf_counter() - t0:.1f}s")
    print(f"  Classes: {yolo_opts.get('custom_classes', [])}")

    # ---- Warmup ----
    print("\n--- Warmup ---")
    warm_img = create_test_image()
    demo_image(model, yolo_opts, warm_img)

    # ---- Camera mode ----
    if args.camera:
        print("\n--- Live Camera (press Q to quit) ---")
        from drivers.camera import make_camera
        cam_cfg = cfg.get("camera", {})
        cam = make_camera(cfg)
        cam.open()
        cam.warm_up(10)

        try:
            while True:
                color, _ = cam.get_frame()
                if color is None:
                    continue
                t0 = time.perf_counter()
                results = model.predict(
                    color, verbose=False,
                    device=yolo_opts.get("device", "cpu"),
                    conf=float(yolo_opts.get("conf", 0.25)),
                    iou=float(yolo_opts.get("iou", 0.45)),
                )
                fps = 1.0 / max(time.perf_counter() - t0, 0.001)
                disp = color.copy()
                cv2.putText(disp, f"{fps:.1f} FPS", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                for r in results:
                    boxes = getattr(r, "boxes", None)
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cls_id = int(box.cls[0].item())
                            name = r.names.get(cls_id, str(cls_id))
                            conf_val = float(box.conf[0].item())
                            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(disp, f"{name} {conf_val:.2f}", (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("YOLO Demo", disp)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
        finally:
            cam.close()
            cv2.destroyAllWindows()
        return 0

    # ---- Image mode ----
    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"ERROR: cannot read {args.image}")
            return 1
        print(f"\n--- Test image: {args.image} ({img.shape[1]}x{img.shape[0]}) ---")
        elapsed = demo_image(model, yolo_opts, img)
        return 0

    # ---- Benchmark mode ----
    print(f"\n--- Benchmark ({args.loops} loops) ---")
    test_img = create_test_image()
    times = []
    for i in range(args.loops):
        t = demo_image(model, yolo_opts, test_img)
        times.append(t)
        time.sleep(0.1)

    times = np.array(times)
    print(f"\n{'='*55}")
    print(f"Benchmark Results ({args.loops} runs)")
    print("=" * 55)
    print(f"  Mean:   {times.mean()*1000:.1f} ms  ({1/times.mean():.1f} FPS)")
    print(f"  Median: {np.median(times)*1000:.1f} ms")
    print(f"  Min:    {times.min()*1000:.1f} ms")
    print(f"  Max:    {times.max()*1000:.1f} ms")
    print(f"  Std:    {times.std()*1000:.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
