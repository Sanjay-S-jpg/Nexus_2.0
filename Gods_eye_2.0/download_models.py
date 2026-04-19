# download_models.py - Download all required models for Gods Eye 2.0

import os
import sys

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


def main():
    print("\n" + "=" * 50)
    print("  GODS EYE 2.0 - Model Downloader")
    print("=" * 50 + "\n")

    from ultralytics import YOLO

    # 1. YOLO11n - main detection + tracking
    yolo_path = os.path.join(MODELS_DIR, "yolo11n.pt")
    if os.path.exists(yolo_path):
        print(f"[OK] yolo11n.pt already exists")
    else:
        print("[1/2] Downloading YOLO11n (detection + tracking)...")
        m = YOLO("yolo11n.pt")
        os.rename("yolo11n.pt", yolo_path)
        print(f"[OK] Saved to {yolo_path}")

    # 2. YOLO11n-pose - fight detection via skeleton
    pose_path = os.path.join(MODELS_DIR, "yolo11n-pose.pt")
    if os.path.exists(pose_path):
        print(f"[OK] yolo11n-pose.pt already exists")
    else:
        print("[2/2] Downloading YOLO11n-pose (fight detection)...")
        m = YOLO("yolo11n-pose.pt")
        os.rename("yolo11n-pose.pt", pose_path)
        print(f"[OK] Saved to {pose_path}")

    print("\n" + "=" * 50)
    print("  All models ready!")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()