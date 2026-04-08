from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor
import time  # Added for timing
from collections import deque  # Added for FPS smoothing

# =========================
# CONFIG
# =========================
BUNDLE_PATH = Path("resnet_prototype_bundle.pt")
INPUT_SIZE = 224
WINDOW_NAME = "Live Prototype Inference"

# USB webcam settings – you can adjust these
USB_CAM_INDEX = 0          # try 0, if fails change to 1
USB_WIDTH = 640            # lower resolution = faster, you can increase
USB_HEIGHT = 480
USB_FPS = 30

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =========================
# UTILS
# =========================
def l2_normalize_torch(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


# =========================
# CAMERA (USB WEBCAM)
# =========================
def open_camera(index=0, width=640, height=480):
    """
    Opens a USB webcam using OpenCV's VideoCapture.
    Tries to set resolution and prints info.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"❌ Could not open USB camera with index {index}")

    # Try to set the desired resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Verify actual resolution
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"✅ USB camera opened. Requested {width}x{height}, got {actual_width}x{actual_height}")

    # Test first frame
    ret, frame = cap.read()
    if not ret or frame is None:
        raise RuntimeError("❌ Camera opened but no frames received")
    print("✅ First frame captured. Shape:", frame.shape)
    return cap


# =========================
# PREPROCESS
# =========================
def preprocess_frame(frame_bgr, device):
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))
    img = img_rgb.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).unsqueeze(0).to(device)


# =========================
# MODEL
# =========================
def load_model_bundle(bundle_path):
    if not bundle_path.is_file():
        raise FileNotFoundError(bundle_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cuda")
    bundle = torch.load(str(bundle_path), map_location=device)

    class_names = list(bundle["class_names"])
    num_classes = len(class_names)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(bundle["model_state_dict"])
    model.to(device).eval()

    feat_extractor = create_feature_extractor(
        model, return_nodes={"avgpool": "emb"}
    ).to(device).eval()

    prototypes = torch.tensor(bundle["prototypes"], dtype=torch.float32).to(device)
    threshold = float(bundle["threshold"])

    return feat_extractor, prototypes, threshold, class_names, device


# =========================
# INFERENCE
# =========================
def predict_frame(frame, feat_extractor, prototypes, threshold, class_names, device):
    x = preprocess_frame(frame, device)

    with torch.no_grad():
        emb = feat_extractor(x)["emb"]
        emb = torch.flatten(emb, 1)
        emb = l2_normalize_torch(emb)

        sims = emb @ prototypes.T
        score, idx = sims.max(1)

    score = float(score)
    idx = int(idx)
    nearest = class_names[idx]

    if score >= threshold:
        return nearest, nearest, score, (0, 255, 0)
    else:
        return "REJECT", nearest, score, (0, 0, 255)


# =========================
# DRAW (modified to show timing)
# =========================
def draw_overlay(frame, label, nearest, score, threshold, color, inference_ms, fps):
    out = frame.copy()

    # Make the black background taller to accommodate the extra line
    cv2.rectangle(out, (10, 10), (600, 180), (0, 0, 0), -1)

    cv2.putText(out, f"Prediction: {label}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.putText(out, f"Nearest: {nearest}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(out, f"Score: {score:.3f} | Thr: {threshold:.3f}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # New line for timing information
    cv2.putText(out, f"Inference: {inference_ms:.1f} ms  ({fps:.1f} FPS)", (20, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    return out


# =========================
# MAIN
# =========================
def main():
    feat_extractor, prototypes, threshold, class_names, device = load_model_bundle(BUNDLE_PATH)

    print("Device:", device)
    print("Classes:", class_names)

    # Try to open USB webcam – if index 0 fails, try index 1
    try:
        cap = open_camera(USB_CAM_INDEX, USB_WIDTH, USB_HEIGHT)
    except RuntimeError as e:
        print(f"Index {USB_CAM_INDEX} failed: {e}")
        print("Trying index 1...")
        cap = open_camera(1, USB_WIDTH, USB_HEIGHT)

    print("Press 'q' to quit")

    # For FPS calculation (rolling average over last 30 frames)
    inference_times = deque(maxlen=30)

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("⚠️ Frame dropped")
            continue

        # Start timing
        t_start = time.perf_counter()

        label, nearest, score, color = predict_frame(
            frame, feat_extractor, prototypes, threshold, class_names, device
        )

        # End timing
        t_end = time.perf_counter()
        inference_ms = (t_end - t_start) * 1000.0

        # Store the inference time for rolling average
        inference_times.append(inference_ms)
        avg_inference_ms = sum(inference_times) / len(inference_times)
        fps = 1000.0 / avg_inference_ms if avg_inference_ms > 0 else 0.0

        # Draw everything (pass timing info)
        vis = draw_overlay(frame, label, nearest, score, threshold, color,
                           avg_inference_ms, fps)

        cv2.imshow(WINDOW_NAME, vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
