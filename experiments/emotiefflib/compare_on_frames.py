"""Offline classifier comparison: deepface labels (from frame filenames) vs EmotiEffLib.

Runs EmotiEffLib on the frames dumped by the app's EMOTION_FRAME_DUMP_DIR feature.
The face is located with the same opencv haar cascade the app's deepface backend used,
so the detector is held constant and the classifier is the only variable.

Run (ephemeral env, keeps the app venv clean; first run downloads the ONNX model):

    uv run --with emotiefflib python compare_on_frames.py [frames_dir]

Frame filenames are expected to look like HHMMSS_<label>.jpg, where <label> is
deepface's classification ("noface" for detector misses).
"""

import sys
from pathlib import Path

import cv2
from emotiefflib.facial_analysis import EmotiEffLibRecognizer

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRAMES_DIR = REPO_ROOT / "logs" / "frames" / "2026-07-21_opencv-webcam"
MODEL_NAME = "enet_b0_8_best_vgaf"
CROP_MARGIN = 0.2  # fraction of box size added on each side; HSEmotion models saw tight crops


def largest_haar_face(gray_img):
    """Return the largest (x, y, w, h) haar detection, or None."""
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: box[2] * box[3])


def crop_with_margin(img, box):
    """Crop the face box plus a safety margin, clamped to image bounds."""
    x, y, w, h = box
    mx, my = int(w * CROP_MARGIN), int(h * CROP_MARGIN)
    y0, y1 = max(0, y - my), min(img.shape[0], y + h + my)
    x0, x1 = max(0, x - mx), min(img.shape[1], x + w + mx)
    return img[y0:y1, x0:x1]


def main() -> None:
    frames_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FRAMES_DIR
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        sys.exit(f"no .jpg frames found in {frames_dir}")

    recognizer = EmotiEffLibRecognizer(engine="onnx", model_name=MODEL_NAME)

    agree = disagree = undetected = 0
    print(f"{'frame':<24} {'deepface':<12} {'emotiefflib':<12} verdict")
    print("-" * 62)
    for path in frames:
        deepface_label = path.stem.split("_", 1)[1]
        img = cv2.imread(str(path))
        box = largest_haar_face(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
        if box is None:
            print(f"{path.name:<24} {deepface_label:<12} {'-':<12} no face (haar)")
            undetected += 1
            continue
        crop_rgb = cv2.cvtColor(crop_with_margin(img, box), cv2.COLOR_BGR2RGB)
        emotions, _scores = recognizer.predict_emotions(crop_rgb, logits=False)
        label = emotions[0].lower()
        same = label == deepface_label
        agree += same
        disagree += not same
        print(f"{path.name:<24} {deepface_label:<12} {label:<12} {'=' if same else 'DIFFERS'}")

    print("-" * 62)
    print(f"agree: {agree}  differ: {disagree}  haar-undetected: {undetected}  model: {MODEL_NAME}")


if __name__ == "__main__":
    main()
