import argparse
from pathlib import Path

import cv2
import numpy as np

from src.ar_overlay import apply_overlay
from src.feature_detection import detect_features
from src.feature_matching import extract_matched_points, filter_matches, match_features
from src.homography import estimate_homography, project_reference_corners
from src.webcam_recognition import REFERENCE_PATH, resize_image

MIN_INLIERS = 40
MIN_INLIER_RATIO = 0.30


def load_overlay(path):
    if path is not None:
        overlay = cv2.imread(str(path))
        if overlay is None or min(overlay.shape[:2]) < 2:
            raise ValueError(f"Could not load a valid overlay image: {path}")
        return overlay

    overlay = np.full((200, 400, 3), (180, 70, 25), dtype=np.uint8)
    cv2.putText(
        overlay, "AR DEMO", (80, 115), cv2.FONT_HERSHEY_SIMPLEX,
        1.5, (255, 255, 255), 3, cv2.LINE_AA,
    )
    return overlay


def track_target(frame, reference_keypoints, reference_descriptors, reference_size):
    frame_keypoints, frame_descriptors = detect_features(frame)
    matches = match_features(reference_descriptors, frame_descriptors)
    good_matches = filter_matches(matches)
    reference_points, frame_points = extract_matched_points(
        reference_keypoints, frame_keypoints, good_matches
    )
    H, mask = estimate_homography(reference_points, frame_points)
    inliers = int(np.count_nonzero(mask)) if mask is not None else 0
    match_count = len(good_matches)

    if H is None or inliers < MIN_INLIERS or inliers / match_count < MIN_INLIER_RATIO:
        return None, inliers, match_count

    corners = project_reference_corners(H, *reference_size)
    if not np.isfinite(corners).all() or not cv2.isContourConvex(corners):
        return None, inliers, match_count

    return corners, inliers, match_count


def main():
    parser = argparse.ArgumentParser(description="Track a planar target using a webcam.")
    parser.add_argument("--overlay", type=Path, help="Image to place on the target.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    args = parser.parse_args()

    reference = cv2.imread(str(REFERENCE_PATH))
    if reference is None:
        raise FileNotFoundError(f"Reference image not found: {REFERENCE_PATH}")
    reference = resize_image(reference)
    height, width = reference.shape[:2]
    keypoints, descriptors = detect_features(reference)
    if descriptors is None or len(descriptors) < MIN_INLIERS:
        raise RuntimeError("Too few SIFT features in the reference image.")

    overlay = load_overlay(args.overlay)
    camera = cv2.VideoCapture(args.camera)
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Unable to open webcam {args.camera}.")
        was_detected = None
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a webcam frame.")

            frame = resize_image(frame)
            corners, inliers, matches = track_target(
                frame, keypoints, descriptors, (width, height)
            )
            detected = corners is not None
            if detected != was_detected:
                ratio = 100 * inliers / matches if matches else 0.0
                state = "DETECTED" if detected else "LOST"
                print(f"{state} | Inliers: {inliers}/{matches} ({ratio:.1f}%)")
                was_detected = detected
            display = frame if corners is None else apply_overlay(frame, overlay, corners)
            status = "Searching for target" if corners is None else "Target detected"
            cv2.putText(
                display, f"{status} | Inliers: {inliers}/{matches}",
                (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255),
                2, cv2.LINE_AA,
            )
            cv2.imshow("Markerless AR", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
