
import cv2
from pathlib import Path

if __package__:
    from .feature_detection import detect_features
    from .feature_matching import (
        match_features,
        filter_matches,
        extract_matched_points,
        evaluate_matches
    )
else:
    from feature_detection import detect_features
    from feature_matching import (
        match_features,
        filter_matches,
        extract_matched_points,
        evaluate_matches
    )

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = ROOT / "assets" / "reference.jpg"

MAX_WIDTH = 960
MIN_MATCHES = 10


def resize_image(image, max_width=MAX_WIDTH):
    """Resize an image while preserving its aspect ratio."""
    height, width = image.shape[:2]

    if width <= max_width:
        return image

    scale = max_width / width
    new_height = int(height * scale)

    return cv2.resize(
        image,
        (max_width, new_height),
        interpolation=cv2.INTER_AREA
    )


def main():
    # Load reference image
    reference = cv2.imread(str(REFERENCE_PATH))

    if reference is None:
        raise FileNotFoundError(
            f"Reference image not found: {REFERENCE_PATH}"
        )

    reference = resize_image(reference)

    # Extract reference features once
    kp_ref, des_ref = detect_features(reference)

    if des_ref is None:
        raise RuntimeError("No SIFT features in reference image.")

    print(f"Reference keypoints: {len(kp_ref)}")

    # Open MacBook webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Unable to access webcam.")

    print("Webcam started. Press Q to quit.")

    try:
        while True:
            ret, frame = cap.read()

            detection_history = []
            WINDOW_SIZE = 5

            if not ret:
                print("Failed to read webcam frame.")
                break

            frame = resize_image(frame)

            # Extract live frame features
            kp_frame, des_frame = detect_features(frame)

            # Find and filter matches
            matches = match_features(des_ref, des_frame)
            good_matches = filter_matches(matches, ratio=0.75)

            # Extract matching coordinates for homography
            reference_pts, frame_pts = extract_matched_points(
                kp_ref,
                kp_frame,
                good_matches
            )

            # Evaluate feature matches using RANSAC
            H, inliers, inlier_ratio = evaluate_matches(
                reference_pts,
                frame_pts
            )

            # Initial experimental thresholds
            MIN_INLIERS = 40
            MIN_INLIER_RATIO = 35.0

            detected = (
                    H is not None
                    and inliers >= MIN_INLIERS
                    and inlier_ratio >= MIN_INLIER_RATIO
            )

            if detected:
                print(
                    f"MAGAZINE DETECTED | "
                    f"Inliers: {inliers} | "
                    f"Ratio: {inlier_ratio:.2f}%"
                )
            else:
                print(
                    f"MAGAZINE NOT DETECTED | "
                    f"Inliers: {inliers} | "
                    f"Ratio: {inlier_ratio:.2f}%"
                )

            print(
                f"Matches: {len(reference_pts)} | "
                f"RANSAC inliers: {inliers} | "
                f"Inlier ratio: {inlier_ratio:.2f}%"
            )

            count = len(reference_pts)

            # Show matched keypoints in current frame
            display = frame.copy()

            for x, y in frame_pts:
                cv2.circle(
                    display,
                    (int(x), int(y)),
                    3,
                    (0, 255, 0),
                    -1
                )

            if detected:
                status = "Magazine detected"
                color = (0, 255, 0)
            else:
                status = "Searching for magazine..."
                color = (0, 0, 255)

            cv2.putText(
                display,
                f"{status}: {count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            cv2.imshow("Markerless AR - Feature Recognition", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
