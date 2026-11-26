
import cv2
import numpy as np
from pathlib import Path

if __package__:
    from .feature_detection import detect_features
else:
    from feature_detection import detect_features


# Project paths
ROOT = Path(__file__).resolve().parent.parent

REFERENCE_PATH = ROOT / "assets" / "reference.jpg"
SCENE_PATH = ROOT / "assets" / "scene.jpg"
OUTPUT_PATH = ROOT / "results" / "feature_matches.jpg"


def match_features(descriptors1, descriptors2):
    """
    Match SIFT descriptors using FLANN.

    Returns:
        List of KNN match pairs (two nearest neighbors).
    """
    if descriptors1 is None or descriptors2 is None:
        return []

    if len(descriptors1) == 0 or len(descriptors2) < 2:
        return []

    # FLANN KD-tree configuration for SIFT
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(
        index_params,
        search_params
    )

    matches = flann.knnMatch(
        np.asarray(descriptors1, dtype=np.float32),
        np.asarray(descriptors2, dtype=np.float32),
        k=2
    )

    return matches


def filter_matches(matches, ratio=0.75):
    """
    Apply Lowe's ratio test to remove ambiguous matches.
    """
    good_matches = []

    for pair in matches:
        if len(pair) == 2:
            m, n = pair

            if m.distance < ratio * n.distance:
                good_matches.append(m)

    return good_matches


def extract_matched_points(kp_ref, kp_frame, good_matches):
    """
    Convert DMatch objects into corresponding (x, y) coordinates.

    Both arrays have shape (N, 2) and dtype float32.
    """
    reference_pts = np.array(
        [kp_ref[m.queryIdx].pt for m in good_matches],
        dtype=np.float32
    ).reshape(-1, 2)

    frame_pts = np.array(
        [kp_frame[m.trainIdx].pt for m in good_matches],
        dtype=np.float32
    ).reshape(-1, 2)

    return reference_pts, frame_pts


def get_matched_points(reference_image, camera_frame, ratio=0.75):
    """
    Main interface for Francisco's homography implementation.

    Args:
        reference_image: BGR reference image
        camera_frame: BGR webcam frame or scene image
        ratio: Lowe's ratio threshold

    Returns:
        reference_pts: NumPy array (N, 2)
        frame_pts: NumPy array (N, 2)
    """
    kp_ref, des_ref = detect_features(reference_image)
    kp_frame, des_frame = detect_features(camera_frame)

    matches = match_features(des_ref, des_frame)
    good_matches = filter_matches(matches, ratio)

    return extract_matched_points(
        kp_ref,
        kp_frame,
        good_matches
    )


def evaluate_matches(reference_pts, frame_pts):
    """
    Evaluate matching quality using RANSAC.
    This is for diagnostic testing before integration.
    """
    if len(reference_pts) < 4:
        return None, 0, 0.0

    H, mask = cv2.findHomography(
        reference_pts,
        frame_pts,
        cv2.RANSAC,
        5.0
    )

    if H is None or mask is None:
        return None, 0, 0.0

    inliers = int(mask.sum())
    inlier_ratio = inliers / len(reference_pts) * 100

    return H, inliers, inlier_ratio


def main():
    # Load reference and scene images
    reference = cv2.imread(str(REFERENCE_PATH))
    scene = cv2.imread(str(SCENE_PATH))

    if reference is None:
        raise FileNotFoundError(
            f"Reference image not found: {REFERENCE_PATH}"
        )

    if scene is None:
        raise FileNotFoundError(
            f"Scene image not found: {SCENE_PATH}"
        )

    # Step 1: Detect SIFT features
    kp_ref, des_ref = detect_features(reference)
    kp_scene, des_scene = detect_features(scene)

    print(f"Reference keypoints: {len(kp_ref)}")
    print(f"Scene keypoints: {len(kp_scene)}")

    # Step 2: Match features using FLANN
    matches = match_features(des_ref, des_scene)
    valid_pairs = [pair for pair in matches if len(pair) == 2]

    print(f"Candidate match pairs: {len(valid_pairs)}")

    # Step 3: Apply Lowe's ratio test
    good_matches = filter_matches(matches, ratio=0.75)

    print(f"Good matches after ratio test: {len(good_matches)}")

    # Step 4: Extract corresponding (x, y) coordinates
    reference_pts, frame_pts = extract_matched_points(
        kp_ref,
        kp_scene,
        good_matches
    )

    print(f"Reference points shape: {reference_pts.shape}")
    print(f"Frame points shape: {frame_pts.shape}")

    # Step 5: Visualize good matches
    good_matches = sorted(
        good_matches,
        key=lambda m: m.distance
    )

    result = cv2.drawMatches(
        reference,
        kp_ref,
        scene,
        kp_scene,
        good_matches[:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Save visualization
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(OUTPUT_PATH), result):
        raise RuntimeError("Failed to save feature matching image.")

    print(f"Matches visualized: {min(50, len(good_matches))}")
    print(f"Result saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
