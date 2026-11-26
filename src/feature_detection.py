
import cv2
from pathlib import Path

# Project directories
ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = ROOT / "assets" / "reference.jpg"
OUTPUT_PATH = ROOT / "results" / "sift_keypoints.jpg"


def detect_features(image):
    """
    Detect SIFT keypoints and compute descriptors.

    Returns:
        keypoints: Detected image features
        descriptors: Feature descriptors
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    return keypoints, descriptors


if __name__ == "__main__":
    # Load reference image
    image = cv2.imread(str(REFERENCE_PATH))

    if image is None:
        raise FileNotFoundError(
            f"Cannot find image: {REFERENCE_PATH}"
        )

    # Extract features
    keypoints, descriptors = detect_features(image)

    print(f"Total keypoints detected: {len(keypoints)}")

    if descriptors is not None:
        print(f"Descriptor shape: {descriptors.shape}")

    # Draw detected SIFT features
    result = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Save visualization
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), result)

    print(f"Result saved to: {OUTPUT_PATH}")
