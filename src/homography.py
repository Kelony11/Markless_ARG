import cv2
import numpy as np
from ar_overlay import apply_overlay


def estimate_homography(ref_pts, frame_pts):
    """Estimate the homography from the reference image to the camera frame."""
    ref_pts = np.asarray(ref_pts, dtype=np.float32)
    frame_pts = np.asarray(frame_pts, dtype=np.float32)

    if len(ref_pts) < 4 or len(ref_pts) != len(frame_pts):
        return None, None

    H, inlier_mask = cv2.findHomography(
        ref_pts,
        frame_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    return H, inlier_mask

def project_reference_corners(H, width, height):
    """Project the reference image corners into the camera frame."""
    corners = np.array(
        [[[0, 0], [width - 1, 0],
          [width - 1, height - 1], [0, height - 1]]],
        dtype=np.float32,
    )
    return cv2.perspectiveTransform(corners, H)[0]


if __name__ == "__main__":
    # Move each reference point 100 pixels right and 50 pixels down.
    ref_pts = [(0, 0), (199, 0), (199, 99), (0, 99), (40, 30), (150, 70)]

    # Simulate a reference surface viewed at an angle.
    source_corners = np.array(ref_pts[:4], dtype=np.float32)
    target_corners = np.array(
        [[85, 45], [315, 75], [275, 185], [110, 155]],
        dtype=np.float32,
    )
    test_transform = cv2.getPerspectiveTransform(source_corners, target_corners)

    # Generate consistent matches for all six reference points.
    reference_points = np.array([ref_pts], dtype=np.float32)
    frame_pts = cv2.perspectiveTransform(
        reference_points, test_transform
    )[0].tolist()

    # Add one intentionally incorrect match.
    ref_pts.append((80, 20))
    frame_pts.append((350, 250))

    H, inlier_mask = estimate_homography(ref_pts, frame_pts)
    if H is None or inlier_mask is None:
        raise RuntimeError("Homography estimation failed.")

    point = np.array([[[40, 30]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(point, H)[0, 0]

    print("Homography matrix:\n", np.round(H, 3))
    print("Projected point:", np.round(projected, 2))
    print("Inlier mask:", inlier_mask.ravel())

    frame_corners = project_reference_corners(H, width=200, height=100)
    print("Projected corners:\n", np.round(frame_corners, 2))

        # Create a blank test frame (height, width, color channels).
    test_frame = np.zeros((240, 400, 3), dtype=np.uint8)

    # Draw the projected reference outline in green.
    outline = np.rint(frame_corners).astype(np.int32)
    cv2.polylines(
        test_frame, [outline], isClosed=True, color=(0, 255, 0), thickness=2
    )

    # Create a test image without loading a file.
    overlay = np.full((100, 200, 3), (180, 70, 25), dtype=np.uint8)
    cv2.putText(
        overlay, "AR DEMO", (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
    )

    result = apply_overlay(test_frame, overlay, frame_corners)
    cv2.imshow("Projected AR overlay", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()