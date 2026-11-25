import cv2
import numpy as np


def apply_overlay(frame, overlay, frame_corners):
    """Warp the overlay into the four projected corners of the frame."""
    height, width = overlay.shape[:2]
    overlay_corners = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(
        overlay_corners, np.asarray(frame_corners, dtype=np.float32)
    )

    frame_height, frame_width = frame.shape[:2]
    output_size = (frame_width, frame_height)
    warped_overlay = cv2.warpPerspective(overlay, transform, output_size)

    # Warp a mask so only the overlay region replaces the frame pixels.
    mask = np.full((height, width), 255, dtype=np.uint8)
    warped_mask = cv2.warpPerspective(
        mask, transform, output_size, flags=cv2.INTER_NEAREST
    )

    result = frame.copy()
    result[warped_mask > 0] = warped_overlay[warped_mask > 0]
    return result