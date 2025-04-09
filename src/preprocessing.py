import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from the given path and check if it was loaded successfully.

    Args:
        image_path: Path to the image file

    Returns:
        The loaded image

    Raises:
        ValueError: If the image cannot be loaded
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not open or find the image: {image_path}")
    return image


def create_binary_mask(image: np.ndarray, threshold: int = 220) -> np.ndarray:
    """
    Convert image to grayscale and create a binary mask based on threshold.

    Args:
        image: Input image in BGR format
        threshold: Pixel value threshold (default: 200)

    Returns:
        Binary mask where pixels above threshold are 255, others 0
    """
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(image_grey, threshold, 255, cv2.THRESH_BINARY)
    return binary_mask


def create_circular_mask(
    height: int, width: int, center: tuple, radius: int
) -> np.ndarray:
    """
    Create a circular mask with specified dimensions and parameters.

    Args:
        height: Height of the mask
        width: Width of the mask
        center: (x, y) coordinates of circle center
        radius: Radius of the circle

    Returns:
        Circular binary mask
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask


def preprocess_image(
    image_path: str,
    max_width: int = 1400,
    circle_center: tuple = (650, 610),
    circle_radius: int = 560,
) -> np.ndarray:
    """
    Perform all preprocessing steps on the image.

    Args:
        image_path: Path to the image file
        max_width: Maximum width to crop the image to
        circle_center: Center coordinates for circular mask
        circle_radius: Radius for circular mask

    Returns:
        Preprocessed grayscale image
    """
    # Load and convert to grayscale
    image = load_image(image_path)
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Crop width if needed
    height, width = image_grey.shape
    cropped_width = min(width, max_width)
    image_grey = image_grey[:, :cropped_width]

    # Adjust circle center x-coordinate if it's beyond the cropped width
    # circle_center_x = min(circle_center[0], cropped_width - 1)
    # adjusted_center = (circle_center_x, circle_center[1])

    # Create and apply circular mask with the correct dimensions
    mask = create_circular_mask(height, cropped_width, circle_center, circle_radius)
    masked_image = cv2.bitwise_and(image_grey, image_grey, mask=mask)

    return masked_image
