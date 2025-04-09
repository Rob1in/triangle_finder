import os
from typing import Tuple

import cv2

from preprocessing import preprocess_image


def extract_and_save_pattern(
    image_path: str,
    box: Tuple[int, int, int, int],
    output_dir: str = "patterns",
    padding: int = 5,
    filename: str = None,
) -> str:
    """
    Extract a pattern from an image based on bounding box coordinates and save it.

    Args:
        image_path: Path to the source image
        box: Tuple of (x, y, w, h) coordinates of the pattern
        output_dir: Directory to save the pattern (default: "patterns")
        padding: Additional padding around the pattern in pixels (default: 5)
        filename: Custom filename for the pattern (default: None, will generate automatically)

    Returns:
        Path to the saved pattern image

    Raises:
        ValueError: If the image cannot be loaded or the box is invalid
    """
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    # Extract box coordinates
    x, y, w, h = box

    # Add padding to the box (ensuring we don't go out of bounds)
    height, width = preprocessed_image.shape
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(width, x + w + padding)
    y_end = min(height, y + h + padding)

    # Crop the pattern from the image
    pattern = preprocessed_image[y_start:y_end, x_start:x_end]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename if not provided
    if filename is None:
        base_image_name = os.path.splitext(os.path.basename(image_path))[0]
        filename = f"{base_image_name}_pattern_{x}_{y}_{w}_{h}.png"

    # Ensure filename has .png extension
    if not filename.endswith(".png"):
        filename += ".png"

    # Save the pattern
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, pattern)

    return output_path


def main():
    """
    Command-line interface for extracting patterns from images.

    Usage:
        python pattern_creation.py <image_path> <x> <y> <w> <h> [--output-dir OUTPUT_DIR] [--padding PADDING] [--filename FILENAME]
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract a pattern from an image based on bounding box coordinates"
    )
    parser.add_argument("image_path", help="Path to the source image")
    parser.add_argument("x", type=int, help="X coordinate of the pattern")
    parser.add_argument("y", type=int, help="Y coordinate of the pattern")
    parser.add_argument("w", type=int, help="Width of the pattern")
    parser.add_argument("h", type=int, help="Height of the pattern")
    parser.add_argument(
        "--output-dir",
        default="patterns",
        help="Directory to save the pattern (default: patterns)",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=5,
        help="Additional padding around the pattern in pixels (default: 5)",
    )
    parser.add_argument(
        "--filename", help="Custom filename for the pattern (default: auto-generated)"
    )

    args = parser.parse_args()

    # Extract the pattern
    box = (args.x, args.y, args.w, args.h)
    output_path = extract_and_save_pattern(
        args.image_path,
        box,
        output_dir=args.output_dir,
        padding=args.padding,
        filename=args.filename,
    )

    print(f"Pattern extracted and saved to: {output_path}")


if __name__ == "__main__":
    import os

    from preprocessing import preprocess_image

    main()
