import os
from typing import Dict, List, Tuple

import cv2
import numpy as np


def load_templates(template_dir: str) -> Dict[str, np.ndarray]:
    """
    Load all template images from a directory for pattern matching.

    Args:
        template_dir: Path to directory containing template images

    Returns:
        Dictionary mapping template names to template images

    Raises:
        ValueError: If the directory cannot be found or has no valid templates
    """
    if not os.path.isdir(template_dir):
        raise ValueError(f"Template directory not found: {template_dir}")

    templates = {}
    valid_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]

    for filename in os.listdir(template_dir):
        # Check if file has a valid image extension
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            template_path = os.path.join(template_dir, filename)
            template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

            if template is not None:
                templates[filename] = template

    if not templates:
        raise ValueError(f"No valid template images found in {template_dir}")

    print(f"Loaded {len(templates)} templates from {template_dir}")
    return templates


def find_matching_patterns(
    image: np.ndarray,
    templates: Dict[str, np.ndarray] = None,
    template: np.ndarray = None,
    threshold: float = 0.6,
) -> List[Tuple[int, int, int, int]]:
    """
    Perform template matching to find patterns in the image using multiple templates.

    Args:
        image: Input image (grayscale)
        templates: Dictionary of templates to match against (from load_templates)
        template: Single template to match against (optional)
        threshold: Matching threshold (0.0 to 1.0)

    Returns:
        List of (x, y, w, h) bounding boxes for all matches found

    Raises:
        ValueError: If neither templates nor template is provided
    """
    if templates is None and template is None:
        raise ValueError("Either templates or template must be provided")

    all_boxes = []

    # Process each template if we have multiple
    if templates:
        for template_name, template_img in templates.items():
            template_h, template_w = template_img.shape

            # Perform template matching
            result = cv2.matchTemplate(image, template_img, cv2.TM_CCOEFF_NORMED)

            # Find locations where the matching score exceeds the threshold
            locations = np.where(result >= threshold)

            # Convert to a list of bounding boxes
            for pt in zip(*locations[::-1]):  # Reverse to get (x, y)
                x, y = pt
                all_boxes.append((x, y, template_w, template_h))

    # Handle single template
    elif template is not None:
        template_h, template_w = template.shape
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)

        for pt in zip(*locations[::-1]):
            x, y = pt
            all_boxes.append((x, y, template_w, template_h))

    # Apply non-maximum suppression to filter out overlapping boxes
    def calculate_iou(box1, box2):
        """Calculate Intersection over Union (IoU) between two boxes."""
        # Extract coordinates
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate intersection coordinates
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        # Check if there is an intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Calculate union area
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area

        # Calculate IoU
        iou = intersection_area / union_area

        return iou

    # Apply non-maximum suppression
    def non_max_suppression(boxes, iou_threshold=0.5):
        """Filter out overlapping boxes using non-maximum suppression."""
        if not boxes:
            return []

        filtered_boxes = []

        for box in boxes:
            should_keep = True

            # Check against all already filtered boxes
            for filtered_box in filtered_boxes:
                if calculate_iou(box, filtered_box) > iou_threshold:
                    should_keep = False
                    break

            if should_keep:
                filtered_boxes.append(box)

        return filtered_boxes

    # Apply non-maximum suppression to filter out overlapping boxes
    filtered_boxes = non_max_suppression(all_boxes, iou_threshold=0.5)

    return filtered_boxes


def draw_bounding_boxes(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes around the matched regions.

    Args:
        image: Image to draw on
        boxes: List of (x, y, w, h) bounding boxes
        color: Box color in BGR format
        thickness: Line thickness

    Returns:
        Image with bounding boxes drawn
    """
    # Convert grayscale to color if needed
    if len(image.shape) == 2:
        result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result_image = image.copy()

    # Draw bounding boxes
    for x, y, w, h in boxes:
        print(f"Match: x={x}, y={y}, w={w}, h={h}")
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, thickness)

    return result_image


def match_template(
    image: np.ndarray,
    template_dir: str,
    threshold: float = 0.6,
    output_dir: str = "results",
    image_filename: str = None,
) -> Tuple[List[Tuple[int, int, int, int]], np.ndarray]:
    """
    Complete template matching workflow: load templates from directory,
    find matches, and save results to a specified directory.

    Args:
        image: Input image (grayscale)
        template_dir: Path to directory containing template images
        threshold: Matching threshold
        output_dir: Directory to save the result image
        image_filename: Name of the current image file (for naming the output)

    Returns:
        Tuple of (list of bounding boxes, image with matches visualized)
    """
    # Load templates from directory
    templates = load_templates(template_dir)

    # Find matching patterns
    boxes = find_matching_patterns(image, templates=templates, threshold=threshold)

    # Create a new image for visualization (don't modify the original)
    if len(image.shape) == 2:
        result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result_image = image.copy()

    # Draw bounding boxes on the new image
    for x, y, w, h in boxes:
        print(f"Match: x={x}, y={y}, w={w}, h={h}")
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Save the result image to the output directory
    if output_dir and image_filename:
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")

        # Generate output filename based on original filename
        base_name = os.path.splitext(os.path.basename(image_filename))[0]
        output_filename = f"results_{base_name}.png"
        output_path = os.path.join(output_dir, output_filename)

        cv2.imwrite(output_path, result_image)
        print(f"Saved result image with {len(boxes)} matches to {output_path}")

    return boxes, result_image
