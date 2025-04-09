import argparse
import os

import cv2
import matplotlib.pyplot as plt

from pattern_matching import match_template

# Update imports to use modules from src directory
from preprocessing import preprocess_image

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Template matching for image processing")
parser.add_argument(
    "--images",
    type=str,
    required=True,
    help="Directory containing input images to process",
)
parser.add_argument(
    "--threshold", type=float, default=0.65, help="Matching threshold (0.0 to 1.0)"
)
parser.add_argument(
    "--patterns",
    type=str,
    default="patterns",
    help="Directory containing template patterns",
)
parser.add_argument(
    "--output",
    type=str,
    default="results",
    help="Directory to save result images",
)
parser.add_argument(
    "--show", action="store_true", help="Show visualizations for each processed image"
)
args = parser.parse_args()

# Get arguments
images_dir = args.images
threshold = args.threshold
template_dir = args.patterns
output_dir = args.output
show_visualizations = args.show

# Check if the input directory exists
if not os.path.isdir(images_dir):
    print(f"Error: Input directory '{images_dir}' not found")
    exit(1)

# Process all images in the input directory
valid_extensions = [".png", ".jpg", ".jpeg"]
processed_count = 0

for filename in os.listdir(images_dir):
    # Check if file is an image
    if any(filename.lower().endswith(ext) for ext in valid_extensions):
        # Full path to the input image
        image_path = os.path.join(images_dir, filename)
        print(f"\nProcessing image: {image_path}")

        try:
            # Preprocess the image
            image_grey = preprocess_image(image_path)

            # Load the template and perform matching
            boxes, result_image = match_template(
                image_grey,
                template_dir,
                threshold=threshold,
                output_dir=output_dir,
                image_filename=filename,  # Pass just the filename
            )

            # Print results
            print(f"Found {len(boxes)} matches in {filename}")
            for x, y, w, h in boxes:
                print(f"  Match at position (x={x}, y={y}), size: {w}x{h}")

            # Display the image if requested
            if show_visualizations:
                plt.figure(figsize=(12, 8))
                plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
                plt.title(f"Matches in {filename}")
                plt.axis("equal")
                plt.show()

            processed_count += 1

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

print(f"\nProcessing complete. Processed {processed_count} images.")
