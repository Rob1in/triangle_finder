# triangle_finder

<p align="center">
  <img src="img/animation.gif" width="50%" alt="example">
</p>


## Setup

### Windows

To set up the project on Windows:

1. Create a virtual environment:
   ```
   python -m venv .venv
   ```

2. Activate the virtual environment:
   ```
   .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r .\requirements
   ```

### Mac/Linux

To set up the project on Mac or Linux:

1. Create a virtual environment:
   ```
   python -m venv .venv
   ```

2. Activate the virtual environment:
   ```
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running triangle detection


Once your venv is activated:


```
python src/main.py --images /path/to/input/data
```



### Command Line Options
#### Required

- `--images`: Directory containing input images to process (required)


#### Optional
- `--patterns`: Directory containing template patterns (default: "patterns")
- `--threshold`: Matching threshold (0.0 to 1.0) - higher values mean stricter matching (default: 0.65)
- `--output`: Directory to save result images (defaults to [images_dir]_results)
- `--show`: Flag to show visualizations for each processed image

### Output

When you run the script, it:

1. **Processes all images** in the specified input directory
2. **Creates an output directory** (by default named after your input directory with "_results" appended)
3. **Generates output files** for each processed image:
   - Files are named `results_[original_filename].png`
   - Each output image shows the original image with bounding boxes around detected triangles

