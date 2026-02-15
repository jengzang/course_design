# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based Stokes polarization analysis tool for analyzing polarized light from images. The project computes Stokes parameters (S0, S1, S2, S3) from polarization images taken at different angles and visualizes the results on a Poincaré sphere.

## Running the Code

### Environment Setup
```bash
# Activate virtual environment
source .venv/bin/activate  # On Windows with Cygwin: source .venv/Scripts/activate

# Install dependencies (if needed)
pip install numpy pandas matplotlib pillow pyvista
```

### Main Scripts

**stokes_pro.py** (Recommended - Advanced version)
```bash
python stokes_pro.py
```
- Interactive GUI with three sampling tools: line tool, ellipse tool, and free drawing tool
- Requires 4 polarization images with suffixes: 9045, 090, 045, 00
- Requires max_i.txt file at: C:\Users\joengzaang\Desktop\课设\max_i.txt
- Generates locations.txt to cache origin points
- Outputs grayscale_stokes_output.txt with results
- Displays 3D Poincaré sphere visualization with PyVista

**stokes.py** (Basic version)
```bash
python stokes.py
```
- Manual point selection (fixed number of points)
- Same input/output requirements as stokes_pro.py

**grayscale_analysis.py**
```bash
python grayscale_analysis.py
```
- Analyzes average grayscale values in selected rectangular regions
- Outputs grayscale_results.txt

**polarization_analysis.py**
```bash
python polarization_analysis.py
```
- Simple calculator for Stokes parameters with hardcoded values
- No file input required

## Architecture

### Data Flow
1. **Image Loading**: User selects 4 polarization images via tkinter file dialog
2. **Origin Selection**: First run prompts user to click origin (0,0) on each image, cached in locations.txt
3. **Point Sampling**: User selects sampling tool and marks points/paths on reference image
4. **Coordinate Transformation**: Points converted to relative coordinates based on origin
5. **Grayscale Extraction**: Extract pixel values at corresponding positions across all 4 images
6. **Intensity Correction**: Multiply grayscale by max intensity from max_i.txt
7. **Stokes Computation**: Calculate S0, S1, S2, S3 using polarization formulas
8. **Normalization**: Compute normalized parameters S0*, S1*, S2*, S3* and polarization degree
9. **Visualization**: Display results on Poincaré sphere and multi-image preview

### Key Components

**stokes_pro.py structure:**
- Lines 22-34: File selection dialog
- Lines 36-44: Load max intensity values from max_i.txt
- Lines 60-102: Origin point selection/loading with caching
- Lines 103-423: Interactive sampling tools (line/ellipse/free)
- Lines 433-467: Grayscale extraction with intensity correction
- Lines 468-490: Stokes parameter computation
- Lines 492-508: Normalization and polarization degree
- Lines 512-541: Polarization state classification
- Lines 551-700: PyVista 3D Poincaré sphere visualization

**Image naming convention:**
- Files must end with: 9045, 090, 045, or 00
- Maps to: gray(90,45), gray(0,90), gray(0,45), gray(0,0)

**Stokes formulas:**
- S0 = gray(0,0) + gray(0,90)
- S1 = gray(0,0) - gray(0,90)
- S2 = 2 * gray(0,45) - S0
- S3 = S0 - 2 * gray(90,45)

## Important Notes

- The project uses matplotlib with Qt5Agg backend for better window interaction
- Chinese text is supported via SimHei font
- locations.txt caching prevents re-selecting origins on subsequent runs
- Delete locations.txt to reset origin points
- max_i.txt path is hardcoded - update line 37 in stokes_pro.py if needed
- PyVista visualization includes interactive checkbox to toggle data table display
