# Detecting Deepfakes Using Error Level Analysis and Transfer Learning

This repository contains the code and LaTeX reports for my final university project on deepfake detection. The project leverages Error Level Analysis (ELA) to visualize JPEG compression inconsistencies, which are then fed into a fine-tuned EfficientNet-B0 neural network to classify images as real or manipulated.

## Project Structure
- `.gitignore`: Excludes the 1GB dataset and Python caches from version control.
- `src/generate_ela.py`: Script to process raw images into ELA heatmaps.
- `src/train_model.py`: PyTorch training script utilizing transfer learning on EfficientNet-B0.
- `best_model.pth`: The saved weights of our best-performing model (86.40% Validation Accuracy).
- `results.txt`: The final evaluation metrics.
- `report/`: Contains the CVPR-style LaTeX source code for the final written report.
- `poster/`: Contains the `tikzposter` LaTeX source code for the presentation poster.

## Installation and Requirements

To run the Python code locally, you need Python 3.8+ and an NVIDIA GPU (recommended for fast training).

1. **Install PyTorch with CUDA support** (Example for CUDA 12.1):
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
2. **Install other dependencies**:
   ```bash
   pip install pillow scikit-learn
   ```

## Usage

### 1. Data Preprocessing (ELA Generation)
Place your raw image dataset inside `1GB_dataset/raw/real` and `1GB_dataset/raw/fake`.
Run the ELA generation script:
```bash
python src/generate_ela.py
```
This will populate the `1GB_dataset/ela/train` and `1GB_dataset/ela/val` directories.

### 2. Model Training
Run the training script to fine-tune the EfficientNet model:
```bash
python src/train_model.py
```
The script will output progress per epoch and save the highest performing model to `best_model.pth`.

## Compiling the LaTeX Documents

If you have a local LaTeX distribution (like TeX Live or MiKTeX), you can compile the `.tex` files using `pdflatex`.

**For the Report:**
```bash
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

**For the Poster:**
```bash
cd poster
pdflatex poster.tex
```

*Tip: You can also zip the `report` or `poster` directories and upload them directly to [Overleaf](https://www.overleaf.com/) to compile them in your browser without installing LaTeX locally.*

## Results
Our optimized model achieved the following metrics on the validation set:
- **Accuracy:** 86.40%
- **Precision:** 88.00%
- **Recall:** 84.30%
- **F1-Score:** 86.11%
