# Optimizer Comparison for Hybrid Positioning Model

This project implements a comprehensive comparison of different optimization algorithms for training a hybrid neural network model designed for 3D positioning tasks using multimodal spectrum data.

## Overview

The project compares four different optimizers:
- **SGD with Momentum** (momentum=0.9)
- **Adam**
- **AdamW** (with weight decay=0.01)
- **Adam with ReduceLROnPlateau** (learning rate scheduler)

The model uses a hybrid architecture that combines:
- **CNN branch**: Processes spatial spectrum images (3 channels, 36×9)
- **MLP branch**: Processes metadata (gateway positions and orientations)
- **Fusion head**: Combines features from both branches for final 3D position prediction

## Features

- **Comprehensive Optimizer Comparison**: Trains the same model with different optimizers and tracks performance metrics
- **Visualization**: Generates detailed comparison plots including:
  - Training/validation loss curves
  - Validation distance error curves
  - Bar charts comparing best performance and training time
- **Automatic Best Model Selection**: Selects the best optimizer based on validation performance and retrains a final model
- **Independent Test Evaluation**: Evaluates the final model on a held-out test set

## Requirements

### Environment Setup

Create a conda environment named `opt`:

```bash
conda create -n opt python=3.9 -y
conda activate opt
conda install pytorch numpy -c pytorch -y
pip install matplotlib tqdm pandas pillow
```

### Dependencies

- Python 3.9+
- PyTorch
- NumPy
- Matplotlib
- tqdm
- pandas
- Pillow (PIL)

## Data Format

The code expects data files in `.t` format (PyTorch tensor files) with the following structure:

- **Labels**: Columns 1-4 (ground truth 3D positions)
- **Spatial Spectrum**: Columns 29+ (reshaped to 3×36×9 for CNN input)
- **Metadata**: 
  - Gateway 1: Position (8-11), Orientation (11-15)
  - Gateway 2: Position (15-18), Orientation (18-22)
  - Gateway 3: Position (22-25), Orientation (25-29)

Required files:
- `train_0.t`: Training and validation data
- `test_0.t`: Independent test set

## Usage

### Running the Comparison

```bash
conda activate opt
python localization_model.py
```

### Output Files

The script generates:

1. **`opt_withtest_results.png`**: Main comparison figure with three subplots:
   - Training loss curves
   - Validation loss curves
   - Validation distance error curves

2. **`optimizer_summary.png`**: Bar chart comparison showing:
   - Best validation distance error for each optimizer
   - Total training time for each optimizer

3. **`final_best_model.pth`**: Saved model weights of the best performing configuration

## Model Architecture

### HybridPositioningModel

- **CNN Branch**:
  - Conv2d(3→16) → BatchNorm → ReLU → MaxPool
  - Conv2d(16→32) → BatchNorm → ReLU → MaxPool
  - Conv2d(32→64) → BatchNorm → ReLU → AdaptiveAvgPool
  - Output: 64-dimensional feature vector

- **Metadata MLP**:
  - Linear(21→64) → ReLU → Linear(64→32)
  - Output: 32-dimensional feature vector

- **Fusion Head**:
  - Concatenates CNN features (64) + Metadata features (32) = 96
  - Linear(96→128) → ReLU → Dropout(0.5)
  - Linear(128→64) → ReLU → Dropout(0.5)
  - Linear(64→3) → 3D position prediction

## Hyperparameters

Default configuration:
- Batch size: 64
- Learning rate: 0.01
- Number of epochs: 25
- Validation split ratio: 0.2 (20% for validation, 80% for training)

These can be modified in the main script.

## Performance Metrics

The script tracks and reports:
- **MSE Loss**: Mean squared error between predicted and ground truth positions
- **Distance Error**: Euclidean distance error in meters
- **Training Time**: Total time taken for each optimizer

## Code Structure

- `localization_model.py`: Main script with English comments (optimizer comparison)
- `comparison.py`: Original script with Chinese comments
- `README.md`: This documentation file

## Notes

- The script automatically detects and uses CUDA if available, otherwise falls back to CPU
- Progress bars are displayed during training and evaluation
- The best model is selected based on the lowest validation distance error
- All visualizations are saved automatically before display

## License

This project is provided as-is for research and educational purposes.

