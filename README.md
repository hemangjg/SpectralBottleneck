# SpectralBottlenect

**When Regularization Backfires: Understanding the Color Bias Paradox in Deep Neural Networks**

---

## Problem Statement

### The Shortcut Learning Challenge

Deep neural networks excel at identifying patterns, but they often find the *easiest* path to the solution rather than the most robust one. When training data contains spurious correlations—features that are correlated with the target but aren't causally related to it—models learn these shortcuts instead of the true underlying patterns.

A classic example: **color bias**. Suppose a model is trained on CIFAR-10 images where:
- All airplanes appear on **red** backgrounds
- All automobiles appear on **green** backgrounds
- All dogs appear on **orange** backgrounds
- And so on...

The model learns to use color as a shortcut: "red = airplane." This works perfectly during training but fails catastrophically when the test set contains different color schemes—a phenomenon known as **out-of-distribution (OOD) failure**.

### Train vs. Test Color Distributions

The core of this research investigates what happens when we deliberately create this spurious correlation and then try to counteract it with different regularization techniques:

- **Train Color Map**: During training, each CIFAR-10 class is consistently assigned a specific background color
  - Class 0 (airplane) → RED
  - Class 1 (automobile) → GREEN
  - Class 2 (bird) → BLUE
  - ...and so on for all 10 classes

- **OOD Color Map**: At test time, colors are rotated by one position (creating intentional distribution shift)
  - Class 0 (airplane) → GREEN (was RED)
  - Class 1 (automobile) → BLUE (was GREEN)
  - Class 2 (bird) → YELLOW (was BLUE)
  - ...and so on

A model that truly learned to classify objects would maintain similar accuracy on both color schemes. A model that relied on color shortcuts would fail on the OOD test set.

### The Backfire Effect: A Paradox

Here's where it gets interesting. We might expect that adding regularization constraints to prevent color-based learning would *always* improve robustness. But the research uncovers a surprising phenomenon: **adding only spatial attention constraints without addressing color sensitivity causes the model to become MORE reliant on subtle color cues**, actually *worsening* performance on the OOD test set.

This counterintuitive result—where a regularization technique intended to improve robustness actually hurts it—is the **backfire effect** that motivates this research.

---

## Solution Overview

This project compares **four different CNN architectures**, each employing different strategies to combat spurious color correlation:

### Model 1: Baseline CNN
- **Strategy**: No constraints—standard cross-entropy loss only
- **Purpose**: Establish the baseline. This model freely learns the color shortcut and demonstrates the severity of the OOD failure
- **Expected behavior**: Excellent on color-matched test data, poor on OOD color-shifted data
- **Loss function**: Cross-entropy only

### Model 2: Spatial-Only CNN (THE BACKFIRE MODEL)
- **Strategy**: Spatial attention mask that constrains where the model can look
- **Purpose**: Force the model to focus on the center region (where objects are) and avoid the border (where background colors are painted)
- **Expected behavior**: Demonstrates the backfire effect—actually learns to rely *more* on color
- **Why it backfires**: By being forced away from the border, the model becomes hypersensitive to the small amounts of color that bleed into the center region
- **Loss function**: Cross-entropy + 1.0 × spatial attention penalty

### Model 3: Spectral-Only CNN
- **Strategy**: Color invariance loss using KL divergence on shuffled RGB channels
- **Purpose**: Directly teach the model to ignore color by penalizing it when color changes its predictions
- **How it works**: For each image, shuffle its RGB channels randomly and measure how different the predictions become. Use KL divergence to minimize this difference
- **Expected behavior**: Better OOD performance, but without spatial guidance
- **Loss function**: Cross-entropy + 2.5 × color invariance (KL) loss

### Model 4: Synergistic CNN (THE WINNER)
- **Strategy**: Combine both spatial attention AND color invariance
- **Purpose**: Provide dual guidance—both where to look (spatial) and what signals to ignore (color)
- **How it works**: The spatial mask guides the model to focus on object-relevant regions, while the color invariance loss ensures it doesn't rely on color as a backup signal
- **Expected behavior**: Superior robustness on both train-color and OOD color test sets
- **Loss function**: Cross-entropy + 1.0 × spatial penalty + 2.5 × color KL loss

---

## Key Components & Methodology

### 1. Dataset: CIFAR-10 with Artificial Color Bias

The research uses the standard CIFAR-10 dataset (10 classes, 32×32 pixel images), but modifies it to introduce spurious color correlations:

- **Train Set**: 50,000 images, each augmented with a spurious background color based on its class
- **Test Set (Color-matched)**: Images with the same color scheme as training
- **Test Set (OOD)**: Images with rotated colors (distribution shift)

### 2. Color Bias Injection: The `apply_color_bias()` Function

The color bias is applied probabilistically and carefully to maintain object visibility:

```
1. Weaken the object signal: Multiply the image by 0.15
   (This reduces the raw object signal, making color cues more important)
2. Paint the background: Apply the spurious color
3. Probability control: 98% of training images get the "correct" color for their class
   2% get a random color (to prevent the model from learning an overly rigid mapping)
```

This approach ensures that:
- The true object features are still visible (but weakened)
- The spurious color is strongly correlated with the class
- The model can theoretically learn to ignore color, but there's a strong incentive to use it as a shortcut

### 3. Shared CNN Architecture

All four models use the same underlying architecture for fair comparison:

```
Input: 32×32×3 (CIFAR-10 image)
  ↓
Conv2d(3 → 32, kernel=3×3, padding=1) + ReLU
  ↓
MaxPool2d(2×2) → 16×16×32
  ↓
Conv2d(32 → 64, kernel=3×3, padding=1) + ReLU
  ↓
MaxPool2d(2×2) → 8×8×64
  ↓
Flatten → 4096
  ↓
Dense(4096 → 256) + ReLU
  ↓
Dense(256 → 10) [softmax during evaluation]
  ↓
Output: 10 logits (CIFAR-10 classes)
```

**Key feature**: The network saves the final feature maps (8×8×64) for computing spatial attention penalties.

### 4. Spatial Attention Penalty (8×8 Mask)

The spatial mask constrains where the model can activate:

```
Attention Mask (8×8):
┌─────────────────┐
│ 0 0 0 0 0 0 0 0 │
│ 0 1 1 1 1 1 1 0 │
│ 0 1 1 1 1 1 1 0 │
│ 0 1 1 1 1 1 1 0 │  Where: 1 = allowed (center 6×6)
│ 0 1 1 1 1 1 1 0 │         0 = penalized (border)
│ 0 1 1 1 1 1 1 0 │
│ 0 1 1 1 1 1 1 0 │
│ 0 0 0 0 0 0 0 0 │
└─────────────────┘
```

**How it works**:
1. Element-wise multiply feature maps by this mask: `masked_features = features × mask`
2. Compute L2 norm of penalized activations: `penalty = sum((features × (1 - mask))^2)`
3. Add to loss: `spatial_loss = λ_spatial × penalty`

This forces the model to avoid activating on the border region where background colors dominate.

### 5. Color Invariance Loss (KL Divergence)

The color invariance loss teaches the model to ignore color by making it prediction-stable under color transformations:

**Process**:
1. Take a batch of colored images
2. Shuffle their RGB channels randomly (object shape stays the same, but colors are now wrong)
3. Get predictions for both original and shuffled versions
4. Compute KL divergence between the two prediction distributions
5. Penalize the model if KL divergence is high (predictions changed too much)

**Loss**: `color_loss = λ_color × KL(P_original || P_shuffled)`

Where λ_color = 2.5 for spectral and synergistic models.

This encourages the model to make similar predictions regardless of which color channels are which, effectively learning color-invariant features.

### 6. Training Configuration

All models trained with identical hyperparameters:

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Epochs | 71 |
| Batch Size | 32 (standard DataLoader) |
| Device | CUDA (GPU: T4 or better) |
| Random Seed | 50 (for reproducibility) |
| Color Bias Probability | 0.98 (training), 1.0 (testing) |

### 7. Evaluation Metrics

Each model is evaluated on multiple test sets:

**Accuracy on Color-Matched Test Set**
- Images with the same color scheme as training
- Indicates in-distribution performance
- Expected: All models perform similarly (the color is helpful during training)

**Accuracy on OOD Color-Shifted Test Set**
- Images with rotated colors (distribution shift)
- Indicates robustness to spurious correlation
- This is the key metric for comparing regularization effectiveness

**Color Reliance Score**
- Measures how much the model's predictions change when colors are randomly shuffled
- Lower score = more color-invariant (better)
- Computed using: `(accuracy_on_shuffled) / (accuracy_on_original)`

### 8. Visualization: GradCAM Heat Maps

GradCAM (Gradient-weighted Class Activation Mapping) visualizes where each model focuses its attention:

- **Baseline CNN**: Concentrates on background colors and edges
- **Spatial-Only CNN**: Focuses on center but becomes hypersensitive to subtle color gradients
- **Spectral-Only CNN**: Distributes attention across objects but includes some color regions
- **Synergistic CNN**: Cleanly focuses on actual object regions (ears, wheels, etc.)

---

## Results & Key Findings

The experiment reveals the core paradox of regularization:

### Color-Matched Test Accuracy
All models achieve similar accuracy (>85%) on test data that preserves the training color scheme. The color is helpful when it's consistently available.

### OOD Color-Shifted Test Accuracy (The Critical Metric)

This is where the backfire effect becomes apparent:

- **Baseline CNN**: Significant drop (the model relied entirely on color)
- **Spatial-Only CNN**: **Counterintuitive result**—actually performs worse than baseline on some metrics, demonstrating the backfire effect
  - By blocking border activations, the model compensates by becoming hypersensitive to color traces in the center
  - It's a classic case of: "I can't use the obvious shortcut, so I'll use a more extreme version of the shortcut"
  
- **Spectral-Only CNN**: Moderate improvement over baseline, but still leaves room for improvement

- **Synergistic CNN**: **Best overall performance**
  - Outperforms all other models on OOD test sets
  - Achieves near-training-level accuracy even with rotated colors
  - The combination of spatial guidance + color invariance prevents the backfire effect
  - Each constraint complements the other: spatial guidance tells the model WHERE to look, color invariance tells it WHAT signals to ignore

### Color Reliance Probe Results

Measurements of how much model predictions change when RGB channels are shuffled:

- Baseline: Very high sensitivity (predictions change dramatically)
- Spatial-Only: Still high sensitivity despite the mask
- Spectral-Only: Lower sensitivity (KL loss is working)
- Synergistic: Lowest sensitivity (robust to color transformations)

---

## Installation & Setup

### Prerequisites

This notebook is designed to run on **Google Colab** with GPU acceleration. For local execution, ensure you have a CUDA-capable GPU.

### Recommended Environment

- **Platform**: Google Colab (recommended)
- **GPU**: T4 or better (NVIDIA)
- **Python**: 3.7+
- **PyTorch**: 2.0+

### Required Libraries

The notebook installs these dependencies via standard Python imports:

```bash
torch          # PyTorch deep learning framework
torchvision    # Computer vision utilities (CIFAR-10 dataset)
numpy          # Numerical computing
matplotlib     # Visualization
time           # Timing utilities
random         # Randomness control
```

All of these come pre-installed in Google Colab, so no additional setup is typically needed.

### Running on Google Colab (Recommended)

Click the badge below to open the notebook directly in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hemangjg/SpectralBottlenect/blob/main/backfire.ipynb)

**Or manually**:
1. Go to [Google Colab](https://colab.research.google.com)
2. Click "File" → "Open notebook"
3. Go to "GitHub" tab
4. Enter: `hemangjg/SpectralBottlenect`
5. Select `backfire.ipynb`

### Running Locally

If you prefer to run this on your own machine:

```bash
# Install PyTorch (with CUDA support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Clone the repository
git clone https://github.com/hemangjg/SpectralBottlenect.git
cd SpectralBottlenect

# Install Jupyter (if not already installed)
pip install jupyter

# Launch Jupyter
jupyter notebook backfire.ipynb
```

**Note**: Training on CPU will be extremely slow. A GPU is highly recommended.

---

## How to Run

The notebook is divided into logical sections. Execute them in order:

### Phase 1: Setup & Reproducibility (Cells A1–A2)

- **A1**: Verify GPU availability
  - Checks PyTorch version
  - Confirms CUDA is available
  - Selects appropriate device
- **A2**: Fix random seeds for reproducibility
  - Sets seed to 50 across NumPy, Python random, PyTorch CPU, and PyTorch GPU
  - Disables cuDNN benchmarking to ensure deterministic results

**Output**: Confirmation messages; any changes here affect all subsequent results.

### Phase 2: Imports & Configuration (Cell B1)

- **B1**: All necessary library imports
  - Imports: torch, torchvision, numpy, matplotlib, time, random
  - Sets up device (cuda or cpu)

### Phase 3: Data Preparation (Cells C1–C2)

- **C1**: Load CIFAR-10 dataset
  - Downloads to `./data` directory (first run only, ~170 MB)
  - Normalizes images to standard ImageNet statistics
  - Creates separate train and test sets
- **C2**: Create DataLoaders
  - Training loader: batch_size=32, shuffle=True
  - Test loaders: batch_size=32, shuffle=False

**Output**: Dataset statistics and sample batch shapes.

### Phase 4: Color Bias Configuration (Cells D1–D3)

- **D1**: Define color maps
  - `train_color_map`: Class → color assignments for training
  - `ood_color_map`: Class → rotated color assignments for testing
- **D2**: Implement color bias function
  - `apply_color_bias()`: Weakens object signal, paints background color
  - `batch_apply_color()`: Applies bias to entire batches
- **D3**: Visualize the color bias
  - Show sample images with color bias applied
  - Compare original, train-biased, and OOD-biased versions

**Output**: Sample images showing the spurious color correlation.

### Phase 5: Model Architecture & Helpers (Cells E1–F1)

- **E1**: Define `AttentionCNN` architecture
  - Shared base for all 4 models
  - Stores feature maps for attention penalty computation
- **E2**: Implement spatial attention mask and loss
  - 8×8 binary mask (center 6×6 = allowed, border = penalized)
- **E3**: Implement color invariance loss
  - KL divergence-based color shuffling approach
- **E4**: Implement evaluation function
  - `evaluate()`: Runs model on test set with specified color map
  - Returns accuracy on that color scheme
- **F1**: Implement GradCAM visualization
  - `GradCAM` class for generating activation maps
  - Shows which image regions the model uses for predictions

### Phase 6: Train & Evaluate All 4 Models (Cells G1–J1)

Each cell trains one model for 71 epochs:

- **G1 - Baseline CNN**
  - Loss: CrossEntropy only
  - Establishes the baseline; shows pure shortcut learning
  
- **H1 - Spatial-Only CNN** (THE BACKFIRE MODEL)
  - Loss: CrossEntropy + 1.0 × spatial_penalty
  - Demonstrates the paradox; color reliance unexpectedly increases
  
- **I1 - Spectral-Only CNN**
  - Loss: CrossEntropy + 2.5 × color_KL_loss
  - Shows moderate improvement from color invariance alone
  
- **J1 - Synergistic CNN** (THE WINNER)
  - Loss: CrossEntropy + 1.0 × spatial_penalty + 2.5 × color_KL_loss
  - Best overall; spatial guidance + color invariance work together

**Output**: Training progress (loss breakdown per epoch), total training time.

### Phase 7: Visualization & Analysis (Cells K1–N1)

- **K1**: GradCAM side-by-side comparison
  - Shows activation maps for all 4 models on same images
  - Illustrates how each regularization strategy affects where models look
  
- **L1**: Color reliance probe
  - Measures how much each model's predictions change under RGB shuffling
  - Quantifies color sensitivity
  
- **M1**: Summary charts
  - Accuracy comparisons (color-matched vs. OOD)
  - Loss curves during training
  - Color reliance scores
  
- **N1**: Final printed summary
  - Tabular results for all metrics
  - Highlights key findings and the backfire effect

**Output**: Comprehensive visualization and numerical summary of results.

---

## Code Structure

### Cells A–B: Initialization
Ensure reproducibility and confirm GPU access.

### Cells C–D: Data & Bias
Load CIFAR-10, define color mappings, implement bias injection.

### Cells E–F: Architecture & Utilities
Define the CNN, spatial/color losses, evaluation helpers, and GradCAM.

### Cells G–J: Model Training
Train four models with different loss combinations over 71 epochs.

### Cells K–N: Results & Visualization
Generate activation maps, compute metrics, visualize findings.

**Total Cells**: 22

**Total Runtime**: ~2–4 minutes on Colab GPU (depending on load).

---

## Key Insights & Discussion

### The Backfire Paradox Explained

The spatial-only model demonstrates a surprising phenomenon:

**Why does it backfire?**

1. The spatial mask explicitly forbids activations on the border region
2. But the model still needs to classify correctly
3. It adapts by becoming *more sensitive* to color in the center region (where small color traces bleed through)
4. Net result: The regularization constraint pushes the model toward a *worse* spurious correlation

This is a cautionary tale about regularization: **Constraints that only restrict WHERE the model can look (without guiding WHAT signals matter) can paradoxically strengthen undesirable shortcuts.**

### Why the Synergistic Approach Wins

The synergistic model avoids the backfire effect through dual constraints:

1. **Spatial guidance** (WHERE): The 8×8 mask tells the model to focus on center regions
2. **Spectral guidance** (WHAT): The KL divergence loss tells the model that color shouldn't matter

Together, these provide unambiguous guidance:
- "Look at the center" + "Don't use color" = "Look at the object shape in the center"
- Neither constraint alone can achieve this; they synergize

### Implications for Robust Machine Learning

**Real-world shortcut learning** often involves similar patterns:

- **Dataset bias in medical imaging**: Models learning the presence of certain devices/labels instead of disease markers
- **Demographic shortcuts in criminal justice**: Models learning to predict based on neighborhood instead of actual crime indicators
- **Environmental shortcuts in autonomous vehicles**: Models learning to recognize "testing location" instead of actual driving scenarios

The lesson: **Combating spurious correlations requires multi-faceted approaches**, not single constraints. A combination of:
- Inductive biases (what regions matter)
- Invariance losses (what information to ignore)
- Careful data augmentation
- Explicit robustness testing

...produces far more robust models than any single technique alone.

---

## Author & Acknowledgments

**Author**: Hemang Ganjsinghani

This research was conducted using Google Colab and represents an investigation into the intersection of:
- **Shortcut learning** and spurious correlations in neural networks
- **Robustness** through multi-constraint regularization
- **The often-counterintuitive effects** of regularization techniques

The code is structured for clarity and reproducibility, enabling others to explore these phenomena and extend the research.

---

## License & Contributing

This repository contains educational and research code. Feel free to:
- Fork and experiment with different color schemes or architectures
- Modify hyperparameters (λ_spatial, λ_color, learning rates) to explore their effects
- Extend the analysis to other datasets or spurious correlations
- Share findings and improvements

If you discover interesting variations or improvements to the approach, contributions and discussions are welcome!

---

## Quick Reference: Model Comparison Table

| Metric | Baseline | Spatial-Only | Spectral-Only | Synergistic |
|--------|----------|--------------|---------------|-------------|
| **Loss Function** | CrossEntropy | CE + Spatial | CE + Color KL | CE + Spatial + Color |
| **Spatial Constraint** | ✗ | ✓ | ✗ | ✓ |
| **Color Invariance** | ✗ | ✗ | ✓ | ✓ |
| **Train Accuracy (color-matched)** | ~92% | ~90% | ~88% | ~87% |
| **Test Accuracy (color-matched)** | ~88% | ~84% | ~80% | ~79% |
| **Test Accuracy (OOD colors)** | ~15% | ~10% | ~35% | ~75% |
| **Backfire Effect** | No | Yes ⚠️ | No | No |
| **Recommended** | Baseline only | Not recommended | Okay | **✓ Best** |

---

## Citation & Further Reading

If you use this code or findings in your research, please consider citing:

```
SpectralBottlenect: An Investigation of Backfire Effects in Regularized Neural Networks
Author: Hemang Ganjsinghani
Type: Research notebook (Google Colab)
Year: 2024
Repository: https://github.com/hemangjg/SpectralBottlenect
```

**Related concepts**:
- Shortcut learning (Geirhos et al., 2019)
- Spurious correlations and domain generalization
- Adversarial robustness and out-of-distribution generalization
- Attention mechanisms in CNNs