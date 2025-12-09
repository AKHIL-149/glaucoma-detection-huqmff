# Glaucoma Detection using H-UQ-MFF

An automated glaucoma screening system using heterogeneous uncertainty-quantified multi-head feature fusion (H-UQ-MFF) on the EyePACS-AIROGS-light-V2 dataset.

## Overview

This project implements a clinical decision-support system for detecting referable glaucoma from color fundus photographs. The approach combines deep learning features with traditional clinical features and incorporates uncertainty quantification to provide reliable predictions for clinical use.

### Key Features

- **Multi-modal Feature Fusion**: Combines deep features (ResNet50), structural features (CDR, ISNT), and texture features (LBP, GLCM)
- **Uncertainty Quantification**: MC-Dropout with temperature scaling for calibrated predictions
- **High Performance**: 89.5% AUC on external validation, 98.1% sensitivity, 97.2% specificity
- **Clinical Ready**: Includes uncertainty bands for clinical decision-making and Grad-CAM visualization

## Architecture

The H-UQ-MFF pipeline consists of:

1. **Preprocessing**: Image resizing, optic disc cropping, augmentation
2. **Feature Extraction**:
   - Deep features: ResNet50 (2048-d)
   - Structural features: CDR, ISNT, rim-to-disc ratio (10-d)
   - Texture features: LBP + GLCM (64-d)
3. **Multi-head Fusion**: Weighted combination with uncertainty-aware fusion
4. **Uncertainty Estimation**: 50-pass MC-Dropout + temperature scaling
5. **Clinical Thresholds**: Risk bands (GREEN/YELLOW/RED) based on uncertainty

## Dataset

The model is trained on **EyePACS-AIROGS-light-V2** and validated on:
- REFUGE dataset
- PAPILA dataset

Dataset links are available in [`Dataset/Dataset links.txt`](Dataset/Dataset%20links.txt)

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/Akhil-149/glaucoma-detection-huqmff.git
cd glaucoma-detection-huqmff

# Install dependencies
pip install torch torchvision
pip install opencv-python albumentations scikit-learn scikit-image
pip install pandas numpy matplotlib jupyter

# Download datasets (see Dataset/Dataset links.txt)
```

## Usage

### Training and Evaluation

Open and run the Jupyter notebook:

```bash
jupyter notebook "Code/Glaucoma_Al_HUQMFF.ipynb"
```

The notebook contains 10 main steps:

1. Load the Dataset
2. Preprocessing
3. Feature Extraction
4. Uncertainty-Aware Fusion (H-UQ-MFF)
5. Ablation Studies
6. Baseline Comparison
7. Cross-Dataset Validation
8. Regulatory Readiness
9. Deployment Optimization
10. Performance Metrics

### Quick Start

```python
# Load and preprocess image
image = preprocess_full(image_path, target_size=256)

# Extract features
deep_feat = extract_deep_features(image)
struct_feat = extract_structural_features(image)
texture_feat = extract_texture_features(image)

# Fuse and predict with uncertainty
fused = fuse_features(deep_feat, struct_feat, texture_feat)
risk_score, uncertainty = predict_with_uncertainty(fused)

# Get clinical action band
action_band = clinical_threshold(uncertainty)  # GREEN/YELLOW/RED
```

## Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| AUC (External) | 0.895 |
| Sensitivity | 98.1% |
| Specificity | 97.2% |
| Accuracy | 97.7% |
| ECE (Calibration) | 0.234 |
| Brier Score | 0.164 |

### Baseline Comparisons

| Model | AUC |
|-------|-----|
| **H-UQ-MFF** | **0.895** |
| Deep Ensemble UQ | 0.857 |
| EfficientNet-B0 | 0.796 |
| ResNet50 | 0.784 |

### Visualizations

Performance visualizations are available in the [`Outputs/`](Outputs/) directory:
- AUC comparisons: `auc_comparison.png`, `external_auc_comparison.png`
- Calibration plots: `ece_comparison.png`, `external_ece_comparison.png`
- Metrics radar plot: `radar_internal_metrics.png`
- Deployment profiles: `deploy_model_profiles.csv`

## Project Structure

```
.
├── Code/
│   └── Glaucoma_Al_HUQMFF.ipynb    # Main implementation notebook
├── Dataset/
│   └── Dataset links.txt            # Links to datasets
├── Outputs/                         # Results, metrics, visualizations
│   ├── metrics_summary.csv
│   ├── baselines_results.csv
│   ├── deploy_api_contract.txt
│   └── *.png                        # Performance plots
├── Base paper/
│   └── Base_paper.pdf               # Reference paper
├── Reference papers/                # Additional references
├── Flow and Novelty explanation/
│   └── Glaucoma_AI_Flowchart.pptx  # Architecture flowchart
└── Results/                         # Empty (CSVs moved to Outputs/)
```

## Clinical Deployment

The model includes deployment-ready features:

- **API Contract**: REST API specification in `Outputs/deploy_api_contract.txt`
- **Uncertainty Bands**: Clinical action thresholds (GREEN/YELLOW/RED)
- **Interpretability**: Grad-CAM heatmaps for visual explanation
- **Quality Checks**: Image quality scoring and failure mode detection
- **Model Optimization**: INT8 quantization and pruning for edge deployment

### API Example

```json
{
  "input": {
    "patient_id": "P12345",
    "image": "base64_encoded_fundus_image",
    "metadata": {
      "age": 65,
      "sex": "M",
      "camera": "Topcon",
      "eye": "OD"
    }
  },
  "output": {
    "risk_score": 0.87,
    "uncertainty": 0.12,
    "action_band": "GREEN",
    "quality_score": 0.94,
    "gradcam_heatmap": "base64_encoded_image"
  }
}
```

## Uncertainty Quantification

The system provides uncertainty estimates for each prediction:

- **GREEN** (U < 0.2): High confidence, routine follow-up
- **YELLOW** (0.2 ≤ U < 0.5): Moderate confidence, clinician assistance recommended
- **RED** (U ≥ 0.5): Low confidence, manual specialist review required

## References

- Base paper: See [`Base paper/Base_paper.pdf`](Base%20paper/Base_paper.pdf)
- Additional references: See [`Reference papers/`](Reference%20papers/)

## Citation

If you use this code in your research, please cite:

```bibtex
@article{glaucoma_huqmff_2024,
  title={Heterogeneous Uncertainty-Quantified Multi-head Feature Fusion for Glaucoma Detection},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is for research and educational purposes. Not intended as a standalone diagnostic device.

## Contact

For questions or collaboration opportunities, please open an issue or contact me at venkatakhil149@gmail.com
