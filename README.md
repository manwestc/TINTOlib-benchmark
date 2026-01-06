# A Comprehensive Benchmark of Spatial Encoding Methods for Tabular Data with Deep Neural Networks

**Authors:** Jiayun Liu, Manuel Castillo-Cara, Raúl García-Castro  
**Affiliation:** Universidad Politécnica de Madrid  
**Published in:** Information Fusion, Vol. 130, 2026 (Open Access)  
**DOI:** https://doi.org/10.1016/j.inffus.2025.104088  
**Project Page:** https://oeg-upm.github.io/TINTOlib/

---

## Overview

Despite the success of deep neural networks on perceptual data, their performance on tabular data remains limited, where traditional models still outperform them. A promising alternative is to transform tabular data into synthetic images, enabling the use of vision architectures such as Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs). However, the literature lacks a large-scale, standardized benchmark evaluating these transformation techniques.

This repository presents **the first comprehensive evaluation of 9 spatial encoding methods across 24 diverse regression and classification datasets**. We assess performance, scalability, and computational trade-offs under a unified framework with rigorous hyperparameter optimization, providing clear guidance for researchers and practitioners on when and how to effectively apply these techniques.

## Video

https://github.com/user-attachments/assets/8b167075-2010-4072-a5ff-dea6fc117437

---

## Research Questions and Findings

**Main Questions:**
- How do different spatial encoding methods compare across diverse datasets and tasks?
- Do vision architectures (CNN, ViT) outperform traditional deep learning on transformed tabular data?
- What role do hybrid models (combining vision and dense layers) play in performance and variance reduction?
- How do sample size (N) and dimensionality (d) affect transformation method effectiveness?

**Key Findings:**
- **REFINED emerges as the most robust transformation** across tasks and datasets
- **Transformation method choice exerts significantly stronger influence** on predictive performance than the chosen vision architecture
- **Performance landscape is structured by data regimes**, defined by sample size (N) and dimensionality (d)
- **Hybrid models (CNN+MLP, ViT+MLP) consistently reduce predictive variance**, offering advantages especially in smaller datasets, yet play a secondary role to transformation choice
- **Transforming tabular data into synthetic images is a powerful, yet data-dependent strategy** — optimal methods vary across different data regimes

---

## Main Contributions

1. **First large-scale standardized benchmark** for spatial encoding methods on tabular data
2. **Comprehensive evaluation** of 9 methods across 24 diverse datasets spanning regression and classification tasks
3. **Unified framework** with rigorous hyperparameter optimization ensuring fair comparison
4. **Data regime analysis** revealing how sample size (N) and dimensionality (d) structure performance
5. **Practical insights** on scalability, computational trade-offs, and architectural interplay
6. **Clear guidance** for researchers and practitioners on method selection strategies

---

## Benchmark Scope

### Spatial Encoding Methods (9)
1. **TINTO** (TINTO_blur) — Tabular data to image transformation with blur enhancement
2. **REFINED** — Robust encoding approach
3. **IGTD** — Image generator for tabular data using distance preservation
4. **FeatureWrap** — Feature wrapping in 2D space
5. **SuperTML** — Supervised tabular-to-image method
6. **BarGraph** — Bar graph visualization encoding
7. **DistanceMatrix** — Distance-based matrix representation
8. **Combination** — Combined multiple methods
9. **BIE** — Binary image encoding

### Architectures Evaluated
- **Baseline:** Linear Regression, Logistic Regression, Random Forest, XGBoost
- **Deep Models:** MLP
- **Vision Models:** CNN, Vision Transformer (ViT)
- **Hybrid:** CNN+MLP, ViT+MLP

### Datasets (24 total)

**Regression (8):** 
- Boston Housing
- California Housing
- Geographical Origin of Music
- Health Insurance
- MIMO* (Ultra-dense indoor massive MIMO CSI)
- Pumadyn32nh
- Student Performance (Portuguese)
- Superconductivity

**Binary Classification (8):** 
- Adult (Census Income)
- Bioresponse
- Credit Approval
- Dengue Chikungunya
- HELOC (Home Equity Line of Credit)
- NOMAO
- QSAR Biodegradability
- Sick (Thyroid disease)

**Multiclass Classification (8):** 
- CMC (Contraceptive Method Choice)
- CNAE-9 (Business text classification)
- Connect-4
- Covertype (Forest cover type)
- DNA (Molecular biology)
- Gas Drift
- Isolet (Speech recognition)
- Student Performance (Portuguese)

**\*Note on MIMO Dataset:**  
The MIMO dataset is exceptionally large and requires separate download and preprocessing:
1. Download from [IEEE Dataport](https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset)
2. Convert using the provided script: `python convert_csi_to_csv_by_antennas.py`

---

## Methodology

### Unified Evaluation Framework
This benchmark establishes a standardized evaluation framework ensuring fair comparison across all methods:
- Consistent data preprocessing and splitting procedures
- Equal computational budget for all method-architecture combinations
- Identical training procedures and convergence criteria

### Hyperparameter Optimization
- **Framework:** Optuna with Tree Parzen Estimator (TPE)
- **Approach:** Rigorous systematic search for each method-architecture combination
- **Fairness:** All models receive equal optimization budget to ensure unbiased comparison

### Evaluation Metrics
- **Regression:** RMSE (Root Mean Squared Error)
- **Classification:** Accuracy, Precision, Recall, F1-score, ROC-AUC

### Statistical Rigor
- Stratified k-fold cross-validation
- Variance analysis across multiple runs
- **Data regime-based analysis:** Performance grouped by sample size (N) and dimensionality (d)
- Scalability and computational trade-off assessment

---

## Repository Structure

```
.
├── 1_data_EDA.ipynb                          # Dataset exploration and analysis
├── 2_Generate_images.ipynb                   # Image generation from tabular data
├── 3_Preprocessing_And_Training_Baseline.ipynb    # Traditional ML baselines
├── 3_Preprocessing_And_Training_Deep.ipynb        # MLP models
├── 3_Preprocessing_And_Training_Deep_Vision.ipynb # CNN and ViT models
├── 3_Preprocessing_And_Training_Hybrid.ipynb      # Hybrid architectures
├── OpenML_datasets.ipynb                    # Dataset loading
├── convert_csi_to_csv_by_antennas.py       # CSI data conversion utility
├── configs/                                 # Configuration files
│   ├── default/                            # Baseline model configs
│   ├── image_generation/                   # Encoding method configs
│   ├── optuna_search/                      # Hyperparameter search spaces
│   └── preprocess/                         # Data preprocessing configs
├── data/                                    # Datasets
│   ├── Binary/
│   ├── Multiclass/
│   └── Regression/
└── logs/                                    # Training results
```

---

## Getting Started

### Installation

```bash
git clone https://github.com/manwestc/TINTOlib-benchmark
cd TINTOlib-benchmark

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Core Dependencies
- TINTOlib (0.0.26) — tabular-to-image transformation
- PyTorch — deep learning framework
- scikit-learn — traditional ML models
- Optuna — hyperparameter optimization
- OpenCV — image processing

### Running the Pipeline

Execute notebooks in order:
1. `1_data_EDA.ipynb` — Explore datasets
2. `2_Generate_images.ipynb` — Create synthetic images using encoding methods
3. `3_Preprocessing_And_Training_*.ipynb` — Train and evaluate models

---

## Citation

```bibtex
@article{LIU2026104088,
  title = {A comprehensive benchmark of spatial encoding methods for tabular data with deep neural networks},
  journal = {Information Fusion},
  volume = {130},
  pages = {104088},
  year = {2026},
  issn = {1566-2535},
  doi = {https://doi.org/10.1016/j.inffus.2025.104088},
  url = {https://www.sciencedirect.com/science/article/pii/S1566253525011509},
  author = {Jiayun Liu and Manuel Castillo-Cara and Raúl García-Castro}
}
```

## License

MIT License — see [LICENSE](LICENSE) file.

## Contact

- Jiayun Liu: jiayun.liu@upm.es
- Manuel Castillo-Cara: manuelcastillo@dia.uned.es
- Raúl García-Castro: r.garcia@upm.es

## Acknowledgments

For more information about TINTOlib and additional resources, visit the official project page: https://oeg-upm.github.io/TINTOlib/

