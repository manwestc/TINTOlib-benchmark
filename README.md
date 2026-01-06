# Benchmark of Spatial Encoding Methods for Tabular Data with Deep Neural Networks

**Authors:** Jiayun Liu, Manuel Castillo-Cara, Raúl García-Castro  
**Affiliation:** Universidad Politécnica de Madrid  
**Published in:** Information Fusion, Vol. 130, 2026  
**DOI:** https://doi.org/10.1016/j.inffus.2025.104088  
**Project Page:** https://oeg-upm.github.io/TINTOlib/

---

## Overview

While deep neural networks excel at learning from images and text, they often underperform on tabular data compared to traditional machine learning methods. This repository presents a comprehensive benchmark evaluating **spatial encoding methods** — techniques that transform tabular data into synthetic images — to enable the use of vision architectures (CNNs and Vision Transformers) on tabular datasets.

This work evaluates 9 spatial encoding methods across 24 diverse datasets using rigorous hyperparameter optimization, providing empirical guidance on when and how to effectively apply image-based deep learning to tabular data.

## Video

https://github.com/user-attachments/assets/8b167075-2010-4072-a5ff-dea6fc117437

---

## Research Questions and Findings

**Main Questions:**
- How do different spatial encoding methods compare across diverse datasets?
- Do vision architectures outperform traditional deep learning on transformed tabular data?
- Can hybrid models (combining vision and dense layers) improve generalization?

**Key Findings:**
- **REFINED** is the most robust transformation method across tasks and data regimes
- Encoding method choice has stronger impact on performance than architecture selection
- Hybrid models (CNN+MLP, ViT+MLP) reduce variance, especially beneficial for smaller datasets
- Performance is structured by data regimes: optimal methods vary with sample size (N) and dimensionality (d)

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

### Hyperparameter Optimization
- Framework: Optuna with Tree Parzen Estimator
- Equal computational budget across all method-architecture combinations
- Systematic search for fair comparison

### Evaluation Metrics
- **Regression:** RMSE
- **Classification:** Accuracy, Precision, Recall, F1-score, ROC-AUC

### Statistical Rigor
- Stratified k-fold cross-validation
- Variance analysis across runs
- Data regime-based performance grouping

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

