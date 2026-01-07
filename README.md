# 🧠 TINTOlib Benchmark: Spatial Encoding for Tabular Data with Deep Learning

[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-View%20Documentation-brightgreen?logo=github)](https://oeg-upm.github.io/TINTOlib/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/oeg-upm/TINTOlib-Documentation/blob/main/LICENSE)
[![Python Version](https://shields.io/badge/python-3.11+-blue)](https://pypi.python.org/pypi/)
[![Documentation Status](https://readthedocs.org/projects/tintolib/badge/?version=latest)](https://tintolib.readthedocs.io/en/latest/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/manwestc/TINTOlib-benchmark)
[![PyPI Downloads](https://static.pepy.tech/badge/tintolib)](https://pepy.tech/projects/tintolib)

---

**👥 Authors:** Jiayun Liu, Manuel Castillo-Cara, Raúl García-Castro  
**🏛️ Affiliation:** Universidad Politécnica de Madrid  
**📘 Published in:** *Information Fusion*, Vol. 130, 2026 (Open Access)  
**🔗 DOI:** [10.1016/j.inffus.2025.104088](https://doi.org/10.1016/j.inffus.2025.104088)  
**📂 Project Page:** [TINTOlib Lubrary](https://github.com/oeg-upm/TINTOlib)

---

<div>
    <p align = "center">
    <img src="https://raw.githubusercontent.com/DCY1117/TEMP-Images/refs/heads/main/TINTOlib-images/logo.svg" alt="TINTO Logo" width="150">
    </p>
</div>


---



## 🔍 Overview

Despite the success of deep learning in vision and language, tabular data remains a challenge where classical models often outperform DNNs. A promising alternative is to transform tabular data into synthetic images to leverage vision architectures like CNNs and ViTs.

This repository presents the **first large-scale benchmark of 9 spatial encoding methods** across **24 regression and classification datasets**, evaluated under a **unified framework** with rigorous optimization and statistical testing.


### 🎬 TINTOlib — Overview Video (English)



---

## 🎯 Research Questions and Key Findings

**Main Questions:**
- How do different spatial encoding methods compare across diverse datasets and tasks?
- Do vision architectures (CNN, ViT) outperform traditional deep learning on transformed tabular data?
- What role do hybrid models (combining vision and dense layers) play in performance and variance reduction?
- How do sample size (N) and dimensionality (d) affect transformation method effectiveness?

**Key Findings:**
- **REFINED emerges as the most robust transformation** across tasks and datasets.
- **Transformation method choice exerts stronger influence** on performance than the chosen vision architecture.
- **Performance landscape is structured by data regimes** (sample size and dimensionality).
- **Hybrid models (CNN+MLP, ViT+MLP) consistently reduce predictive variance**.
- **Transforming tabular data into synthetic images is a powerful, data-dependent strategy**.


---

## 🧪 Main Contributions

1. **First large-scale standardized benchmark** of spatial encoding methods on tabular data.
2. **Comprehensive evaluation** of 9 encoding techniques over 24 regression and classification datasets.
3. **Unified framework** with fair and rigorous hyperparameter optimization.
4. **Insights into data regimes**, scalability, and architecture-performance interplay.
5. **Clear guidelines** for applying tabular-to-image transformations.

---

## 🧬 Methods and Architectures

### Spatial Encoding Methods
- TINTO
- REFINED
- IGTD
- FeatureWrap
- SuperTML
- BarGraph
- DistanceMatrix
- Combination
- BIE

### Models Benchmarked
- Traditional: Linear Regression, Logistic Regression, Random Forest, XGBoost
- Deep Learning: MLP, CNN, ViT
- Hybrid: CNN+MLP, ViT+MLP

- **Synthetic images using Hybrid Neural Network with ViT (HyViT)**  
  ![Tabular-to-Image HyNNViT](imgs/HybridViT.png)

---

## 📊 Datasets

**Regression (8):**  
Boston Housing, California Housing, Geographical Origin of Music, Health Insurance, MIMO*, Pumadyn32nh, Student Performance (Portuguese), Superconductivity

**Binary Classification (8):**  
Adult (Census Income), Bioresponse, Credit Approval, Dengue Chikungunya, HELOC, NOMAO, QSAR Biodegradability, Sick (Thyroid disease)

**Multiclass Classification (8):**  
CMC, CNAE-9, Connect-4, Covertype, DNA, Gas Drift, Isolet, Student Performance

**\*Note on MIMO Dataset:**  
Requires separate download and preprocessing:
1. Download from [IEEE Dataport](https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset)
2. Convert using `python convert_csi_to_csv_by_antennas.py`

---

## 🧰 Methodology and Framework

### Evaluation Setup
- Standard preprocessing and splitting across all datasets
- Equal optimization budget per model
- Fixed training protocols and early stopping

### Hyperparameter Optimization
- Optuna (TPE) with cross-validation
- Same number of trials per method

### Metrics
- Regression: RMSE
- Classification: Accuracy, Precision, Recall, F1, ROC-AUC

### Statistical Analysis
- Friedman test with Nemenyi post-hoc
- Variance analysis over runs
- Data regime stratification

---

## 🗂️ Repository Structure

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

## 💬 More information

- For more detailed information, refer to the **[TINTOlib ReadTheDocs](https://tintolib.readthedocs.io/en/latest/)**.  
- GitHub repository: **[TINTOlib Repository](https://github.com/oeg-upm/TINTOlib)**.
- PyPI: **[PyPI](https://pypi.org/project/TINTOlib/)**.

---

## 🚀 Getting Started

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
- `tintolib`
- `torch`
- `scikit-learn`
- `optuna`
- `opencv-python`

---

## ▶️ Execution Pipeline
1. `1_data_EDA.ipynb` — Dataset exploration
2. `2_Generate_images.ipynb` — Generate synthetic images
3. `3_Preprocessing_And_Training_Baseline.ipynb` — Traditional ML models
4. `3_Preprocessing_And_Training_Deep.ipynb` — MLPs
5. `3_Preprocessing_And_Training_Deep_Vision.ipynb` — CNNs / ViTs
6. `3_Preprocessing_And_Training_Hybrid.ipynb` — HyNNs

---

## 📄 Citation

If you use TINTOlib in your research, please cite our papers: 

- [SoftwareX journal](https://doi.org/10.1016/j.softx.2025.102444):

```bibtex
  @article{LIU2025102444,
    title = {TINTOlib: A Python library for transforming tabular data into synthetic images for deep neural networks},
    journal = {SoftwareX},
    volume = {32},
    pages = {102444},
    year = {2025},
    issn = {2352-7110},
    doi = {https://doi.org/10.1016/j.softx.2025.102444},
    author = {Jiayun Liu and David González-Fernández and Manuel Castillo-Cara and Raúl García-Castro}
  }
```


- [Information Fusion journal](https://doi.org/10.1016/j.inffus.2025.104088):

```
  @article{LIU2026104088,
    title = {A comprehensive benchmark of spatial encoding methods for tabular data with deep neural networks},
    journal = {Information Fusion},
    volume = {130},
    pages = {104088},
    year = {2026},
    issn = {1566-2535},
    doi = {https://doi.org/10.1016/j.inffus.2025.104088},
    author = {Jiayun Liu and Manuel Castillo-Cara and Raúl García-Castro}
  }
```

---


## 🛡️ License

TINTOlib is available under the **[Apache License 2.0](https://github.com/manwestc/TINTOlib-benchmark/blob/main/LICENSE)**.

---

## 👥 Authors
- **[Jiayun Liu](https://github.com/DCY1117)**
- **[Manuel Castillo-Cara](https://github.com/manwestc)**
- **[Raúl García-Castro](https://github.com/rgcmme)**

---

## 🏛️ Contributors

<div>
<p align = "center">
<kbd><img src="https://raw.githubusercontent.com/DCY1117/TEMP-Images/refs/heads/main/TINTOlib-images/logo-oeg.png" alt="Ontology Engineering Group" width="150"></kbd> <kbd><img src="https://raw.githubusercontent.com/DCY1117/TEMP-Images/refs/heads/main/TINTOlib-images/logo-upm.png" alt="Universidad Politécnica de Madrid" width="150"></kbd> <kbd><img src="https://raw.githubusercontent.com/DCY1117/TEMP-Images/refs/heads/main/TINTOlib-images/logo-uned-.jpg" alt="Universidad Nacional de Educación a Distancia" width="231"></kbd> 
</p>
</div>
