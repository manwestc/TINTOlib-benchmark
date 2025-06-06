# Improving the Use of Deep Neural Networks with Tabular Data by Exploiting Synthetic Images

## Overview
This repository contains the resources and code for the study: **"Improving the Use of Deep Neural Networks with Tabular Data by Exploiting Synthetic Images."** It benchmarks eight tabular-to-image transformation techniques and evaluates their effectiveness in combination with hybrid architectures (CNN+MLP and ViT+MLP) across diverse datasets and machine learning tasks.

## Key Features
- **Comprehensive Benchmark**: Evaluation of eight transformation techniques for tabular data:
  - TINTO
  - REFINED
  - IGTD
  - FeatureWrap
  - SuperTML
  - BarGraph
  - DistanceMatrix
  - Combination
- **Hybrid Architectures**: Analysis of CNN+MLP and ViT+MLP combinations.
- **Diverse Datasets**: Includes regression, binary classification, and multiclass classification tasks.
- **Metrics**: Performance evaluation using RMSE, Accuracy, Precision, Recall, and F1-score.

## Methodology

### Datasets
The experiments span a variety of datasets, including:
- **Regression**: Boston Housing, California Housing, MIMO
- **Binary Classification**: Dengue/Chikungunya, HELOC
- **Multiclass Classification**: Covertype, GAS

To download MIMO Dataset: https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset

And use convert_csi_to_csv_by_antennas.py

## Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/manwestc/TINTOlib-benchmark

