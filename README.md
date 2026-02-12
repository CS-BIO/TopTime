# TopTime: Persistent Topological Representations for Enhancing Time-Series Modeling and Interpretation

This repository provides the official implementation of **TopTime**, a topology-guided representation framework for time-series analysis.

TopTime integrates persistent topological descriptors with deep temporal encoders to capture the global geometric structure of temporal dynamics.

---

## Dependencies

The main Python packages required to run the code are:

- Python ≥ 3.9
- PyTorch ≥ 2.6.0
- PyTorch Geometric ≥ 2.6.1
- NumPy ≥ 1.26
- SciPy ≥ 1.13
- pandas ≥ 2.2
- scikit-learn ≥ 1.5
- matplotlib ≥ 3.9
- RDKit ≥ 2025.3

We recommend using a conda environment:

conda create -n toptime python=3.9  
conda activate toptime

---

## Dataset Preparation

### Step 1 — Download UCR Archive

Download the UCR Time-Series Classification Archive:

https://www.cs.ucr.edu/~eamonn/time_series_data/

---

### Step 2 — Organize Dataset

Place the datasets into the following directory:

project_root/  
│── UCR_data/  
│     ├── Dataset_1/  
│     ├── Dataset_2/  
│     └── ...

---

### Step 3 — Create Feature Storage Folder

Create a directory for storing topological descriptors:

project_root/  
│── Topo_feature/

This folder will automatically store computed topological features.

---

## Usage

After preparing the datasets, run:

python main2.py

---

## Workflow Overview

The execution pipeline follows:

1. Load raw time series from `UCR_data/`
2. Takens delay embedding
3. Persistent homology computation
4. Topological descriptor extraction
5. Feature storage in `Topo_feature/`
6. Model training and evaluation

---

## Notes

- Topological features are computed dynamically
- Features are cached for efficiency
- The framework is dataset-agnostic

---

## Contact

For questions regarding the code:

Cong Shen  
Email: cshen@amss.ac.cn
