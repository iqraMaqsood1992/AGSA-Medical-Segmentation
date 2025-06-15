# AGSA: Annotation-Guided Sparse Attention for Medical Image Segmentation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Annotation-Guided Sparse Attention for Accurate Small Structure Segmentation in Medical Images**
> 
> Iqra Maqsood¹, Mohd Halim Mohd Noor²  
> ¹²Universiti Sains Malaysia

## 🎯 Overview

This repository contains the official implementation of **AGSA (Annotation-Guided Sparse Attention)**, a novel attention mechanism that significantly enhances both efficiency and accuracy of medical image segmentation by focusing computational resources on clinically relevant regions, particularly small anatomical structures.


### Installation

```bash
# Clone the repository
git clone https://github.com/iqraMaqsood1992/AGSA-Medical-Segmentation.git
cd AGSA-Medical-Segmentation

# Create conda environment
conda create -n agsa python=3.8
conda activate agsa

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

```bash
# Download Synapse Multi-organ Dataset
# 1. Register at https://www.synapse.org/#!Synapse:syn3193805/wiki/89480


# Organize data structure
mkdir -p data/Synapse
# Extract your data following this structure:
# data/Synapse/
# ├── Training-Training/
# │   ├── img/
# │   └── label/
# └── Training-Testing/
#     ├── img/
#     └── label/
```

### Training and Testing

```bash
# Train AGSA model on Synapse dataset
python scripts/train.py
python scripts/test.py


