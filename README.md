# PGM2Net

## Overview

PGM2Net (Physics-Guided Manifold Mamba Network) is a state-of-the-art deep learning model designed for Optics-Guided thermal UAV image super-resolution. This model leverages physics-guided principles and manifold learning techniques combined with the powerful Mamba architecture to enhance the resolution of thermal UAV images.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Architecture](#architecture)
- [Training](#training)
- [Results](#results)
- [Predicted Maps](#predicted-maps)
- [Model Checkpoints](#model-checkpoints)
- [Usage](#usage)

## Requirements

### Hardware Requirements

- CUDA 11.8 or higher
- NVIDIA GPU with at least 8GB VRAM

### Software Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0.0
- For detailed dependencies, refer to `requirements.txt`

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Yuride0404127/PGM2Net.git
cd PGM2Net
```

1. Install dependencies:

```bash
pip install -r requirements.txt
```

## Architecture

### Model Structure

!\[PGM2Net]\(<https://github.com/Yuride0404127/PGM2Net/blob/main/Picture/PGM2Net.png> null)

The PGM2Net architecture integrates:

- Physics-Informed Diffusion Mamba (PID-Mamba)
- **Sparse Manifold Reconstruction (SMR)**
- **Consistency-Gated Refinement (CGR)**

### Training Algorithm

!\[Algorithm1]\(<https://github.com/Yuride0404127/PGM2Net/blob/main/Picture/Algorithm1.png> null)

## Results

### Quantitative Results

!\[Table1]\(<https://github.com/Yuride0404127/PGM2Net/blob/main/Picture/Table1.png> null)

!\[Table2]\(<https://github.com/Yuride0404127/PGM2Net/blob/main/Picture/table2.png> null)

### Qualitative Results

!\[Fig1]\(<https://github.com/Yuride0404127/PGM2Net/blob/main/Picture/Fig1.png> null)

!\[Fig2]\(<https://github.com/Yuride0404127/PGM2Net/blob/main/Picture/fig2.png> null)

## Predicted Maps

The visualization results of the model's predictions can be viewed at the following cloud storage link. These visualizations demonstrate the model's ability to generate high-resolution thermal images, with improved clarity and detail.

## Model Checkpoints

### Baidu Netdisk

- Link: \[BaiduNet disk] ( <https://pan.baidu.com/s/1acQ3UusQfuPWxDoEyFp5WQ?pwd=268s>)
- Password: [268s](https://pan.baidu.com/s/1acQ3UusQfuPWxDoEyFp5WQ?pwd=268s)
- Includes: Model checkpoints and predicted maps

### Google Drive

- Link: \[Google Drive ( <https://drive.google.com/file/d/10dQnu2C34W4T5aQuVlcu3PO3vwkM5-XL/view?usp=drive_link>)
- Includes: Model checkpoints and predicted maps

## Usage

### Inference

```python
from pgm2net import PGM2Net

# Load the model
model = PGM2Net()
model.load_state_dict(torch.load('path/to/checkpoint.pth'))
model.eval()

# Perform super-resolution
low_res_image = torch.load('path/to/low_res_image.pt')
high_res_infrared_image = model(low_res_infrared_image, high_res_optical_image)
```

### Testing

```bash
python test.py --config config.yaml
```

