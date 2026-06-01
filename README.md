# CropGuard — AI Crop Disease Detection

CropGuard is a deep learning web application that identifies diseases in crop leaves from a photo. Upload a leaf image, and the system returns the disease name, confidence score, severity level, treatment steps, and prevention tips.

---

## How It Works

```
User uploads image
        |
        v
Flask receives the image (app.py)
        |
        v
PIL decodes + resizes to 160x160
        |
        v
EfficientNet-B0 runs inference (predict.py)
        |
        v
Softmax → top-5 class probabilities
        |
        v
Disease database lookup (DISEASE_INFO dict)
        |
        v
JSON response → rendered in browser (index.html)
```

##Dataset

You can download the dataset from [Top Agriculture Crop Disease India](https://www.kaggle.com/)

---

## Setup

### Requirements

- Python 3.10+
- NVIDIA GPU with CUDA (recommended) — CPU also works but is slower
- ~4 GB free disk space (PyTorch with CUDA is ~2.5 GB)

### Install Dependencies

```powershell
pip install -r requirements.txt
```

### Install PyTorch with GPU support (NVIDIA GPU only)

```powershell
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

---

## Usage

### Step 1 — Train the Model

```powershell
cd e:\cropguard
python src\train.py
```

The best model is saved automatically to `models/cropguard_best.pth`.

### Step 2 — Start the Web App

```powershell
cd e:\cropguard
python src\app.py
```

Open your browser at **http://localhost:5000**

### Step 3 — Detect Disease

1. Drag and drop a crop leaf image onto the page (or click "Choose Image")
2. The model analyzes the image
3. Results show:
   - Disease name and crop type
   - Confidence percentage
   - Severity level
   - Description and symptoms
   - Treatment recommendation
   - Prevention tips
   - Top-5 alternative predictions

---

## Results

| Metric | Value |
|---|---|
| Model | EfficientNet-B0 |
| Input size | 160 × 160 px |
| Parameters | ~5.3M |
| Best validation accuracy | ~94.5% |
| Inference time (CPU) | ~200ms |
| Inference time (GPU) | ~20ms |

---

## Supported Diseases

| # | Class | Crop | Type |
|---|---|---|---|
| 1 | Common Rust | Corn | Disease |
| 2 | Gray Leaf Spot | Corn | Disease |
| 3 | Northern Leaf Blight | Corn | Disease |
| 4 | Healthy | Corn | Healthy |
| 5 | Early Blight | Potato | Disease |
| 6 | Late Blight | Potato | Disease |
| 7 | Healthy | Potato | Healthy |
| 8 | Brown Spot | Rice | Disease |
| 9 | Leaf Blast | Rice | Disease |
| 10 | Neck Blast | Rice | Disease |
| 11 | Healthy | Rice | Healthy |
| 12 | Bacterial Blight | Sugarcane | Disease |
| 13 | Red Rot | Sugarcane | Disease |
| 14 | Healthy | Sugarcane | Healthy |
| 15 | Brown Rust | Wheat | Disease |
| 16 | Yellow Rust | Wheat | Disease |
| 17 | Healthy | Wheat | Healthy |
