# GeoEye: Satellite Image Land-Use Classifier & Temporal Change Detector

This repository contains a complete computer vision system to classify land-use types from satellite imagery and detect temporal changes between two time periods using deep learning embeddings and cosine similarity.

### 🚀 Interactive Geo-Dashboard
When running locally, you can access the interactive dashboard here:
👉 **[http://localhost:8501](http://localhost:8501)**

## Project Structure

```
satellite_change_detection/
├── requirements.txt         # Package dependencies
├── download_data.py        # Dataset downloader (EuroSAT, UC Merced, Spatial Splits)
├── data_pipeline.py        # PyTorch datasets, dataloaders, and transforms
├── run_pipeline.py         # Runs pipeline visualizations (distributions, class samples)
├── models.py               # Baseline CNN and ResNet-18 model architectures
├── train_baseline.py       # Trains the scratch baseline CNN on EuroSAT
├── train_transfer.py       # Fine-tunes ResNet-18 and runs UC Merced ablation
├── change_detector.py      # Embedding-based cosine similarity change detector & ROC evaluation
├── run_experiments.py      # Spatial Leakage, Class Imbalance, and t-SNE visualizations
├── app.py                  # Interactive Streamlit geo-dashboard (includes GradCAM & toggles)
├── generate_report.py      # ReportLab script to compile the final evaluation PDF
├── README.md               # Project documentation
├── data/                   # Dataset directories (downloaded automatically)
│   ├── EuroSAT_RGB/
│   └── UCMerced_LandUse/
├── splits/                 # Predefined spatial split files (downloaded automatically)
├── checkpoints/            # Saved model weights (.pt files)
└── results/                # Metrics, reports, and visualization plots
```

## Setup & Installation

The project uses the fast Python package manager `uv` for setup.

1. **Clone or navigate to the workspace directory**:
   ```bash
   cd satellite_change_detection
   ```

2. **Create a virtual environment**:
   ```bash
   uv venv
   ```

3. **Install dependencies**:
   ```bash
   uv pip install -r requirements.txt
   ```

---

## Execution Workflow

Run the following scripts in order to execute the full data processing, training, evaluation, and visualization workflow:

### 1. Data Acquisition
Downloads and extracts EuroSAT RGB, UC Merced Land Use datasets, and the spatial train/val/test splits:
```bash
.venv\Scripts\python download_data.py
```

### 2. Data Pipeline Visualization
Generates the class distribution plot and class sample grids:
```bash
.venv\Scripts\python run_pipeline.py
```

### 3. Train Baseline CNN
Trains a 3-layer scratch CNN on EuroSAT (64x64 inputs) as a performance floor:
```bash
.venv\Scripts\python train_baseline.py
```

### 4. Train Transfer Learning (ResNet-18)
Fine-tunes a pretrained ResNet-18 model using a two-phase strategy on EuroSAT (224x224 inputs), evaluates on EuroSAT validation and the UC Merced holdout (applying the semantic class mapping), and outputs classification reports and confusion matrices:
```bash
.venv\Scripts\python train_transfer.py
```

### 5. Change Detector
Extracts 512D embeddings, simulates a temporal time series of 50 regions, computes cosine similarities, plots the ROC curve, selects operating points, and outputs change heatmaps:
```bash
.venv\Scripts\python change_detector.py
```

### 6. Run Experiments
Executes three core experiments:
- **Spatial Leakage**: Quantifies performance gap (random vs. block split).
- **Class Imbalance**: Downsamples minority classes and evaluates Weighted Cross Entropy Loss mitigation.
- **Embedding Visualisation**: Projects embeddings to 2D using t-SNE (scratch CNN vs. fine-tuned ResNet-18) and plots comparison.
```bash
.venv\Scripts\python run_experiments.py
```

### 7. Compile Evaluation Report
Compiles all evaluation results, experiment logs, tables, and plots into a professional PDF report:
```bash
.venv\Scripts\python generate_report.py
```

### 8. Launch Streamlit Geo-Dashboard
Launches the interactive local dashboard to classify uploaded tile pairs, compute cosine similarities, overlay GradCAM heatmaps, and toggle thresholds dynamically:
```bash
.venv\Scripts\streamlit run app.py
```

---

## Methodologies

### 1. Spatial Block Split
To prevent **spatial leakage** (artificial inflation of accuracy due to spatial autocorrelation in neighboring satellite tiles), the training splits are geographically partitioned (e.g. by longitude bounds) ensuring the model generalizes to unseen geographical locations.

### 2. Temporal Change Detection
Instead of pixel-wise subtraction, we leverage the fine-tuned ResNet-18 as a **feature extractor** (stripping the final classification layer) to retrieve 512-dimensional embeddings. Cosine similarity between T1 (Before) and T2 (After) embeddings quantifies change, which is then thresholded based on our ROC analysis.

### 3. GradCAM
Uses PyTorch hooks to capture feature maps and gradients from `model.layer4`. The weighted combination of feature activations shows which spatial regions drove the land-use classification.
