import os
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
from data_pipeline import EuroSATSpatialDataset, get_transforms, EUROSAT_CLASSES
from models import get_resnet18_model, ResNet18Embedder

# Configurations
WORKSPACE = "C:/Users/alok1/.gemini/antigravity-ide/scratch/satellite_change_detection"
DATA_EUROSAT = os.path.join(WORKSPACE, "data", "EuroSAT_RGB")
SPLITS_DIR = os.path.join(WORKSPACE, "splits")
CHECKPOINT_DIR = os.path.join(WORKSPACE, "checkpoints")
RESULTS_DIR = os.path.join(WORKSPACE, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_cosine_similarity(emb1, emb2):
    # emb1, emb2 shape: (N, D)
    dot_product = np.sum(emb1 * emb2, axis=1)
    norm1 = np.linalg.norm(emb1, axis=1)
    norm2 = np.linalg.norm(emb2, axis=1)
    return dot_product / (norm1 * norm2 + 1e-8)


def extract_embeddings(model, file_paths, transform):
    model.eval()
    embedder = ResNet18Embedder(model).to(device)
    embedder.eval()
    
    embeddings = []
    with torch.no_grad():
        for path in file_paths:
            img = Image.open(path).convert('RGB')
            img_t = transform(img).unsqueeze(0).to(device)
            emb = embedder(img_t)
            embeddings.append(emb.cpu().numpy()[0])
            
    return np.array(embeddings)


def stitch_grid(image_paths, grid_size=(6, 6), thumb_size=(64, 64)):
    cols, rows = grid_size
    grid_img = Image.new('RGB', (cols * thumb_size[0], rows * thumb_size[1]))
    for idx, path in enumerate(image_paths):
        if idx >= cols * rows:
            break
        img = Image.open(path).convert('RGB').resize(thumb_size)
        x = (idx % cols) * thumb_size[0]
        y = (idx // cols) * thumb_size[1]
        grid_img.paste(img, (x, y))
    return grid_img


def main():
    # Load model
    model = get_resnet18_model(num_classes=10, pretrained=False)
    model_path = os.path.join(CHECKPOINT_DIR, "resnet18_fine_tuned.pt")
    if not os.path.exists(model_path):
        # Fallback to frozen model if fine-tuned is not trained yet
        model_path = os.path.join(CHECKPOINT_DIR, "resnet18_frozen.pt")
    
    if not os.path.exists(model_path):
        print(f"Error: No model checkpoint found at {model_path}. Train the model first.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded feature extractor from {model_path}")
    
    # Load test split
    test_split_file = os.path.join(SPLITS_DIR, "eurosat-spatial-test.txt")
    with open(test_split_file, 'r') as f:
        filenames = [line.strip() for line in f if line.strip()]
        
    # Group test files by class
    files_by_class = {cls: [] for cls in EUROSAT_CLASSES}
    for fn in filenames:
        cls = fn.split('_')[0]
        full_path = os.path.join(DATA_EUROSAT, cls, fn)
        files_by_class[cls].append(full_path)
        
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Define 50 regions, each a 6x6 grid (36 tiles)
    num_regions = 50
    grid_size = (6, 6)
    tiles_per_region = grid_size[0] * grid_size[1]
    
    regions_t1 = []
    regions_t2 = []
    change_labels = [] # Ground truth change: 0 = No Change, 1 = Change
    
    print("Simulating temporal change datasets...")
    for r in range(num_regions):
        t1_paths = []
        t2_paths = []
        labels = []
        
        for i in range(tiles_per_region):
            # Pick a random class for T1
            cls1 = random.choice(EUROSAT_CLASSES)
            img1 = random.choice(files_by_class[cls1])
            t1_paths.append(img1)
            
            # Simulate 20% change probability
            is_change = random.random() < 0.20
            if is_change:
                # Change: pick a different class for T2
                cls2 = random.choice([c for c in EUROSAT_CLASSES if c != cls1])
                img2 = random.choice(files_by_class[cls2])
                labels.append(1)
            else:
                # No change: pick a different image of the same class
                img2 = random.choice(files_by_class[cls1])
                labels.append(0)
                
            t2_paths.append(img2)
            
        regions_t1.append(t1_paths)
        regions_t2.append(t2_paths)
        change_labels.extend(labels)
        
    # Extract embeddings
    _, val_transform, _ = get_transforms()
    
    print("Extracting embeddings for T1 and T2...")
    flat_t1_paths = [path for region in regions_t1 for path in region]
    flat_t2_paths = [path for region in regions_t2 for path in region]
    
    emb_t1 = extract_embeddings(model, flat_t1_paths, val_transform)
    emb_t2 = extract_embeddings(model, flat_t2_paths, val_transform)
    
    # Compute Cosine Similarities
    similarities = compute_cosine_similarity(emb_t1, emb_t2)
    change_labels = np.array(change_labels)
    
    # Evaluate Change Detector (distance = 1 - cosine_similarity)
    # Changed tile pairs should have LOWER similarity, meaning HIGHER distance
    distances = 1.0 - similarities
    
    fpr, tpr, thresholds = roc_curve(change_labels, distances)
    roc_auc = auc(fpr, tpr)
    print(f"Embedding Change Detection AUC: {roc_auc:.4f}")
    
    # Determine operating thresholds
    # 1. Balanced: maximizes F1-score on change detection
    precision, recall, pr_thresholds = precision_recall_curve(change_labels, distances)
    # Avoid division by zero
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    # Note: distance threshold
    balanced_dist_thresh = pr_thresholds[min(best_idx, len(pr_thresholds)-1)]
    balanced_sim_thresh = 1.0 - balanced_dist_thresh
    
    # 2. High Recall: TPR >= 95%
    hr_idx = np.where(tpr >= 0.95)[0][0]
    hr_dist_thresh = thresholds[hr_idx]
    hr_sim_thresh = 1.0 - hr_dist_thresh
    
    # 3. High Precision: Precision >= 95%
    # find first index where precision is >= 0.95
    hp_indices = np.where(precision >= 0.95)[0]
    if len(hp_indices) > 0:
        hp_dist_thresh = pr_thresholds[min(hp_indices[0], len(pr_thresholds)-1)]
    else:
        hp_dist_thresh = pr_thresholds[np.argmax(precision)]
    hp_sim_thresh = 1.0 - hp_dist_thresh
    
    # Save threshold reports
    thresh_report = (
        "=== Cosine Similarity Change Detection Operating Points ===\n\n"
        f"AUC-ROC: {roc_auc:.4f}\n\n"
        f"Operating Point 1: Balanced (Max F1-Score)\n"
        f"  - Distance Threshold: {balanced_dist_thresh:.4f}\n"
        f"  - Similarity Threshold: {balanced_sim_thresh:.4f}\n"
        f"  - Expected Precision: {precision[best_idx]:.4f}\n"
        f"  - Expected Recall: {recall[best_idx]:.4f}\n"
        f"  - Expected F1-Score: {f1_scores[best_idx]:.4f}\n\n"
        f"Operating Point 2: High Recall (TPR >= 95%)\n"
        f"  - Distance Threshold: {hr_dist_thresh:.4f}\n"
        f"  - Similarity Threshold: {hr_sim_thresh:.4f}\n"
        f"  - True Positive Rate: {tpr[hr_idx]:.4f}\n"
        f"  - False Positive Rate: {fpr[hr_idx]:.4f}\n\n"
        f"Operating Point 3: High Precision (Precision >= 95%)\n"
        f"  - Distance Threshold: {hp_dist_thresh:.4f}\n"
        f"  - Similarity Threshold: {hp_sim_thresh:.4f}\n"
    )
    print("\n" + thresh_report)
    with open(os.path.join(RESULTS_DIR, "change_detection_thresholds.txt"), 'w') as f:
        f.write(thresh_report)
        
    # Plot ROC Curve
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # Plot our chosen Balanced threshold point
    # Find matching FPR/TPR for balanced threshold
    bal_idx = np.argmin(np.abs(thresholds - balanced_dist_thresh))
    plt.plot(fpr[bal_idx], tpr[bal_idx], 'go', markersize=10, label=f'Balanced Thresh ({balanced_sim_thresh:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "change_roc_curve.png"), dpi=300)
    plt.close()
    
    # ----------------------------------------------------
    # Generate Heatmaps for 5 Sample Regions
    # ----------------------------------------------------
    print("Generating change heatmaps for 5 sample regions...")
    for r in range(5):
        t1_p = regions_t1[r]
        t2_p = regions_t2[r]
        
        # Stitch Before/After grid images
        before_grid_img = stitch_grid(t1_p, grid_size=grid_size)
        after_grid_img = stitch_grid(t2_p, grid_size=grid_size)
        
        # Extract similarities for this region
        start_idx = r * tiles_per_region
        end_idx = start_idx + tiles_per_region
        region_sims = similarities[start_idx:end_idx].reshape(grid_size)
        region_labels = change_labels[start_idx:end_idx].reshape(grid_size)
        
        # Calculate predicted changes at Balanced threshold
        region_preds = (1.0 - region_sims) > balanced_dist_thresh
        
        # Plot side-by-side
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # 1. Before Grid
        axes[0, 0].imshow(before_grid_img)
        axes[0, 0].set_title(f"Region {r+1} Before (T1)", fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # 2. After Grid
        axes[0, 1].imshow(after_grid_img)
        axes[0, 1].set_title(f"Region {r+1} After (T2)", fontsize=12, fontweight='bold')
        axes[0, 1].axis('off')
        
        # 3. Similarity Heatmap
        sns.heatmap(region_sims, annot=True, fmt=".2f", cmap="RdYlGn", ax=axes[1, 0], cbar=True, vmin=0.5, vmax=1.0)
        axes[1, 0].set_title("Cosine Similarity Heatmap", fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel("Grid X")
        axes[1, 0].set_ylabel("Grid Y")
        
        # 4. Binary Change Flagged (vs Ground Truth)
        # We can draw True Positives, False Positives, True Negatives, False Negatives
        # 0: TN (No change, correctly predicted no change) - Green
        # 1: FN (Change, predicted no change) - Yellow
        # 2: FP (No change, predicted change) - Orange
        # 3: TP (Change, correctly predicted change) - Red
        eval_grid = np.zeros(grid_size)
        for x in range(grid_size[0]):
            for y in range(grid_size[1]):
                true_c = region_labels[y, x]
                pred_c = region_preds[y, x]
                if true_c == 1 and pred_c == 1:
                    eval_grid[y, x] = 3 # TP (Red)
                elif true_c == 0 and pred_c == 1:
                    eval_grid[y, x] = 2 # FP (Orange)
                elif true_c == 1 and pred_c == 0:
                    eval_grid[y, x] = 1 # FN (Yellow)
                else:
                    eval_grid[y, x] = 0 # TN (Green)
                    
        cmap_eval = sns.color_palette(["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"])
        sns.heatmap(eval_grid, cmap=cmap_eval, annot=True, cbar=False, ax=axes[1, 1])
        axes[1, 1].set_title("Change Prediction (0:TN, 1:FN, 2:FP, 3:TP)", fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel("Grid X")
        axes[1, 1].set_ylabel("Grid Y")
        
        plt.suptitle(f"Region {r+1} Change Analysis (Simulated)", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"region_{r+1}_change_heatmap.png"), dpi=150)
        plt.close()
        print(f"Saved region {r+1} analysis to results/region_{r+1}_change_heatmap.png")

if __name__ == "__main__":
    main()
