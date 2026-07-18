import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from data_pipeline import EuroSATSpatialDataset, UCMercedDataset, get_transforms, EUROSAT_CLASSES, EUROSAT_CLASS_TO_IDX
from models import get_resnet18_model

# Configurations
WORKSPACE = "C:/Users/alok1/.gemini/antigravity-ide/scratch/satellite_change_detection"
DATA_EUROSAT = os.path.join(WORKSPACE, "data", "EuroSAT_RGB")
DATA_UCMERCED = os.path.join(WORKSPACE, "data", "UCMerced_LandUse", "Images")
SPLITS_DIR = os.path.join(WORKSPACE, "splits")
CHECKPOINT_DIR = os.path.join(WORKSPACE, "checkpoints")
RESULTS_DIR = os.path.join(WORKSPACE, "results")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def evaluate_model(model, dataloader, dataset_name):
    """
    Evaluates a model and returns predictions, true labels, macro F1, and accuracy.
    Handles both EuroSAT and UCMerced (ground-truth is mapped in UCMercedDataset).
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Compute metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return all_preds, all_labels, accuracy, macro_f1


def save_confusion_matrix(y_true, y_pred, classes, save_path, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def main():
    # 1. Transforms & Dataloaders
    train_transform, val_transform, _ = get_transforms()
    
    train_dataset = EuroSATSpatialDataset(
        root_dir=DATA_EUROSAT, 
        split_file=os.path.join(SPLITS_DIR, "eurosat-spatial-train.txt"), 
        transform=train_transform
    )
    val_dataset = EuroSATSpatialDataset(
        root_dir=DATA_EUROSAT, 
        split_file=os.path.join(SPLITS_DIR, "eurosat-spatial-val.txt"), 
        transform=val_transform
    )
    # UC Merced is our holdout evaluation set
    ucm_dataset = UCMercedDataset(
        root_dir=DATA_UCMERCED, 
        transform=val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
    ucm_loader = DataLoader(ucm_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    print(f"EuroSAT Spatial Train: {len(train_dataset)} images")
    print(f"EuroSAT Spatial Val: {len(val_dataset)} images")
    print(f"UC Merced Holdout: {len(ucm_dataset)} images")
    
    # Initialize Pretrained ResNet-18
    model = get_resnet18_model(num_classes=10, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # ----------------------------------------------------
    # PHASE 1: Train Classifier Head Only (3 Epochs)
    # ----------------------------------------------------
    print("\n--- PHASE 1: Training Classifier Head Only (3 Epochs) ---")
    # Freeze backbone
    for name, param in model.named_parameters():
        if "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    optimizer_head = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    phase1_epochs = 3
    for epoch in range(phase1_epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer_head.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_head.step()
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        print(f"Phase 1 - Epoch {epoch+1}/{phase1_epochs} | Loss: {epoch_loss:.4f} | Time: {time.time()-t0:.1f}s")
        
    # Save Phase 1 Checkpoint
    frozen_checkpoint = os.path.join(CHECKPOINT_DIR, "resnet18_frozen.pt")
    torch.save(model.state_dict(), frozen_checkpoint)
    print(f"Saved Frozen Backbone checkpoint to {frozen_checkpoint}")
    
    # Evaluate Phase 1 Model
    print("Evaluating Frozen Backbone Model...")
    _, _, val_acc_f, val_f1_f = evaluate_model(model, val_loader, "EuroSAT Val")
    _, _, ucm_acc_f, ucm_f1_f = evaluate_model(model, ucm_loader, "UC Merced Holdout")
    print(f"Frozen Model - EuroSAT Val Acc: {val_acc_f:.4f}, Macro-F1: {val_f1_f:.4f}")
    print(f"Frozen Model - UC Merced Acc: {ucm_acc_f:.4f}, Macro-F1: {ucm_f1_f:.4f}")
    
    # ----------------------------------------------------
    # PHASE 2: Fine-Tune Backbone Blocks 3 & 4 (5 Epochs)
    # ----------------------------------------------------
    print("\n--- PHASE 2: Fine-Tuning Backbone layer3/4 (5 Epochs) ---")
    # Unfreeze layer3, layer4, and fc
    for name, param in model.named_parameters():
        if "layer3" in name or "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    # Reduce learning rate by 10x
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    
    phase2_epochs = 5
    for epoch in range(phase2_epochs):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer_ft.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_ft.step()
            running_loss += loss.item() * images.size(0)
            
        epoch_loss = running_loss / len(train_dataset)
        print(f"Phase 2 - Epoch {epoch+1}/{phase2_epochs} | Loss: {epoch_loss:.4f} | Time: {time.time()-t0:.1f}s")
        
    # Save Phase 2 Checkpoint (Final Fine-Tuned Model)
    ft_checkpoint = os.path.join(CHECKPOINT_DIR, "resnet18_fine_tuned.pt")
    torch.save(model.state_dict(), ft_checkpoint)
    print(f"Saved Fine-Tuned Model checkpoint to {ft_checkpoint}")
    
    # Evaluate Phase 2 Model
    print("Evaluating Fine-Tuned (Unfrozen) Model...")
    val_preds_u, val_labels_u, val_acc_u, val_f1_u = evaluate_model(model, val_loader, "EuroSAT Val")
    ucm_preds_u, ucm_labels_u, ucm_acc_u, ucm_f1_u = evaluate_model(model, ucm_loader, "UC Merced Holdout")
    print(f"Unfrozen Model - EuroSAT Val Acc: {val_acc_u:.4f}, Macro-F1: {val_f1_u:.4f}")
    print(f"Unfrozen Model - UC Merced Acc: {ucm_acc_u:.4f}, Macro-F1: {ucm_f1_u:.4f}")
    
    # ----------------------------------------------------
    # Generate Deliverables: Tables and Confusion Matrices
    # ----------------------------------------------------
    # 1. Ablation comparison table
    ablation_summary = (
        "=== Ablation Study: Frozen vs Unfrozen Backbone ===\n\n"
        f"{'Model/Split':<25}{'EuroSAT Val Acc':<18}{'EuroSAT Val F1':<18}{'UC Merced Acc':<15}{'UC Merced F1':<15}\n"
        f"{'-'*91}\n"
        f"{'Frozen Backbone (Ph1)':<25}{val_acc_f:<18.4f}{val_f1_f:<18.4f}{ucm_acc_f:<15.4f}{ucm_f1_f:<15.4f}\n"
        f"{'Fine-tuned (Ph2)':<25}{val_acc_u:<18.4f}{val_f1_u:<18.4f}{ucm_acc_u:<15.4f}{ucm_f1_u:<15.4f}\n"
    )
    print("\n" + ablation_summary)
    
    with open(os.path.join(RESULTS_DIR, "transfer_ablation_study.txt"), "w") as f:
        f.write(ablation_summary)
        
    # 2. Detailed per-class report for fine-tuned model
    with open(os.path.join(RESULTS_DIR, "transfer_classification_report.txt"), "w") as f:
        f.write("=== Fine-tuned ResNet-18 EuroSAT Validation Report ===\n")
        f.write(classification_report(val_labels_u, val_preds_u, target_names=EUROSAT_CLASSES))
        f.write("\n=== Fine-tuned ResNet-18 UC Merced Holdout Report (Mapped Labels) ===\n")
        f.write(classification_report(ucm_labels_u, ucm_preds_u, target_names=EUROSAT_CLASSES))
        
    # 3. Save confusion matrices
    save_confusion_matrix(
        val_labels_u, val_preds_u, EUROSAT_CLASSES, 
        os.path.join(RESULTS_DIR, "transfer_eurosat_cm.png"), 
        "EuroSAT Validation Confusion Matrix (Fine-tuned ResNet-18)"
    )
    save_confusion_matrix(
        ucm_labels_u, ucm_preds_u, EUROSAT_CLASSES, 
        os.path.join(RESULTS_DIR, "transfer_uc_merced_cm.png"), 
        "UC Merced Holdout Confusion Matrix (Fine-tuned ResNet-18)"
    )

if __name__ == "__main__":
    main()
