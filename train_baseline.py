import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
from data_pipeline import EuroSATSpatialDataset, get_transforms, EUROSAT_CLASSES, EUROSAT_CLASS_TO_IDX
from models import BaselineCNN

# Configurations
WORKSPACE = "C:/Users/alok1/.gemini/antigravity-ide/scratch/satellite_change_detection"
DATA_DIR = os.path.join(WORKSPACE, "data", "EuroSAT_RGB")
SPLITS_DIR = os.path.join(WORKSPACE, "splits")
CHECKPOINT_DIR = os.path.join(WORKSPACE, "checkpoints")
RESULTS_DIR = os.path.join(WORKSPACE, "results")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    # 1. Transforms for Baseline CNN (natively 64x64 is faster and works well for scratch CNN)
    train_transform = T.Compose([
        T.Resize((64, 64)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 2. Datasets & Loaders
    train_dataset = EuroSATSpatialDataset(
        root_dir=DATA_DIR, 
        split_file=os.path.join(SPLITS_DIR, "eurosat-spatial-train.txt"), 
        transform=train_transform
    )
    val_dataset = EuroSATSpatialDataset(
        root_dir=DATA_DIR, 
        split_file=os.path.join(SPLITS_DIR, "eurosat-spatial-val.txt"), 
        transform=val_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
    
    # 3. Model, Loss, Optimizer
    model = BaselineCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 10
    train_losses = []
    val_losses = []
    
    print("Starting Baseline CNN training...")
    for epoch in range(epochs):
        t0 = time.time()
        
        # Training Phase
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_train_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item() * images.size(0)
                
        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)
        
        epoch_time = time.time() - t0
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Time: {epoch_time:.1f}s")
        
    # Save checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "baseline_cnn.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved Baseline CNN checkpoint to {checkpoint_path}")
    
    # Plot loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", marker='o')
    plt.plot(val_losses, label="Val Loss", marker='x')
    plt.title("Baseline CNN Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True)
    loss_curve_path = os.path.join(RESULTS_DIR, "baseline_loss_curves.png")
    plt.savefig(loss_curve_path, dpi=300)
    plt.close()
    print(f"Saved loss curves to {loss_curve_path}")
    
    # 4. Final Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Compute metrics
    report = classification_report(all_labels, all_preds, target_names=EUROSAT_CLASSES, output_dict=True)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # Save classification report summary
    report_path = os.path.join(RESULTS_DIR, "baseline_classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("=== Baseline CNN Validation Evaluation ===\n")
        f.write(classification_report(all_labels, all_preds, target_names=EUROSAT_CLASSES))
        f.write(f"\nMacro F1-Score: {macro_f1:.4f}\n")
        
    print(f"Saved evaluation report to {report_path}")
    print(f"Baseline CNN Macro F1: {macro_f1:.4f}")

if __name__ == "__main__":
    main()
