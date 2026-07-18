import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# Class Mapping from UC Merced (21 classes) to EuroSAT (10 classes)
UCM_TO_EUROSAT = {
    'agricultural': 'AnnualCrop',
    'airplane': 'Industrial',
    'baseballdiamond': 'Residential',
    'beach': 'HerbaceousVegetation',
    'buildings': 'Industrial',
    'chaparral': 'HerbaceousVegetation',
    'denseresidential': 'Residential',
    'forest': 'Forest',
    'freeway': 'Highway',
    'golfcourse': 'Pasture',
    'harbor': 'Industrial',
    'intersection': 'Highway',
    'mediumresidential': 'Residential',
    'mobilehomepark': 'Residential',
    'overpass': 'Highway',
    'parkinglot': 'Industrial',
    'river': 'River',
    'runway': 'Highway',
    'sparseresidential': 'Residential',
    'storagetanks': 'Industrial',
    'tenniscourt': 'Residential'
}

# Standard EuroSAT classes sorted alphabetically
EUROSAT_CLASSES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 
    'River', 'SeaLake'
]
EUROSAT_CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(EUROSAT_CLASSES)}

class EuroSATSpatialDataset(Dataset):
    def __init__(self, root_dir, split_file, transform=None):
        """
        root_dir: path to 'data/EuroSAT_RGB'
        split_file: path to the spatial split text file (e.g. splits/eurosat-spatial-train.txt)
        """
        self.root_dir = root_dir
        self.transform = transform
        
        with open(split_file, 'r') as f:
            self.filenames = [line.strip() for line in f if line.strip()]
            
        self.classes = EUROSAT_CLASSES
        self.class_to_idx = EUROSAT_CLASS_TO_IDX

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        class_name = filename.split('_')[0]
        img_path = os.path.join(self.root_dir, class_name, filename)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # Fallback to an empty/black image if read fails
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (64, 64), color='black')
            
        label = self.class_to_idx[class_name]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class UCMercedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: path to 'data/UCMerced_LandUse/Images'
        """
        self.root_dir = root_dir
        self.transform = transform
        
        self.classes = sorted(os.listdir(root_dir))
        self.samples = []
        
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(('.tif', '.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cls_dir, fname), cls))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, ucm_cls_name = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (256, 256), color='black')
            
        # Map class to EuroSAT label
        eurosat_cls_name = UCM_TO_EUROSAT.get(ucm_cls_name, 'AnnualCrop') # Default fallback
        label = EUROSAT_CLASS_TO_IDX[eurosat_cls_name]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, ucm_cls_name, eurosat_cls_name


def get_transforms():
    # Pretrained ImageNet models expect 224x224 inputs, normalized
    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Raw transforms for visualization
    viz_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    
    return train_transform, val_transform, viz_transform


def plot_class_distribution(root_dir, splits_dir, save_path):
    """
    Plots and saves class distribution across spatial train/val/test splits.
    """
    data = []
    for split in ['train', 'val', 'test']:
        split_file = os.path.join(splits_dir, f"eurosat-spatial-{split}.txt")
        with open(split_file, 'r') as f:
            filenames = [line.strip() for line in f if line.strip()]
        for fn in filenames:
            cls = fn.split('_')[0]
            data.append({'Class': cls, 'Split': split.capitalize()})
            
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='Class', hue='Split', order=EUROSAT_CLASSES, palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title('EuroSAT Spatial Split Class Distribution')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved class distribution plot to {save_path}")


def visualize_class_samples(root_dir, save_dir, n_samples=5):
    """
    Visualizes n_samples from each of the 10 classes and saves the figure.
    """
    fig, axes = plt.subplots(10, n_samples, figsize=(n_samples * 2.5, 25))
    
    for i, cls in enumerate(EUROSAT_CLASSES):
        cls_dir = os.path.join(root_dir, cls)
        img_names = sorted([f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:n_samples]
        
        for j in range(n_samples):
            ax = axes[i, j]
            if j < len(img_names):
                img_path = os.path.join(cls_dir, img_names[j])
                img = Image.open(img_path)
                ax.imshow(img)
                if j == 0:
                    ax.set_ylabel(cls, rotation=0, labelpad=50, fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            
    plt.suptitle('EuroSAT Dataset Class Samples', fontsize=18, fontweight='bold', y=0.99)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "eurosat_class_samples.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved class sample visualization to {save_dir}/eurosat_class_samples.png")
