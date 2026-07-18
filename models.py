import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class BaselineCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
        # Adaptive pooling ensures the shape is always 4x4 before flattening
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Layer 1
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        # Layer 2
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        # Layer 3
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        return x


def get_resnet18_model(num_classes=10, pretrained=True):
    if pretrained:
        weights = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    else:
        model = resnet18(weights=None)
        
    # Replace classification head
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


class ResNet18Embedder(nn.Module):
    def __init__(self, fine_tuned_model):
        super(ResNet18Embedder, self).__init__()
        # Strip the last fc layer
        self.backbone = nn.Sequential(*list(fine_tuned_model.children())[:-1])
        
    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1) # Shape: (batch, 512)
        return features
